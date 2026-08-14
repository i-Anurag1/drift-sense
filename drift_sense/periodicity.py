"""
EXPERIMENTAL -- NOT INTEGRATED, DID NOT VALIDATE.

Self-Referential Periodicity Fingerprinting with Residual-Anomaly Confidence:
an attempted mechanism to distinguish a true site from decoy periodic
repeats by folding a candidate patch against its own estimated pitch and
scoring what fails to cancel out.

Honest result: tested against ground truth on the DRAM dev set, residual
energy at the true site was NOT reliably distinguishable from decoys
(e.g. true site 0.0722 vs. decoy 0.0732 -- higher). Sensor noise dominates
the signal at the scales this project operates at; folding two
independently-noisy patches together does not suppress noise enough to
reveal the underlying discriminating structure. Three variants were tried
(native scale, upsampled patches, axis-separated shifts) with no reliable
improvement.

This module is kept in the repo, unintegrated, as a documented negative
result and a starting point for future work (e.g. it may work better with
more shift samples, a learned denoiser prior to folding, or on real SEM
data with a different noise profile than the synthetic model used here).
It is NOT called by matcher.py, inference.py, or any other production path.
See docs/INVENTION_DISCLOSURE.md, section "Explored and rejected directions".
"""
import numpy as np
import cv2


def _1d_pitch(signal, min_pitch, max_pitch):
    signal = signal - signal.mean()
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode="full")[n - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-8)

    local_max = np.zeros_like(autocorr, dtype=bool)
    local_max[1:-1] = (autocorr[1:-1] >= autocorr[:-2]) & (autocorr[1:-1] >= autocorr[2:])

    valid = local_max.copy()
    valid[:min_pitch] = False
    if max_pitch < len(valid):
        valid[max_pitch:] = False

    if not valid.any():
        return 0.0, 0.0

    candidates = np.where(valid)[0]
    best = candidates[np.argmax(autocorr[candidates])]
    return float(best), float(autocorr[best])


def estimate_periodicity_vector(gray_img, min_pitch=8, max_pitch=150):
    """Estimate the dominant 2D periodicity (pitch_x, pitch_y) of an image
    directly from its own structure, with no prior pitch knowledge. Uses
    1D axis-projection autocorrelation (column-sum for pitch_x, row-sum for
    pitch_y) rather than full 2D autocorrelation, since axis-aligned grid/
    fin layouts give a far higher signal-to-noise ratio under projection
    than a raw 2D autocorrelation peak search, which is easily dominated by
    fine sensor-noise texture at small pitch."""
    img = gray_img.astype(np.float32)

    col_profile = img.sum(axis=0)
    row_profile = img.sum(axis=1)

    pitch_x, strength_x = _1d_pitch(col_profile, min_pitch, max_pitch)
    pitch_y, strength_y = _1d_pitch(row_profile, min_pitch, max_pitch)

    if strength_x < 0.15 and strength_y < 0.15:
        return None

    strength = max(strength_x, strength_y)
    return {"pitch_x": pitch_x if strength_x >= 0.15 else 0.0,
            "pitch_y": pitch_y if strength_y >= 0.15 else 0.0,
            "strength": strength}


def periodic_fold_residual(patch, pitch_x, pitch_y, n_shifts=2):
    """Fold a patch against its predicted periodicity vector and return the
    residual energy -- the fraction of the patch's energy that does NOT
    repeat according to the periodic lattice. High residual = locally
    unique / non-periodic content (candidate true site or real defect)."""
    if abs(pitch_x) < 1 and abs(pitch_y) < 1:
        return 0.0

    patch_f = patch.astype(np.float32)
    h, w = patch_f.shape
    accum = patch_f.copy()
    count = 1

    for k in range(1, n_shifts + 1):
        for sign in (1, -1):
            M = np.float32([[1, 0, sign * k * pitch_x], [0, 1, sign * k * pitch_y]])
            shifted = cv2.warpAffine(patch_f, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
            accum += shifted
            count += 1

    periodic_avg = accum / count
    residual = patch_f - periodic_avg

    total_energy = float(np.sum(patch_f ** 2)) + 1e-8
    residual_energy = float(np.sum(residual ** 2))
    return residual_energy / total_energy


def pncs_rerank(candidates, search_gray, ref_periodicity, patch_half=24, alpha=2.0):
    """Re-rank candidate (x, y, vote_weight) tuples using a combined
    Periodicity-Normalized Confidence Score: vote_weight * (1 + alpha *
    normalized_residual_energy). Returns candidates sorted best-first with
    their PNCS scores attached."""
    if ref_periodicity is None:
        return [(x, y, w, w) for x, y, w in candidates]

    pitch_x, pitch_y = ref_periodicity["pitch_x"], ref_periodicity["pitch_y"]
    h, w_img = search_gray.shape[:2]

    residuals = []
    for x, y, vote_weight in candidates:
        x0, y0 = int(max(0, x - patch_half)), int(max(0, y - patch_half))
        x1, y1 = int(min(w_img, x + patch_half)), int(min(h, y + patch_half))
        patch = search_gray[y0:y1, x0:x1]
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            residuals.append(0.0)
            continue
        residuals.append(periodic_fold_residual(patch, pitch_x, pitch_y))

    max_res = max(residuals) if residuals and max(residuals) > 0 else 1.0
    scored = []
    for (x, y, vote_weight), res in zip(candidates, residuals):
        norm_res = res / max_res
        pncs = vote_weight * (1 + alpha * norm_res)
        scored.append((x, y, vote_weight, pncs))

    scored.sort(key=lambda c: c[3], reverse=True)
    return scored

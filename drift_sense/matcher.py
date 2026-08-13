import numpy as np
import cv2
from dataclasses import dataclass, field
from skimage.feature import peak_local_max


@dataclass
class MatchResult:
    x: float
    y: float
    score: float
    scale: float
    angle: float
    ambiguous: bool
    n_candidates: int
    candidates: list = field(default_factory=list)


def _clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(img)


def _prep(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return _clahe(img)


def _rotate_template(tmpl, angle_deg):
    if angle_deg == 0:
        return tmpl
    h, w = tmpl.shape
    diag = int(np.ceil(np.hypot(h, w)))
    canvas = np.full((diag, diag), int(np.median(tmpl)), dtype=np.uint8)
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = tmpl
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                              borderValue=int(np.median(tmpl)))
    return rotated


def _orb_estimate(ref_gray, search_gray, top_k=500):
    orb = cv2.ORB_create(nfeatures=top_k)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(search_gray, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.78 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask.sum() < 8:
        return None
    h, w = ref_gray.shape
    center = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H)
    return float(proj[0, 0, 0]), float(proj[0, 0, 1]), int(mask.sum())


def _cluster_votes(votes, radius):
    clusters = []
    for x, y, score in votes:
        placed = False
        for c in clusters:
            if np.hypot(c["cx"] - x, c["cy"] - y) < radius:
                c["members"].append((x, y, score))
                total = sum(s for _, _, s in c["members"])
                c["cx"] = sum(px * s for px, py, s in c["members"]) / total
                c["cy"] = sum(py * s for px, py, s in c["members"]) / total
                c["weight"] = total
                placed = True
                break
        if not placed:
            clusters.append({"cx": x, "cy": y, "weight": score, "members": [(x, y, score)]})
    return clusters


def locate_reference_naive(reference_img, search_img,
                            scale_range=(0.055, 0.17), scale_steps=12,
                            angle_range=(-6, 6), angle_steps=5):
    """Single-best-peak baseline (no voting/clustering) — the classical
    template-matching approach this project is designed to improve on.
    Kept only to quantify how much the voting scheme in locate_reference
    helps on periodic layouts; not used by inference.py."""
    ref = _prep(reference_img)
    search = _prep(search_img)

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = np.linspace(angle_range[0], angle_range[1], angle_steps)

    best_score = -1.0
    best_xy = (search.shape[1] / 2, search.shape[0] / 2)

    for scale in scales:
        th, tw = int(ref.shape[0] * scale), int(ref.shape[1] * scale)
        if th < 6 or tw < 6 or th >= search.shape[0] or tw >= search.shape[1]:
            continue
        resized = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)

        for angle in angles:
            tmpl = _rotate_template(resized, angle)
            if tmpl.shape[0] >= search.shape[0] or tmpl.shape[1] >= search.shape[1]:
                continue
            corr = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(corr)
            if max_val > best_score:
                best_score = max_val
                best_xy = (max_loc[0] + tmpl.shape[1] / 2, max_loc[1] + tmpl.shape[0] / 2)

    return MatchResult(x=best_xy[0], y=best_xy[1], score=best_score,
                        scale=0.0, angle=0.0, ambiguous=False, n_candidates=1)


def locate_reference_ablation(reference_img, search_img,
                               use_multiscale=True, use_multiangle=True,
                               use_voting=True, use_periodicity=True,
                               scale_range=(0.055, 0.17), scale_steps=12,
                               angle_range=(-6, 6), angle_steps=5,
                               ambiguity_ratio=0.85, cluster_radius_frac=0.35):
    """Single configurable matcher used to isolate the contribution of each
    stage: plain NCC -> + multi-scale -> + multi-angle -> + voting ->
    + periodicity-aware disambiguation. See ablation.py."""
    ref = _prep(reference_img)
    search = _prep(search_img)

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps) if use_multiscale \
        else np.array([0.5 * (scale_range[0] + scale_range[1])])
    angles = np.linspace(angle_range[0], angle_range[1], angle_steps) if use_multiangle \
        else np.array([0.0])

    votes = []
    tmpl_sizes = []
    best_single = (-1.0, None, None)

    for scale in scales:
        th, tw = int(ref.shape[0] * scale), int(ref.shape[1] * scale)
        if th < 6 or tw < 6 or th >= search.shape[0] or tw >= search.shape[1]:
            continue
        resized = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)

        for angle in angles:
            tmpl = _rotate_template(resized, angle)
            if tmpl.shape[0] >= search.shape[0] or tmpl.shape[1] >= search.shape[1]:
                continue
            corr = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            tmpl_sizes.append(tmpl.shape)

            if use_voting:
                top_peaks = peak_local_max(corr, min_distance=max(5, min(tmpl.shape) // 3),
                                            num_peaks=3, exclude_border=False)
                for py, px in top_peaks:
                    score = float(corr[py, px])
                    if score <= 0:
                        continue
                    cy, cx = py + tmpl.shape[0] / 2, px + tmpl.shape[1] / 2
                    votes.append((cx, cy, score, scale, angle))
            else:
                _, max_val, _, max_loc = cv2.minMaxLoc(corr)
                if max_val > best_single[0]:
                    cx, cy = max_loc[0] + tmpl.shape[1] / 2, max_loc[1] + tmpl.shape[0] / 2
                    best_single = (max_val, cx, cy)

    if not use_voting:
        if best_single[1] is None:
            h, w = search_img.shape[:2]
            return MatchResult(w / 2, h / 2, 0.0, 1.0, 0.0, False, 0)
        return MatchResult(x=best_single[1], y=best_single[2], score=best_single[0],
                            scale=0.0, angle=0.0, ambiguous=False, n_candidates=1)

    if not votes:
        h, w = search_img.shape[:2]
        return MatchResult(w / 2, h / 2, 0.0, 1.0, 0.0, True, 0)

    avg_tmpl = np.mean([max(s) for s in tmpl_sizes]) if tmpl_sizes else 40
    radius = max(8, avg_tmpl * cluster_radius_frac)

    clusters = _cluster_votes([(x, y, s) for x, y, s, _, _ in votes], radius)
    clusters.sort(key=lambda c: c["weight"], reverse=True)

    top_weight = clusters[0]["weight"]

    if use_periodicity:
        strong = [c for c in clusters if c["weight"] >= ambiguity_ratio * top_weight]
        search_center = np.array([search.shape[1] / 2, search.shape[0] / 2])
        strong.sort(key=lambda c: np.hypot(c["cx"] - search_center[0], c["cy"] - search_center[1]))
        chosen = strong[0]
        ambiguous = len(strong) > 1
        n_candidates = len(strong)
    else:
        chosen = clusters[0]
        ambiguous = False
        n_candidates = 1

    best_vote = max(chosen["members"], key=lambda m: m[2])

    return MatchResult(
        x=chosen["cx"], y=chosen["cy"], score=best_vote[2],
        scale=0.0, angle=0.0, ambiguous=ambiguous, n_candidates=n_candidates,
    )


def locate_reference(reference_img, search_img,
                      scale_range=(0.055, 0.17), scale_steps=12,
                      angle_range=(-6, 6), angle_steps=5,
                      ambiguity_ratio=0.85, cluster_radius_frac=0.35,
                      verbose=False):
    ref = _prep(reference_img)
    search = _prep(search_img)

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = np.linspace(angle_range[0], angle_range[1], angle_steps)

    votes = []
    tmpl_sizes = []

    for scale in scales:
        th, tw = int(ref.shape[0] * scale), int(ref.shape[1] * scale)
        if th < 6 or tw < 6 or th >= search.shape[0] or tw >= search.shape[1]:
            continue
        resized = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)

        for angle in angles:
            tmpl = _rotate_template(resized, angle)
            if tmpl.shape[0] >= search.shape[0] or tmpl.shape[1] >= search.shape[1]:
                continue
            corr = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            tmpl_sizes.append(tmpl.shape)

            top_peaks = peak_local_max(corr, min_distance=max(5, min(tmpl.shape) // 3),
                                        num_peaks=3, exclude_border=False)
            for py, px in top_peaks:
                score = float(corr[py, px])
                if score <= 0:
                    continue
                cy, cx = py + tmpl.shape[0] / 2, px + tmpl.shape[1] / 2
                votes.append((cx, cy, score, scale, angle))

    if not votes:
        h, w = search_img.shape[:2]
        return MatchResult(w / 2, h / 2, 0.0, 1.0, 0.0, True, 0)

    avg_tmpl = np.mean([max(s) for s in tmpl_sizes]) if tmpl_sizes else 40
    radius = max(8, avg_tmpl * cluster_radius_frac)

    clusters = _cluster_votes([(x, y, s) for x, y, s, _, _ in votes], radius)
    clusters.sort(key=lambda c: c["weight"], reverse=True)

    top_weight = clusters[0]["weight"]
    strong = [c for c in clusters if c["weight"] >= ambiguity_ratio * top_weight]

    search_center = np.array([search.shape[1] / 2, search.shape[0] / 2])
    strong.sort(key=lambda c: np.hypot(c["cx"] - search_center[0], c["cy"] - search_center[1]))
    chosen = strong[0]

    best_vote = max(chosen["members"], key=lambda m: m[2])
    match_idx = [i for i, (x, y, s, sc, an) in enumerate(votes)
                 if abs(x - best_vote[0]) < 1e-6 and abs(y - best_vote[1]) < 1e-6]
    scale_used = votes[match_idx[0]][3] if match_idx else scales[0]
    angle_used = votes[match_idx[0]][4] if match_idx else 0.0

    ambiguous = len(strong) > 1

    result = MatchResult(
        x=chosen["cx"], y=chosen["cy"], score=best_vote[2],
        scale=float(scale_used), angle=float(angle_used),
        ambiguous=ambiguous, n_candidates=len(strong),
        candidates=[(c["cx"], c["cy"], c["weight"]) for c in clusters[:10]],
    )

    if verbose:
        est = _orb_estimate(ref, search)
        if est is not None:
            result.candidates.append(("orb_crosscheck",) + est)

    return result


def locate_reference_dl(reference_img, search_img, net,
                         scale_range=(0.055, 0.17), scale_steps=8,
                         angle_range=(-6, 6), angle_steps=3):
    """CNN feature-space localization: extracts the trained embedding
    network's dense feature map once for the search image, then correlates
    resized/rotated reference feature maps against it (same multi-scale
    search as the classical matcher, but matching in learned feature space
    instead of raw pixel intensity)."""
    from drift_sense.dl_features import extract_feature_map

    ref = _prep(reference_img)
    search = _prep(search_img)

    search_feat = extract_feature_map(search, net).astype(np.float32)

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = np.linspace(angle_range[0], angle_range[1], angle_steps)

    best_score = -1.0
    best_xy = (search.shape[1] / 2, search.shape[0] / 2)
    stride = net.conv1.stride * net.conv2.stride

    for scale in scales:
        th, tw = int(ref.shape[0] * scale), int(ref.shape[1] * scale)
        if th < stride * 4 or tw < stride * 4:
            continue
        resized = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)

        for angle in angles:
            tmpl = _rotate_template(resized, angle)
            tmpl_feat = extract_feature_map(tmpl, net).astype(np.float32)
            if (tmpl_feat.shape[0] >= search_feat.shape[0] or
                    tmpl_feat.shape[1] >= search_feat.shape[1] or
                    tmpl_feat.shape[0] < 2 or tmpl_feat.shape[1] < 2):
                continue

            cross = np.zeros(
                (search_feat.shape[0] - tmpl_feat.shape[0] + 1,
                 search_feat.shape[1] - tmpl_feat.shape[1] + 1), dtype=np.float32)
            for c in range(search_feat.shape[2]):
                cross += cv2.matchTemplate(search_feat[:, :, c], tmpl_feat[:, :, c], cv2.TM_CCORR)

            sq = (search_feat ** 2).sum(axis=2)
            window_energy = cv2.boxFilter(sq, ddepth=-1, ksize=(tmpl_feat.shape[1], tmpl_feat.shape[0]),
                                           normalize=False, anchor=(0, 0))
            window_energy = window_energy[:cross.shape[0], :cross.shape[1]]
            tmpl_norm = np.sqrt((tmpl_feat ** 2).sum()) + 1e-8
            corr = cross / (np.sqrt(np.maximum(window_energy, 0)) * tmpl_norm + 1e-8)

            max_val = float(corr.max())
            max_loc = np.unravel_index(np.argmax(corr), corr.shape)
            max_loc = (max_loc[1], max_loc[0])
            if max_val > best_score:
                best_score = max_val
                fx = (max_loc[0] + tmpl_feat.shape[1] / 2) * stride
                fy = (max_loc[1] + tmpl_feat.shape[0] / 2) * stride
                best_xy = (fx, fy)

    return MatchResult(x=best_xy[0], y=best_xy[1], score=float(best_score),
                        scale=0.0, angle=0.0, ambiguous=False, n_candidates=1)

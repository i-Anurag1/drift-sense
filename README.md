# Drift-Sense

AI/CV-powered Navigation-Error Recovery for wafer inspection tools. Given a high-magnification
**reference** image of a die site and a lower-magnification **search** image that contains that
same site shrunk ~10x somewhere inside it, Drift-Sense returns the pixel-accurate `(x, y)` center
of the matching region in the search image — even when the surrounding layout is a highly periodic
DRAM or FinFET array full of near-identical repeating structure.

Built for Applied Materials Problem Statement 2.

## How it works

1. **Multi-scale, multi-angle search.** The reference image is resized across a range of candidate
   scale factors (~0.055x-0.17x, centered on the nominal 10x demagnification) and rotated across a
   small angle range (±6°), and each variant is correlated against the search image with
   normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`).
2. **Multi-hypothesis voting.** Instead of trusting a single "best" scale/angle combination (which
   can overfit to noise in one particular trial and lock onto the wrong periodic repeat), every
   trial casts votes for its top local correlation peaks. Votes are spatially clustered, and the
   cluster with the largest total support wins — a location has to be a strong match *consistently
   across neighboring scales/rotations*, not just a one-off spike, which is exactly what defeats
   naive template matching on periodic DRAM/FinFET arrays.
3. **Periodicity-aware disambiguation.** If more than one vote cluster is nearly as strong as the
   winner (a genuinely ambiguous, highly periodic region), Drift-Sense reports
   `ambiguous_site_detected: true` and — per the spec — returns the cluster closest to the center
   of the search image.
4. **Optional ORB cross-check** (`verbose=True` in `locate_reference`) independently verifies the
   estimate with keypoint matching + RANSAC homography, useful for auditing low-confidence calls.

This is a classical/geometric computer-vision pipeline rather than a trained deep network. That is
a deliberate choice for this problem: DRAM/FinFET reference sites are extremely well-defined
geometric templates, not object classes with texture/semantic variation, so a well-designed
multi-hypothesis correlation search is more robust and more interpretable than a CNN trained on a
necessarily small, synthetic dataset — and it ships with zero risk of overfitting to synthetic
noise statistics that won't match Applied Materials' real held-out test set. See
`citations/CITATIONS.md` for the literature backing every noise/augmentation/matching choice.

## Repository layout

```
drift-sense/
├── README.md
├── requirements.txt
├── dataset_generator.py        # generates synthetic reference/search pairs + ground truth
├── inference.py                 # THE script Applied Materials will run on their test data
├── evaluate.py                  # self-evaluation + light-themed HTML report
├── drift_sense/
│   ├── structures.py             # DRAM / FinFET synthetic die-pattern generators
│   ├── degrade.py                # SEM noise, edge brightening, blur, rotation
│   └── matcher.py                # multi-scale/angle NCC + voting localization engine
├── citations/
│   └── CITATIONS.md              # references for every augmentation/noise/matching choice
└── data/, results/                # generated at runtime (gitignored)
```

## Setup

```bash
git clone <this-repo-url>
cd drift-sense
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

Tested on Python 3.10+.

## 1. Generate a synthetic dataset

```bash
python3 dataset_generator.py --style both --num-pairs 30 --output-dir ./data --seed 42
```

Arguments:

| Flag | Description |
|---|---|
| `--style` | `dram`, `finfet`, or `both` (alternates per pair) |
| `--num-pairs` | number of image pairs to generate (default 30) |
| `--output-dir` | output directory (default `./data`) |
| `--seed` | RNG seed for reproducibility |

Output:

```
data/
├── reference/<style>_<id>_reference.png
├── search/<style>_<id>_search.png
└── ground_truth.json     # per-pair true (x, y) center in the search image, plus style/scale/seed
```

## 2. Run localization on a single pair

```bash
python3 inference.py --reference data/reference/dram_000_reference.png \
                      --search data/search/dram_000_search.png \
                      --json --visualize ./results/dram_000_overlay.png
```

Output (JSON mode):

```json
{
  "x": 424.03,
  "y": 888.51,
  "score": 0.4251,
  "scale": 0.1081,
  "angle_deg": 3.0,
  "ambiguous_site_detected": false,
  "candidate_count": 1
}
```

`--visualize` optionally writes an annotated copy of the search image with the predicted center
marked. This is the exact script Applied Materials will run on the official test set — it takes
only a reference path and a search path and needs no manual edits.

## 3. Self-evaluate on the generated dataset

```bash
python3 evaluate.py --data-dir ./data --output ./results/report.html
```

This runs `locate_reference` on every generated pair, computes pixel-distance error against ground
truth, and writes a light-themed HTML report (`results/report.html`) with per-pair error, mean /
median / P90 error, ambiguous-site count, and average runtime — open it directly in a browser.

On the bundled 30-pair self-eval set (seed 42): **mean error ≈ 20px, median ≈ 19px** on a
1000×1000 search image (≈2% of frame width), with the periodicity-ambiguity flag correctly firing
on repeat-heavy regions.

## Design notes / failure-mode awareness

- The dataset generator intentionally reproduces the exact failure mode Applied Materials calls
  out: large regions of the search image are visually near-identical (same pitch, same via/fin
  pattern), so a purely local best-correlation approach can and does lock onto the wrong periodic
  repeat under heavy noise. The voting/clustering step in `matcher.py` is the direct response to
  that failure mode, and `ambiguous_site_detected` surfaces exactly when it's genuinely uncertain.
- Reference and search images use **independently seeded** noise generators (never the same noise
  applied to both), per the task's data-generation requirement.
- The search image is always degraded with more noise/blur than the reference image.
- All structural parameters (pitch, line widths, gate width, fiducial size) and every augmentation
  (noise model, edge brightening, blur, rotation) are cited in `citations/CITATIONS.md`.

## Extending

- `drift_sense/structures.py` — add new `generate_<style>_canvas()` functions and register them in
  `GENERATORS` to support additional device architectures.
- `drift_sense/matcher.py` — `locate_reference(..., verbose=True)` returns an ORB-based
  cross-check estimate in `MatchResult.candidates` for auditing.

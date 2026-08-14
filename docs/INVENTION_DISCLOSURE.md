# Invention Disclosure (Draft)

**Title:** Multi-Hypothesis Cross-Transform Consistency Voting for Disambiguating
Periodic Pattern Matches in Image-Based Navigation-Error Recovery

**Status:** Internal engineering draft, prepared to give a patent attorney a
concrete starting point. This document is **not legal advice**, does not
constitute a patentability opinion, and has not been checked against issued
patents. Semiconductor metrology and periodic-pattern image alignment is a
heavily patented field — including, plausibly, by Applied Materials itself —
and no claim here should be relied upon without a professional prior-art
search and review by a registered patent attorney or agent.

---

## 1. Field of the invention

Image-based site re-registration ("navigation-error recovery") on
semiconductor wafer inspection tools, specifically the problem of locating a
previously-recorded reference site inside a lower-magnification search image
when the surrounding die layout is highly periodic (e.g. DRAM word-line/
bit-line arrays, FinFET fin/gate arrays), such that many locations in the
search image are visually near-identical to the true site.

## 2. Background / problem with existing approaches

Classical template matching (normalized cross-correlation, phase correlation,
feature-keypoint matching such as SIFT/ORB) locates a reference pattern by
finding the single location in a search image with the highest similarity
score. On a periodic layout, this is unreliable: many locations spaced at
multiples of the device pitch produce correlation scores that are close to,
or under sensor noise can exceed, the score at the true site. A single-best-
peak matcher therefore has no principled way to avoid confidently reporting
the wrong repeat of the pattern — it cannot distinguish "this is definitely
the right place" from "this happened to score slightly higher this time."

We built and measured this failure directly. On a 30-pair synthetic
DRAM/FinFET dev set (see `evaluate.py` in this repository), a single-best-
peak multi-scale/multi-angle NCC matcher mislocalized the site by more than
100 pixels (i.e., landed on a different unit cell of the periodic array
entirely) on **8 of 30 pairs (27%)**. This is the specific, quantified
failure mode the invention addresses.

## 3. Summary of the invention

Rather than trusting the single highest-scoring correlation trial, the
matcher runs the same normalized cross-correlation search across a swept
grid of candidate geometric transforms (scale factors and rotation angles),
and instead of keeping only the global best score, it:

1. Extracts the top-K local correlation peaks from **every** transform trial
   in the sweep (not just the single best trial's single best peak).
2. Casts each peak as a spatial "vote" at its corresponding location in the
   search image, weighted by that peak's correlation score.
3. Spatially clusters votes that fall within a proximity radius derived from
   the template size, accumulating each cluster's total weighted support
   across all contributing transform trials.
4. Selects the match location as the cluster with the greatest **aggregate,
   cross-transform support**, rather than the single highest individual
   correlation value.
5. If multiple clusters carry near-equal aggregate support (a configurable
   ambiguity ratio), the location is flagged as ambiguous, and the tie is
   broken by proximity to the center of the search image — reflecting that,
   absent other evidence, the site closest to where the tool's navigation
   system expected to land is the more probable true site.

The key insight distinguishing this from generic multi-hypothesis matching
or standard non-max-suppression: **a location must be corroborated by
multiple independent, nearby-but-distinct geometric transform trials to be
trusted**, not merely by one favorable trial. On a periodic layout, a false
match at a decoy repeat typically wins outright at only one particular
(scale, angle) combination by chance (a noise-driven coincidence), whereas a
match at the true site tends to remain competitive across a neighborhood of
similar transforms, because the underlying image content that identifies it
is real and transform-stable, not a noise artifact. Requiring cross-trial
consensus is therefore a mechanism for statistically discounting
noise-driven false positives specifically in a periodic-matching context —
distinct from ensembling for its own sake, and distinct from post-hoc
non-max suppression on a single trial's correlation surface.

## 4. Detailed description / reduction to practice

Reference implementation: `drift_sense/matcher.py`, function
`locate_reference()` (production path) and `locate_reference_ablation()`
(instrumented variant used to isolate each stage's contribution).

```
for scale in scale_grid:                       # e.g. 12 steps over ~0.055x-0.17x
    resized_ref = resize(reference, scale)
    for angle in angle_grid:                    # e.g. 5 steps over +/-6 deg
        template = rotate(resized_ref, angle)
        corr_map = normalized_cross_correlation(search, template)
        for each of top-3 local maxima in corr_map:
            cast vote at (peak_x, peak_y) weighted by peak_score

cluster votes by spatial proximity (radius ~ template_size * 0.35)
rank clusters by summed weighted support
if top cluster's support is within ambiguity_ratio of the runner-up:
    flag ambiguous; choose whichever near-tied cluster is closest to
    the search image's center
else:
    choose the top cluster outright
```

### 4.1 Measured results supporting the mechanism

All numbers below are produced by scripts in this repository and are
reproducible (`ablation.py`, `evaluate.py`, `benchmark.py`).

**Ablation across 20 dev pairs** (`ablation.py`) — isolating exactly what
cross-transform voting contributes, holding the rest of the pipeline fixed:

| Stage | Mean err | Fails &gt;100px |
|---|---|---|
| Fixed scale/angle NCC (no search) | 13.2px | 0/20 |
| + multi-scale search, single best peak, no voting | 191.8px | 6/20 |
| + multi-scale + multi-angle search, still no voting | 149.7px | 5/20 |
| **+ cross-transform voting (this invention)** | **18.4px** | **0/20** |

This is the central empirical result: adding more transform search
*without* voting makes the matcher **worse**, not better (191.8px vs.
13.2px mean error) — because a wider search gives noise more chances to win
outright at some one favorable transform. Voting is what converts a wider
search from a liability into an asset, by requiring consensus rather than a
single lucky trial.

**Held-out generalization** (`benchmark.py`, 110 pairs, unseen random seed
never used during development): mean error 17.3px, **0/110 catastrophic
failures** (&gt;100px), on a 1000×1000-pixel search frame — i.e., the
mechanism never silently returned a wrong-unit-cell match on this set.

**Direct comparison to a single-best-peak baseline** (`evaluate.py`,
30 pairs): 20.1px mean error / 0 failures for the voting method vs. 170.9px
mean error / 8 failures (27%) for the otherwise-identical single-best-peak
matcher.

## 5. Candidate claim language (for attorney review, not a filed claim)

*Independent claim, method form (illustrative only):*

> A computer-implemented method for locating a reference image pattern
> within a search image containing periodically repeating structure,
> comprising: (a) computing, for each of a plurality of candidate
> geometric transforms of the reference image, a normalized cross-
> correlation between the transformed reference image and the search
> image; (b) identifying, for each candidate transform, a plurality of
> local maxima in the resulting correlation surface; (c) recording each
> local maximum as a weighted spatial vote at its corresponding location
> in the search image; (d) clustering the recorded votes by spatial
> proximity to form a plurality of candidate location clusters, each
> having an aggregate weight computed from votes contributed by multiple
> distinct candidate geometric transforms; (e) selecting, as the location
> of the reference pattern, the candidate location cluster having the
> greatest aggregate weight; and (f) responsive to a second-ranked
> candidate location cluster having an aggregate weight within a
> predetermined ratio of the greatest aggregate weight, designating the
> location as ambiguous and selecting among the near-tied clusters based
> on proximity to a reference point in the search image.

*Possible dependent claims (illustrative only):* the geometric transforms
comprising a swept range of scale factors corresponding to an expected
magnification ratio between the reference and search images; the geometric
transforms further comprising a swept range of rotation angles; the
reference point in (f) being the geometric center of the search image; the
method further comprising an independent keypoint-based verification pass
(e.g. ORB + RANSAC) used to audit low-confidence determinations; application
specifically to DRAM word-line/bit-line or FinFET fin/gate periodic device
structures in semiconductor wafer inspection.

**These are drafting starting points, not vetted claims.** Claim scope,
validity over prior art, and patentability are determinations for a patent
attorney following a professional prior-art search.

## 6. Known related art (non-exhaustive, not a legal search)

The individual techniques underlying this mechanism are independently well
established and are cited with sources in `citations/CITATIONS.md`:
normalized cross-correlation (Lewis, 1995), keypoint-based verification
(ORB: Rublee et al., 2011), and the general principle that multiple weak
hypotheses can be combined into a more reliable estimate (a very old idea in
computer vision generally, e.g. the Hough transform's use of accumulator-
based voting for line/shape detection, and RANSAC-style consensus voting for
outlier rejection). What may be novel, subject to a real prior-art search,
is the **specific combination and application**: using cross-*transform*
(not cross-feature or cross-frame) consensus, harvested from multiple local
peaks per trial across a swept scale/angle grid, specifically to
statistically discount decoy matches at repeat-distance offsets in a
periodic semiconductor device layout, with a defined ambiguity-detection and
center-proximity tie-break policy suited to a navigation-error-recovery
use case. A patent attorney's search should specifically check prior art in:
semiconductor overlay/CD-SEM metrology patents (Applied Materials, KLA,
ASML, and peers commonly hold patents in this exact space), general
multi-hypothesis tracking and Hough-voting literature, and Siamese/
correlation-filter visual tracking patents.

## 7. Explored and rejected directions

In the course of this project, an additional mechanism —
**self-referential periodicity fingerprinting with residual-anomaly
scoring** (`drift_sense/periodicity.py`) — was designed and implemented to
try to further improve disambiguation by estimating a reference image's
dominant pitch from its own autocorrelation and using it to fold candidate
search patches against their predicted periodic lattice, scoring the
residual that fails to cancel out as a "uniqueness confidence."

This was tested directly against ground truth rather than assumed to work.
**It did not reliably discriminate the true site from decoy repeats** —
residual energy at the true site was not consistently higher than at
decoys (e.g., in one measured case, 0.0722 at the true site vs. 0.0732 at a
decoy). The most likely cause: sensor noise dominates the residual signal
at the pitch/patch scales this project operates at, and folding two
independently-noisy patches together does not suppress noise sufficiently
to expose the underlying discriminating structure. The code is retained,
unintegrated, as a documented negative result and a starting point should
future work revisit it (e.g., with more fold samples, a denoising step
before folding, or on real SEM data with a different noise profile than the
synthetic model used in this project).

This section is included deliberately: an honest disclosure record of what
was tried and did not work is standard practice and protects against later
claims of undisclosed prior attempts.

## 8. Inventor / authorship note

This document was drafted with AI assistance (Claude, Anthropic) based on
work performed in this repository. It is a technical description of a
mechanism that was designed, implemented, and empirically validated as part
of this project — it is offered as a starting point for a human inventor
and patent attorney to review, refine, verify against prior art, and decide
whether and how to pursue formal IP protection. No claim of patentability,
novelty, or freedom-to-operate is made or implied by this document or by
Claude.

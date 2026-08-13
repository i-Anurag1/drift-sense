# Citations & Augmentation Justification

Every synthetic-data choice in `dataset_generator.py` and `drift_sense/degrade.py` is grounded in
published SEM imaging and semiconductor-metrology literature. This file maps each augmentation to
its justification.

## 1. Sensor noise model (`apply_sensor_noise`)

SEM micrographs are formed by counting secondary electrons, so they are dominated by shot noise
(signal-dependent, Poisson-distributed) with an additive Gaussian floor from the detector/amplifier
chain. We model noise as `0.35 * Poisson(signal) + Gaussian(0, sigma)`.

- A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian, "Practical Poisson-Gaussian Noise Modeling and
  Fitting for Single-Image Raw-Data," *IEEE Transactions on Image Processing*, 2008.
- M.T. Postek, A.E. Vladár, "Modeling for Quantitative and Accurate SEM Metrology," *NIST /
  Proc. SPIE Metrology, Inspection, and Process Control for Microlithography*.
- P. Brunner et al., "Impact of Image Noise on CD Measurement in CD-SEM," *Proc. SPIE
  Metrology, Inspection, and Process Control for Microlithography XX*, 2006.

We independently instantiate the noise generator (separate `numpy.random.Generator` seeds) for the
reference and search images, matching the task requirement that both are separate physical
acquisitions and per the noise-independence assumption used in the CD-SEM references above. The
search image uses a lower `poisson_scale` (more shot noise) and higher Gaussian sigma than the
reference image, consistent with the requirement that the lower-magnification / faster navigation
scan is noisier than the reference capture.

## 2. Edge brightening (`apply_edge_brightening`)

SEM images show characteristically bright rims at feature edges because the secondary-electron
escape probability rises sharply where the beam grazes a sidewall (the "edge effect"). We
approximate this with a Sobel-gradient-magnitude boost added back onto the image.

- L. Reimer, *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*,
  2nd ed., Springer Series in Optical Sciences, 1998 (edge-brightening / topographic contrast
  mechanism, Ch. 4-5).
- J. Goldstein et al., *Scanning Electron Microscopy and X-Ray Microanalysis*, 4th ed.,
  Springer, 2017 (secondary-electron yield vs. surface tilt/edge geometry).

## 3. Blur (`apply_blur`)

Finite probe size, astigmatism, and defocus impose a near-Gaussian point-spread function on the
recorded image. A small Gaussian blur (sigma randomized per image) reproduces this, with the search
image receiving a larger blur range because navigation/recovery frames are typically captured at
faster scan speeds and lower dwell time.

- L. Reimer, *Scanning Electron Microscopy*, Springer, 1998 (probe diameter and its convolution
  effect on image sharpness).
- J.S. Villarrubia, "Algorithms for Critical Dimension Measurement," *Journal of Research of NIST*,
  Vol. 106, 2001 (treats the SEM imaging system response as a blurring kernel for CD extraction).

## 4. Rotation (`apply_rotation`)

Small stage/tool orientation offsets accumulate between visits due to mechanical slack and
imperfect re-registration, which is exactly the "Navigation-Error Recovery" problem this project
targets. We apply an independent small random rotation (a few degrees) to both the reference and
search images to reflect this drift.

- M. Adel et al., "Diffraction Order Control in Overlay Metrology," *Proc. SPIE Metrology,
  Inspection, and Process Control for Microlithography*, 2008 (overlay/registration error budget
  including rotational misalignment between tool visits).
- G.E. McGuire, *Semiconductor Materials and Process Technology Handbook*, Noyes, 1988
  (stage-repeatability and thermal-drift error sources in wafer inspection tools).

## 5. Scale / magnification jitter (`downsample` randomization in `dataset_generator.py`)

Magnification calibration on a real tool is not perfectly deterministic between sessions, so the
same nominal "10x" demagnification varies slightly pair to pair (we randomize the downsample factor
in the 8x-10.5x range rather than fixing it at exactly 10x).

- M.T. Postek, A.E. Vladár, "Modeling for Quantitative and Accurate SEM Metrology," NIST
  (magnification calibration error and its propagation into CD/registration measurements).
- Semiconductor Equipment and Materials International (SEMI) E89 guideline on SEM magnification
  calibration procedures (industry practice motivating why magnification is treated as a
  calibrated-but-imperfect parameter).

## 6. Periodic device structures (DRAM word-line/bit-line grid, FinFET fin/gate array)

- S.M. Sze, K.K. Ng, *Physics of Semiconductor Devices*, 3rd ed., Wiley, 2007 (DRAM cell array
  layout: orthogonal word-lines/bit-lines with a storage-node contact at each intersection).
- C. Auth et al., "A 22nm High Performance and Low-Power CMOS Technology Featuring Fully-Depleted
  Tri-Gate Transistors," *IEEE International Electron Devices Meeting (IEDM)*, 2012 (parallel fin
  arrays crossed by orthogonal gate lines, the canonical FinFET layout motif we replicate).

## 7. Site-identifying fiducial

Real re-navigation on a periodic array only works because the target die/cell site is
distinguished from its neighbors by some non-repeating local feature (an alignment mark, a
boundary of the periodic block, a unique via/defect, etc.) — otherwise the site genuinely cannot be
disambiguated from an image alone, periodic or not. We inject a small L-shaped fiducial mark at the
true target site to stand in for this real-world uniqueness cue while keeping ground truth exact
and machine-checkable.

- M. Adel et al., "Diffraction Order Control in Overlay Metrology," *Proc. SPIE*, 2008
  (use of dedicated fiducial/alignment targets for site re-registration in wafer metrology tools).
- P. Leray et al., "Overlay Metrology for Double Patterning Processes," *Proc. SPIE Advanced
  Lithography*, 2010 (fiducial-mark-based registration in high-density periodic layouts).

## 8. Matching algorithm design choices

- J.P. Lewis, "Fast Normalized Cross-Correlation," *Vision Interface*, 1995 (basis for the
  `TM_CCOEFF_NORMED` multi-scale/rotation search used in `drift_sense/matcher.py`).
- E. Rublee, V. Rabaud, K. Konolige, G. Bradski, "ORB: An Efficient Alternative to SIFT or SURF,"
  *IEEE International Conference on Computer Vision (ICCV)*, 2011 (keypoint cross-check used as an
  optional independent verification pass, `--visualize`/`verbose` mode).
- D.G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *International Journal
  of Computer Vision*, 2004 (general justification for local, distinctive-feature-based
  disambiguation of repeated structure, motivating the multi-hypothesis voting/clustering scheme
  used to resist periodic false matches).

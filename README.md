# Pitch-locked self-repetition in wild common dolphin whistles

Reproducibility code for "A chorus, not a conversation" — analysis of self-repetition
patterns in wild common dolphin (*Delphinus delphis*) whistles, with cross-species
replication on common marmoset (*Callithrix jacchus*) phee calls.

## Key findings

- Dolphins repeat their own whistle shapes at 1.21x the rate expected by chance
  (within-minute-class null, p < 0.001)
- The same framework applied to marmosets finds 1.17x (p < 0.001) — across
  85 million years of evolutionary separation
- The triangle statistic (exact pitch match) shows self-repeat is pitch-locked,
  not just shape-similar
- A null model designed to test turn-taking (Null B) returns 1.00 on data where
  turn-taking is undeniable at z = +18 — the instrument is blind to the phenomenon

## Data sources

This repository contains analysis code only. The two public datasets must be downloaded separately;
everything else (feature files, DTW matrices, shape classes) is regenerated from them by `dolphin/p00_prepare.py`.

### Dolphin data
**Lehnhoff et al. 2025** — *A dataset of acoustic recordings of wild free-ranging short-beaked common dolphins*
(DOLPHINFREE project, Bay of Biscay; Earth System Science Data 17, 4495–4509, doi:10.5194/essd-17-4495-2025).
- Zenodo concept DOI: [10.5281/zenodo.14637674](https://doi.org/10.5281/zenodo.14637674) (always resolves to the latest version;
  at the time of writing v3, "Improved behavioural data", DOI [10.5281/zenodo.15681697](https://doi.org/10.5281/zenodo.15681697)).
- The deposit is one 39.9 GB archive, `DOLPHINFREE_public.zip`. Only two folders are used (about 7 MB):
  `Single_hydrophone/Whistle_annotations/` (whistle contour JSONs) and `Single_hydrophone/Visual_observation/` (the three `.xlsx`
  tables). The audio and the Tetra array data are not needed. You can extract just those folders from the archive
  (`unzip DOLPHINFREE_public.zip 'DOLPHINFREE_public/Single_hydrophone/Whistle_annotations/*' 'DOLPHINFREE_public/Single_hydrophone/Visual_observation/*'`).
- Point `DOLPHINFREE_DIR` at the extracted folder that contains `Single_hydrophone/`.

### Marmoset data
**Grijseels, Fairbank & Miller 2024** — *A model of marmoset monkey vocal turn-taking* (Dryad, 1.7 MB zip).
- Dryad DOI: [10.5061/dryad.9ghx3ffpx](https://doi.org/10.5061/dryad.9ghx3ffpx)
- Extract it so that `data/marmoset/vocal_turn_taking_data/` exists (containing `metadata/`, `one_monkey/`, `two_monkey/`,
  `three_monkey/`), or point `MARMOSET_RAW_DIR` at that folder.

## Setup

Tested with Python 3.13 (numpy 2.5, scipy 1.18, pandas 3.0, scikit-learn 1.9, matplotlib 3.11, dtaidistance 2.5). Any recent
Python 3 with the versions in `requirements.txt` should work; the DTW step uses a compiled extension and
parallelises across cores.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Where files live

Nothing needs to be edited: all paths default to locations inside the repository and can be overridden with environment variables.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DOLPHINFREE_DIR` | *(required for p00)* | extracted Zenodo deposit (folder containing `Single_hydrophone/`) |
| `DOLPHIN_DATA_DIR` | `data/dolphin` | derived dolphin inputs and JSON results |
| `DOLPHIN_FIG_DIR` | `figures` | figure output |
| `MARMOSET_RAW_DIR` | `data/marmoset/vocal_turn_taking_data` | extracted Dryad dataset |
| `MARMOSET_DATA_DIR` | `data/marmoset` | derived marmoset tables and JSON results |
| `MARMOSET_FIG_DIR` | `figures` | figure output |

After the full run the data directories contain:

```
data/dolphin/
  whistles_obs.csv          p00  one row per whistle (timing, frequency summary, session, visual-observation metadata)
  features.pkl              p00  resampled contours (X, Xs, Xk) and raw contours (C)
  dtw_abs.npy               p00  pairwise DTW distance, absolute-kHz contours
  dtw_shape.npy             p00  pairwise DTW distance, z-scored shape
  v02_spectral_labels.csv   p00  four shape classes (arch / rise / fall / U)
  gap_vs_null.csv           p01  overlap/gap histogram vs null A (used by m06)
  d0*_*.json, d05_condition_encounters.csv   results of the d-scripts
data/marmoset/
  calls.csv, calls2.csv, sessions.csv        m01
  m0*_*.json, m0*_*.csv                      results of the m-scripts
figures/                                     PNG/PDF outputs
```

## Running the analysis

Run everything from the repository root. The dolphin pipeline goes first (`m06` reads dolphin results); each block
is independent otherwise. Timings are for a 16-core machine.

### Dolphin analysis

```bash
export DOLPHINFREE_DIR=/path/to/DOLPHINFREE_public      # folder containing Single_hydrophone/
cd dolphin

# p00: build every derived input from the Zenodo deposit (contours, observations, DTW matrices, shape classes)   ~10 s
python p00_prepare.py
# p01: gap histogram vs null A (used by the cross-species figure)                                                 <1 s
python p01_gap_null.py

# d01: Self-repeat decomposition — shape match at short lag vs long lag                                            ~25 s
python d01_repeats.py
# d02: Decomposition by shape class, frequency offset, duration ratio                                              ~3 min
python d02_repeats_decomp.py
# d03: Triangle statistic — exact pitch-locked pairs vs transposed                                                 ~5 s
python d03_triangles.py
# d04: Within-minute null (the primary result), threshold sweep, lead-figure triangle counts, article figures
python d04_renull.py                                                                                               # ~5 s
python d04_threshold.py                                                                                            # ~1.5 min
python d04_triangles.py       # must run before d04_figures.py
python d04_figures.py
# d05: Condition analysis (behaviour, group size, distance) and Janik 2-3 s window
python d05_condition.py
python d05_lag23.py
```

### Marmoset replication

```bash
cd marmoset        # data/marmoset/vocal_turn_taking_data/ must exist (or set MARMOSET_RAW_DIR)

python m01_load.py             # parse Dryad .mat files, build calls.csv / calls2.csv / sessions.csv
python m02_gaps.py             # gap distributions, inter-onset intervals
python m03_selfrepeat.py       # self-repeat analysis (mirrors dolphin d01/d04)
python m03b_localnull.py       # local (within-minute) null                 (~2 min)
python m03c_exact_local.py     # exact pitch match, local null
python m04_overlap.py          # call overlap analysis
python m05_pointprocess.py     # point-process models (Hawkes, burstiness, Fano factor)
python m06_crossspecies.py     # cross-species figures — needs the dolphin steps above (d04_renull, p01_gap_null)
```

### Reproducibility check

The whole chain — from the raw Zenodo/Dryad files to every JSON result — was re-run from a fresh clone into empty
output directories (Python 3.13, package versions above). All dolphin JSON results (`d02`–`d05`, `gap_vs_null.csv`) and the marmoset
results of `m03`–`m05` were byte-identical to the original analysis runs; the derived inputs (`dtw_*.npy`, `features.pkl`,
shape classes, `whistles_obs.csv` columns) matched exactly. The only difference is the Monte-Carlo null table in
`m02_gaps.json`, which agrees to within simulation noise (differences of about 0.3 counts in bins of 30–800).
Headline numbers to check: `d04_renull` file-level (within-minute) pair ratio 1.21x; `m03c_exact_local` 5–10 s ratio 1.17x.

## Script reference

### Dolphin (`dolphin/`)

| Script | Purpose |
|--------|---------|
| `p00_prepare.py` | Zenodo deposit → `whistles_obs.csv`, `features.pkl`, `dtw_abs.npy`, `dtw_shape.npy`, `v02_spectral_labels.csv` |
| `p01_gap_null.py` | Gap histogram vs burst-preserving null A → `gap_vs_null.csv` |
| `d01_lib.py` | Shared loader, whistle classification, plotting style |
| `d01_repeats.py` | Self-repeat at short lag — shape match excess over long-lag null |
| `d02_repeats_decomp.py` | Decomposition by shape class, frequency offset, duration ratio |
| `d03_triangles.py` | Triangle statistic: exact vs transposed pitch-locked pairs |
| `d04_lib.py` | Utilities for the within-minute null framework |
| `d04_renull.py` | Within-minute-class null — the primary result (1.21x excess) |
| `d04_threshold.py` | Sensitivity: sweep DTW threshold, verify stability |
| `d04_triangles.py` | Triangle counts for the lead figure (exact vs transposed triples, two nulls) → `d04_triangles.json` |
| `d04_figures.py` | Article-quality figures (PNG 300dpi + PDF) |
| `d05_condition.py` | Does repeat rate track behaviour, group size, or distance? (No) |
| `d05_lag23.py` | Janik's 2-3s signature whistle window — no pitch-shifted excess |

### Marmoset (`marmoset/`)

| Script | Purpose |
|--------|---------|
| `m00_lib.py` | Shared loader for Grijseels .mat files, plotting style |
| `m01_load.py` | Parse Dryad dataset, version selection, build calls.csv |
| `m02_gaps.py` | Inter-call gap distributions, IOI analysis |
| `m03_selfrepeat.py` | Self-repeat replication (mirrors d01/d04 framework) |
| `m03b_localnull.py` | Local null: within-minute blocks |
| `m03c_exact_local.py` | Exact self-repeat with local null (1.17x excess) |
| `m04_overlap.py` | Call overlap quantification |
| `m05_pointprocess.py` | Hawkes process, burstiness index, Fano factor |
| `m06_crossspecies.py` | Cross-species comparison figures |

## Method

The core method is **type-free pitch matching**: instead of classifying whistles into
discrete types (signature whistles, etc.), we compute pairwise DTW distances on
pitch contours and ask whether short-lag pairs (< 10s) are more similar than
expected from same-session, same-minute-class baselines.

The null model shuffles whistle identities within 60-second windows of the same
shape class — controlling for temporal autocorrelation in repertoire use while
testing whether specific contours recur beyond chance.

The triangle statistic tests whether recurring shapes are pitch-locked (same absolute
frequency) or pitch-shifted (same shape, different register). The answer is:
pitch-locked. This points toward self-repetition (same animal) rather than
vocal matching (different animal copying the shape).

## Citation

If you use this code:

```bibtex
@software{squeak2026dolphin,
  author = {Squeak},
  title = {Pitch-locked self-repetition in wild common dolphin whistles},
  year = {2026},
  url = {https://github.com/squeak-ai/dolphin-self-repeat}
}
```

## License

MIT — see [LICENSE](LICENSE).

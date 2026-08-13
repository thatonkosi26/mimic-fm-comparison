# MIMIC-III Time-Series Foundation Model Comparison

Code for _Evaluating Time-Series Foundation Models for Early Warning in
Intensive Care Units_ (Wits BSc Honours dissertation, Big Data Analytics).

**Status: Chapter 3 (Sections 3.2–3.6) is complete.** All five
task-specific models, plus Chronos zero-shot and fine-tuned, have been
trained and evaluated on both imputation conditions. Full evaluation
(bootstrap CIs, McNemar imputation-sensitivity tests) is done. See
[PROGRESS.md](PROGRESS.md) for the complete stage-by-stage log — exact
commands run, real output numbers, every issue hit and how it was fixed.

## Folder structure

```
mimic-fm-comparison/
├── config.py                    # all paths + shared constants
├── requirements.txt
├── LICENSE                       # MIT (code only — see DATA_ACCESS.md for MIMIC-III itself)
├── DATA_ACCESS.md                # how to get your own credentialed MIMIC-III access
├── PROGRESS.md                   # full run log: commands, real numbers, issues + fixes
├── .gitignore                    # excludes all MIMIC-III data, per data use agreement
│
├── data/
│   ├── raw/                      # (gitignored) mimic3-benchmarks output, if pointed here
│   └── processed/                # this pipeline's output
│       ├── forward_fill/         # Condition A (Harutyunyan-style)
│       └── linear_interp/        # Condition B
│
├── preprocessing/
│   ├── imputation.py             # Condition A / B implementations
│   ├── feature_extraction.py     # 102-dim static features for LR/RF/XGBoost
│   └── build_dataset.py          # orchestrates preprocessing end to end
│
├── models/
│   ├── baselines.py              # LR / Random Forest / XGBoost (Section 3.5.1)
│   ├── lstm.py                   # channel-wise LSTM (Section 3.5.2)
│   ├── tft.py                    # Temporal Fusion Transformer (Section 3.5.2)
│   ├── chronos_eval.py           # Chronos zero-shot (Section 3.5.3)
│   └── chronos_finetune.py       # Chronos fine-tuned (Section 3.5.3)
│
├── evaluation/
│   ├── metrics.py                # AUROC/AUPRC/F1/ECE, bootstrap CI, McNemar
│   └── evaluate.py               # pulls all 7 models' predictions, computes everything (Section 3.6)
│
├── results/                      # gitignored — all model outputs, predictions, final tables
│   ├── baselines/<condition>/
│   ├── lstm/<condition>/
│   ├── tft/<condition>/
│   ├── chronos/<condition>/      # zero-shot + fine-tuned outputs, plus checkpoints/
│   └── evaluation/                # full_results.csv, imputation_sensitivity.csv, summary.md
│
├── scripts/
│   ├── decompress_mimic.py       # one-time .csv.gz -> .csv step
│   ├── find_bad_events_csv.py    # diagnostic: scan events.csv files for parse errors
│   ├── verify_benchmark_output.py # sanity-check mimic3-benchmarks output before the full build
│   └── benchmark_chronos.py      # times Chronos model sizes on your hardware before committing
│
└── tests/
    └── test_pipeline_output.py   # shape/leakage/NaN checks on build_dataset.py output
```

## Step 0 — prerequisites

- CITI training + PhysioNet data use agreement (see [DATA_ACCESS.md](DATA_ACCESS.md)).
- MIMIC-III v1.4 downloaded. If your files look like
  `...\mimic-iii-clinical-database-1.4\...\PRESCRIPTIONS.csv.gz`, you have
  the gzipped CSV distribution (not Postgres) — that's what this pipeline
  assumes.
- Python 3.13, Windows. `pip install -r requirements.txt`.

### Windows / gzip note

`mimic3-benchmarks`'s extraction scripts read plain `.csv`, not `.csv.gz`:

```powershell
python scripts\decompress_mimic.py
```

Reads `config.RAW_MIMIC_GZ_DIR`, writes plain CSVs to
`config.RAW_MIMIC_CSV_DIR`. One-time step; large tables (CHARTEVENTS,
LABEVENTS) are several GB each — budget a few minutes and ~2x the
compressed size in free disk space.

### Windows / OpenMP note (needed before running any PyTorch/transformers script)

Running the Chronos scripts can trigger a silent crash (no traceback,
process just exits) caused by a duplicate OpenMP runtime conflict
between torch/transformers/numpy on Windows. Always set this first:

```powershell
set KMP_DUPLICATE_LIB_OK=TRUE
```

Consider setting it as a permanent Windows environment variable rather
than re-typing it every session.

## Step 1 — cohort extraction (use the existing benchmark repo, don't rebuild it)

```bash
git clone https://github.com/YerevaNN/mimic3-benchmarks.git
cd mimic3-benchmarks
python -m mimic3benchmark.scripts.extract_subjects {MIMIC_III_CSV_DIR} data/root/
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3models.split_train_val data/in-hospital-mortality/
```

Reproduces the exact Harutyunyan et al. (2019) cohort, 17 variables, and
train/val/test split. Long-running (multiple hours for the largest
tables) — see PROGRESS.md Stage 2 for what to expect and how the two
interruptions that happened here were diagnosed and resolved.

Before running the full build in Step 2, verify the output matches what
`config.VARIABLES` expects:

```powershell
python scripts\verify_benchmark_output.py
```

## Step 2 — preprocessing

1. Edit `config.py`: set `BENCHMARK_ROOT` to Step 1's
   `data/in-hospital-mortality/` output.
2. `python preprocessing\build_dataset.py`
3. `python -m pytest tests\test_pipeline_output.py -v` — confirms shapes,
   no NaNs, no train/val/test subject leakage, plausible mortality rates.

Produces, for **each** imputation condition:

```
data/processed/<condition>/train/sequences.npy      (N, 48, 17) — LSTM/TFT/Chronos
data/processed/<condition>/train/static_feats.npy   (N, 102)    — LR/RF/XGBoost
data/processed/<condition>/train/labels.npy         (N,)
data/processed/<condition>/train/subject_ids.npy    (N,)
... same for val/ and test/
data/processed/<condition>/norm_stats.npz
```

## Step 3 — train all models

```powershell
python models\baselines.py          # LR/RF/XGBoost, both conditions (~minutes)
python models\lstm.py               # channel-wise LSTM (~1-2 hours CPU)
python models\tft.py                # TFT, 4-config grid search (~hours CPU)
python models\chronos_eval.py       # Chronos zero-shot (~18 hours CPU, Chronos-Small)
python models\chronos_finetune.py   # Chronos fine-tuned (~multi-day CPU, checkpointed)
```

For the two Chronos scripts specifically:

- Run with `CHRONOS_QUICK_MODE=1` first (small subset) to verify
  correctness and get a real timing estimate on your hardware before
  committing to the full run.
- Both checkpoint automatically (per-channel for zero-shot, per-batch
  for fine-tuning) — if interrupted, just re-run the same command and it
  resumes rather than restarting.
- `scripts\benchmark_chronos.py` times mini/small/base on your machine
  if you want to reconsider the model-size choice.

## Step 4 — evaluate everything

```powershell
python evaluation\evaluate.py
```

Loads every model's saved val/test predictions, computes AUROC/AUPRC/F1/
ECE with bootstrap CIs (n=1000), and runs the McNemar
imputation-sensitivity test two ways (own-threshold, per Section 3.6.3's
literal spec, and shared-threshold, isolating genuine score-level
disagreement from threshold-selection artifacts). Output:

```
results/evaluation/full_results.csv
results/evaluation/imputation_sensitivity.csv
results/evaluation/full_results_summary.md
```

## Results summary

All seven model configurations, test set, both imputation conditions —
full numbers and confidence intervals in
`results/evaluation/full_results.csv` (see PROGRESS.md Stage 10 for the
complete table):

| Model                | Test AUROC (forward_fill / linear_interp) |
| -------------------- | ----------------------------------------- |
| Logistic regression  | 0.810 / 0.816                             |
| Random Forest        | 0.825 / 0.830                             |
| XGBoost              | 0.833 / 0.837                             |
| LSTM                 | 0.828 / 0.822                             |
| TFT                  | 0.826 / 0.816                             |
| Chronos (zero-shot)  | 0.769 / 0.773                             |
| Chronos (fine-tuned) | 0.776 / 0.777                             |

**Headline findings:**

- Zero-shot Chronos clearly underperforms every task-specific model
  (~0.77 vs ~0.81–0.84 AUROC), directly reproducing Gu et al. (2025) and
  Rockenschaub et al. (2024)'s finding that foundation models don't
  transfer well to clinical time-series zero-shot — now shown on the
  standard MIMIC-III mortality benchmark specifically.
- Fine-tuning gives only a marginal improvement (<1 percentage point)
  and does not close the gap to task-specific models, under the
  resource-constrained fine-tuning budget used here (3 epochs,
  Chronos-Small, CPU-only).
- AUROC is essentially robust to imputation strategy across every model
  (deltas under 0.01). However, individual classification decisions are
  not: 5 of 7 models show genuine, threshold-independent sensitivity to
  imputation choice; 2 (logistic regression, and borderline LSTM) have
  their apparent sensitivity substantially explained by
  threshold-selection shifts rather than the underlying risk scores.

## Design decisions worth knowing before you use this in your writeup

- **Normalisation and imputation fallback means are fit on train only**,
  applied unchanged to val/test (Section 3.3; protects against leakage).
- **The 102 static features are computed on the imputed grid**, so
  traditional ML and deep models see a consistent underlying
  representation — differences trace to the model, not the data view.
- **Missingness fraction** (17 of 102 features) comes from the
  observation mask _before_ imputation, preserving the "was this ever
  measured" signal (Section 2.6's informative-missingness point).
- **TFT is natively a forecaster, not a classifier** — adapted via
  CrossEntropy loss (`output_size=2`) over a 47-hour encoder + 1-step
  target. See `models/tft.py`'s docstring for the full rationale.
- **Chronos zero-shot uses sampling** to estimate the predictive
  distribution's mean/variance (matching the proposal's literal
  wording); **Chronos fine-tuning uses an analytic (softmax-expectation)
  version instead**, since sampling isn't differentiable and the encoder
  needs gradient updates. Verified via a local test that gradients from
  this analytic path genuinely reach every encoder parameter. See
  `models/chronos_finetune.py`'s docstring.
- **Model sizes and hyperparameter grids were chosen under real CPU-only
  constraints** (no GPU access) — Chronos-Small over Large, small TFT/RF
  search grids, 3-epoch Chronos fine-tuning budget. Each choice is
  benchmarked and documented at the point it's made, not asserted
  without justification — see PROGRESS.md for the reasoning and numbers
  behind each one.
- GCS fields are label-encoded from their ordinal string form
  (`preprocessing/imputation.py`) — double-check this matches your
  actual benchmark CSV format if you re-run extraction with a different
  `mimic3-benchmarks` version.

## Reproducibility

All random seeds fixed via `config.RANDOM_SEED` (42), used consistently
across numpy, torch, and sklearn `random_state` arguments throughout.
Full run log with exact commands and real output at every stage:
[PROGRESS.md](PROGRESS.md).

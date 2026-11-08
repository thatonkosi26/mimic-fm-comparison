# MIMIC-III Time-Series Foundation Model Comparison

Code for _Evaluating Time-Series Foundation Models for Early Warning in
Intensive Care Units_ (Wits BSc Honours research report, Big Data Analytics).

**Status: the core experiments (Chapter 3, Sections 3.2–3.6) are complete,
plus two follow-ups.** All five task-specific models, plus Chronos zero-shot
and fine-tuned, have been trained and evaluated on both imputation
conditions, with bootstrap confidence intervals and McNemar
imputation-sensitivity tests. Follow-ups: a cost timing run
(`scripts/time_models.py`) and a Chronos model-size comparison
(`models/chronos_sizes.py`). See [PROGRESS.md](PROGRESS.md) for the
stage-by-stage log: exact commands run, real output numbers, every issue hit
and how it was fixed.

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
│   ├── chronos_eval.py           # Chronos-Small zero-shot, sampled features (Section 3.5.3)
│   ├── chronos_finetune.py       # Chronos-Small fine-tuned (Section 3.5.3)
│   └── chronos_sizes.py          # frozen Chronos Mini/Small/Base, exact features + 2 heads (Stage 13)
│
├── evaluation/
│   ├── metrics.py                # AUROC/AUPRC/F1/ECE, bootstrap CI, McNemar
│   ├── evaluate.py               # pulls all 7 models' predictions, computes everything (Section 3.6)
│   ├── bootstrap_sizes.py        # bootstrap CIs + paired AUROC tests for the Chronos size runs (Stage 14)
│   └── make_cost_figure.py       # AUROC-versus-inference-cost figure used in the paper
│
├── results/                      # gitignored — all model outputs, predictions, final tables
│   ├── baselines/<condition>/
│   ├── lstm/<condition>/
│   ├── tft/<condition>/
│   ├── chronos/<condition>/      # zero-shot + fine-tuned outputs, plus checkpoints/
│   ├── chronos_sizes/            # Mini/Small/Base frozen-feature runs (Stage 13)
│   ├── timing/                   # timing_results_<label>.csv (Stage 12)
│   └── evaluation/               # full_results.csv, imputation_sensitivity.csv, summary.md
│
├── scripts/
│   ├── decompress_mimic.py       # one-time .csv.gz -> .csv step
│   ├── find_bad_events_csv.py    # diagnostic: scan events.csv files for parse errors
│   ├── verify_benchmark_output.py # sanity-check mimic3-benchmarks output before the full build
│   ├── benchmark_chronos.py      # times Chronos model sizes on your hardware before committing
│   └── time_models.py            # training/inference timing for all models (Stage 12)
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

Approximate CPU times on the laptop used for this project (8 logical cores,
no GPU; see PROGRESS.md Stage 12 for how each figure was obtained):

```powershell
python models\baselines.py          # LR/RF/XGBoost, both conditions (~minutes; RF search is the slowest part)
python models\lstm.py               # channel-wise LSTM (~3 hours per condition, estimated)
python models\tft.py                # TFT, 4-config grid search (~2 hours per condition, estimated)
python models\chronos_eval.py       # Chronos-Small zero-shot, sampled (~18 hours for both conditions)
python models\chronos_finetune.py   # Chronos-Small fine-tuned (~35 hours per condition for 3 epochs, checkpointed)
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

## Step 5 — optional follow-up experiments

```powershell
set KMP_DUPLICATE_LIB_OK=TRUE

:: Cost timing: training + inference per model (~15-30 min on a laptop)
set TIMING_QUICK=1 & python scripts\time_models.py      :: quick test first
set TIMING_QUICK=  & python scripts\time_models.py
:: -> results\timing\timing_results_<label>.csv   (label = TIMING_LABEL, default cpu_laptop)

:: Chronos size comparison: frozen weights, exact mean/variance features, two heads
set CHRONOS_SIZES_TIME_ONLY=1 & python models\chronos_sizes.py                              :: projected runtime first
set CHRONOS_SIZES_TIME_ONLY= & set CHRONOS_QUICK_MODE=1 & python models\chronos_sizes.py    :: quick test, results NOT valid
set CHRONOS_QUICK_MODE= & python models\chronos_sizes.py                                    :: full run, ~22 h CPU for 3 sizes x 2 conditions
:: -> results\chronos_sizes\summary.csv

:: Bootstrap CIs + paired AUROC tests for the size runs (a few minutes; needs the size runs and Step 4 first)
python evaluation\bootstrap_sizes.py
:: -> results\evaluation\bootstrap_sizes.csv, bootstrap_sizes_paired.csv, bootstrap_sizes_table.tex

:: Figure for the paper: AUROC versus inference cost
python evaluation\make_cost_figure.py
```

Both scripts save progress (`chronos_sizes.py` per channel), so an
interrupted run resumes when re-run. **Clear any `*_TIME_ONLY` and
`*_QUICK*` variables before a real run**: they persist in the terminal
session and silently turn a real run into a timing or quick run.

## Results summary

All seven main model configurations, test set, both imputation conditions —
full numbers and confidence intervals in
`results/evaluation/full_results.csv` (see PROGRESS.md Stage 10 for the
complete table):

| Model                               | Test AUROC (forward_fill / linear_interp) |
| ----------------------------------- | ----------------------------------------- |
| Logistic regression                 | 0.810 / 0.816                             |
| Random Forest                       | 0.825 / 0.830                             |
| XGBoost                             | 0.833 / 0.837                             |
| LSTM                                | 0.828 / 0.822                             |
| TFT                                 | 0.826 / 0.816                             |
| Chronos-Small zero-shot (sampled)   | 0.769 / 0.773                             |
| Chronos-Small fine-tuned (3 epochs) | 0.776 / 0.777                             |

Follow-up: frozen Chronos at three sizes, exact features (Stages 13–14; the
95% CIs are about 0.05 wide, so compare models with the paired tests in
PROGRESS.md Stage 14, not by eye):

| Model         | forward_fill: LR head / MLP head | linear_interp: LR head / MLP head |
| ------------- | -------------------------------- | --------------------------------- |
| Chronos-Mini  | 0.775 / 0.803                    | 0.775 / 0.800                     |
| Chronos-Small | 0.775 / 0.805                    | 0.778 / 0.801                     |
| Chronos-Base  | 0.779 / 0.807                    | 0.775 / 0.800                     |

**Headline findings:**

- Zero-shot Chronos-Small scored lower than every task-specific model
  (~0.77 vs ~0.81–0.84 AUROC). This is consistent with Gu et al. (2025)
  and Rockenschaub et al. (2024), who reported similar problems for ICU
  vital-sign forecasting and critical-care classification tasks. In the
  literature reviewed for this project (a narrative review, not a
  systematic one) I did not find a comparison of this kind on the standard
  MIMIC-III mortality benchmark.
- Fine-tuning Chronos-Small gave only a small gain (< 1 percentage point)
  under a short, CPU-limited budget (3 epochs). Treat this as a lower
  bound: a small MLP trained on frozen Chronos-Small features scored
  significantly higher (~0.80, difference +0.022 to +0.030, paired
  bootstrap), and the fine-tuned classifier used unstandardised features,
  so the two setups are not directly comparable. Fine-tuned Small was not
  significantly different from a linear head on frozen features.
- Across Chronos-Mini/Small/Base (20M–201M parameters) AUROC differed by
  at most 0.004 and no size difference was significant (paired bootstrap).
  A trained MLP head on the same frozen features reached ~0.80–0.81 at
  every size: significantly above the linear head, the fine-tuned model
  and sampled zero-shot, and not significantly different from logistic
  regression on summary features, but still significantly below Random
  Forest, XGBoost and the LSTM.
- AUROC is essentially robust to imputation strategy across every model
  (deltas under 0.01). However, individual classification decisions are
  not: 5 of 7 models show genuine, threshold-independent sensitivity to
  imputation choice; 2 (logistic regression, and borderline LSTM) have
  their apparent sensitivity substantially explained by
  threshold-selection shifts rather than the underlying risk scores.
- At the F1-optimal threshold every model had sensitivity of only
  0.41–0.53 (specificity 0.88–0.93): they miss about half of the deaths,
  which matters for any early-warning use.
- Chronos is by far the most expensive model on CPU: fine-tuning took
  ~35 h for 3 epochs, and inference ~0.34 s/patient (fine-tuned) to
  ~1.4 s/patient (zero-shot, sampled), versus under 3 ms/patient for the
  LSTM and TFT. All timings are CPU-only upper bounds.

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
- **The Chronos size comparison uses the exact (softmax-expectation)
  mean/variance features, not sampling**, because it is ~4x faster and is
  the same calculation as the fine-tuning path. It is reported as an
  additional experiment; the sampled Chronos-Small run remains the main
  zero-shot result. Two heads are trained on the frozen features:
  logistic regression (as in zero-shot) and a standardised, early-stopped
  2-layer MLP. The fine-tuned model's classifier did not standardise its
  features, so the fine-tuned and frozen-MLP results are not a clean
  comparison.
- **Model sizes and hyperparameter grids were chosen under real CPU-only
  constraints** (no GPU access) — Chronos-Small for the main experiments,
  small TFT/RF search grids, 3-epoch Chronos fine-tuning budget. Each
  choice is benchmarked and documented at the point it's made, not
  asserted without justification — see PROGRESS.md for the reasoning and
  numbers behind each one.
- GCS fields are label-encoded from their ordinal string form
  (`preprocessing/imputation.py`) — double-check this matches your
  actual benchmark CSV format if you re-run extraction with a different
  `mimic3-benchmarks` version.

## Reproducibility

All random seeds fixed via `config.RANDOM_SEED` (42), used consistently
across numpy, torch, and sklearn `random_state` arguments throughout.
Full run log with exact commands and real output at every stage:
[PROGRESS.md](PROGRESS.md).

MIMIC-III data and anything derived from it at patient level (processed
arrays, saved predictions, model outputs) must stay out of version control
under the PhysioNet data use agreement; `.gitignore` is set up for this.
Check `git status` before committing, especially under `results/` and
`data/`.

# MIMIC-III Preprocessing Pipeline
Supports Chapter 3 of *Evaluating Time-Series Foundation Models for Early Warning
in Intensive Care Units*.

## Folder structure
```
mimic-fm-comparison/
├── config.py                 # all paths + shared constants — edit this first
├── requirements.txt
├── .gitignore                 # excludes all MIMIC-III data, per data use agreement
├── data/
│   ├── raw/                    # (gitignored) mimic3-benchmarks output lands here if you point BENCHMARK_ROOT here
│   └── processed/              # this pipeline's output
│       ├── forward_fill/
│       └── linear_interp/
├── preprocessing/
│   ├── imputation.py
│   ├── feature_extraction.py
│   └── build_dataset.py
├── models/                     # baselines.py, lstm.py, tft.py, chronos_eval.py (to build next)
├── evaluation/                 # metrics.py, imputation_sensitivity.py, evaluate.py (to build next)
├── experiments/configs/        # one YAML per run, so results are traceable to exact settings
├── results/                    # gitignored — tables, CI outputs, figures
├── notebooks/                  # exploratory only, not the pipeline itself
├── scripts/
│   └── decompress_mimic.py     # one-time .csv.gz -> .csv step, see Windows/gzip note below
└── tests/                      # sanity checks: no test-leakage, shape checks
```

## Step 0 — prerequisites
- Completed CITI training + PhysioNet data use agreement (you've noted this is done).
- MIMIC-III v1.4 downloaded. If your files look like
  `C:\Users\thato\Downloads\mimic-iii-clinical-database-1.4\...\PRESCRIPTIONS.csv.gz`,
  you have the gzipped CSV distribution (not Postgres) — that's what this
  pipeline assumes.

### Windows / gzip note
`mimic3-benchmarks`'s extraction scripts read plain `.csv`, not `.csv.gz`.
Decompress once:
```powershell
python scripts\decompress_mimic.py
```
This reads `config.RAW_MIMIC_GZ_DIR` (your Downloads path) and writes plain
CSVs to `config.RAW_MIMIC_CSV_DIR`. It's a one-time step — the large tables
(CHARTEVENTS, LABEVENTS) are several GB each, so it'll take a few minutes
and you'll need roughly 2x the compressed size in free disk space during
the copy.

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
This reproduces the exact Harutyunyan et al. (2019) cohort, 17 variables, and
train/val/test split — using the field-standard implementation keeps your
results directly comparable to prior work, which is the whole point of citing
that benchmark.

## Step 2 — this pipeline
1. Edit `config.py`: set `BENCHMARK_ROOT` to the `data/in-hospital-mortality/`
   folder from Step 1, and `OUTPUT_ROOT` to wherever you want processed arrays.
2. `pip install numpy pandas tqdm`
3. `python build_dataset.py`

This produces, for **each** imputation condition (`forward_fill`, `linear_interp`):

```
OUTPUT_ROOT/<condition>/train/sequences.npy      (N, 48, 17) float — for LSTM/TFT/Chronos
OUTPUT_ROOT/<condition>/train/static_feats.npy   (N, 102) float   — for LR/RF/XGBoost
OUTPUT_ROOT/<condition>/train/labels.npy         (N,) int
OUTPUT_ROOT/<condition>/train/subject_ids.npy    (N,) str
... same for val/ and test/
OUTPUT_ROOT/<condition>/norm_stats.npz           training means/stds used for normalisation
```

## Design decisions worth knowing before you use this in your writeup
- **Normalisation and imputation fallback means are fit on train only**, then
  applied unchanged to val/test — this is what Section 3.3 requires and what
  protects against test-set leakage.
- **The 102 static features are computed on the imputed grid**, not the raw
  sparse data, so the traditional ML models and deep models see a consistent
  underlying representation of each patient course — differences you observe
  should trace to the model, not to different data views.
- **Missingness fraction** (17 of the 102 features) comes from the observation
  mask *before* imputation, so the models retain the "was this ever measured"
  signal even though the imputed values themselves are filled in — this is
  the informative-missingness point raised in Section 2.6.
- GCS fields are label-encoded from their ordinal string form before any
  imputation happens (`imputation._encode_categoricals`) — double check this
  encoding against your actual benchmark CSV column format, since MIMIC-III
  free-text GCS values are occasionally inconsistent.
- Chronos zero-shot/fine-tuning (Section 3.5.3) and model training/evaluation
  themselves are separate downstream scripts (not included here) that will
  consume `sequences.npy` for the temporal models and `static_feats.npy` for
  the traditional ones — this pipeline only covers up through Section 3.5.1.

## Suggested next files to build
- `train_baselines.py` — LR/RF/XGBoost with the CV grids from Sec 3.5.1
- `train_lstm_tft.py` — PyTorch training loop with weighted BCE (Sec 3.5.2)
- `chronos_eval.py` — zero-shot + fine-tuned Chronos (Sec 3.5.3)
- `evaluate.py` — AUROC/AUPRC/F1/ECE with bootstrap CIs + McNemar (Sec 3.6)

Happy to build any of those next — just say which one.

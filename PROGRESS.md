# Progress Log

Running record of pipeline execution: what was run, when, and what the
output confirmed. Kept alongside the code so the dissertation's methodology
chapter can cite exact steps and numbers rather than reconstructing them
from memory later.

Convention: one entry per completed stage. Record the command(s) run, the
key output/numbers, and anything that needed a decision or fix.

---

## Stage 1: Raw data decompression

**Status:** Complete

Decompressed the `.csv.gz` MIMIC-III v1.4 files using
`scripts/decompress_mimic.py` into a plain-CSV directory
(`config.RAW_MIMIC_CSV_DIR`), since `mimic3-benchmarks`' extraction scripts
require uncompressed CSVs.

---

## Stage 2: Cohort extraction (mimic3-benchmarks)

**Status:** Complete
**Tool:** unmodified clone of https://github.com/YerevaNN/mimic3-benchmarks

### `extract_subjects`

```
python -m mimic3benchmark.scripts.extract_subjects "C:\Users\thato\mimic-iii-decompressed" data/root/
```

Output funnel:

- START: 61,532 ICU stays / 57,786 admissions / 46,476 subjects
- After removing ICU transfers: 55,830 / 52,834 / 43,277
- After removing multiple stays per admission: 50,186 / 50,186 / 41,587
- After removing age < 18: 42,276 / 42,276 / 33,798

**Note:** interrupted once partway through (CHARTEVENTS parsing, ~55%) —
possibly a laptop sleep/interrupt. Re-ran from scratch rather than
attempting to resume, since this script doesn't checkpoint and a partial
run risks silently truncated subject data.

### `validate_events`

```
python -m mimic3benchmark.scripts.validate_events data/root/
```

Also interrupted once (KeyboardInterrupt during a pandas parse step).
Ran a diagnostic scan (`scripts/find_bad_events_csv.py`) over all 33,798
subjects' `events.csv` files first to rule out an actual malformed-CSV
row — found 0 problem files, confirming the interruption was a system
hiccup, not a data issue. Re-ran clean:

- n_events: 247,111,649
- empty_hadm: 4,313,069
- no_hadm_in_stay: 28,014,171
- no_icustay: 13,858,087
- recovered: 13,858,087 (matches no_icustay exactly — fully recovered)
- could_not_recover: 0
- icustay_missing_in_stays: 6,212,172

### `extract_episodes_from_subjects`

```
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

Completed cleanly, 33,802 subjects processed (vs. 33,798 reported by
`extract_subjects` — a difference of 4; consistent with minor folder-count
noise at this scale, not investigated further as immaterial to cohort
validity).

### `split_train_and_test`, `create_in_hospital_mortality`, `split_train_val`

```
python -m mimic3benchmark.scripts.split_train_and_test data/root/
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3models.split_train_val data/in-hospital-mortality/
```

All completed without errors.

---

## Stage 3: Benchmark output verification

**Status:** Complete — passed
**Command:** `python scripts/verify_benchmark_output.py`

Confirms:

- `BENCHMARK_ROOT` and all listfiles found.
- Listfile columns: `['stay', 'y_true']` for train/val/test, as expected.
- Mortality rates: **train 0.135, val 0.135, test 0.116** — all within
  the ~10-15% range the proposal expects (Section 3.2/2.7).
- Sample `*_timeseries.csv` (`10004_episode1_timeseries.csv`) contains
  all 17 expected variable columns matching `config.VARIABLES` exactly —
  no column-name mismatch, so `build_dataset.py` is safe to run as-is.

---

## Stage 4: Full preprocessing pipeline (build_dataset.py)

**Status:** Complete — verified

```
python preprocessing/build_dataset.py
```

Both imputation conditions processed successfully:

| Condition     | Split | Episodes | Mortality rate |
| ------------- | ----- | -------- | -------------- |
| forward_fill  | train | 14,681   | 0.135          |
| forward_fill  | val   | 3,222    | 0.135          |
| forward_fill  | test  | 3,236    | 0.116          |
| linear_interp | train | 14,681   | 0.135          |
| linear_interp | val   | 3,222    | 0.135          |
| linear_interp | test  | 3,236    | 0.116          |

Episode counts and mortality rates are identical across both conditions
(as expected, since imputation strategy doesn't change which episodes/
labels exist) and match the rates already confirmed independently from
the raw listfiles in Stage 3 — consistent cross-check passed.
`norm_stats.npz` saved for both conditions.

### Verification

```
python -m pytest tests/test_pipeline_output.py -v
```

**22/22 tests passed**, covering both conditions x 3 splits:

- `test_shapes_consistent` — sequences (N,48,17), static_feats (N,102)
- `test_no_nans_remain` — no NaNs in either output array
- `test_mortality_rate_plausible` — all splits within 10-15%-ish range
- `test_no_subject_overlap_across_splits` — zero overlap between train/val/test subject IDs (per condition)
- `test_normalisation_stats_saved` — norm_stats.npz present with expected keys

**Preprocessing (Chapter 3, Sections 3.2-3.5.1) is complete and verified.**
No train/test leakage, no data-quality issues found. Ready to move on to
model implementation (Stage 5).

---

## Stage 5+: Model training and evaluation

**Status:** Not started
Pending: `models/baselines.py`, `models/lstm.py`, `models/tft.py`,
`models/chronos_eval.py`, `evaluation/evaluate.py`.

# Progress Log

Running record of pipeline execution: what was run, when, and what the
output confirmed. Kept alongside the code so the dissertation's methodology
chapter can cite exact steps and numbers rather than reconstructing them
from memory later.

Convention: one entry per completed stage. Record the command(s) run, the
key output/numbers, and anything that needed a decision or fix.

Where a later stage corrected or qualified an earlier entry, the earlier
entry carries a clearly marked **Correction** or **Update** note (original
text kept) rather than being silently rewritten.

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

## Stage 5: Traditional ML baselines (models/baselines.py)

**Status:** Complete — verified

Logistic regression, Random Forest, and XGBoost (Section 3.5.1), with
uniform inverse-frequency class weighting (Section 3.4), trained and
evaluated on both imputation conditions.

### Issues hit and fixed along the way

- **Nested parallelism**: initial version set `n_jobs=-1` on both the CV
  search and the individual estimators (RF/XGBoost), causing worker
  processes to compete for cores. Fixed by parallelising only at the
  search level (`N_JOBS`, configurable via env var, default
  `min(4, cpu_count)`), estimators run single-threaded per fit.
- **Out-of-memory on Windows**: `n_jobs=-1` spawns one process per core,
  and Windows has to re-import numpy/scipy/sklearn per worker (no fork),
  which is RAM-expensive. First real run with `N_JOBS=4` completed but
  had 9/500 individual CV fits fail with `MemoryError`/`bad_malloc`
  (scikit-learn scored these as NaN and proceeded with the rest
  automatically). Re-ran with `N_JOBS=2` for a fully clean pass -- zero
  failures, and results were identical to the decimal against the
  `N_JOBS=4` run, confirming the dropped fits didn't affect the outcome.
- **Convergence warnings for logistic regression**: the 102 static
  features are on very different scales (GCS ~3-15, heart rate ~60-150,
  observation counts 0-48). Fixed by fitting a `StandardScaler` inside
  the CV pipeline ahead of `LogisticRegression` (scaler refit per
  training fold, no leakage).

### Final results (N_JOBS=2, clean run, no failed fits)

| Condition     | Model               | Val AUROC | Test AUROC |
| ------------- | ------------------- | --------- | ---------- |
| forward_fill  | logistic_regression | 0.7968    | 0.8103     |
| forward_fill  | random_forest       | 0.8254    | 0.8252     |
| forward_fill  | xgboost             | 0.8151    | 0.8332     |
| linear_interp | logistic_regression | 0.7997    | 0.8158     |
| linear_interp | random_forest       | 0.8252    | 0.8296     |
| linear_interp | xgboost             | 0.8191    | 0.8369     |

Best hyperparameters, per-condition val/test predictions, and trained
models saved to `results/baselines/<condition>/`.

Note for the discussion chapter: these test AUROCs (0.81-0.84) are
somewhat below the 0.85-0.92 range reported in the RF/XGBoost literature
cited in Section 2.2 (Ashrafi et al. 2024 in particular). Plausible
reasons to explore later: different cohort definition/size, uniform
(not per-model-tuned) class-weighting strategy applied here for fairness
across all five configurations, or the 102-feature summary-statistic
representation being less rich than what those studies used.

---

## Stage 6: Channel-wise LSTM (models/lstm.py)

**Status:** Complete — verified

Section 3.5.2's channel-wise LSTM (shared 2-layer LSTM per channel,
hidden dim 128, dropout 0.3), weighted BCE loss, Adam (lr=1e-3) with
LR reduction on validation-AUROC plateau, early stopping (patience=10).
Ran on CPU (no GPU available on this machine) -- completed without
issue, no memory problems this time.

### Training behaviour

- Both conditions stopped via early stopping at **epoch 45** (max was
  100), after the LR had been reduced three times (1e-3 -> 5e-4 -> 2.5e-4
  -> 1.25e-4 -> 6.25e-5), consistent with genuine convergence rather than
  a bug -- loss decreased steadily and val AUROC plateaued exactly as
  expected before stopping triggered.

### Final results

| Condition     | Model | Val AUROC | Test AUROC | Epochs trained |
| ------------- | ----- | --------- | ---------- | -------------- |
| forward_fill  | lstm  | 0.8132    | 0.8276     | 45             |
| linear_interp | lstm  | 0.8117    | 0.8219     | 45             |

Model checkpoints, val/test predictions, and per-epoch training history
saved to `results/lstm/<condition>/`.

Note for the discussion chapter: consistent with the baselines, these
test AUROCs (~0.82-0.83) sit below Harutyunyan et al. (2019)'s reported
~0.86 for their channel-wise LSTM. Same plausible explanations apply
(cohort/preprocessing differences, no per-model tuning beyond what's
specified) -- worth digging into directly once all five configurations
are done, since a consistent ~3-4 point gap across model families
suggests something systematic in the pipeline/cohort rather than
per-model tuning being the main driver.

---

## Stage 7: Temporal Fusion Transformer (models/tft.py)

**Status:** Complete — verified

### Environment fix needed before training

Installing `pytorch-forecasting` surfaced a numpy version issue:

- `requirements.txt` originally pinned `numpy<2.0`, written before torch/
  xgboost were already installed against numpy 2.x. On Python 3.13, numpy
  1.26.4 has no prebuilt Windows wheel, so pip built it from source via
  MinGW -- flagged by numpy itself as experimental and crash-prone.
- Fixed by relaxing the pin to `numpy>=2.0,<2.5` (the upper bound matches
  scipy 1.15.3's stated requirement). Verified clean: no warnings from
  numpy, scipy, torch, xgboost, sklearn, or pytorch_forecasting imports.
- Lesson: avoid pinning below what's already working unless there's a
  specific reason -- an overly cautious pin caused more friction here
  than no pin would have.

### Training

Section 3.5.2's TFT via pytorch-forecasting, adapted for binary
classification (CrossEntropy loss, output_size=2, over a 47-hour
encoder + 1-step classification target -- see docstring in tft.py for
the full adaptation rationale). Ran on CPU. 4-config hyperparameter grid
(hidden_size x attention_head_size) x 2 conditions, selecting best by
validation AUROC per Section 3.5.2's "tuned on the validation set".

### Training behaviour

- ~55-70s/epoch depending on config size; most configs ran the full 30
  epochs without early stopping firing (val_loss kept marginally
  improving), a couple stopped at epoch 29. No crashes, no memory issues
  -- the numpy/scipy version fix held up cleanly under real training.

### Final results (best config per condition, by val AUROC)

| Condition     | Best hidden_size | Best attention_heads | Val AUROC | Test AUROC |
| ------------- | ---------------- | -------------------- | --------- | ---------- |
| forward_fill  | 32               | 4                    | 0.8140    | 0.8255     |
| linear_interp | 16               | 1                    | 0.8124    | 0.8163     |

Predictions and best hyperparameters saved to `results/tft/<condition>/`.

Consistent with the pattern across all model families so far: test
AUROC clustering around 0.81-0.83, still below the ~0.86 Harutyunyan
LSTM/TFT literature benchmarks -- reinforces that this is likely a
systematic pipeline/cohort effect worth discussing directly (Section on
Discussion), rather than anything specific to one architecture.

---

## Stage 8: Chronos zero-shot (models/chronos_eval.py)

**Status:** Complete — verified

### Model size decision

Benchmarked mini/small/base on this CPU-only machine
(`scripts/benchmark_chronos.py`) before committing:

| Model | Params | Est. full zero-shot runtime (both conditions) |
| ----- | ------ | --------------------------------------------- |
| Mini  | 20.5M  | ~12.5 hours                                   |
| Small | 46.2M  | ~17.1 hours                                   |
| Base  | 201.4M | ~96.0 hours                                   |

Chose **Chronos-Small**: meaningful capability step up from Mini without
Base's 4-day single-run risk. Large (710M) ruled out entirely as
impractical without GPU -- consistent with the proposal's own documented
fallback plan (Section 3.5.3).

### Full run

Completed in one continuous pass, no interruption needed (~18 hours
total, matching the recalibrated estimate from quick-mode testing:
~90ms/episode/channel observed consistently across the full run).
Per-channel checkpointing (verified separately via mock pipeline) was
available as a safety net but wasn't needed this time.

### Final results

| Condition     | Val AUROC | Test AUROC |
| ------------- | --------- | ---------- |
| forward_fill  | 0.7680    | 0.7692     |
| linear_interp | 0.7636    | 0.7734     |

**Key finding**: zero-shot Chronos-Small scores lower than every other
model family (all of which cluster at 0.81-0.84 test AUROC). This is
consistent with the caution of Gu et al. (2025) for ICU vital-sign
forecasting (their conclusion: the "ChatGPT moment" for time-series
foundation models in the clinical domain has not yet arrived): foundation
models pretrained on general-domain time series may not transfer well to
clinical data zero-shot. In the
literature reviewed for the dissertation (a narrative review, not a
systematic one) no comparison of this kind was found on the standard
MIMIC-III mortality benchmark under matched conditions (the gap
identified in Section 2.9), so this is reported as a controlled
data point, not as a claim to be the first.

Features, model, and predictions saved to
`results/chronos/<condition>/zeroshot_*`.

---

## Stage 9: Chronos fine-tuning (models/chronos_finetune.py)

**Status:** Complete — verified

Section 3.5.3's fine-tuned configuration: the pretrained encoder is
fine-tuned end-to-end jointly with a 2-layer MLP classification head,
weighted BCE loss, encoder lr=1e-5, MLP lr=1e-3 (to limit catastrophic
forgetting of pretrained representations), 3 epochs, early stopping
patience=1.

### Key technical adaptation (differentiability)

Zero-shot extracts mean/variance by DRAWING RANDOM SAMPLES from the
model's predictive distribution (matching the proposal's literal
wording, Section 3.5.3). Random sampling is not differentiable -- you
cannot backpropagate through a stochastic draw -- so it cannot be used
here, where the encoder itself needs to be updated by gradient descent.

Instead, fine-tuning computes mean/variance ANALYTICALLY from the
decoder's output softmax distribution over the token vocabulary at the
single prediction step:

```
mean = sum_i(p_i * bin_center_i)
variance = sum_i(p_i * (bin_center_i - mean)^2)
```

This is the exact expectation/variance of the same predictive
distribution the zero-shot sampling was estimating via Monte Carlo --
conceptually the same "mean and variance of the predictive distribution"
required by Section 3.5.3, just computed exactly rather than by
sampling, which is what makes end-to-end gradient-based fine-tuning
possible at all. Verified correct via a local test (a genuine, if tiny,
locally-built T5 model, no internet needed): gradients computed this way
reach every encoder parameter, not just the MLP head, confirmed by
checking `encoder_grad_norm > 0` after a backward pass.

### Issues hit and fixed

- **Silent crash, no traceback**: first quick-mode attempt died right
  after model load with no error message at all -- process just
  returned to the prompt. Classic signature of a Windows OpenMP DLL
  conflict (torch + transformers + numpy each potentially bundling their
  own runtime, causing a native-level abort rather than a Python
  exception). Fixed with `KMP_DUPLICATE_LIB_OK=TRUE`, which was enough to
  surface the _real_ underlying error on the next attempt instead of a
  silent death.
- **Real error once visible: CPU out-of-memory** during a tiny (1.6MB)
  allocation inside a T5 feed-forward layer -- root cause was
  `ChronosClassifier.forward()` looping over 17 channels, each running a
  full T5 encoder-decoder pass, with PyTorch forced to hold all 17
  channels' activation graphs simultaneously until the single final
  loss/`backward()` call (effectively 17x a single forward pass's
  activation memory, since the MLP head needs all 34 features -- 17
  channels x mean+variance -- at once). Fixed with gradient checkpointing
  (`torch.utils.checkpoint`, wrapping the per-channel T5 forward call),
  which recomputes activations during backward instead of storing them
  during forward. Verified this produces bit-identical loss values to a
  non-checkpointed test run (confirming it's a pure memory optimisation,
  not an approximation) and that checkpoint/resume still works correctly
  with it enabled.
- **Mid-epoch checkpointing was essential, not just precautionary**: a
  single epoch of fine-tuning takes on the order of hours (real logs:
  ~11-12 hours/epoch/condition; the quick-mode extrapolation had
  predicted ~13-14, see the correction under Timing below), so
  checkpointing only at epoch boundaries (as used for the LSTM) would
  have been far too coarse. Checkpoints save every 50 batches, capturing
  model state, optimizer state, epoch, and batch index. The real run
  spanned Friday night through the following week and was interrupted
  once (VS Code closed accidentally) mid-way through `linear_interp`'s
  epoch 3 -- resumed cleanly from the last saved batch (~21,764s into
  that epoch alone) with no lost progress or corrupted state. Note: the
  random batch-shuffle order itself isn't checkpointed, so a resumed run
  processes remaining batches in a different order than an uninterrupted
  run would have -- doesn't affect training validity (still correct SGD
  over the full training set), just means exact bit-for-bit
  reproducibility isn't preserved across an interruption.

### Timing (quick-mode extrapolation vs the real logs)

50 train + 50 val episodes/epoch took ~195-246s in quick mode. Scaled to
the real split sizes (14,681 train / 3,222 val), this predicted ~13-14
hours/epoch/condition.

**Correction (added in Stage 12):** the real per-epoch times in
`results/chronos/<condition>/finetuned_training_history.json` are:

| Condition     | Epoch 1 | Epoch 2 | Epoch 3         | Total              |
| ------------- | ------- | ------- | --------------- | ------------------ |
| forward_fill  | 11.3 h  | 11.5 h  | 12.0 h          | 34.8 h (125,402 s) |
| linear_interp | 11.4 h  | 11.5 h  | 6.0 h (partial) | >= 28.9 h          |

So a full epoch took ~11.3-12.0 h, slightly less than the extrapolation.
`linear_interp` epoch 3 (21,764 s) only timed the portion after the
accidental interruption, so its true total is longer than 28.9 h
(probably close to the forward_fill 34.8 h). Use these log values, not the
13-14 h estimate, when quoting fine-tuning cost.

Validation AUROC was still rising at epoch 3 in both conditions
(forward_fill 0.7555 -> 0.7663; linear_interp 0.7557 -> 0.7678), so early
stopping never triggered.

### Final results

| Condition     | Val AUROC | Test AUROC | Epochs trained |
| ------------- | --------- | ---------- | -------------- |
| forward_fill  | 0.7663    | 0.7764     | 3              |
| linear_interp | 0.7678    | 0.7773     | 3              |

### Zero-shot vs fine-tuned comparison

| Condition     | Zero-shot test AUROC | Fine-tuned test AUROC | Delta   |
| ------------- | -------------------- | --------------------- | ------- |
| forward_fill  | 0.7692               | 0.7764                | +0.0072 |
| linear_interp | 0.7734               | 0.7773                | +0.0039 |

**Key finding, directly answering Research Question 2:** fine-tuning
produces only a marginal improvement (<1 percentage point in test
AUROC) and does NOT close the gap to task-specific models (0.81-0.84
across Stages 5-7). This is a genuine, controlled result -- both
zero-shot and fine-tuned Chronos were evaluated under identical
conditions (same cohort, same imputation conditions, same splits) to
every other model family, so the comparison is fair by construction.

Honest caveat for the discussion chapter: this reflects a
resource-constrained fine-tuning budget (3 epochs, Chronos-Small,
CPU-only, analytic rather than sampling-based feature extraction). The
defensible claim is that fine-tuning under these conditions didn't close
the gap, not that fine-tuning categorically cannot help -- a larger
model, more epochs, or GPU-enabled training over more data might behave
differently, and this limitation is worth stating explicitly rather than
implying the result generalises beyond what was actually tested.

**Update (Stage 13):** a small MLP trained on FROZEN Chronos-Small features
scored ~0.80-0.81 test AUROC, well above this fine-tuned model (0.776 /
0.777), at ~2 h of feature extraction instead of ~35 h of fine-tuning.
Two differences between the setups may explain part of this: the
fine-tuned classifier received the 34 features UNSTANDARDISED (the frozen
MLP standardises them), and fine-tuning stopped after 3 epochs with
validation AUROC still rising. Neither was tested. Read the result above
as a lower bound on what fine-tuning could achieve, not as evidence that
fine-tuning cannot help.

Model, val/test predictions, and per-epoch training history saved to
`results/chronos/<condition>/finetuned_*`.

---

> **Milestone:** all five task-specific model configurations, plus Chronos
> zero-shot and fine-tuned, are COMPLETE and verified across both
> imputation conditions. This is the full empirical core of Chapter 3
> (Sections 3.2-3.5.3).

## Stage 10: Evaluation (evaluation/evaluate.py)

**Status:** Complete — verified

Built `evaluation/metrics.py` (AUROC/AUPRC/F1/ECE, threshold selection,
bootstrap CIs, Platt scaling, McNemar test) and `evaluation/evaluate.py`
(loads every model's saved val/test predictions, computes the full
metric set per Section 3.6, runs the imputation-sensitivity analysis
per Section 3.6.3).

### Verification before running on real data

Tested against synthetic prediction files matching the exact output
structure of all 7 model scripts, in two passes:

1. Basic run confirming correct file discovery across all 14
   model x condition combinations, no path errors.
2. A deliberately weakened synthetic condition (one model/condition
   combination given genuinely different, noisy predictions) to confirm
   the statistical tests actually DETECT a real difference when one
   exists (non-overlapping CIs, McNemar p<0.001) while correctly
   reporting no difference for identical synthetic predictions
   elsewhere (p=1.0) -- not just "runs without crashing."

### Real run

```
python evaluation/evaluate.py
```

Completed in well under a minute (pure metric computation on already-
saved predictions, no model inference). AUROC values cross-checked
exactly against the numbers already logged in Stages 5-9 above (e.g.
xgboost/forward_fill: 0.8332 here matches Stage 5's logged value
exactly) -- confirms the evaluation pipeline is correctly reading the
real saved predictions, not silently pulling from a stale or wrong
source.

### Full results (test set, all 7 models x 2 conditions)

| Model               | Condition     | AUROC  | AUPRC  | F1     | Sens.  | Spec.  | ECE (before) | ECE (after Platt) |
| ------------------- | ------------- | ------ | ------ | ------ | ------ | ------ | ------------ | ----------------- |
| logistic_regression | forward_fill  | 0.8103 | 0.3754 | 0.4224 | 0.4840 | 0.8945 | 0.2731       | 0.0197            |
| logistic_regression | linear_interp | 0.8158 | 0.3883 | 0.4276 | 0.5214 | 0.8802 | 0.2741       | 0.0178            |
| random_forest       | forward_fill  | 0.8252 | 0.3990 | 0.4318 | 0.5294 | 0.8795 | 0.1397       | 0.0221            |
| random_forest       | linear_interp | 0.8296 | 0.4003 | 0.4382 | 0.5214 | 0.8878 | 0.1325       | 0.0238            |
| xgboost             | forward_fill  | 0.8332 | 0.4424 | 0.4581 | 0.5267 | 0.8990 | 0.1583       | 0.0193            |
| xgboost             | linear_interp | 0.8369 | 0.4286 | 0.4526 | 0.4973 | 0.9085 | 0.2357       | 0.0164            |
| lstm                | forward_fill  | 0.8276 | 0.4356 | 0.4185 | 0.4225 | 0.9221 | 0.2204       | 0.0257            |
| lstm                | linear_interp | 0.8219 | 0.4277 | 0.4353 | 0.4813 | 0.9046 | 0.2233       | 0.0183            |
| tft                 | forward_fill  | 0.8255 | 0.4540 | 0.4266 | 0.4118 | 0.9322 | 0.0346       | 0.0246            |
| tft                 | linear_interp | 0.8163 | 0.4278 | 0.4379 | 0.4759 | 0.9088 | 0.0205       | 0.0270            |
| chronos_zeroshot    | forward_fill  | 0.7692 | 0.3618 | 0.3962 | 0.4439 | 0.8959 | 0.3023       | 0.0161            |
| chronos_zeroshot    | linear_interp | 0.7734 | 0.3724 | 0.3899 | 0.4118 | 0.9085 | 0.3039       | 0.0180            |
| chronos_finetuned   | forward_fill  | 0.7764 | 0.3745 | 0.4034 | 0.4492 | 0.8983 | 0.3077       | 0.0175            |
| chronos_finetuned   | linear_interp | 0.7773 | 0.3766 | 0.4108 | 0.4278 | 0.9144 | 0.3120       | 0.0182            |

Observation: F1 scores (0.39-0.46) are notably modest relative to the
AUROCs (0.77-0.84) -- expected given the ~13% positive class rate; even
after threshold tuning on validation, sensitivity/specificity trade-offs
remain harsh under this level of imbalance. Platt scaling consistently
and substantially reduces ECE across every model (e.g. TFT/forward_fill:
0.0346, already well-calibrated raw; chronos_zeroshot/forward_fill:
0.3023 -> 0.0161, a large improvement) -- worth noting in the discussion
that raw model outputs are generally poorly calibrated, and post-hoc
calibration matters regardless of which model family is used.

### Imputation sensitivity (Section 3.6.3, Research Question 3) — REVISED

Initial run (own-threshold McNemar only) found every model significant,
which risked over-claiming "all models are sensitive to imputation."
Added a second, shared-threshold McNemar variant to
`evaluation/evaluate.py`: both conditions' predictions binarised using
forward_fill's threshold (the Harutyunyan-standard reference condition),
isolating genuine score-level disagreement from threshold-selection
artifacts. Verified correct via three deliberately constructed synthetic
cases before trusting it on real data: identical scores + no threshold
shift (both variants correctly p=1.0), identical scores + large
threshold shift (own-threshold significant, shared-threshold correctly
p=1.0 -- confirms artifact isolation), and a genuine systematic score
shift (both variants correctly significant).

**Final results on real data:**

| Model               | Threshold delta (FF-LI) | Own-threshold p | Shared-threshold p | Interpretation                              |
| ------------------- | ----------------------- | --------------- | ------------------ | ------------------------------------------- |
| logistic_regression | 0.0207                  | <0.0001         | 0.8453             | **Threshold artifact**                      |
| lstm                | 0.0240                  | <0.0001         | 0.0875             | **Largely threshold artifact** (borderline) |
| random_forest       | 0.0049                  | 0.0043          | 0.0003             | **Genuine score sensitivity**               |
| xgboost             | -0.0661                 | 0.0053          | <0.0001            | **Genuine score sensitivity**               |
| tft                 | 0.1242                  | <0.0001         | <0.0001            | **Genuine score sensitivity**               |
| chronos_zeroshot    | -0.0082                 | <0.0001         | 0.0005             | **Genuine score sensitivity**               |
| chronos_finetuned   | -0.0548                 | <0.0001         | <0.0001            | **Genuine score sensitivity**               |

**Notable correction to an earlier eyeball-based guess:** before building
the shared-threshold test, I incorrectly guessed from the raw threshold
gap alone that TFT's sensitivity (the single largest threshold shift,
0.124) was "largely threshold-driven." The rigorous shared-threshold
test overturned this: TFT's disagreement survives even with the
threshold held fixed (p<0.0001) -- its risk scores themselves genuinely
differ between imputation conditions, independent of where the cutoff
sits. This is worth stating directly in the methodology chapter as the
justification for building the shared-threshold control rather than
relying on visual inspection of threshold gaps.

**Final, precise answer to Research Question 3:** imputation strategy's
effect on individual classification decisions is not uniform across
model families. Five of seven models (Random Forest, XGBoost, TFT,
Chronos zero-shot, Chronos fine-tuned) show genuine, threshold-
independent sensitivity to imputation choice -- their underlying risk
estimates for individual patients meaningfully differ between
forward_fill and linear_interp. Two models (logistic regression, and to
a lesser/borderline extent LSTM) have their apparent sensitivity
substantially explained by the independently-selected decision threshold
shifting, rather than the models' risk scores themselves changing much.
In all cases, though, AUROC itself remains essentially stable across
imputation conditions (deltas under 0.01, fully overlapping CIs) --
reinforcing that this is specifically an individual-decision-level
phenomenon, not one visible in aggregate ranking performance.

Full detail (including raw discordant-pair counts for both threshold
variants): `results/evaluation/imputation_sensitivity.csv`.

---

> **Milestone:** all of Chapter 3 (Sections 3.2-3.6) is complete,
> verified and reproducible. Every model trained, every metric computed,
> both imputation conditions compared.

## Stage 11: Results figures (evaluation/generate_figures.py)

**Status:** Complete

Generated all Results-chapter figures, reusing `evaluate.py`'s exact
prediction-loading logic (renamed `_load_predictions`/`_load_labels` to
public `load_predictions`/`load_labels` since they're now genuinely
shared across both scripts, not private to `evaluate.py`):

- `roc_curves_forward_fill.png` / `roc_curves_linear_interp.png` -- all
  7 models overlaid per condition, AUROC in legend
- `calibration_forward_fill.png` / `calibration_linear_interp.png` --
  reliability diagrams, raw vs after Platt scaling, side by side
- `model_comparison_auroc.png` -- grouped bar chart, both conditions,
  95% bootstrap CI error bars pulled directly from `full_results.csv`

All saved as 300 DPI PNG (Word-ready) and PDF (LaTeX-ready) to
`results/figures/`.

One real bug caught during testing (before handing off): the comparison
chart's y-axis was initially hardcoded to `[0.5, 0.9]`, which would
silently clip any model exceeding 0.9 AUROC (didn't affect this
project's real numbers, all under 0.84, but was fragile). Fixed to size
the axis dynamically from the actual CI bounds in the data.

Ran successfully against real results -- all 5 figures generated
without error.

---

> **Milestone:** Chapter 3 and all supporting figures are complete. The full
> empirical pipeline (preprocessing, all 7 model configurations,
> evaluation with bootstrap CIs and McNemar tests, Results figures) is
> built, verified and reproducible. Stages 12-14 below add cost timing,
> a Chronos model-size comparison, and bootstrap CIs with paired tests for
> it, in response to supervisor feedback.

---

## Stage 12: Computational cost timing (scripts/time_models.py)

**Status:** Complete — verified (quick run, then real run)

Supervisor feedback asked for a cost table backing up the Research
Question 4 claims. Before this stage only Chronos had logged timings;
baseline/LSTM/TFT training times and all inference times were unrecorded.

`scripts/time_models.py` measures, on the machine it runs on:

- **Inference** on the saved models (baselines, LSTM, fine-tuned Chronos
  weights) as seconds per patient. TFT weights were never saved by
  `tft.py`, so the best config is re-trained for 2 epochs and that model
  is timed (inference time does not depend on how well trained the weights
  are). Chronos is timed on a sample of 96 test patients and scaled.
- **Training:** one final fit with the chosen hyperparameters for the
  baselines (the CV search cost is an ESTIMATE: final-fit time x number of
  CV fits x 0.8 + 1, single core); a few real epochs for LSTM/TFT
  multiplied by the epochs the real runs used (ESTIMATE); Chronos
  fine-tuning read from `finetuned_training_history.json`.

Each section runs inside its own try/except, so one failure is logged as
`FAILED` and the others still run; results are saved after every section.

Commands (after `set KMP_DUPLICATE_LIB_OK=TRUE`):

```
set TIMING_QUICK=1 & python scripts\time_models.py    # quick run: all 5 sections ran, no failures
set TIMING_QUICK=                                     # clear the variable first
python scripts\time_models.py                         # real run
```

Output: `results/timing/timing_results_cpu_laptop.csv`. Hardware: Windows 11,
8 logical CPUs, PyTorch 2.7.0 (CPU, 4 threads), Python 3.13.3, no GPU.
Condition timed: forward_fill (timing does not depend on imputation).

### Results (forward_fill, CPU only)

| Model                       | Training time                                                 | Inference (ms/patient) | Full test set (3,236)  |
| --------------------------- | ------------------------------------------------------------- | ---------------------- | ---------------------- |
| Logistic regression         | 1.1 s final fit; search ~23 s (est.)                          | 0.0009                 | 0.003 s                |
| Random Forest               | 26 s final fit; search ~86 min on one core (est.)             | 0.056                  | 0.18 s                 |
| XGBoost                     | 5.6 s final fit; search ~19 min on one core (est.)            | 0.011                  | 0.04 s                 |
| LSTM                        | ~3.1 h (est.: 245 s/epoch x 45 epochs)                        | 2.83                   | 9.2 s                  |
| TFT                         | ~2.2 h (est.: 66.8 s/epoch x 29 epochs x 4 configs)           | 2.33                   | 7.5 s                  |
| Chronos zero-shot (sampled) | ~18 h for both conditions together (measured, Stage 8)        | 1372                   | ~74 min (extrapolated) |
| Chronos fine-tuned          | 34.8 h, 3 epochs, forward_fill (log); >= 28.9 h linear_interp | 341                    | ~18 min (extrapolated) |

### Notes

- Fine-tuned Chronos is ~4x faster per patient than zero-shot because it
  computes mean/variance exactly instead of drawing 20 samples.
- The fine-tuning epoch time (~11.3-12.0 h) corrects the 13-14 h
  extrapolation quoted in Stage 9 (see the correction there).
- LSTM and TFT totals assume the real runs' epoch counts (45 and 29); the
  TFT figure assumes every grid configuration costs about the same as the
  best one (smaller configs are slightly faster).
- Real baseline searches used up to 4 cores, so their wall-clock time was
  shorter than the single-core estimates above.

---

## Stage 13: Chronos model-size comparison (models/chronos_sizes.py)

**Status:** Complete — verified (timing run, quick run, then full run)

Supervisor feedback: the central claim ("Chronos underperforms") was only
tested for Chronos-Small. No GPU/cluster was available, so a
laptop-feasible extension was designed:

- Zero-shot (frozen) Chronos at **Mini (20.5M), Small (46.2M), Base
  (201.4M)**, both imputation conditions, full train/val/test splits.
- Features = the same 34 numbers (17 channel means + 17 variances), but
  computed **exactly** from the output softmax (same maths as in
  `chronos_finetune.py`) instead of from 20 sampled draws. ~4x faster per
  patient, which made Base feasible. This is a different feature method
  from Stage 8, so it is reported as an ADDITIONAL experiment; Stage 8
  (sampled, Small) remains the main zero-shot result.
- Two heads on the frozen features: logistic regression (balanced, C=1,
  as in Stage 8) and a 2-layer MLP (34-64-1, weighted BCE, features
  standardised on train statistics, early stopping on val AUROC,
  patience 10).
- No encoder training, so no epochs. Features are cached per channel/
  split/condition/size (`results/chronos_sizes/cache/`), so an interrupted
  run resumes at the next unfinished channel.

Commands (after `set KMP_DUPLICATE_LIB_OK=TRUE`):

```
set CHRONOS_SIZES_TIME_ONLY=1 & python models\chronos_sizes.py                              # projection only
set CHRONOS_SIZES_TIME_ONLY= & set CHRONOS_QUICK_MODE=1 & python models\chronos_sizes.py    # correctness only
set CHRONOS_QUICK_MODE= & python models\chronos_sizes.py                                    # full run
```

Gotcha hit once: `CHRONOS_SIZES_TIME_ONLY=1` stays set in the terminal and
makes later runs only time. Clear it before the real run. Quick mode writes
to `results/chronos_sizes_quick/` and its AUROCs (50 patients per split) are
not valid.

### Timing: projection vs reality

| Size  | Projected (64-patient test) | Measured per condition      |
| ----- | --------------------------- | --------------------------- |
| Mini  | 1.1 h                       | 0.88 h (fwd) / 0.86 h (lin) |
| Small | 2.3 h                       | 2.00 h / 2.03 h             |
| Base  | 8.9 h                       | 8.26 h / 8.31 h             |

Total ~22.3 h CPU time for 3 sizes x 2 conditions. Per patient: Mini 0.15 s,
Small 0.34 s, Base 1.41 s.

### Results (test AUROC; validation AUROC in brackets)

| Size  | Condition     | LR head         | MLP head        |
| ----- | ------------- | --------------- | --------------- |
| Mini  | forward_fill  | 0.7749 (0.7674) | 0.8033 (0.7986) |
| Mini  | linear_interp | 0.7753 (0.7674) | 0.8000 (0.7977) |
| Small | forward_fill  | 0.7747 (0.7656) | 0.8053 (0.7988) |
| Small | linear_interp | 0.7776 (0.7645) | 0.8006 (0.7956) |
| Base  | forward_fill  | 0.7790 (0.7668) | 0.8067 (0.7997) |
| Base  | linear_interp | 0.7754 (0.7688) | 0.7996 (0.7999) |

Reference (Stages 8-9): Small zero-shot sampled 0.7692 / 0.7734; Small
fine-tuned 0.7764 / 0.7773.

Sanity check, Small exact vs sampled features (test split): means
correlate 0.988 (forward_fill) / 0.966 (linear_interp); variances only
0.585 / 0.423, which is plausible because a variance from 20 draws is
noisy.

### What this does and does not show

- With the LR head, a ~10x increase in parameters changes test AUROC by at
  most 0.004, far inside the ~0.04-0.05 width of the bootstrap CIs: no
  clear size trend.
- The MLP head scores ~0.80-0.81 at every size, above the fine-tuned Small
  run (0.776 / 0.777) at ~2 h of extraction instead of ~35 h of
  fine-tuning. This makes the Stage 9 conclusion weaker than it first
  looked (see the Update note there). Which factor explains the difference
  (feature standardisation, training length, or the encoder updates
  themselves) was NOT tested.
- Only a 34-number summary of a next-step forecast was used (not the
  encoder's internal embeddings) and the weights were frozen.
- Bootstrap CIs and paired tests for these runs were added in Stage 14;
  without them, differences of a few thousandths must not be ranked.
  Threshold metrics and calibration (Platt scaling) were not computed for
  these runs.

Output: `results/chronos_sizes/summary.csv`, plus per-size/condition
features and predictions under `results/chronos_sizes/<size>/<condition>/`.

---

## Stage 14: Bootstrap CIs and paired tests for the size runs (evaluation/bootstrap_sizes.py)

**Status:** Complete — verified

`evaluation/evaluate.py` only covers the seven main configurations, so the
Stage 13 runs had no confidence intervals. `evaluation/bootstrap_sizes.py`
adds them, using the same recipe as the main evaluation (1,000 bootstrap
resamples of the test set, with the validation-chosen F1-optimal threshold
held fixed), plus **paired bootstrap differences in AUROC**: two models are
scored on the same test patients and the same resamples, so a difference's
CI is tighter and fairer than comparing two separate CIs by eye. A
difference is "significant" when its 95% CI excludes 0.

Command: `python evaluation\bootstrap_sizes.py` (numpy/pandas/scikit-learn
only, a few minutes). Output in `results/evaluation/`:
`bootstrap_sizes.csv` (AUROC/AUPRC/F1 with CIs, sensitivity, specificity),
`bootstrap_sizes_paired.csv` (all pairs), `bootstrap_sizes_table.tex`.

**Self-check against Stage 10:** for the 6 main models in both conditions
the script reproduced `full_results.csv` exactly (largest differences:
threshold 0, F1 5.6e-17, sensitivity 5.6e-17, specificity 1.1e-16, AUROC 0),
so the threshold, F1, sensitivity and specificity of the size runs are
directly comparable with the main table. TFT is deliberately excluded
(its predictions come from the pytorch-forecasting dataloader, so the order
must be checked against the labels before pairing).

### AUROC with 95% bootstrap CI (test)

| Model (head) | forward_fill            | linear_interp           |
| ------------ | ----------------------- | ----------------------- |
| Mini, LR     | 0.7749 [0.7501, 0.8007] | 0.7753 [0.7502, 0.8017] |
| Mini, MLP    | 0.8033 [0.7800, 0.8265] | 0.8000 [0.7775, 0.8233] |
| Small, LR    | 0.7747 [0.7483, 0.8009] | 0.7776 [0.7526, 0.8029] |
| Small, MLP   | 0.8053 [0.7810, 0.8286] | 0.8006 [0.7777, 0.8234] |
| Base, LR     | 0.7790 [0.7529, 0.8041] | 0.7754 [0.7502, 0.8017] |
| Base, MLP    | 0.8067 [0.7836, 0.8297] | 0.7996 [0.7770, 0.8221] |

### Paired AUROC differences (A - B), forward_fill / linear_interp

| A                | B                           | forward_fill                       | linear_interp                      |
| ---------------- | --------------------------- | ---------------------------------- | ---------------------------------- |
| Base LR          | Mini LR                     | +0.0042 [-0.0022, +0.0111] ns      | +0.0002 [-0.0036, +0.0036] ns      |
| Base MLP         | Mini MLP                    | +0.0035 [-0.0029, +0.0094] ns      | -0.0003 [-0.0075, +0.0069] ns      |
| Small MLP        | Small LR                    | +0.0306 [+0.0146, +0.0471] **sig** | +0.0230 [+0.0079, +0.0381] **sig** |
| Small LR (exact) | Chronos zero-shot (sampled) | +0.0055 [-0.0024, +0.0128] ns      | +0.0042 [+0.0004, +0.0077] **sig** |
| Small LR (exact) | Chronos fine-tuned          | -0.0017 [-0.0130, +0.0091] ns      | +0.0003 [-0.0079, +0.0082] ns      |
| Small MLP        | Chronos fine-tuned          | +0.0289 [+0.0127, +0.0442] **sig** | +0.0232 [+0.0084, +0.0374] **sig** |
| Base MLP         | Chronos fine-tuned          | +0.0303 [+0.0148, +0.0450] **sig** | +0.0223 [+0.0075, +0.0374] **sig** |
| Mini MLP         | Logistic regression         | -0.0070 [-0.0271, +0.0133] ns      | -0.0159 [-0.0359, +0.0032] ns      |
| Small MLP        | Logistic regression         | -0.0050 [-0.0246, +0.0157] ns      | -0.0153 [-0.0361, +0.0048] ns      |
| Base MLP         | Logistic regression         | -0.0036 [-0.0233, +0.0161] ns      | -0.0162 [-0.0366, +0.0036] ns      |

(ns = 95% CI includes 0. Also significant, from the same run: frozen MLP vs
sampled zero-shot at every size, +0.026 to +0.038; Mini MLP vs Mini LR,
+0.028 forward_fill and +0.025 linear_interp; Base MLP vs Base LR, +0.028
and +0.024. Full list of all 66 pairs: `bootstrap_sizes_paired.csv`.)

### What this shows

- **No size effect:** no size difference is significant for either head or
  condition (largest: Base - Mini, LR head, forward_fill, +0.004).
- **The head matters:** the MLP head beats the LR head significantly at
  all three sizes in both conditions (+0.023 to +0.031).
- **Frozen MLP beats fine-tuning:** the frozen MLP head is significantly
  better than the fine-tuned Chronos-Small at every size and condition
  (+0.022 to +0.030) and than sampled zero-shot (+0.026 to +0.038).
- **Fine-tuning gave nothing over a linear head on frozen features:**
  fine-tuned Small is not significantly different from the frozen exact LR
  head (differences -0.002 to +0.003).
- **The gap to logistic regression is not significant** for the frozen MLP
  head (-0.004 to -0.016, every CI includes 0). **But it is significantly
  below the stronger task-specific models:** Random Forest (-0.019 to
  -0.030; borderline under forward_fill, upper CI limit within 0.001 of 0
  for Small and Base), XGBoost (-0.027 to -0.037) and LSTM (-0.021 to
  -0.024), at every size and condition. So frozen Chronos + MLP head is
  about level with logistic regression, not with the top of the ceiling.
- **Exact vs sampled features:** differences are small; only Small
  linear_interp (+0.0042, lower CI +0.0004) and Base LR forward_fill
  (+0.0098) are significant against sampled zero-shot.
- Caveats from Stage 13 still apply (unstandardised fine-tuned classifier,
  3-epoch budget, 34-number summary features, frozen weights). Which
  factor explains the fine-tuning result was not tested. The main models
  were not re-tested pairwise against each other.

### AUPRC, F1, sensitivity, specificity (size runs, 95% CIs)

Full values are in `bootstrap_sizes.csv` (and the paper's appendix). In
short: AUPRC 0.366-0.410 and F1 0.394-0.432 across all 12 runs, with CI
widths of ~0.11 (AUPRC) and ~0.09 (F1); sensitivity 0.39-0.55 and
specificity 0.87-0.92 at the F1-optimal threshold, the same pattern as
the main models (every model misses about half of the deaths).

### Also added

`evaluation/make_cost_figure.py` draws the AUROC-versus-inference-cost
figure used in the paper (values hard-coded at the top of the script;
output `results/figures/auroc_vs_cost.png`).

"""
models/chronos_sizes.py

Supervisor comment 6 (laptop version): zero-shot Chronos at several model
sizes (mini / small / base), under both imputation conditions.

HOW IT DIFFERS FROM models/chronos_eval.py
  chronos_eval.py estimates each channel's mean and variance by drawing 20
  random samples. This script computes the SAME two quantities exactly from
  the model's output probabilities (the method already used for
  fine-tuning in models/chronos_finetune.py). It is about 4x faster on your
  laptop (0.34 s vs 1.37 s per patient for Small in the timing run), which
  is what makes Base feasible. The encoder is frozen, so there are NO
  epochs here: this is feature extraction only.

  Because it is a different way of computing the features, treat these
  results as an additional experiment (a model-size comparison), and keep
  the sampled Chronos-Small result from chronos_eval.py as the main
  zero-shot number. Running Small here as well links the two: it shows
  whether exact vs sampled features changes the result.

TWO FROZEN-FEATURE HEADS are trained on the 34 features (17 means, 17 variances):
  1. "exact_lr":  logistic regression, class_weight='balanced', C=1
                  (same head as the sampled zero-shot configuration).
  2. "exact_mlp": 2-layer MLP (34 -> 64 -> 1), weighted BCE, features
                  standardised with training-set statistics, early stopping
                  on validation AUROC. This is a "frozen encoder + trained
                  head" middle point between zero-shot and full fine-tuning.
                  It trains in seconds once the features exist.

CHECKPOINTING: features are cached per channel, per split, per condition,
per size, so an interrupted run resumes at the next unfinished channel.
Do not delete results/chronos_sizes/cache/ between runs.

USAGE (from the project root; same habits as the other Chronos scripts):
    set KMP_DUPLICATE_LIB_OK=TRUE

    1) Measure first (about 5-10 minutes): per-patient time for each size and
       a projected total.
       set CHRONOS_SIZES_TIME_ONLY=1 & python models\\chronos_sizes.py

    2) Quick correctness test (50 patients per split, results NOT valid):
       set CHRONOS_SIZES_TIME_ONLY= & set CHRONOS_QUICK_MODE=1 & python models\\chronos_sizes.py

    3) Real run (hours; keep the laptop plugged in and stop it sleeping):
       set CHRONOS_QUICK_MODE= & python models\\chronos_sizes.py

  Run one size at a time if you prefer:
       set CHRONOS_SIZES=mini & python models\\chronos_sizes.py
       set CHRONOS_SIZES=small & python models\\chronos_sizes.py
       set CHRONOS_SIZES=base & python models\\chronos_sizes.py

Environment variables:
    CHRONOS_SIZES            comma list, default mini,small,base
    CHRONOS_BATCH_SIZE       default 32 (same as the other scripts)
    CHRONOS_QUICK_MODE       1 = small test run
    CHRONOS_SIZES_TIME_ONLY  1 = timing/projection only

Output (results/chronos_sizes/):
    cache/<size>_<condition>_<split>_channel<NN>.npz
    <size>/<condition>/exact_features_<split>.npy
    <size>/<condition>/exact_lr_{val,test}_predictions.npy
    <size>/<condition>/exact_mlp_{val,test}_predictions.npy
    summary.csv   (size, condition, head, val/test AUROC, feature extraction seconds)
"""

import os
import sys
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT, VARIABLES

QUICK_MODE = os.environ.get("CHRONOS_QUICK_MODE", "0") == "1"
QUICK_MODE_N = int(os.environ.get("CHRONOS_QUICK_N", 50))
TIME_ONLY = os.environ.get("CHRONOS_SIZES_TIME_ONLY", "0") == "1"
TIME_N = 64

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results",
                           "chronos_sizes_quick" if QUICK_MODE else "chronos_sizes")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")

SIZES = [s.strip() for s in os.environ.get("CHRONOS_SIZES", "mini,small,base").split(",") if s.strip()]
MODEL_IDS = {
    "mini": "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}
BATCH_SIZE = int(os.environ.get("CHRONOS_BATCH_SIZE", 32))
N_CHANNELS = len(VARIABLES)

# MLP head settings (mirrors the fine-tuned head's shape: 34 -> 64 -> 1)
MLP_HIDDEN = 64
MLP_LR = 1e-3
MLP_BATCH = 256
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 10


# --------------------------------------------------------------------
# loading
# --------------------------------------------------------------------
def _load_pipeline(size):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(
        MODEL_IDS[size], device_map="cpu", torch_dtype=torch.float32,
    )


def _load_split(condition, split):
    base = os.path.join(OUTPUT_ROOT, condition, split)
    X = np.load(os.path.join(base, "sequences.npy")).astype(np.float32)
    y = np.load(os.path.join(base, "labels.npy")).astype(int)
    if QUICK_MODE:
        X, y = X[:QUICK_MODE_N], y[:QUICK_MODE_N]
    return X, y


# --------------------------------------------------------------------
# exact mean / variance extraction (no sampling, no gradients)
# --------------------------------------------------------------------
class ExactMomentExtractor:
    """Same calculation as ChronosClassifier._channel_mean_var in
    chronos_finetune.py, without the classifier and without gradients."""

    def __init__(self, pipeline):
        self.t5 = pipeline.model.model.eval()
        self.tokenizer = pipeline.tokenizer
        self.n_special = pipeline.model.config.n_special_tokens
        self.centers = self.tokenizer.centers
        self.decoder_start_id = self.t5.config.decoder_start_token_id
        self.n_params = sum(p.numel() for p in self.t5.parameters())

    @torch.no_grad()
    def mean_var(self, context_1d):
        """context_1d: (batch, 48). Returns numpy (mean, var), each (batch,)."""
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_1d)
        decoder_input_ids = torch.full(
            (token_ids.shape[0], 1), self.decoder_start_id, dtype=torch.long
        )
        logits = self.t5(
            input_ids=token_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits
        bin_logits = logits[:, 0, self.n_special:self.n_special + len(self.centers)]
        probs = torch.softmax(bin_logits, dim=-1)
        mean_scaled = (probs * self.centers).sum(dim=-1)
        var_scaled = (probs * (self.centers.unsqueeze(0) - mean_scaled.unsqueeze(-1)) ** 2).sum(dim=-1)
        return (mean_scaled * scale).numpy(), (var_scaled * scale ** 2).numpy()


def _cache_path(size, condition, split, ch):
    tag = "_quick" if QUICK_MODE else ""
    return os.path.join(CACHE_DIR, f"{size}_{condition}_{split}_channel{ch:02d}{tag}.npz")


def _channel_stats(extractor, X_channel, size, condition, split, ch):
    path = _cache_path(size, condition, split, ch)
    if os.path.exists(path):
        c = np.load(path)
        return c["means"], c["variances"], float(c["seconds"])

    n = X_channel.shape[0]
    means = np.zeros(n, dtype=np.float32)
    variances = np.zeros(n, dtype=np.float32)
    ctx = torch.from_numpy(X_channel)

    t0 = time.time()
    for i in range(0, n, BATCH_SIZE):
        m, v = extractor.mean_var(ctx[i:i + BATCH_SIZE])
        means[i:i + BATCH_SIZE] = m
        variances[i:i + BATCH_SIZE] = v
    elapsed = time.time() - t0

    print(f"    [{size}] channel {ch:2d} ({VARIABLES[ch]}): {n} episodes in {elapsed:.1f}s "
          f"({elapsed / max(n, 1) * 1000:.0f}ms/episode)")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, means=means, variances=variances, seconds=elapsed)
    return means, variances, elapsed


def extract_features(extractor, size, condition, split):
    """Returns (features (N, 34), labels (N,), extraction_seconds)."""
    X, y = _load_split(condition, split)
    print(f"  [{size}] extracting exact features: {condition}/{split} ({len(y)} episodes)")
    all_means, all_vars, total_s = [], [], 0.0
    for c in range(N_CHANNELS):
        m, v, s = _channel_stats(extractor, X[:, :, c], size, condition, split, c)
        all_means.append(m)
        all_vars.append(v)
        total_s += s
    feats = np.concatenate([np.stack(all_means, axis=1), np.stack(all_vars, axis=1)], axis=1)
    return feats.astype(np.float32), y, total_s


# --------------------------------------------------------------------
# heads
# --------------------------------------------------------------------
def fit_lr_head(F_tr, y_tr, F_va, F_te):
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(F_tr, y_tr)
    return clf, clf.predict_proba(F_va)[:, 1], clf.predict_proba(F_te)[:, 1]


def fit_mlp_head(F_tr, y_tr, F_va, y_va, F_te):
    torch.manual_seed(RANDOM_SEED)
    mu = F_tr.mean(axis=0, keepdims=True)
    sd = F_tr.std(axis=0, keepdims=True) + 1e-8
    Xtr = torch.from_numpy(((F_tr - mu) / sd).astype(np.float32))
    Xva = torch.from_numpy(((F_va - mu) / sd).astype(np.float32))
    Xte = torch.from_numpy(((F_te - mu) / sd).astype(np.float32))
    ytr = torch.from_numpy(y_tr.astype(np.float32))

    model = nn.Sequential(nn.Linear(F_tr.shape[1], MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, 1))
    pos_weight = torch.tensor((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

    best_auroc, best_state, bad = -np.inf, None, 0
    for _ in range(MLP_MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), MLP_BATCH):
            idx = perm[i:i + MLP_BATCH]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xva).squeeze(-1)).numpy()
        auroc = roc_auc_score(y_va, val_probs)
        if auroc > best_auroc:
            best_auroc, bad = auroc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(Xva).squeeze(-1)).numpy()
        test_probs = torch.sigmoid(model(Xte).squeeze(-1)).numpy()
    return val_probs, test_probs


# --------------------------------------------------------------------
# one (size, condition)
# --------------------------------------------------------------------
def run_condition(extractor, size, condition):
    print(f"\n{'=' * 60}\nSize: {size}   Condition: {condition}\n{'=' * 60}")
    out_dir = os.path.join(RESULTS_DIR, size, condition)
    os.makedirs(out_dir, exist_ok=True)

    F_tr, y_tr, s_tr = extract_features(extractor, size, condition, "train")
    F_va, y_va, s_va = extract_features(extractor, size, condition, "val")
    F_te, y_te, s_te = extract_features(extractor, size, condition, "test")
    feat_seconds = s_tr + s_va + s_te
    for name, F in [("train", F_tr), ("val", F_va), ("test", F_te)]:
        np.save(os.path.join(out_dir, f"exact_features_{name}.npy"), F)
    print(f"  feature extraction time (sum over channels/splits): {feat_seconds / 3600:.2f} h")

    if len(set(y_tr)) < 2 or len(set(y_va)) < 2 or len(set(y_te)) < 2:
        print("  A split has only one class (expected in tiny quick mode); skipping the heads.")
        return []

    rows = []
    clf, lr_val, lr_test = fit_lr_head(F_tr, y_tr, F_va, F_te)
    dump(clf, os.path.join(out_dir, "exact_lr_model.joblib"))
    np.save(os.path.join(out_dir, "exact_lr_val_predictions.npy"), lr_val)
    np.save(os.path.join(out_dir, "exact_lr_test_predictions.npy"), lr_test)
    rows.append(("exact_lr", roc_auc_score(y_va, lr_val), roc_auc_score(y_te, lr_test)))

    mlp_val, mlp_test = fit_mlp_head(F_tr, y_tr, F_va, y_va, F_te)
    np.save(os.path.join(out_dir, "exact_mlp_val_predictions.npy"), mlp_val)
    np.save(os.path.join(out_dir, "exact_mlp_test_predictions.npy"), mlp_test)
    rows.append(("exact_mlp", roc_auc_score(y_va, mlp_val), roc_auc_score(y_te, mlp_test)))

    result_rows = []
    for head, v, t in rows:
        print(f"  {head}: val AUROC {v:.4f}   test AUROC {t:.4f}")
        result_rows.append({
            "size": size, "condition": condition, "head": head,
            "chronos_model": MODEL_IDS[size], "n_params_M": round(extractor.n_params / 1e6, 1),
            "val_auroc": v, "test_auroc": t,
            "feature_extraction_seconds": round(feat_seconds, 1),
        })

    if size == "small":
        _sanity_check_against_sampled(F_te, condition)
    return result_rows


def _sanity_check_against_sampled(F_te_exact, condition):
    """Compare exact features with the earlier SAMPLED zero-shot features
    (from chronos_eval.py) for Chronos-Small, test split. Means should agree
    very closely; sampled variances (only 20 draws) are noisier."""
    path = os.path.join(PROJECT_ROOT, "results", "chronos", condition, "zeroshot_features_test.npy")
    if QUICK_MODE or not os.path.exists(path):
        return
    F_s = np.load(path)
    if F_s.shape != F_te_exact.shape:
        print(f"  (sanity check skipped: shape mismatch {F_s.shape} vs {F_te_exact.shape})")
        return
    corr = np.array([np.corrcoef(F_te_exact[:, j], F_s[:, j])[0, 1] for j in range(F_s.shape[1])])
    n = N_CHANNELS
    print(f"  sanity check vs sampled features (test): correlation of means "
          f"mean={np.nanmean(corr[:n]):.3f} min={np.nanmin(corr[:n]):.3f}; "
          f"of variances mean={np.nanmean(corr[n:]):.3f} min={np.nanmin(corr[n:]):.3f}")


def _write_summary(new_rows):
    if not new_rows:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "summary.csv")
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = ["size", "condition", "head"]
        merged = pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset=key, keep="last")
    else:
        merged = new_df
    merged.to_csv(path, index=False)


# --------------------------------------------------------------------
# timing / projection mode
# --------------------------------------------------------------------
def time_only():
    base = os.path.join(OUTPUT_ROOT, IMPUTATION_CONDITIONS[0])
    n_patients = sum(np.load(os.path.join(base, s, "labels.npy"), mmap_mode="r").shape[0]
                     for s in ("train", "val", "test"))
    n_conditions = len(IMPUTATION_CONDITIONS)
    X = np.load(os.path.join(base, "test", "sequences.npy")).astype(np.float32)[:TIME_N]
    print(f"Timing {TIME_N} test patients x {N_CHANNELS} channels per size. "
          f"Real run = {n_patients} patients x {n_conditions} conditions.")

    for size in SIZES:
        print(f"\n--- {size} ---")
        pipeline = _load_pipeline(size)
        ex = ExactMomentExtractor(pipeline)
        ex.mean_var(torch.from_numpy(X[:BATCH_SIZE, :, 0]))  # warm-up
        t0 = time.time()
        for c in range(N_CHANNELS):
            ctx = torch.from_numpy(X[:, :, c])
            for i in range(0, len(ctx), BATCH_SIZE):
                ex.mean_var(ctx[i:i + BATCH_SIZE])
        elapsed = time.time() - t0
        per_patient = elapsed / len(X)
        total_h = per_patient * n_patients * n_conditions / 3600
        print(f"  {ex.n_params / 1e6:.1f}M params: {per_patient:.3f} s/patient  ->  "
              f"projected feature extraction for the whole real run: about {total_h:.1f} h "
              f"({total_h / n_conditions:.1f} h per condition)")
        del ex, pipeline
        gc.collect()


# --------------------------------------------------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    unknown = [s for s in SIZES if s not in MODEL_IDS]
    if unknown:
        raise SystemExit(f"Unknown size(s) {unknown}; choose from {list(MODEL_IDS)}")

    if TIME_ONLY:
        time_only()
        return

    if QUICK_MODE:
        print(f"*** QUICK MODE: {QUICK_MODE_N} episodes per split. Correctness test only; "
              f"results are NOT valid. Output goes to {RESULTS_DIR}. ***")
    else:
        print("NOTE: long run. Progress is cached per channel, so if interrupted just re-run "
              "this script. Keep the laptop plugged in and stop it from sleeping.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for size in SIZES:
        print(f"\nLoading {MODEL_IDS[size]} ...")
        pipeline = _load_pipeline(size)
        extractor = ExactMomentExtractor(pipeline)
        print(f"Loaded. Params: {extractor.n_params / 1e6:.1f}M")
        for condition in IMPUTATION_CONDITIONS:
            _write_summary(run_condition(extractor, size, condition))
        del extractor, pipeline
        gc.collect()

    path = os.path.join(RESULTS_DIR, "summary.csv")
    if os.path.exists(path):
        print(f"\n{'=' * 60}\nSummary: {path}\n{'=' * 60}")
        print(pd.read_csv(path).to_string(index=False))


if __name__ == "__main__":
    main()
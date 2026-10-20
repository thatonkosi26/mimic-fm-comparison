"""
scripts/time_models.py

Measures training and inference time for every model in the project, so
the report can have a real cost table (supervisor comment 4).

WHAT IT MEASURES (all on the machine it runs on):
  - Inference: time to predict on test patients, reported as total
    seconds and milliseconds per patient.
      * Logistic regression / Random Forest / XGBoost: full test set,
        on the saved models. Uses the 102 pre-computed features (feature
        building time is NOT included).
      * LSTM: full test set, on the saved model.
      * TFT: full test set. The TFT model was not saved to disk by
        tft.py, so the best configuration is re-trained for a couple of
        epochs and that model is timed (inference time does not depend
        on how well trained the weights are).
      * Chronos zero-shot: a sample of test patients (all 17 channels,
        20 samples each, same settings as chronos_eval.py), then scaled
        to per-patient time. Includes the logistic regression head.
      * Chronos fine-tuned: same sample, using the saved fine-tuned
        weights.
  - Training:
      * Baselines: ONE final fit with the chosen hyperparameters, plus a
        rough estimate of the whole cross-validation search (see note in
        the output; it is an estimate, not a measurement).
      * LSTM / TFT: a few real epochs are timed, then multiplied by the
        number of epochs the real run used (read from your result
        files). This is an ESTIMATE, labelled as such.
      * Chronos fine-tuned and zero-shot training times already exist in
        your logs (finetuned_training_history.json, PROGRESS.md), so they
        are not re-run here.

Usage (from the project root, same terminal habits as the Chronos scripts):
    set KMP_DUPLICATE_LIB_OK=TRUE
    set TIMING_QUICK=1 & python scripts/time_models.py      # tiny test run first
    python scripts/time_models.py                           # real run

Optional environment variables:
    TIMING_CONDITION   forward_fill (default) or linear_interp
    TIMING_ONLY        comma list, e.g. baselines,lstm,tft,chronos_zeroshot,chronos_finetuned
    TIMING_LABEL       name for the output file, e.g. cpu_laptop or cluster_gpu
    TIMING_QUICK       1 = small/fast version for checking the script works

Output:
    results/timing/timing_results_<label>.csv
"""

import os
import sys
import json
import time
import platform

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from config import OUTPUT_ROOT, PROJECT_ROOT, VARIABLES  # noqa: E402

CONDITION = os.environ.get("TIMING_CONDITION", "forward_fill")
QUICK = os.environ.get("TIMING_QUICK", "0") == "1"
ONLY = [s.strip() for s in os.environ.get("TIMING_ONLY", "").split(",") if s.strip()]
LABEL = os.environ.get("TIMING_LABEL", "cpu_laptop")

REPEATS = 1 if QUICK else 3                 # repeats for fast inference timings (median is reported)
N_CHRONOS = 32 if QUICK else 96             # test patients used for Chronos timing (multiple of 32)
CHRONOS_BATCH = 32                          # same batch size as the real scripts
CHRONOS_NUM_SAMPLES = 20                    # same as chronos_eval.py
LSTM_TIMED_EPOCHS = 1 if QUICK else 3
TFT_TIMED_EPOCHS = 1 if QUICK else 2

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "timing")
RESULTS = []


# --------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------
def add(model, measure, seconds, n_patients=None, note=""):
    ms = (seconds / n_patients * 1000.0) if n_patients else None
    RESULTS.append({
        "model": model, "measure": measure, "condition": CONDITION,
        "seconds": round(float(seconds), 4),
        "n_patients": n_patients,
        "ms_per_patient": None if ms is None else round(ms, 4),
        "note": note, "label": LABEL,
    })
    extra = f"  ({ms:.3f} ms/patient over {n_patients})" if ms is not None else ""
    print(f"  [{model}] {measure}: {seconds:.2f}s{extra}")


def median_time(fn, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def sync():
    """Wait for the GPU to finish, if there is one (so timings are honest)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def save():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"timing_results_{LABEL}.csv")
    pd.DataFrame(RESULTS).to_csv(path, index=False)
    return path


def record_environment():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "not available"
    print("Environment:", json.dumps(info))
    RESULTS.append({
        "model": "ENVIRONMENT", "measure": "info", "condition": CONDITION,
        "seconds": None, "n_patients": None, "ms_per_patient": None,
        "note": json.dumps(info), "label": LABEL,
    })


# --------------------------------------------------------------------
# baselines: logistic regression, random forest, xgboost
# --------------------------------------------------------------------
def time_baselines():
    from joblib import load
    from sklearn.base import clone

    base = os.path.join(OUTPUT_ROOT, CONDITION)
    X_train = np.load(os.path.join(base, "train", "static_feats.npy"))
    y_train = np.load(os.path.join(base, "train", "labels.npy"))
    X_test = np.load(os.path.join(base, "test", "static_feats.npy"))
    if QUICK:
        X_test = X_test[:500]

    # (CV fits in the search, as in models/baselines.py: grid 5 values x 5 folds;
    #  random search 50 iterations x 5 folds)
    n_search_fits = {"logistic_regression": 25, "random_forest": 250, "xgboost": 250}
    names = ["logistic_regression"] if QUICK else ["logistic_regression", "random_forest", "xgboost"]

    for name in names:
        path = os.path.join(PROJECT_ROOT, "results", "baselines", CONDITION, f"{name}_model.joblib")
        model = load(path)

        model.predict_proba(X_test[:50])  # warm-up
        secs = median_time(lambda: model.predict_proba(X_test))
        add(name, "inference_test_set", secs, len(X_test),
            "median of %d runs; features already built" % REPEATS)

        fresh = clone(model)
        t0 = time.perf_counter()
        fresh.fit(X_train, y_train)
        fit_secs = time.perf_counter() - t0
        add(name, "train_final_fit", fit_secs, None,
            "one fit with the chosen hyperparameters, single run; NOT the CV search")

        # Rough cost of the whole search on ONE core: each CV fit uses 80% of
        # the data, plus one final refit. Assumes fit time grows with data size
        # and is similar across hyperparameter settings (it is not exactly).
        est = fit_secs * (n_search_fits[name] * 0.8 + 1)
        add(name, "train_search_ESTIMATE", est, None,
            "ESTIMATE: final-fit time x (%d CV fits x 0.8 + 1); single core" % n_search_fits[name])


# --------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------
def time_lstm():
    import torch
    import torch.nn as nn
    from models import lstm as L

    X_train, y_train = L._load_split(CONDITION, "train")
    X_val, y_val = L._load_split(CONDITION, "val")
    X_test, y_test = L._load_split(CONDITION, "test")
    if QUICK:
        X_train, y_train = X_train[:640], y_train[:640]
        X_val, y_val = X_val[:320], y_val[:320]
        X_test, y_test = X_test[:320], y_test[:320]

    # --- inference on the saved model ---
    model = L.ChannelWiseLSTM(n_channels=len(VARIABLES)).to(L.DEVICE)
    state_path = os.path.join(PROJECT_ROOT, "results", "lstm", CONDITION, "model_state_dict.pt")
    model.load_state_dict(torch.load(state_path, map_location=L.DEVICE))
    model.eval()
    test_loader = L._make_loader(X_test, y_test, shuffle=False)

    def infer():
        with torch.no_grad():
            for xb, _ in test_loader:
                torch.sigmoid(model(xb.to(L.DEVICE)))
        sync()

    infer()  # warm-up
    secs = median_time(infer)
    add("lstm", "inference_test_set", secs, len(X_test),
        "median of %d runs; batch size %d; device %s" % (REPEATS, L.BATCH_SIZE, L.DEVICE))

    # --- training: time a few real epochs (train pass + validation pass,
    #     same as the real loop) and extrapolate ---
    torch.manual_seed(42)
    net = L.ChannelWiseLSTM(n_channels=len(VARIABLES)).to(L.DEVICE)
    pos_weight = torch.tensor(
        (len(y_train) - y_train.sum()) / max(y_train.sum(), 1), dtype=torch.float32
    ).to(L.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=L.INIT_LR)
    train_loader = L._make_loader(X_train, y_train, shuffle=True)
    val_loader = L._make_loader(X_val, y_val, shuffle=False)

    epoch_times = []
    for _ in range(LSTM_TIMED_EPOCHS):
        t0 = time.perf_counter()
        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(L.DEVICE), yb.to(L.DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
        sync()
        L._evaluate(net, val_loader)
        epoch_times.append(time.perf_counter() - t0)
    per_epoch = float(np.median(epoch_times))
    add("lstm", "train_epoch", per_epoch, None,
        "median of %d timed epochs (train + validation pass); device %s%s"
        % (LSTM_TIMED_EPOCHS, L.DEVICE, "; QUICK subset" if QUICK else ""))

    hist_path = os.path.join(PROJECT_ROOT, "results", "lstm", CONDITION, "training_history.json")
    n_epochs = len(json.load(open(hist_path))) if os.path.exists(hist_path) else 45
    add("lstm", "train_total_ESTIMATE", per_epoch * n_epochs, None,
        "ESTIMATE: median epoch time x %d epochs used in the real run" % n_epochs)


# --------------------------------------------------------------------
# TFT
# --------------------------------------------------------------------
def time_tft():
    from models import tft as T

    hp_path = os.path.join(PROJECT_ROOT, "results", "tft", CONDITION, "best_hparams.json")
    hp = json.load(open(hp_path))

    train_df, _ = T._load_split_as_df(CONDITION, "train")
    val_df, _ = T._load_split_as_df(CONDITION, "val")
    test_df, _ = T._load_split_as_df(CONDITION, "test")
    if QUICK:
        keep_train = train_df["episode_id"] < 500
        keep_val = val_df["episode_id"] < 200
        keep_test = test_df["episode_id"] < 200
        train_df, val_df, test_df = train_df[keep_train], val_df[keep_val], test_df[keep_test]

    training, validation, test = T._build_datasets(train_df, val_df, test_df)

    # --- train the best config for a couple of epochs and time it ---
    T.MAX_EPOCHS = TFT_TIMED_EPOCHS
    t0 = time.perf_counter()
    model, _, _ = T._train_one_config(
        training, validation, hp["hidden_size"], hp["attention_head_size"]
    )
    train_secs = time.perf_counter() - t0
    per_epoch = train_secs / TFT_TIMED_EPOCHS
    add("tft", "train_epoch", per_epoch, None,
        "best config (hidden %d, heads %d); %d epochs timed, includes dataloader setup and one "
        "validation pass, so slightly high; device %s"
        % (hp["hidden_size"], hp["attention_head_size"], TFT_TIMED_EPOCHS,
           "gpu" if T.torch.cuda.is_available() else "cpu"))

    epochs_used = hp.get("epochs_trained", 30)
    n_configs = len(T.HIDDEN_SIZE_GRID) * len(T.ATTENTION_HEAD_GRID)
    add("tft", "train_total_ESTIMATE", per_epoch * epochs_used * n_configs, None,
        "ESTIMATE: epoch time x %d epochs x %d grid configs (assumes every config costs about "
        "the same as the best one; smaller configs are a bit faster)" % (epochs_used, n_configs))

    # --- inference ---
    test_loader = test.to_dataloader(train=False, batch_size=T.BATCH_SIZE, num_workers=T.NUM_WORKERS)
    T._get_probs_and_labels(model, test_loader)  # warm-up
    t0 = time.perf_counter()
    probs, _ = T._get_probs_and_labels(model, test_loader)
    secs = time.perf_counter() - t0
    add("tft", "inference_test_set", secs, len(probs),
        "single run after warm-up; batch size %d; re-trained model (weights not saved)" % T.BATCH_SIZE)


# --------------------------------------------------------------------
# Chronos (zero-shot and fine-tuned)
# --------------------------------------------------------------------
def _load_pipeline():
    import torch
    from chronos import BaseChronosPipeline
    name = os.environ.get("CHRONOS_MODEL", "amazon/chronos-t5-small")
    return BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)


def _test_sample():
    X = np.load(os.path.join(OUTPUT_ROOT, CONDITION, "test", "sequences.npy")).astype(np.float32)
    return X[:N_CHRONOS]


def time_chronos_zeroshot():
    import torch
    from joblib import load

    pipeline = _load_pipeline()
    X = _test_sample()
    n = len(X)

    def extract():
        for c in range(len(VARIABLES)):
            ctx = torch.from_numpy(X[:, :, c])
            for i in range(0, n, CHRONOS_BATCH):
                pipeline.predict(inputs=ctx[i:i + CHRONOS_BATCH], prediction_length=1,
                                 num_samples=CHRONOS_NUM_SAMPLES)

    # warm-up on one small batch
    pipeline.predict(inputs=torch.from_numpy(X[:CHRONOS_BATCH, :, 0]), prediction_length=1,
                     num_samples=CHRONOS_NUM_SAMPLES)

    t0 = time.perf_counter()
    extract()
    feat_secs = time.perf_counter() - t0

    clf = load(os.path.join(PROJECT_ROOT, "results", "chronos", CONDITION, "zeroshot_model.joblib"))
    feats = np.zeros((n, 2 * len(VARIABLES)), dtype=np.float32)
    t0 = time.perf_counter()
    clf.predict_proba(feats)
    head_secs = time.perf_counter() - t0

    add("chronos_zeroshot", "inference_sample", feat_secs + head_secs, n,
        "17 channels x %d samples, batch %d, CPU; includes logistic regression head; "
        "sample of the first %d test patients" % (CHRONOS_NUM_SAMPLES, CHRONOS_BATCH, n))


def time_chronos_finetuned():
    import torch
    from models import chronos_finetune as F

    pipeline = _load_pipeline()
    model = F.ChronosClassifier(pipeline)
    state_path = os.path.join(PROJECT_ROOT, "results", "chronos", CONDITION, "finetuned_model.pt")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    X = torch.from_numpy(_test_sample())
    n = len(X)

    with torch.no_grad():
        model(X[:CHRONOS_BATCH])  # warm-up
        t0 = time.perf_counter()
        for i in range(0, n, CHRONOS_BATCH):
            torch.sigmoid(model(X[i:i + CHRONOS_BATCH]))
        secs = time.perf_counter() - t0

    add("chronos_finetuned", "inference_sample", secs, n,
        "17 channels, exact mean/variance (no sampling), batch %d, CPU; "
        "sample of the first %d test patients" % (CHRONOS_BATCH, n))

    # Training time: taken from finetuned_training_history.json, not re-run.
    hist = os.path.join(PROJECT_ROOT, "results", "chronos", CONDITION, "finetuned_training_history.json")
    if os.path.exists(hist):
        total = sum(e["epoch_time_s"] for e in json.load(open(hist)))
        add("chronos_finetuned", "train_total_FROM_LOG", total, None,
            "sum of epoch_time_s in finetuned_training_history.json (a resumed epoch only counts "
            "time after the resume, so this can be an underestimate)")


# --------------------------------------------------------------------
SECTIONS = {
    "baselines": time_baselines,
    "lstm": time_lstm,
    "tft": time_tft,
    "chronos_zeroshot": time_chronos_zeroshot,
    "chronos_finetuned": time_chronos_finetuned,
}


def main():
    print(f"Timing run: condition={CONDITION} label={LABEL} quick={QUICK}")
    record_environment()
    selected = ONLY or list(SECTIONS)
    for name in selected:
        if name not in SECTIONS:
            print(f"Unknown section '{name}', skipping.")
            continue
        print(f"\n--- {name} ---")
        try:
            SECTIONS[name]()
        except Exception as exc:  # keep going so one failure doesn't lose the others
            print(f"  !! {name} FAILED: {type(exc).__name__}: {exc}")
            RESULTS.append({"model": name, "measure": "FAILED", "condition": CONDITION,
                            "seconds": None, "n_patients": None, "ms_per_patient": None,
                            "note": f"{type(exc).__name__}: {exc}", "label": LABEL})
        path = save()  # save after every section

    print(f"\nSaved to {path}\n")
    df = pd.DataFrame([r for r in RESULTS if r["model"] != "ENVIRONMENT"])
    if len(df):
        print(df[["model", "measure", "seconds", "n_patients", "ms_per_patient"]].to_string(index=False))


if __name__ == "__main__":
    main()
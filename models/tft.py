"""
models/tft.py

Implements Section 3.5.2's TFT baseline using pytorch-forecasting's
TemporalFusionTransformer, adapted for binary classification.

IMPORTANT DESIGN NOTE: pytorch-forecasting is built for multi-horizon
FORECASTING (predicting future continuous/categorical values), not
sequence classification. To adapt it per the proposal ("replacing the
quantile forecasting output head with a sigmoid binary classification
output"), this implementation:
  - Treats the 48-hour episode as 47 encoder steps (hours 0-46) + 1
    "prediction" step (hour 47), matching pytorch-forecasting's
    encoder/decoder framing.
  - Sets the target at every hour to the (constant) mortality label,
    and uses CrossEntropy loss with output_size=2 (i.e. a 2-class
    softmax classification head at the single prediction step) rather
    than a literal sigmoid -- this is pytorch-forecasting's native
    mechanism for classification-type targets and is mathematically
    equivalent (softmax over 2 classes vs. sigmoid over 1 logit).
  - LR reduction uses pytorch-forecasting's built-in
    reduce_on_plateau_patience (monitors val_loss, the library's native
    mechanism), rather than a custom val-AUROC-based scheduler as used
    for the LSTM. Early stopping also monitors val_loss. This is a
    deliberate, documented deviation from literally re-using the LSTM's
    AUROC-based schedule/stopping -- val_loss-based scheduling is
    standard practice for this library and avoids fragile custom
    Lightning callbacks. Val/test AUROC are still computed and used for
    model selection and reporting, consistent with Section 3.6.
  - "Hidden size, attention head size, dropout tuned on validation set"
    (Section 3.5.2) is implemented as a small grid search (see
    HIDDEN_SIZE_GRID / ATTENTION_HEAD_GRID below), selecting the
    combination with the best validation AUROC.

Usage:
    python models/tft.py

Output (same layout as baselines.py / lstm.py):
    results/tft/<condition>/best_hparams.json
    results/tft/<condition>/val_predictions.npy
    results/tft/<condition>/test_predictions.npy
    results/tft/summary.csv
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")  # pytorch-forecasting/lightning are quite verbose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT, VARIABLES

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tft")

SEQ_LEN = 48
ENCODER_LEN = SEQ_LEN - 1   # 47 hours of input
PREDICTION_LEN = 1          # 1 "step" carrying the classification target
DROPOUT = 0.3               # matches the LSTM's dropout, per Section 3.5.2
INIT_LR = 1e-3
LR_REDUCE_PATIENCE = 3
EARLY_STOP_PATIENCE = 10
MAX_EPOCHS = int(os.environ.get("TFT_MAX_EPOCHS", 30))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

# Small grid for "tuned on the validation set" (Section 3.5.2). Kept
# deliberately small since this runs on CPU by default -- set
# TFT_QUICK_MODE=1 to skip the search entirely and use only the first
# combination, if runtime is a concern on your machine.
HIDDEN_SIZE_GRID = [16, 32]
ATTENTION_HEAD_GRID = [1, 4]
QUICK_MODE = os.environ.get("TFT_QUICK_MODE", "0") == "1"


def _sequences_to_long_df(X, y):
    """Vectorised conversion of (N, 48, 17) arrays into the long-format
    dataframe pytorch-forecasting's TimeSeriesDataSet requires. Avoids a
    slow Python-level loop over hundreds of thousands of rows."""
    N, T, V = X.shape
    assert T == SEQ_LEN and V == len(VARIABLES)

    episode_ids = np.repeat(np.arange(N), T)
    time_idx = np.tile(np.arange(T), N)
    labels = np.repeat(y.astype(int), T)

    data = {"episode_id": episode_ids, "time_idx": time_idx, "label": labels}
    X_flat = X.reshape(N * T, V)
    for j, name in enumerate(VARIABLES):
        data[name] = X_flat[:, j]
    return pd.DataFrame(data)


def _load_split_as_df(condition, split):
    base = os.path.join(OUTPUT_ROOT, condition, split)
    X = np.load(os.path.join(base, "sequences.npy")).astype(np.float32)
    y = np.load(os.path.join(base, "labels.npy")).astype(int)
    return _sequences_to_long_df(X, y), y


def _build_datasets(train_df, val_df, test_df):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="label",
        group_ids=["episode_id"],
        min_encoder_length=ENCODER_LEN,
        max_encoder_length=ENCODER_LEN,
        min_prediction_length=PREDICTION_LEN,
        max_prediction_length=PREDICTION_LEN,
        time_varying_unknown_reals=VARIABLES,
        target_normalizer=NaNLabelEncoder(),
        categorical_encoders={"label": NaNLabelEncoder(add_nan=False)},
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=False, stop_randomization=True
    )
    test = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=False, stop_randomization=True
    )
    return training, validation, test


def _get_probs_and_labels(model, dataloader):
    raw = model.predict(dataloader, mode="raw", return_y=True)
    logits = raw.output.prediction              # (batch, 1, 2)
    probs = torch.softmax(logits, dim=-1)[:, 0, 1].numpy()
    labels = raw.y[0].numpy().reshape(-1)
    return probs, labels


def _train_one_config(training, validation, hidden_size, attention_head_size):
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import CrossEntropy
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    torch.manual_seed(RANDOM_SEED)

    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=DROPOUT,
        hidden_continuous_size=max(hidden_size // 2, 4),
        loss=CrossEntropy(),
        output_size=2,
        learning_rate=INIT_LR,
        reduce_on_plateau_patience=LR_REDUCE_PATIENCE,
        log_interval=-1,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, mode="min")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto",
        enable_progress_bar=True, logger=False, enable_checkpointing=False,
        callbacks=[early_stop], gradient_clip_val=0.1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_probs, val_labels = _get_probs_and_labels(model, val_loader)
    from sklearn.metrics import roc_auc_score
    val_auroc = roc_auc_score(val_labels, val_probs)
    return model, val_auroc, trainer.current_epoch


def run_condition(condition):
    print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
    train_df, _ = _load_split_as_df(condition, "train")
    val_df, _ = _load_split_as_df(condition, "val")
    test_df, _ = _load_split_as_df(condition, "test")

    training, validation, test = _build_datasets(train_df, val_df, test_df)
    test_loader = test.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    grid = [(HIDDEN_SIZE_GRID[0], ATTENTION_HEAD_GRID[0])] if QUICK_MODE else [
        (h, a) for h in HIDDEN_SIZE_GRID for a in ATTENTION_HEAD_GRID
    ]

    best_model, best_val_auroc, best_hparams, best_epochs = None, -np.inf, None, None
    for hidden_size, attn_heads in grid:
        print(f"\n--- Training TFT: hidden_size={hidden_size}, "
              f"attention_head_size={attn_heads} ---")
        model, val_auroc, epochs_trained = _train_one_config(
            training, validation, hidden_size, attn_heads
        )
        print(f"  val AUROC: {val_auroc:.4f}  (epochs trained: {epochs_trained})")
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model = model
            best_hparams = {"hidden_size": hidden_size, "attention_head_size": attn_heads}
            best_epochs = epochs_trained

    print(f"\nBest config: {best_hparams}  (val AUROC {best_val_auroc:.4f})")

    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_probs, val_labels = _get_probs_and_labels(best_model, val_loader)
    test_probs, test_labels = _get_probs_and_labels(best_model, test_loader)

    from sklearn.metrics import roc_auc_score
    val_auroc = roc_auc_score(val_labels, val_probs)
    test_auroc = roc_auc_score(test_labels, test_probs)
    print(f"  final val AUROC: {val_auroc:.4f}   test AUROC: {test_auroc:.4f}")

    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "val_predictions.npy"), val_probs)
    np.save(os.path.join(out_dir, "test_predictions.npy"), test_probs)
    with open(os.path.join(out_dir, "best_hparams.json"), "w") as f:
        json.dump({**best_hparams, "epochs_trained": best_epochs}, f, indent=2)

    return {
        "condition": condition, "model": "tft",
        "val_auroc": val_auroc, "test_auroc": test_auroc,
        "best_hparams": json.dumps(best_hparams),
    }


def main():
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device}")
    if device == "CPU":
        print("  NOTE: TFT is heavier per-epoch than the LSTM (attention + "
              "gating networks). With the default hyperparameter grid "
              f"({len(HIDDEN_SIZE_GRID) * len(ATTENTION_HEAD_GRID)} configs "
              "x 2 conditions), this will take considerably longer than "
              "the LSTM step. Set TFT_QUICK_MODE=1 to skip the grid search "
              "and train a single config if runtime is a concern.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = [run_condition(cond) for cond in IMPUTATION_CONDITIONS]

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nSummary written to {summary_path}\n{'=' * 60}")
    print(summary[["condition", "model", "val_auroc", "test_auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
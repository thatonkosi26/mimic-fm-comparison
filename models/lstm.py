"""
models/lstm.py

Implements Section 3.5.2's LSTM baseline:
  - Each of the 17 variables processed by a SHARED two-layer LSTM
    (hidden dim 128) -- i.e. one LSTM's weights applied independently to
    each of the 17 channels, not 17 separate LSTMs. This matches the
    proposal's literal wording in 3.5.2 ("a shared two-layer LSTM").
  - The 17 resulting hidden state vectors are concatenated, passed
    through dropout (p=0.3), then a sigmoid-activated linear classifier.
  - Weighted binary cross-entropy (pos_weight = neg/pos training ratio,
    per Section 3.4's uniform class-imbalance handling).
  - Adam, initial lr 1e-3, reduced by factor 0.5 when validation AUROC
    plateaus, early stopping with patience of 10 epochs.

Trains and evaluates on BOTH imputation conditions.

Usage:
    python models/lstm.py

Output (mirrors models/baselines.py's layout for consistency):
    results/lstm/<condition>/model_state_dict.pt
    results/lstm/<condition>/val_predictions.npy
    results/lstm/<condition>/test_predictions.npy
    results/lstm/<condition>/training_history.json
    results/lstm/summary.csv
"""

import os
import sys
import json
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT, VARIABLES

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lstm")

# --- hyperparameters, exactly as specified in Section 3.5.2 ---
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
INIT_LR = 1e-3
LR_REDUCE_FACTOR = 0.5
LR_PATIENCE = 3          # epochs of no val-AUROC improvement before reducing LR
EARLY_STOP_PATIENCE = 10  # epochs of no val-AUROC improvement before stopping
MAX_EPOCHS = 100

# --- practical knobs, overridable via env vars (same pattern as baselines.py) ---
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))  # 0 avoids multiprocessing issues on Windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


class ChannelWiseLSTM(nn.Module):
    """One shared LSTM applied independently to each of the 17 channels.
    Input: (batch, 48, 17). Output: raw logits (batch,) -- apply sigmoid
    externally (kept as logits here so BCEWithLogitsLoss can be used,
    which is numerically more stable than sigmoid + BCELoss)."""

    def __init__(self, n_channels=17, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_channels * hidden_dim, 1)

    def forward(self, x):
        batch, seq_len, n_channels = x.shape
        assert n_channels == self.n_channels, (
            f"expected {self.n_channels} channels, got {n_channels}"
        )
        # (batch, 48, 17) -> (batch*17, 48, 1): each channel becomes its
        # own "sample" fed through the SAME shared LSTM weights.
        x = x.permute(0, 2, 1).reshape(batch * n_channels, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                                  # (batch*17, hidden_dim)
        last_hidden = last_hidden.reshape(batch, n_channels * self.hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.classifier(out).squeeze(-1)              # (batch,)
        return logits


def _load_split(condition, split):
    base = os.path.join(OUTPUT_ROOT, condition, split)
    X = np.load(os.path.join(base, "sequences.npy")).astype(np.float32)
    y = np.load(os.path.join(base, "labels.npy")).astype(np.float32)
    return X, y


def _make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)


def _evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auroc = roc_auc_score(labels, probs)
    return auroc, probs


def train_lstm(X_train, y_train, X_val, y_val):
    _set_seed()
    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)

    model = ChannelWiseLSTM(n_channels=len(VARIABLES)).to(DEVICE)

    pos_weight = torch.tensor(
        (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        dtype=torch.float32,
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_REDUCE_FACTOR, patience=LR_PATIENCE
    )

    best_val_auroc = -np.inf
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_auroc, _ = _evaluate(model, val_loader)
        scheduler.step(val_auroc)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_auroc > best_val_auroc
        if improved:
            best_val_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"  epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}  "
              f"val_auroc={val_auroc:.4f}  lr={current_lr:.2e}"
              f"{'  *best*' if improved else ''}")

        history.append({
            "epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
            "val_auroc": float(val_auroc), "lr": float(current_lr),
        })

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    return model, best_val_auroc, history


def run_condition(condition):
    print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
    X_train, y_train = _load_split(condition, "train")
    X_val, y_val = _load_split(condition, "val")
    X_test, y_test = _load_split(condition, "test")

    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)

    model, best_val_auroc, history = train_lstm(X_train, y_train, X_val, y_val)

    val_loader = _make_loader(X_val, y_val, shuffle=False)
    test_loader = _make_loader(X_test, y_test, shuffle=False)
    val_auroc, val_probs = _evaluate(model, val_loader)
    test_auroc, test_probs = _evaluate(model, test_loader)
    print(f"  final val AUROC: {val_auroc:.4f}   test AUROC: {test_auroc:.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    np.save(os.path.join(out_dir, "val_predictions.npy"), val_probs)
    np.save(os.path.join(out_dir, "test_predictions.npy"), test_probs)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {
        "condition": condition, "model": "lstm",
        "val_auroc": val_auroc, "test_auroc": test_auroc,
        "epochs_trained": len(history),
    }


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("  NOTE: no GPU detected -- this will be considerably slower "
              "than on a GPU machine. Budget real time for this, same "
              "power-management precautions as the earlier long steps.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = [run_condition(cond) for cond in IMPUTATION_CONDITIONS]

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nSummary written to {summary_path}\n{'=' * 60}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
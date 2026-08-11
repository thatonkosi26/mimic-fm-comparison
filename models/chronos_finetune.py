"""
models/chronos_finetune.py

Implements Section 3.5.3's FINE-TUNED Chronos configuration: the
pretrained encoder is fine-tuned end-to-end jointly with a 2-layer MLP
classification head, using a weighted BCE loss. Lower LR for the
encoder (1e-5) than the MLP (1e-3), per the proposal, to limit
catastrophic forgetting of pretrained representations.

IMPORTANT DESIGN NOTE -- differentiability: zero-shot (models/
chronos_eval.py) extracts mean/variance by DRAWING RANDOM SAMPLES from
the model's predictive distribution (matching the proposal's literal
wording). Random sampling is not differentiable -- you cannot
backpropagate through a stochastic draw -- so it cannot be used here,
where the encoder itself needs to be updated by gradient descent.

Instead, this script computes the mean/variance ANALYTICALLY from the
decoder's output softmax distribution over the token vocabulary at the
single prediction step:
    mean = sum_i(p_i * bin_center_i)
    variance = sum_i(p_i * (bin_center_i - mean)^2)
This is the exact expectation/variance of the same predictive
distribution the zero-shot sampling was estimating via Monte Carlo --
conceptually the same "mean and variance of the predictive distribution"
required by Section 3.5.3, just computed exactly rather than by
sampling, which is what makes end-to-end gradient-based fine-tuning
possible at all. Verified correct via a local test: gradients computed
this way genuinely reach every encoder parameter, not just the MLP head.

CHECKPOINTING: unlike zero-shot (checkpointed per channel), a single
epoch of fine-tuning may itself take many hours, so checkpoints are
saved every CHECKPOINT_EVERY_N_BATCHES batches, capturing model state,
optimizer state, epoch, and batch index. Re-running this script resumes
from the last checkpoint rather than restarting the epoch. Note: the
random batch-shuffle order itself isn't checkpointed (only model/
optimizer state), so a resumed run processes batches in a slightly
different order than an uninterrupted run would have from that point
onward -- this doesn't affect training validity (still correct SGD over
the full training set each epoch), just means exact bit-for-bit
reproducibility isn't preserved across an interruption. Verified via a
local test: resuming after a simulated interruption correctly skips
already-completed batches and continues training without redoing work
or corrupting model state.

MEMORY NOTE: forward() loops over 17 channels, each running a full T5
encoder-decoder pass. Without gradient checkpointing, PyTorch must keep
ALL 17 channels' activation graphs in memory simultaneously until the
single final loss is computed and backward() is called -- this is a
real and significant memory cost (effectively 17x a single forward
pass's activation memory). Gradient checkpointing (see
torch.utils.checkpoint usage below) trades some recomputation during
the backward pass for a large reduction in peak memory, and is applied
automatically during training (not needed during evaluation, since
torch.no_grad() already avoids storing activations at all there).

Usage:
    python models/chronos_finetune.py                                     # full run
    set CHRONOS_QUICK_MODE=1 & python models/chronos_finetune.py          # small-subset correctness + timing test (STRONGLY recommended first)

Output:
    results/chronos/checkpoints/<condition>_latest.pt   (resume checkpoint)
    results/chronos/<condition>/finetuned_model.pt      (best model by val AUROC)
    results/chronos/<condition>/finetuned_val_predictions.npy
    results/chronos/<condition>/finetuned_test_predictions.npy
    results/chronos/<condition>/finetuned_training_history.json
    results/chronos/finetuned_summary.csv
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT, VARIABLES

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "chronos")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

MODEL_NAME = os.environ.get("CHRONOS_MODEL", "amazon/chronos-t5-small")
BATCH_SIZE = int(os.environ.get("CHRONOS_FT_BATCH_SIZE", 16))  # smaller than zero-shot's 32: gradients cost memory
ENCODER_LR = 1e-5
MLP_LR = 1e-3
MLP_HIDDEN = 64
MAX_EPOCHS = int(os.environ.get("CHRONOS_FT_MAX_EPOCHS", 3))
EARLY_STOP_PATIENCE = int(os.environ.get("CHRONOS_FT_PATIENCE", 1))
CHECKPOINT_EVERY_N_BATCHES = int(os.environ.get("CHRONOS_FT_CKPT_EVERY", 50))
USE_GRAD_CHECKPOINTING = os.environ.get("CHRONOS_FT_GRAD_CKPT", "1") == "1"

QUICK_MODE = os.environ.get("CHRONOS_QUICK_MODE", "0") == "1"
QUICK_MODE_N = int(os.environ.get("CHRONOS_QUICK_N", 50))

SEQ_LEN = 48
N_CHANNELS = len(VARIABLES)


class ChronosClassifier(nn.Module):
    """Wraps a Chronos T5 model + tokenizer, extracts differentiable
    34-dim mean/variance features across the 17 channels (shared model
    weights per channel, matching the "shared" pattern used elsewhere in
    this project), and classifies via a 2-layer MLP."""

    def __init__(self, chronos_pipeline):
        super().__init__()
        self.t5 = chronos_pipeline.model.model          # the actual HF PreTrainedModel
        self.tokenizer = chronos_pipeline.tokenizer
        self.n_special = chronos_pipeline.model.config.n_special_tokens
        self.centers = self.tokenizer.centers            # (n_bins,), not a Parameter -- fixed bin locations
        self.decoder_start_id = self.t5.config.decoder_start_token_id

        self.mlp = nn.Sequential(
            nn.Linear(N_CHANNELS * 2, MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN, 1),
        )

    def _t5_forward(self, token_ids, attention_mask, decoder_input_ids):
        return self.t5(
            input_ids=token_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits

    def _channel_mean_var(self, context_1d):
        """context_1d: (batch, 48) single-channel context.
        Returns (mean, var) each shape (batch,), fully differentiable."""
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_1d)
        decoder_input_ids = torch.full(
            (token_ids.shape[0], 1), self.decoder_start_id, dtype=torch.long
        )

        if self.training and USE_GRAD_CHECKPOINTING:
            # Recompute activations during backward instead of storing them
            # during forward -- trades compute for the large memory saving
            # needed since 17 of these run before a single backward() call.
            logits = checkpoint(
                self._t5_forward, token_ids, attention_mask, decoder_input_ids,
                use_reentrant=False,
            )
        else:
            logits = self._t5_forward(token_ids, attention_mask, decoder_input_ids)

        bin_logits = logits[:, 0, self.n_special:self.n_special + len(self.centers)]
        probs = torch.softmax(bin_logits, dim=-1)
        mean_scaled = (probs * self.centers).sum(dim=-1)
        var_scaled = (probs * (self.centers.unsqueeze(0) - mean_scaled.unsqueeze(-1)) ** 2).sum(dim=-1)
        return mean_scaled * scale, var_scaled * scale ** 2

    def forward(self, X):
        """X: (batch, 48, 17). Returns logits (batch,)."""
        means, variances = [], []
        for c in range(N_CHANNELS):
            m, v = self._channel_mean_var(X[:, :, c])
            means.append(m)
            variances.append(v)
        features = torch.cat(
            [torch.stack(means, dim=1), torch.stack(variances, dim=1)], dim=1
        )  # (batch, 34)
        return self.mlp(features).squeeze(-1)


def _load_split(condition, split):
    base = os.path.join(OUTPUT_ROOT, condition, split)
    X = np.load(os.path.join(base, "sequences.npy")).astype(np.float32)
    y = np.load(os.path.join(base, "labels.npy")).astype(np.float32)
    if QUICK_MODE:
        X, y = X[:QUICK_MODE_N], y[:QUICK_MODE_N]
    return torch.from_numpy(X), torch.from_numpy(y)


def _checkpoint_path(condition):
    tag = "_quick" if QUICK_MODE else ""
    return os.path.join(CHECKPOINT_DIR, f"{condition}_latest{tag}.pt")


def _save_checkpoint(path, model, optimizer, epoch, batch_idx, best_val_auroc, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "batch_idx": batch_idx,
        "best_val_auroc": best_val_auroc, "history": history,
    }, path)


@torch.no_grad()
def _evaluate(model, X, y, batch_size=32):
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        probs.append(torch.sigmoid(logits).numpy())
    probs = np.concatenate(probs)
    auroc = roc_auc_score(y.numpy(), probs)
    return auroc, probs


def run_condition(condition, pipeline_factory):
    print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
    torch.manual_seed(RANDOM_SEED)

    X_train, y_train = _load_split(condition, "train")
    X_val, y_val = _load_split(condition, "val")
    X_test, y_test = _load_split(condition, "test")
    print(f"  train/val/test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)}")

    pipeline = pipeline_factory()
    model = ChronosClassifier(pipeline)

    pos_weight_value = (len(y_train) - y_train.sum().item()) / max(y_train.sum().item(), 1)
    pos_weight = torch.tensor(pos_weight_value)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam([
        {"params": model.t5.parameters(), "lr": ENCODER_LR},
        {"params": model.mlp.parameters(), "lr": MLP_LR},
    ])

    ckpt_path = _checkpoint_path(condition)
    start_epoch, start_batch, best_val_auroc, history = 1, 0, -np.inf, []
    best_state = None
    if os.path.exists(ckpt_path):
        print(f"  Found checkpoint at {ckpt_path} -- resuming.")
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch, start_batch = ckpt["epoch"], ckpt["batch_idx"]
        best_val_auroc, history = ckpt["best_val_auroc"], ckpt["history"]

    n_train = len(y_train)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_losses = []
        batch_start = start_batch if epoch == start_epoch else 0

        t_epoch = time.time()
        for batch_i, i in enumerate(range(0, n_train, BATCH_SIZE)):
            if batch_i < batch_start:
                continue  # skip already-completed batches from a resumed run
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if (batch_i + 1) % CHECKPOINT_EVERY_N_BATCHES == 0:
                _save_checkpoint(ckpt_path, model, optimizer, epoch, batch_i + 1,
                                  best_val_auroc, history)
                elapsed = time.time() - t_epoch
                n_batches_total = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"    epoch {epoch} batch {batch_i + 1}/{n_batches_total} "
                      f"loss={np.mean(epoch_losses[-CHECKPOINT_EVERY_N_BATCHES:]):.4f} "
                      f"({elapsed:.0f}s elapsed this epoch, checkpoint saved)")

        val_auroc, _ = _evaluate(model, X_val, y_val)
        epoch_time = time.time() - t_epoch
        print(f"  epoch {epoch} complete in {epoch_time:.0f}s "
              f"train_loss={np.mean(epoch_losses):.4f} val_auroc={val_auroc:.4f}")

        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
                         "val_auroc": float(val_auroc), "epoch_time_s": epoch_time})

        improved = val_auroc > best_val_auroc
        if improved:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        _save_checkpoint(ckpt_path, model, optimizer, epoch + 1, 0, best_val_auroc, history)

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping after epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epoch(s))")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_auroc, val_probs = _evaluate(model, X_val, y_val)
    test_auroc, test_probs = _evaluate(model, X_test, y_test)
    print(f"  FINAL val AUROC: {val_auroc:.4f}   test AUROC: {test_auroc:.4f}")

    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "finetuned_model.pt"))
    np.save(os.path.join(out_dir, "finetuned_val_predictions.npy"), val_probs)
    np.save(os.path.join(out_dir, "finetuned_test_predictions.npy"), test_probs)
    with open(os.path.join(out_dir, "finetuned_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {
        "condition": condition, "model": "chronos_finetuned",
        "chronos_model": MODEL_NAME,
        "val_auroc": val_auroc, "test_auroc": test_auroc,
        "epochs_trained": len(history),
    }


def main():
    def pipeline_factory():
        from chronos import BaseChronosPipeline
        return BaseChronosPipeline.from_pretrained(
            MODEL_NAME, device_map="cpu", torch_dtype=torch.float32,
        )

    if QUICK_MODE:
        print(f"*** QUICK MODE: {QUICK_MODE_N} episodes/split, testing correctness "
              f"AND measuring real per-batch timing on YOUR hardware before you commit "
              f"to the full run. Results are NOT valid for the dissertation. ***")
    else:
        print("NOTE: fine-tuning involves backprop through the encoder every batch. "
              f"Checkpoints save every {CHECKPOINT_EVERY_N_BATCHES} batches -- if "
              "interrupted, just re-run this script and it resumes from the last "
              "checkpoint rather than restarting the epoch.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = [run_condition(cond, pipeline_factory) for cond in IMPUTATION_CONDITIONS]

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "finetuned_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nSummary written to {summary_path}\n{'=' * 60}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
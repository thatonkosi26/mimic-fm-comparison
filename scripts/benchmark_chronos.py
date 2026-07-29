"""
scripts/benchmark_chronos.py

Times actual Chronos inference on YOUR machine, so we can pick a model
size and estimate total runtime based on real numbers rather than a
guess. Run this BEFORE committing to a full Chronos run.

Usage:
    python scripts/benchmark_chronos.py
"""

import time
import torch

from chronos import BaseChronosPipeline

N_TRIALS = 10
CONTEXT_LEN = 47          # matches the 47-hour encoder window in models/tft.py
N_EPISODES_TOTAL = 21139  # approx: 14681 train + 3222 val + 3236 test (your real counts)
N_CHANNELS = 17
N_CONDITIONS = 2

MODELS_TO_TEST = [
    "amazon/chronos-t5-mini",   # 20M params
    "amazon/chronos-t5-small",  # 46M params
    "amazon/chronos-t5-base",   # 200M params
]


def benchmark(model_name):
    print(f"\n=== {model_name} ===")
    t0 = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name, device_map="cpu", torch_dtype=torch.float32,
    )
    load_time = time.time() - t0
    n_params = sum(p.numel() for p in pipeline.model.parameters())
    print(f"  load time: {load_time:.1f}s, params: {n_params / 1e6:.1f}M")

    context = torch.randn(CONTEXT_LEN)
    _ = pipeline.predict(inputs=context, prediction_length=1)  # warmup

    t0 = time.time()
    for _ in range(N_TRIALS):
        _ = pipeline.predict(inputs=context, prediction_length=1)
    per_call = (time.time() - t0) / N_TRIALS
    print(f"  per single-channel forward pass: {per_call * 1000:.1f}ms")

    per_episode = per_call * N_CHANNELS
    total_seconds = per_episode * N_EPISODES_TOTAL * N_CONDITIONS
    total_hours = total_seconds / 3600
    print(f"  estimated per-episode (17 channels): {per_episode * 1000:.1f}ms")
    print(f"  ESTIMATED TOTAL for full zero-shot pass "
          f"(all episodes x both conditions): {total_hours:.1f} hours")

    del pipeline
    return total_hours


if __name__ == "__main__":
    results = {}
    for model_name in MODELS_TO_TEST:
        try:
            results[model_name] = benchmark(model_name)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for name, hours in results.items():
        print(f"  {name}: ~{hours:.1f} hours for full zero-shot feature extraction")
    print("\nNote: this estimates ZERO-SHOT feature extraction only. "
          "Fine-tuning (Section 3.5.3) involves backward passes too and "
          "will take meaningfully longer per epoch on top of this.")
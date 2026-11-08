"""evaluation/make_cost_figure.py
Draws the AUROC-versus-inference-cost figure used in the paper.
Values: forward-fill test AUROC (results/evaluation/full_results.csv and
results/chronos_sizes/summary.csv) and inference ms/patient from
results/timing/timing_results_cpu_laptop.csv (frozen-feature rows: total
feature extraction seconds / 21,139 patients). Hard-coded here so the figure
is reproducible without the result files; update if the numbers change.
Usage (from the project root): python evaluation\\make_cost_figure.py
Output: results/figures/auroc_vs_cost.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as ml

# (label, ms per patient, forward-fill test AUROC, marker, colour, label offset in points)
PTS = [
    ("Logistic Reg.", 0.0009, 0.8103, "o", "tab:blue", (4, -9)),
    ("XGBoost",       0.011,  0.8332, "o", "tab:blue", (-12, 5)),
    ("Random Forest", 0.056,  0.8252, "o", "tab:blue", (-24, -10)),
    ("TFT",           2.33,   0.8255, "s", "tab:green", (-4, -10)),
    ("LSTM",          2.83,   0.8276, "s", "tab:green", (-4, 5)),
    ("ZS",            1372,   0.7692, "^", "tab:red", (-4, -10)),
    ("FT",            341,    0.7764, "^", "tab:red", (-4, 5)),
    ("Mini",          150,    0.8033, "D", "tab:orange", (-10, -10)),
    ("Small",         341,    0.8053, "D", "tab:orange", (-8, 6)),
    ("Base",          1407,   0.8067, "D", "tab:orange", (-8, 6)),
]

fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
for label, x, y, m, c, off in PTS:
    ax.scatter(x, y, marker=m, color=c, s=22, zorder=3)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off, fontsize=5.5)
ax.set_xscale("log")
ax.set_xlim(3e-4, 6e3)
ax.set_ylim(0.755, 0.845)
ax.set_xlabel("Inference time per patient (ms, log scale, CPU)", fontsize=6.5)
ax.set_ylabel("Test AUROC (forward-fill)", fontsize=6.5)
ax.tick_params(labelsize=6)
ax.grid(alpha=0.3, zorder=0)
handles = [ml.Line2D([], [], marker=m, color=c, linestyle="", markersize=4, label=l)
           for m, c, l in [("o", "tab:blue", "Traditional"),
                           ("s", "tab:green", "LSTM / TFT"),
                           ("^", "tab:red", "Chronos-Small: zero-shot (ZS), fine-tuned (FT)"),
                           ("D", "tab:orange", "Frozen + MLP head: Mini, Small, Base (no CI)")]]
ax.legend(handles=handles, fontsize=4.8, loc="lower left", frameon=True)
fig.tight_layout()
try:  # inside the project: save next to the other Results figures
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PROJECT_ROOT
    out = os.path.join(PROJECT_ROOT, "results", "figures", "auroc_vs_cost.png")
except Exception:  # standalone use
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auroc_vs_cost.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print("saved", out)
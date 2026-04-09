#!/usr/bin/env python3
"""
Plot sub-batch rotation tradeoff: Qwen2.5-7B + Mistral-7B side by side.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
})

# Paper palette
C_THR  = "#A8D5BA"; EC_THR  = "#6BA882"
C_LAT  = "#E8B4A8"; EC_LAT  = "#C07868"

# Data
K = [1, 2, 4, 8]
k_labels = ["$K$=1", "$K$=2", "$K$=4", "$K$=8"]

qwen_thr =    [290, 178, 119, 119]
qwen_sp =     [2.45, 1.50, 1.00, 1.00]
qwen_p50 =    [21.7, 42.6, 58.6, 58.6]

mistral_thr = [195, 61, 56, 53]
mistral_sp =  [3.66, 1.15, 1.05, 1.00]
mistral_p50 = [22.5, 79.5, 83.5, 87.9]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)

width = 0.35
x = np.arange(len(K))

for ax, thr, sp, p50, title, kv_info in [
    (ax1, qwen_thr, qwen_sp, qwen_p50,
     "Qwen2.5-7B (56 KB/tok)", "Bud/KV: 89%→11%"),
    (ax2, mistral_thr, mistral_sp, mistral_p50,
     "Mistral-7B (128 KB/tok)", "Bud/KV: 38%→6%"),
]:
    bars_t = ax.bar(x - width/2, thr, width,
                    color=C_THR, edgecolor=EC_THR, linewidth=0.7, zorder=3)
    ax.set_ylabel("Throughput (tok/s)", color=EC_THR, fontsize=9)
    ax.tick_params(axis='y', labelcolor=EC_THR)
    ax.set_ylim(0, max(thr) * 1.25)

    for i, (bar, s) in enumerate(zip(bars_t, sp)):
        if s > 1.03:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f"{s:.2f}×", ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', color=EC_THR)

    ax_r = ax.twinx()
    ax_r.bar(x + width/2, p50, width,
             color=C_LAT, edgecolor=EC_LAT, linewidth=0.7, zorder=3)
    ax_r.set_ylabel("P50 latency (ms)", color=EC_LAT, fontsize=9)
    ax_r.tick_params(axis='y', labelcolor=EC_LAT)
    ax_r.set_ylim(0, max(p50) * 1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels, fontsize=8)
    ax.set_xlabel("Sub-batch size", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    ax.grid(axis='y', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

fig.legend(
    [plt.Rectangle((0,0),1,1, fc=C_THR, ec=EC_THR),
     plt.Rectangle((0,0),1,1, fc=C_LAT, ec=EC_LAT)],
    ["Throughput (tok/s)", "P50 step latency (ms)"],
    loc='upper center', bbox_to_anchor=(0.5, 1.06),
    ncol=2, fontsize=8.5, frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_subbatch_tradeoff.{ext}")
plt.close(fig)
print(f"Wrote {OUT}/fig_subbatch_tradeoff.pdf/png")

#!/usr/bin/env python3
"""Plot tail latency comparison: FIFO vs OrchKvCache vs OrchKv-noattn."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE

C_GPU  = "#A8D5BA"; EC_GPU  = "#6BA882"
C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"
C_NOAT = "#C8B8D8"; EC_NOAT = "#9080A8"

BAR_KW = dict(linewidth=0.7)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

modes   = ["GPU-Only", "FIFO", "OrchKvCache", "OrchKv-noattn"]
colors  = [C_GPU, C_FIFO, C_ORKV, C_NOAT]
edges   = [EC_GPU, EC_FIFO, EC_ORKV, EC_NOAT]

p50  = [17.5, 281.4, 181.1, 181.1]
p95  = [20.8, 386.2, 188.8, 189.4]
p99  = [26.3, 567.6, 192.3, 190.5]

x = np.arange(len(modes))
w = 0.25

fig, ax = plt.subplots(figsize=(8, 3.5))

bars_p50 = ax.bar(x - w, p50, w, color=[c for c in colors],
                  edgecolor=[e for e in edges], alpha=0.5, label="P50", **BAR_KW)
bars_p95 = ax.bar(x,     p95, w, color=[c for c in colors],
                  edgecolor=[e for e in edges], alpha=0.75, label="P95", **BAR_KW)
bars_p99 = ax.bar(x + w, p99, w, color=[c for c in colors],
                  edgecolor=[e for e in edges], alpha=1.0, label="P99", **BAR_KW)

for bar in bars_p99:
    h = bar.get_height()
    if h > 100:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10,
                f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.annotate("2.95×", xy=(1 + w, 567.6), xytext=(1.8, 520),
            fontsize=10, fontweight="bold", color="#C07868",
            arrowprops=dict(arrowstyle="->", color="#C07868", lw=1.5),
            ha="center")

ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=9)
ax.set_ylabel("Per-token latency (ms)")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#888888", alpha=0.5, label="P50"),
    Patch(facecolor="#888888", alpha=0.75, label="P95"),
    Patch(facecolor="#888888", alpha=1.0, label="P99"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.3)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_tail_latency.{ext}")
plt.close(fig)
print(f"Saved fig_tail_latency.pdf/png to {OUT}")

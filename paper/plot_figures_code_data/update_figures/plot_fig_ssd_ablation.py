#!/usr/bin/env python3
"""SSD write-mode ablation: GPU+DRAM vs per-block SSD vs batched SSD."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

configs = ["DRAM\nonly", "SSD\nper-block", "SSD\nbatched-8"]
tok_s   = [175.4, 179.1, 179.3]
d2s     = [0, 256, 256]
s2d     = [0, 256, 256]

colors_bar = ["#A8D5BA", "#E8B4A8", "#9CC0D8"]
edges_bar  = ["#6BA882", "#C07868", "#5A90B0"]

C_D2S = "#E8B4A8"; EC_D2S = "#C07868"
C_S2D = "#9CC0D8"; EC_S2D = "#5A90B0"

fig = plt.figure(figsize=(6.5, 3.2))

gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.8, 15], hspace=0.02, wspace=0.4)

ax1 = fig.add_subplot(gs[2, 0])
ax2 = fig.add_subplot(gs[2, 1])
ax_title_b = fig.add_subplot(gs[0, 1])
ax_title_b.axis("off")
ax_leg = fig.add_subplot(gs[1, 1])
ax_leg.axis("off")

x = np.arange(len(configs))
bars = ax1.bar(x, tok_s, 0.55, color=colors_bar, edgecolor=edges_bar, linewidth=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=8)
ax1.set_ylabel("Throughput (tok/s)", fontsize=9)
ax1.set_ylim(170, 183)
for i, v in enumerate(tok_s):
    ax1.text(i, v + 0.4, f"{v}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)
ax1.set_title("(a) E2E throughput", fontsize=10)

w = 0.25
ax2.bar(x - w/2, d2s, w, color=C_D2S, edgecolor=EC_D2S, linewidth=0.7)
ax2.bar(x + w/2, s2d, w, color=C_S2D, edgecolor=EC_S2D, linewidth=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=8)
ax2.set_ylabel("Migration count", fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax_title_b.text(0.5, 0.0, "(b) SSD migrations", ha="center", va="bottom", fontsize=10, transform=ax_title_b.transAxes)

legend_handles = [
    Patch(facecolor=C_D2S, edgecolor=EC_D2S, linewidth=0.7, label="DRAM\u2192SSD"),
    Patch(facecolor=C_S2D, edgecolor=EC_S2D, linewidth=0.7, label="SSD\u2192DRAM"),
]
ax_leg.legend(handles=legend_handles, loc="center", ncol=2,
              frameon=False, fontsize=7, columnspacing=1.0,
              handletextpad=0.3, handlelength=1.2)

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_ssd_ablation.{ext}")
plt.close(fig)
print(f"Saved fig_ssd_ablation.pdf/png to {HERE}")

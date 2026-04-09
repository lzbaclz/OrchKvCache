#!/usr/bin/env python3
"""Line chart: precision@K under multi-request contention."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

caps = [5, 10, 20, 30]
ema     = [1.000, 1.000, 1.000, 1.000]
noattn  = [0.640, 0.800, 0.900, 0.933]
recency = [0.640, 0.800, 0.900, 0.933]

fig, ax = plt.subplots(figsize=(4.5, 3.0))

ax.plot(caps, ema, "o-", color="#5A90B0", linewidth=2, markersize=7,
        label="Full EMA ($\\alpha{=}0.7$)", zorder=3)
ax.plot(caps, noattn, "s--", color="#C07868", linewidth=1.8, markersize=6,
        label="No-attn ($\\alpha{=}0$)")
ax.plot(caps, recency, "^:", color="#909090", linewidth=1.5, markersize=6,
        label="Recency-only (LRU)")

ax.fill_between(caps, noattn, ema, alpha=0.12, color="#5A90B0")

ax.annotate("36% miss", xy=(5, 0.640), xytext=(8, 0.55),
            fontsize=9, color="#C07868", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C07868", lw=1.2))

ax.set_xlabel("GPU capacity (% of total blocks)")
ax.set_ylabel("Precision@K")
ax.set_ylim(0.5, 1.05)
ax.set_xticks(caps)
ax.set_xticklabels([f"{c}%" for c in caps])
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.grid(axis="y", alpha=0.3)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_contention_identity.{ext}")
plt.close(fig)
print(f"Saved fig_contention_identity.pdf/png to {OUT}")

#!/usr/bin/env python3
"""
Fig 19: Realistic workload throughput bar chart.
Exact same bar architecture as fig01.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "realistic_workload.json"

C_GPU  = "#A8D5BA"; EC_GPU  = "#6BA882"
C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"
BAR_KW = dict(linewidth=0.7)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

GROUP_ORDER = [
    ("longcontext-mix", "LLaMA-2-7B"),
    ("longcontext-mix", "Qwen2.5-7B"),
    ("sharegpt-like", "LLaMA-2-7B"),
    ("sharegpt-like", "Qwen2.5-7B"),
]
GROUP_LABELS = [f"{m}\n{wl}" for wl, m in GROUP_ORDER]

def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    modes  = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    fills  = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    edges  = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(GROUP_ORDER))
    w = 0.25

    for i, mode in enumerate(modes):
        vals = []
        for wl, m in GROUP_ORDER:
            v = next((r["avg_throughput"] for r in data
                      if r["model"] == m and r["workload"] == wl and r["mode"] == mode), 0)
            vals.append(v)
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label=labels[mode],
                      color=fills[mode], edgecolor=edges[mode], **BAR_KW)
        for bar, v in zip(bars, vals):
            if v > 0 and mode != "baseline":
                ax.text(bar.get_x() + bar.get_width() / 2, v + 8,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=8)
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.5)
    ax.grid(axis="y", alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig19_realistic_throughput.{ext}")
    plt.close(fig)
    print(f"Saved fig19 to {HERE}")

if __name__ == "__main__":
    main()

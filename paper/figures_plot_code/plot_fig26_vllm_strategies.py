#!/usr/bin/env python3
"""
Fig 26: vLLM victim-selection strategies bar chart.
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
DATA_PATH = HERE.parent / "exp_vllm_multi_pressure.json"

C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_PROG = "#A8D5BA"; EC_PROG = "#6BA882"
C_BLKS = "#9CC0D8"; EC_BLKS = "#5A90B0"
BAR_KW = dict(linewidth=0.7)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    gpu_utils = sorted({r["gpu_util"] for r in data})
    prompts = sorted({r["num_prompts"] for r in data})
    GROUP_ORDER = [(gu, np_) for gu in gpu_utils for np_ in prompts]
    GROUP_LABELS = [f"u={gu}\nn={np_}" for gu, np_ in GROUP_ORDER]

    strats = ["fifo", "progress", "block_score"]
    labels = {"fifo": "FIFO", "progress": "Progress", "block_score": "Block-score"}
    fills  = {"fifo": C_FIFO, "progress": C_PROG, "block_score": C_BLKS}
    edges  = {"fifo": EC_FIFO, "progress": EC_PROG, "block_score": EC_BLKS}

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(GROUP_ORDER))
    w = 0.25

    for i, s in enumerate(strats):
        vals = []
        for gu, np_ in GROUP_ORDER:
            v = next((r["avg_throughput"] for r in data
                      if r["gpu_util"] == gu and r["num_prompts"] == np_ and r["strategy"] == s), 0)
            vals.append(v)
        ax.bar(x + (i - 1) * w, vals, w,
               label=labels[s],
               color=fills[s], edgecolor=edges[s], **BAR_KW)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=7)
    ax.set_ylabel("Avg throughput (tok/s)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.5)
    ax.grid(axis="y", alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig26_vllm_strategies.{ext}")
    plt.close(fig)
    print(f"Saved fig26 to {HERE}")

if __name__ == "__main__":
    main()

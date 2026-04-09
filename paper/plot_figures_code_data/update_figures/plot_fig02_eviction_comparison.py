#!/usr/bin/env python3
"""
Fig 02: 4-model eviction comparison bar chart (log scale).

Data:   multimodel_e2e_4models.json  (same directory)
Output: fig02_eviction_comparison.pdf/png (same directory)

Config matching paper: budget=50MB, seq=2048, nreq=8.
Models: Qwen2.5-7B, Mistral-7B, LLaMA-2-7B, LLaMA-2-13B

Demonstrates 139-597x migration reduction by OrchKvCache vs FIFO.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "multimodel_e2e_4models.json"

C_GPU  = "#A8D5BA"; EC_GPU  = "#6BA882"
C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"

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

MODEL_ORDER = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]

DISPLAY_NAMES = {
    "Qwen2.5-7B":  "Qwen2.5-7B",
    "Mistral-7B":   "Mistral-7B",
    "LLaMA-2-7B":  "LLaMA-2-7B",
    "LLaMA-2-13B": "LLaMA-2-13B",
}


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    sel = [r for r in data
           if r["seq_len"] == 2048
           and r.get("n_requests", r.get("num_requests")) == 8
           and r.get("gpu_budget_mb", r.get("budget_mb")) == 50]

    modes  = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    fills  = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    edges  = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(MODEL_ORDER))
    w = 0.25

    for i, mode in enumerate(modes):
        for j, m in enumerate(MODEL_ORDER):
            v = next((r["total_evictions"] for r in sel
                      if r["model"] == m and r["mode"] == mode), 0)
            if v > 0:
                ax.bar(x[j] + (i - 1) * w, v, w,
                       color=fills[mode], edgecolor=edges[mode], **BAR_KW)

    legend_handles = [Patch(facecolor=fills[mode], edgecolor=edges[mode],
                            linewidth=0.7, label=labels[mode]) for mode in modes]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.5)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_ylabel("Total evictions (log scale)")
    ax.grid(axis="y", alpha=0.3)

    for i, m in enumerate(MODEL_ORDER):
        fifo_v = next((r["total_evictions"] for r in sel
                       if r["model"] == m and r["mode"] == "naive"), 0)
        orkv_v = next((r["total_evictions"] for r in sel
                       if r["model"] == m and r["mode"] == "orchkv"), 1)
        if fifo_v > 0 and orkv_v > 0:
            ratio = fifo_v / orkv_v
            ax.text(i, fifo_v * 1.8, f"{ratio:.0f}\u00d7",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

    HERE.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig02_eviction_comparison.{ext}")
    plt.close(fig)
    print(f"Saved fig02_eviction_comparison.pdf/png to {HERE}")

    print("\n--- Data Summary (budget=50MB, seq=2048, nreq=8) ---")
    print(f"{'Model':<16} {'GPU-Only':>10} {'FIFO':>12} {'OrchKv':>10} {'Ratio':>8}")
    print("-" * 60)
    for m in MODEL_ORDER:
        base = next((r["total_evictions"] for r in sel
                     if r["model"] == m and r["mode"] == "baseline"), 0)
        fifo = next((r["total_evictions"] for r in sel
                     if r["model"] == m and r["mode"] == "naive"), 0)
        orkv = next((r["total_evictions"] for r in sel
                     if r["model"] == m and r["mode"] == "orchkv"), 0)
        ratio = fifo / orkv if orkv > 0 else float("inf")
        print(f"{m:<16} {base:>10,} {fifo:>12,} {orkv:>10,} {ratio:>7.0f}x")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fig 20: Realistic workload eviction bar chart (log scale).
Exact same bar architecture as fig01.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "realistic_workload.json"

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

    modes  = ["naive", "orchkv"]
    labels = {"naive": "FIFO", "orchkv": "OrchKvCache"}
    fills  = {"naive": C_FIFO, "orchkv": C_ORKV}
    edges  = {"naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(GROUP_ORDER))
    w = 0.30

    for i, mode in enumerate(modes):
        vals = []
        for wl, m in GROUP_ORDER:
            v = next((max(r["total_evictions"], 1) for r in data
                      if r["model"] == m and r["workload"] == wl and r["mode"] == mode), 1)
            vals.append(v)
        ax.bar(x + (i - 0.5) * w, vals, w,
               color=fills[mode], edgecolor=edges[mode], **BAR_KW)

    # Annotate reduction ratio
    for gi, (wl, m) in enumerate(GROUP_ORDER):
        naive_v = next((r["total_evictions"] for r in data
                        if r["model"] == m and r["workload"] == wl and r["mode"] == "naive"), 0)
        orch_v = next((r["total_evictions"] for r in data
                       if r["model"] == m and r["workload"] == wl and r["mode"] == "orchkv"), 1)
        if naive_v > 0 and orch_v > 0:
            ratio = naive_v / orch_v
            ax.text(x[gi] + 0.5 * w, orch_v * 1.8, f"{ratio:.0f}×",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2d6a2e")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=8)
    ax.set_ylabel("Total evictions (log scale)")
    legend_handles = [Patch(facecolor=fills[m], edgecolor=edges[m], linewidth=0.7, label=labels[m])
                      for m in modes]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, fontsize=9, columnspacing=1.5)
    ax.grid(axis="y", alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig20_realistic_eviction.{ext}")
    plt.close(fig)
    print(f"Saved fig20 to {HERE}")

if __name__ == "__main__":
    main()

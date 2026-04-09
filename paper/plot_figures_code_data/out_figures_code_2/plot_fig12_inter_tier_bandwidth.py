#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent  # points to plot_figures_code_data/
OUT = Path(__file__).resolve().parent           # output in same dir as this script

C_GPU = "#A8D5BA";  EC_GPU = "#6BA882"
C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"
C_GRAY = "#C8C8C8"; EC_GRAY = "#909090"
C_ORANGE = "#E8C8A0"; EC_ORANGE = "#C09060"
C_PURPLE = "#C8B8D8"; EC_PURPLE = "#9080A8"

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

def _load(name: str):
    with open(BASE / name) as f:
        return json.load(f)

def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/{stem}.pdf/png")

def _top_legend(ax, ncol=3):
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, frameon=False, fontsize=9, columnspacing=1.5,
    )

def _short(m: str) -> str:
    return (m.replace("Qwen2.5-", "Qwen ").replace("Llama-2-7b-hf", "LLaMA-2-7B")
             .replace("LLaMA-2-", "LLaMA ").replace("meta-llama/", "")
             .replace("-7B", " 7B").replace("-13B", " 13B").strip())

def fig12():
    data = _load("benchmark_e8_storage_bw.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    gd = data["gpu_dram"]
    sizes = [r["size_mb"] for r in gd]
    d2h = [r["d2h_gbps"] for r in gd]; h2d = [r["h2d_gbps"] for r in gd]
    xi = np.arange(len(sizes)); w = 0.35
    ax1.bar(xi - w/2, d2h, w, label="D2H", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax1.bar(xi + w/2, h2d, w, label="H2D", color=C_GPU, edgecolor=EC_GPU, **BAR_KW)
    ax1.set_xticks(xi); ax1.set_xticklabels([f"{s:g}" for s in sizes])
    ax1.set_xlabel("Transfer size (MB)"); ax1.set_ylabel("Bandwidth (GB/s)")
    _top_legend(ax1, ncol=2); ax1.grid(axis="y", alpha=0.3)

    ds = data["dram_storage"]
    sizes2 = [r["size_mb"] for r in ds]
    wr = [r["write_gbps"] for r in ds]; rd = [r["read_gbps"] for r in ds]
    xi2 = np.arange(len(sizes2))
    ax2.bar(xi2 - w/2, wr, w, label="Write", color=C_FIFO, edgecolor=EC_FIFO, **BAR_KW)
    ax2.bar(xi2 + w/2, rd, w, label="Read", color=C_ORANGE, edgecolor=EC_ORANGE, **BAR_KW)
    ax2.set_xticks(xi2); ax2.set_xticklabels([f"{s:g}" for s in sizes2])
    ax2.set_xlabel("Transfer size (MB)"); ax2.set_ylabel("Bandwidth (GB/s)")
    _top_legend(ax2, ncol=2); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig12_inter_tier_bandwidth")



if __name__ == "__main__":
    fig12()

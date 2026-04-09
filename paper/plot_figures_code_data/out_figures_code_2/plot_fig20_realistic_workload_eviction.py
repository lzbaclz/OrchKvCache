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

def fig20():
    rows = _load("realistic_workload.json")
    workloads = sorted({r["workload"] for r in rows})
    models = sorted({r["model"] for r in rows})
    modes = [("baseline", "GPU-Only", C_GPU, EC_GPU),
             ("naive", "FIFO", C_FIFO, EC_FIFO),
             ("orchkv", "OrchKvCache", C_ORKV, EC_ORKV)]
    groups = [(wl, m) for wl in workloads for m in models]
    w = 0.22

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 3.8))
    for gi, (wl, m) in enumerate(groups):
        center = gi * 1.35
        for mi, (mk, label, fc, ec) in enumerate(modes):
            v = next((r["total_evictions"] for r in rows
                      if r["workload"] == wl and r["model"] == m and r["mode"] == mk), 0)
            offset = (mi - 1) * w
            if v > 0:
                ax.bar(center + offset, v, w * 0.9,
                       color=fc, edgecolor=ec, **BAR_KW)
    legend_handles = [Patch(facecolor=fc, edgecolor=ec, linewidth=0.7, label=label)
                      for _, label, fc, ec in modes]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.5)
    ax.set_yscale("log")
    ax.set_ylim(bottom=500, top=2e6)
    ax.set_xticks([i * 1.35 for i in range(len(groups))])
    ax.set_xticklabels([f"{_short(m)}\n{wl[:16]}" for wl, m in groups], fontsize=7)
    ax.set_ylabel("Total evictions (log scale)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig20_realistic_eviction")



if __name__ == "__main__":
    fig20()

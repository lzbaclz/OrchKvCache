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

def fig07():
    rows = _load("multimodel_ablation.json")
    models = sorted({r["model"] for r in rows})
    configs = [("gpu-only", "GPU-Only", C_GPU, EC_GPU),
               ("naive-fifo", "FIFO", C_FIFO, EC_FIFO),
               ("orchkv", "OrchKvCache", C_ORKV, EC_ORKV)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    x = np.arange(len(models)); w = 0.25
    for i, (key, label, fc, ec) in enumerate(configs):
        tp = [next((r["throughput"] for r in rows if r["model"] == m and r["config"] == key), 0) for m in models]
        ev = [next((r["evictions"] for r in rows if r["model"] == m and r["config"] == key), 0) for m in models]
        ax1.bar(x + (i-1)*w, tp, w, label=label, color=fc, edgecolor=ec, **BAR_KW)
        ax2.bar(x + (i-1)*w, ev, w, label=label, color=fc, edgecolor=ec, **BAR_KW)
    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models])
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Throughput (tok/s)")
    ax2.set_ylabel("Evictions")
    _top_legend(ax1); _top_legend(ax2)
    plt.tight_layout()
    _save(fig, "fig07_ablation")



if __name__ == "__main__":
    fig07()

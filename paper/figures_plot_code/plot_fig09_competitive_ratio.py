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

    ax.set_ylim(bottom=0)
    _save(fig, stem: str):
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

def fig09():
    data = _load("exp_competitive_ratio.json")
    rows = data["results"]
    blocks = [r["n_blocks"] for r in rows]
    strats = ["OPT", "EMA", "LFU", "LRU", "FIFO"]
    fill_map = {"OPT": C_ORKV, "EMA": C_GPU, "LFU": C_ORANGE, "LRU": C_FIFO, "FIFO": C_PURPLE}
    edge_map = {"OPT": EC_ORKV, "EMA": EC_GPU, "LFU": EC_ORANGE, "LRU": EC_FIFO, "FIFO": EC_PURPLE}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xi = np.arange(len(blocks)); w = 0.15
    for si, s in enumerate(strats):
        vals = [r[f"{s}_evictions"] for r in rows]
        ax.bar(xi + (si - 2) * w, vals, w, label=s,
               color=fill_map[s], edgecolor=edge_map[s], **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("n_blocks"); ax.set_ylabel("Eviction count")
    _top_legend(ax, ncol=5); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig09_competitive_ratio")



if __name__ == "__main__":
    fig09()

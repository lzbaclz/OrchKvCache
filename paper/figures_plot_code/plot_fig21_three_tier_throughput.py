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

def fig21():
    rows = _load("w4_tier_throughput.json")
    models = sorted({r["model"] for r in rows})
    configs = [("GPU-Only", C_GPU, EC_GPU), ("GPU+DRAM", C_FIFO, EC_FIFO), ("GPU+DRAM+SSD", C_ORKV, EC_ORKV)]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xi = np.arange(len(models)); w = 0.25
    for i, (cfg, fc, ec) in enumerate(configs):
        vals = [next((r["avg_throughput_tok_s"] for r in rows if r["model"] == m and r["config"] == cfg), 0) for m in models]
        ax.bar(xi + (i - 1) * w, vals, w, label=cfg, color=fc, edgecolor=ec, **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("Throughput (tok/s)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig21_three_tier_throughput")



if __name__ == "__main__":
    fig21()

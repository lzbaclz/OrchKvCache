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

def fig17():
    rows = _load("exp_selective_restore.json")
    pcts = [1, 3, 5, 10, 20, 30, 50, 70, 90]
    rows = rows[:2]

    fig, axes = plt.subplots(1, len(rows), figsize=(6.5, 3.2), squeeze=False)
    axes = axes[0]
    for idx, r in enumerate(rows):
        ax = axes[idx]
        ema = r.get("avg_ema_coverage", {})
        rnd = r.get("avg_random_coverage", {})
        ema_vals = [float(ema.get(f"top{p}pct", 0)) for p in pcts]
        rand_vals = [float(rnd.get(f"top{p}pct", 0)) for p in pcts]
        xi = np.arange(len(pcts)); w = 0.35
        ax.bar(xi - w/2, ema_vals, w, label="EMA (ours)", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
        ax.bar(xi + w/2, rand_vals, w, label="Random", color=C_ORANGE, edgecolor=EC_ORANGE, **BAR_KW)
        ax.set_xticks(xi); ax.set_xticklabels([f"{p}%" for p in pcts], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Top-K% restored"); ax.set_ylabel("Attn coverage (%)")
        ax.set_ylim(0, 115); ax.axhline(100, color=EC_GRAY, linestyle=":", linewidth=0.8)
        _top_legend(ax, ncol=2); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig17_selective_restore")



if __name__ == "__main__":
    fig17()

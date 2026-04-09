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

def fig10():
    rows = _load("benchmark_e5_policy_sweep.json")
    by_ab = defaultdict(list)
    for r in rows:
        a, b = r.get("alpha"), r.get("beta")
        if a is not None and b is not None:
            by_ab[(float(a), float(b))].append(float(r.get("hot_ratio", 0)))
    alphas = sorted({k[0] for k in by_ab}); betas = sorted({k[1] for k in by_ab})
    mat = np.full((len(alphas), len(betas)), np.nan)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            if (a, b) in by_ab:
                mat[i, j] = np.mean(by_ab[(a, b)])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\alpha$")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Hot ratio")
    _save(fig, "fig10_policy_sweep_heatmap")



if __name__ == "__main__":
    fig10()

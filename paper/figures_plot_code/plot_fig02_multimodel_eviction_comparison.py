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

def fig02():
    data = _load("multimodel_e2e.json")
    sel = [r for r in data if r["seq_len"] == 2048 and r["n_requests"] == 8 and r["gpu_budget_mb"] == 50]
    models = sorted({r["model"] for r in sel})
    modes = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    fills = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    edges = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(models)); w = 0.25
    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            v = next((r["total_evictions"] for r in sel if r["model"] == m and r["mode"] == mode), 0)
            vals.append(v)
        plot_vals = [max(v, 1) for v in vals]
        bars = ax.bar(x + (i - 1) * w, plot_vals, w, label=labels[mode],
                      color=fills[mode], edgecolor=edges[mode], **BAR_KW)
        for j, v in enumerate(vals):
            if v == 0:
                bx = bars[j].get_x() + bars[j].get_width() / 2
                ax.text(bx, 1.5, "0", ha="center", va="bottom", fontsize=8, fontstyle="italic")

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("Total evictions (log scale)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)

    for i, m in enumerate(models):
        fifo_v = next((r["total_evictions"] for r in sel if r["model"] == m and r["mode"] == "naive"), 0)
        orkv_v = next((r["total_evictions"] for r in sel if r["model"] == m and r["mode"] == "orchkv"), 1)
        ratio = fifo_v / orkv_v
        ax.text(i, fifo_v * 1.8, f"{ratio:.0f}×", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333")

    _save(fig, "fig02_multimodel_eviction")



if __name__ == "__main__":
    fig02()

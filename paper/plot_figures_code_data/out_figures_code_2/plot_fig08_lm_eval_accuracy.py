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

def fig08():
    data = _load("exp_lm_eval.json")
    tasks = ["piqa", "rte", "copa", "openbookqa"]
    task_labels = ["PIQA", "RTE", "COPA", "OBQA"]

    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 3.2), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for idx, r in enumerate(data):
        ax = axes[idx]
        xi = np.arange(len(tasks)); w = 0.35
        gpu_vals = [r["gpu_only"].get(t, 0) for t in tasks]
        orch_vals = [r["orchkv"].get(t, 0) for t in tasks]
        ax.bar(xi - w/2, gpu_vals, w, label="GPU-Only", color=C_GPU, edgecolor=EC_GPU, **BAR_KW)
        ax.bar(xi + w/2, orch_vals, w, label="OrchKvCache", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
        for i in range(len(tasks)):
            if gpu_vals[i] == orch_vals[i] and gpu_vals[i] > 0:
                ax.annotate("=", xy=(xi[i], max(gpu_vals[i], orch_vals[i]) + 0.5),
                            ha="center", fontsize=9, fontweight="bold", color="#333")
        ax.set_xticks(xi); ax.set_xticklabels(task_labels, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_ylim(0, 105)
        _top_legend(ax, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig08_lm_eval_accuracy")



if __name__ == "__main__":
    fig08()

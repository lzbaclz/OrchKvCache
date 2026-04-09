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

def fig26():
    rows = _load("exp_vllm_multi_pressure.json")
    gpu_utils = sorted({r["gpu_util"] for r in rows})
    prompts = sorted({r["num_prompts"] for r in rows})
    strats = ["fifo", "progress", "block_score"]
    strat_labels = {"fifo": "FIFO", "progress": "Progress", "block_score": "Block-score"}
    strat_fill = {"fifo": C_FIFO, "progress": C_GPU, "block_score": C_ORKV}
    strat_edge = {"fifo": EC_FIFO, "progress": EC_GPU, "block_score": EC_ORKV}

    groups = [(gu, np_) for gu in gpu_utils for np_ in prompts]
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.9), 3.5))
    w = 0.25
    for gi, (gu, np_) in enumerate(groups):
        center = gi * 1.2
        for si, s in enumerate(strats):
            v = next((r["avg_throughput"] for r in rows
                      if r["gpu_util"] == gu and r["num_prompts"] == np_ and r["strategy"] == s), 0)
            ax.bar(center + (si - 1) * w, v, w * 0.9,
                   color=strat_fill[s], edgecolor=strat_edge[s],
                   label=strat_labels[s] if gi == 0 else None, **BAR_KW)
    ax.set_xticks([i * 1.2 for i in range(len(groups))])
    ax.set_xticklabels([f"u={gu}\nn={np_}" for gu, np_ in groups], fontsize=6)
    ax.set_ylabel("Avg throughput (tok/s)")
    ax.set_ylim(bottom=0)
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig26_vllm_strategies")



if __name__ == "__main__":
    fig26()

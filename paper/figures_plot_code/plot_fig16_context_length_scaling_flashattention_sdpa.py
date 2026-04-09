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

def fig16():
    rows = _load("exp_scale_flashattn.json")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    n = len(by_model)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
    axes = axes[0]
    for ax, (model, mrows) in zip(axes, sorted(by_model.items())):
        mrows = sorted(mrows, key=lambda r: r["seq_len"])
        ctx = [r["seq_len"] for r in mrows]
        labels_c = [f"{c//1024}K" if c >= 1024 else str(c) for c in ctx]
        gpu = [r.get("gpu_only_tok_s", 0) for r in mrows]
        fifo = [r.get("fifo_tok_s", 0) for r in mrows]
        orch = [float(r.get("orchkv_tok_s", 0) or 0) for r in mrows]
        xi = np.arange(len(ctx))
        ax.plot(xi, gpu, marker="s", markersize=5, label="GPU-Only", color=EC_GPU, linewidth=1.5)
        ax.plot(xi, fifo, marker="^", markersize=5, label="FIFO", color=EC_FIFO, linewidth=1.5)
        ax.plot(xi, orch, marker="o", markersize=5, label="OrchKvCache", color=EC_ORKV, linewidth=1.5)
        ax.set_xticks(xi); ax.set_xticklabels(labels_c)
        ax.set_xlabel("Context length"); ax.set_ylabel("Throughput (tok/s)")
        _top_legend(ax)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig16_scale_context")



if __name__ == "__main__":
    fig16()

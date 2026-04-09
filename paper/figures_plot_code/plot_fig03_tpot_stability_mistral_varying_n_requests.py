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

def fig03():
    data = _load("multimodel_e2e.json")
    m0 = "Mistral-7B"
    sub = [r for r in data if r["model"] == m0 and r["seq_len"] == 2048 and r["gpu_budget_mb"] == 50]
    reqs = sorted({r["n_requests"] for r in sub})
    modes = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    colors = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for mode in modes:
        xs, ys = [], []
        for nr in reqs:
            r = next((r for r in sub if r["mode"] == mode and r["n_requests"] == nr), None)
            if r:
                xs.append(nr); ys.append(r["avg_tpot_ms"])
        ax.plot(xs, ys, marker="o", markersize=5, label=labels[mode], color=colors[mode])
    ax.set_xlabel("Concurrent requests"); ax.set_ylabel("Avg TPOT (ms)")
    _top_legend(ax); ax.grid(alpha=0.3)
    _save(fig, "fig03_tpot_stability")



if __name__ == "__main__":
    fig03()

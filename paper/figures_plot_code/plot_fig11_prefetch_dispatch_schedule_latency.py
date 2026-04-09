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

def fig11():
    rows = _load("benchmark_e7_prefetch.json")
    rows = sorted(rows, key=lambda r: r["prefetch_budget"])
    budgets = [r["prefetch_budget"] for r in rows]
    disp = [r["avg_prefetches_dispatched"] for r in rows]
    lat = [r["avg_schedule_us"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    xi = np.arange(len(budgets))
    ax1.bar(xi - 0.2, disp, 0.4, label="Prefetches dispatched",
            color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax1.set_xlabel("Prefetch budget"); ax1.set_ylabel("Dispatches", color=EC_ORKV)
    ax1.set_xticks(xi); ax1.set_xticklabels([str(int(b)) for b in budgets])
    ax1.tick_params(axis="y", labelcolor=EC_ORKV)

    ax2 = ax1.twinx()
    ax2.plot(xi, lat, color=EC_GRAY, marker="o", markersize=5, linewidth=2, label="Schedule latency (us)")
    ax2.set_ylabel("Avg schedule latency (us)", color=EC_GRAY)
    ax2.tick_params(axis="y", labelcolor=EC_GRAY)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2,
               loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=9)
    _save(fig, "fig11_prefetch_dispatch")



if __name__ == "__main__":
    fig11()

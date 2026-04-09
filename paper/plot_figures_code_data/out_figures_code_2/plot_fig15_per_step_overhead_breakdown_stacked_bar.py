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

def fig15():
    data = _load("exp_overhead_breakdown.json")
    results = data["results"]
    part_keys = [
        ("forward_avg_ms", "Forward", C_GPU, EC_GPU),
        ("build_past_kv_avg_ms", "Build KV", C_ORKV, EC_ORKV),
        ("append_token_avg_ms", "Append token", C_ORANGE, EC_ORANGE),
        ("report_attn_avg_ms", "Report attn", C_FIFO, EC_FIFO),
        ("step_schedule_avg_ms", "Schedule", C_PURPLE, EC_PURPLE),
    ]
    labels_x = [r["label"].replace("OrchKvCache ", "OrchKv ") for r in results]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    xi = np.arange(len(results))
    bottoms = np.zeros(len(results))
    for key, label, fc, ec in part_keys:
        vals = np.array([float(r.get(key, 0) or 0) for r in results])
        if vals.sum() > 0:
            ax.bar(xi, vals, 0.6, bottom=bottoms, label=label,
                   color=fc, edgecolor=ec, **BAR_KW)
            bottoms += vals
    ax.set_xticks(xi); ax.set_xticklabels(labels_x, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Time per step (ms)")
    _top_legend(ax, ncol=5); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig15_overhead_breakdown")



if __name__ == "__main__":
    fig15()

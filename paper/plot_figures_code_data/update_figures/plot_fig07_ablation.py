#!/usr/bin/env python3
"""
Fig 07: 4-model ablation — throughput + eviction (log scale).

Data:   multimodel_ablation_4models.json  (same directory)
Output: fig07_ablation.pdf/png (same directory)

Config: seq_len=2048, max_new=128, budget=50MB, single request.
Models: Qwen2.5-7B, Mistral-7B, LLaMA-2-7B, LLaMA-2-13B
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "multimodel_ablation_4models.json"

C_GPU  = "#A8D5BA"; EC_GPU  = "#6BA882"
C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"

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

MODEL_ORDER = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]

CONFIGS = [
    ("gpu-only",    "GPU-Only",     C_GPU,  EC_GPU),
    ("naive-fifo",  "FIFO",         C_FIFO, EC_FIFO),
    ("orchkv",      "OrchKvCache",  C_ORKV, EC_ORKV),
]


def main():
    with open(DATA_PATH) as f:
        rows = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(MODEL_ORDER))
    w = 0.25

    for i, (key, label, fc, ec) in enumerate(CONFIGS):
        tp = []
        ev = []
        for m in MODEL_ORDER:
            r = next((r for r in rows
                      if r["model"] == m and r["config"] == key), None)
            tp.append(r["throughput"] if r else 0)
            ev.append(r["evictions"] if r else 0)

        ax1.bar(x + (i - 1) * w, tp, w,
                label=label, color=fc, edgecolor=ec, **BAR_KW)

        bars = ax2.bar(x + (i - 1) * w, ev, w,
                       label=label, color=fc, edgecolor=ec, **BAR_KW)
        for j, v in enumerate(ev):
            if v == 0:
                bx = x[j] + (i - 1) * w
                ax2.text(bx, 1.2, "0", ha="center", va="bottom",
                         fontsize=7, fontweight="bold")

    ax2.set_yscale("log")
    ax2.set_ylim(bottom=0.5)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, frameon=False, fontsize=10, columnspacing=1.5)

    ax1.set_ylabel("Throughput (tok/s)")
    ax2.set_ylabel("Evictions (log scale)")

    for i, m in enumerate(MODEL_ORDER):
        fifo = next((r["evictions"] for r in rows
                     if r["model"] == m and r["config"] == "naive-fifo"), 0)
        orkv = next((r["evictions"] for r in rows
                     if r["model"] == m and r["config"] == "orchkv"), 1)
        if fifo > 0 and orkv > 0:
            ratio = fifo / orkv
            ax2.text(i, fifo * 1.8, f"{ratio:.0f}\u00d7",
                     ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color="#333333")

    ax1.set_title("(a) Throughput", fontsize=11, pad=24)
    ax2.set_title("(b) Evictions", fontsize=11, pad=24)

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig07_ablation.{ext}")
    plt.close(fig)
    print(f"Saved fig07_ablation.pdf/png to {HERE}")

    print("\n--- Ablation Summary ---")
    print(f"{'Model':<16} {'Config':<12} {'Tput':>8} {'Evict':>10} {'Ratio':>8}")
    print("-" * 58)
    for m in MODEL_ORDER:
        for key, label, _, _ in CONFIGS:
            r = next((r for r in rows
                      if r["model"] == m and r["config"] == key), None)
            t = r["throughput"] if r else 0
            e = r["evictions"] if r else 0
            ratio_s = ""
            if key == "orchkv":
                fifo_e = next((r["evictions"] for r in rows
                               if r["model"] == m
                               and r["config"] == "naive-fifo"), 0)
                if e > 0:
                    ratio_s = f"{fifo_e/e:.0f}x"
            print(f"{m:<16} {label:<12} {t:>8.1f} {e:>10,} {ratio_s:>8}")


if __name__ == "__main__":
    main()

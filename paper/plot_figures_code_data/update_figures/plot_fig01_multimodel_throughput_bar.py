#!/usr/bin/env python3
"""
Fig 01: 4-model E2E throughput bar chart.

Data:   multimodel_e2e_4models.json  (same directory)
Output: fig01_multimodel_throughput.pdf/png (same directory)

Config matching paper: budget=50MB, seq=2048, nreq=4.
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
DATA_PATH = HERE / "multimodel_e2e_4models.json"

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

DISPLAY_NAMES = {
    "Qwen2.5-7B":  "Qwen2.5-7B",
    "Mistral-7B":   "Mistral-7B",
    "LLaMA-2-7B":  "LLaMA-2-7B",
    "LLaMA-2-13B": "LLaMA-2-13B",
}


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    sel = [r for r in data
           if r["seq_len"] == 2048
           and r.get("n_requests", r.get("num_requests")) == 4
           and r.get("gpu_budget_mb", r.get("budget_mb")) == 50]

    modes  = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    fills  = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    edges  = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(MODEL_ORDER))
    w = 0.25

    for i, mode in enumerate(modes):
        vals = []
        for m in MODEL_ORDER:
            v = next((r["avg_throughput"] for r in sel
                      if r["model"] == m and r["mode"] == mode), 0)
            vals.append(v)
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label=labels[mode],
                      color=fills[mode], edgecolor=edges[mode], **BAR_KW)
        for bar, v in zip(bars, vals):
            if v > 0 and mode != "baseline":
                ax.text(bar.get_x() + bar.get_width() / 2, v + 15,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.5)
    ax.grid(axis="y", alpha=0.3)

    HERE.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig01_multimodel_throughput.{ext}")
    plt.close(fig)
    print(f"Saved fig01_multimodel_throughput.pdf/png to {HERE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fig 7: ablation throughput + eviction from multimodel_ablation.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "multimodel_ablation.json"

C_GPU = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"

EC_GPU = "#6BA882"
EC_FIFO = "#C07868"
EC_ORKV = "#5A90B0"

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

CONFIG_ORDER = [
    ("gpu-only", "GPU-Only", C_GPU, EC_GPU),
    ("naive-fifo", "FIFO", C_FIFO, EC_FIFO),
    ("orchkv", "OrchKvCache", C_ORKV, EC_ORKV),
]


def _top_legend(ax, ncol=3):
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, frameon=False, fontsize=9, columnspacing=1.5,
    )


def main():
    with open(JSON_PATH) as f:
        rows = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    models = sorted({r.get("model", "?") for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    x = np.arange(len(models))
    w = 0.25
    for i, (key, label, color, ec) in enumerate(CONFIG_ORDER):
        tp = []
        ev = []
        for m in models:
            t = e = 0.0
            for r in rows:
                if r.get("model") == m and r.get("config", "").lower() == key:
                    t = float(r.get("throughput", r.get("avg_throughput", 0)) or 0)
                    e = float(r.get("evictions", r.get("total_evictions", 0)) or 0)
                    break
            tp.append(t)
            ev.append(e)
        ax1.bar(x + (i - 1) * w, tp, w, label=label, color=color, edgecolor=ec, **BAR_KW)
        ax2.bar(x + (i - 1) * w, ev, w, label=label, color=color, edgecolor=ec, **BAR_KW)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_ylabel("Throughput (tok/s)")
    _top_legend(ax1, ncol=3)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.set_ylabel("Evictions")
    _top_legend(ax2, ncol=3)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig07_multimodel_ablation.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig07_multimodel_ablation.pdf/png")


if __name__ == "__main__":
    main()

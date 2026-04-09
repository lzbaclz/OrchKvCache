#!/usr/bin/env python3
"""Table 6: 3-tier throughput bar from w4_tier_throughput.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "w4_tier_throughput.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"

EC_GPU  = "#6BA882"
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


def main():
    with open(JSON_PATH) as f:
        rows = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    models = sorted({r.get("model", "?") for r in rows})
    configs = ["GPU-Only", "GPU+DRAM", "GPU+DRAM+SSD"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(models))
    w = 0.25
    bar_colors = [C_GPU, C_FIFO, C_ORKV]
    bar_edges = [EC_GPU, EC_FIFO, EC_ORKV]
    for i, cfg in enumerate(configs):
        vals = []
        for m in models:
            v = 0.0
            for r in rows:
                if r.get("model") == m and str(r.get("config", "")) == cfg:
                    v = float(r.get("avg_throughput_tok_s", r.get("avg_throughput", 0)) or 0)
                    break
            vals.append(v)
        ax.bar(
            x + (i - 1) * w,
            vals,
            w,
            label=cfg,
            color=bar_colors[i],
            edgecolor=bar_edges[i],
            **BAR_KW,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table6_w4_tier_throughput.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table6_w4_tier_throughput.pdf/png")


if __name__ == "__main__":
    main()

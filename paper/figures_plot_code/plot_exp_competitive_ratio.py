#!/usr/bin/env python3
"""Table 5 / Fig: competitive ratio — eviction counts by strategy vs n_blocks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_competitive_ratio.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"
C_GRAY = "#C8C8C8"
C_ORANGE = "#E8C8A0"

EC_GPU  = "#6BA882"
EC_FIFO = "#C07868"
EC_ORKV = "#5A90B0"
EC_GRAY = "#909090"
EC_ORANGE = "#C09060"

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

STRATS = ["OPT", "EMA", "LFU", "LRU", "FIFO"]


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        rows = []

    blocks = []
    series = {s: [] for s in STRATS}
    for r in rows:
        nb = r.get("n_blocks")
        if nb is None:
            continue
        blocks.append(int(nb))
        for s in STRATS:
            key = f"{s}_evictions"
            series[s].append(float(r.get(key, 0) or 0))

    if not blocks:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"table5_competitive_ratio_evictions.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/table5_competitive_ratio_evictions.pdf/png (empty)")
        return

    x = np.arange(len(blocks))
    w = 0.15
    colors = [C_ORKV, C_GPU, C_ORANGE, C_FIFO, C_GRAY]
    edges = [EC_ORKV, EC_GPU, EC_ORANGE, EC_FIFO, EC_GRAY]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, s in enumerate(STRATS):
        ax.bar(
            x + (i - 2) * w,
            series[s],
            w,
            label=s,
            color=colors[i % len(colors)],
            edgecolor=edges[i % len(edges)],
            **BAR_KW,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("n_blocks")
    ax.set_ylabel("Eviction count")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table5_competitive_ratio_evictions.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table5_competitive_ratio_evictions.pdf/png")


if __name__ == "__main__":
    main()

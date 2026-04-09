#!/usr/bin/env python3
"""Table 12: PPL comparison (InfiniGen-style table) as bar chart."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_ppl_infinigen_comparison.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"
C_ORANGE = "#E8C8A0"

EC_GPU  = "#6BA882"
EC_FIFO = "#C07868"
EC_ORKV = "#5A90B0"
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


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    table = data.get("infinigen_table2", {})
    if not table:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No infinigen_table2", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"table12_ppl_infinigen_comparison.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/table12_ppl_infinigen_comparison.pdf/png (empty)")
        return

    modes = ["full_cache", "80pct_fifo", "80pct_lru", "80pct_counter"]
    labels = ["Full", "80% FIFO", "80% LRU", "80% counter"]
    fill_colors = [C_GPU, C_FIFO, C_ORANGE, C_ORKV]
    edge_colors = [EC_GPU, EC_FIFO, EC_ORANGE, EC_ORKV]

    models = list(table.keys())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(models))
    w = 0.2
    for i, (mode, lab, c, ec) in enumerate(zip(modes, labels, fill_colors, edge_colors)):
        vals = []
        for m in models:
            row = table.get(m, {})
            vals.append(float(row.get(mode, 0) or 0))
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c, edgecolor=ec, **BAR_KW)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Perplexity")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table12_ppl_infinigen_comparison.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table12_ppl_infinigen_comparison.pdf/png")


if __name__ == "__main__":
    main()

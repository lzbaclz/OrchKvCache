#!/usr/bin/env python3
"""Table 9: fair baseline throughput landscape."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_fair_baseline.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"
C_GRAY = "#C8C8C8"

EC_GPU  = "#6BA882"
EC_FIFO = "#C07868"
EC_ORKV = "#5A90B0"
EC_GRAY = "#909090"

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

    rows = data.get("results", [])
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"table9_fair_baseline_landscape.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/table9_fair_baseline_landscape.pdf/png (empty)")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    metrics = [
        ("gpu_only_eager", "GPU-only (eager)", C_GPU, EC_GPU),
        ("fast_fifo", "Fast FIFO", C_FIFO, EC_FIFO),
        ("fast_orchkv", "Fast OrchKv", C_ORKV, EC_ORKV),
        ("gpu_only_sdpa", "GPU-only (SDPA)", C_GRAY, EC_GRAY),
    ]

    n_models = len(rows)
    x = np.arange(n_models)
    w = 0.2
    for i, (key, label, color, ec) in enumerate(metrics):
        vals = [float(r.get(key, 0) or 0) for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color, edgecolor=ec, **BAR_KW)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r.get("model", "?")) for r in rows], rotation=15, ha="right")
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table9_fair_baseline_landscape.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table9_fair_baseline_landscape.pdf/png")


if __name__ == "__main__":
    main()

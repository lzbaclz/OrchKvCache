#!/usr/bin/env python3
"""Fig 10: scheduling latency vs blocks from benchmark_e9_scalability.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "benchmark_e9_scalability.json"

EC_GRAY = "#909090"
EC_ORANGE = "#C09060"
EC_ORKV = "#5A90B0"

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

    xs = []
    p50 = []
    p99 = []
    mean = []
    for r in rows:
        nb = r.get("n_blocks")
        if nb is None:
            continue
        xs.append(int(nb))
        p50.append(float(r.get("p50_us", r.get("p50_schedule_us", 0)) or 0))
        p99.append(float(r.get("p99_us", r.get("p99_schedule_us", 0)) or 0))
        mean.append(float(r.get("mean_us", r.get("avg_schedule_us", 0)) or 0))

    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    p50 = [p50[i] for i in order]
    p99 = [p99[i] for i in order]
    mean = [mean[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, p50, marker="s", label="p50", color=EC_ORKV)
    ax.plot(xs, p99, marker="^", label="p99", color=EC_GRAY)
    ax.plot(xs, mean, marker="o", label="mean", color=EC_ORANGE)
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("Schedule latency (μs)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig10_scheduling_latency_scalability.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig10_scheduling_latency_scalability.pdf/png")


if __name__ == "__main__":
    main()

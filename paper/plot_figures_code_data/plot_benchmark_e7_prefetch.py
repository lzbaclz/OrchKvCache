#!/usr/bin/env python3
"""Fig 9: prefetch dispatch + overhead from benchmark_e7_prefetch.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "benchmark_e7_prefetch.json"

C_ORKV = "#9CC0D8"

EC_GRAY = "#909090"
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

    budgets = []
    disp = []
    lat = []
    for r in rows:
        b = r.get("budget", r.get("prefetch_budget"))
        if b is None:
            continue
        budgets.append(float(b))
        disp.append(float(r.get("dispatches", r.get("avg_prefetches_dispatched", 0)) or 0))
        lat.append(float(r.get("avg_latency_us", r.get("avg_schedule_us", 0)) or 0))

    order = np.argsort(budgets)
    budgets = [budgets[i] for i in order]
    disp = [disp[i] for i in order]
    lat = [lat[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(budgets))
    ax1.bar(x - 0.2, disp, 0.4, label="Prefetches dispatched", color=C_ORKV,
            edgecolor=EC_ORKV, **BAR_KW)
    ax1.set_xlabel("Prefetch budget")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(b)) for b in budgets])
    ax1.set_ylabel("Dispatches (count)", color=EC_ORKV)
    ax1.tick_params(axis="y", labelcolor=EC_ORKV)

    ax2 = ax1.twinx()
    ax2.plot(x, lat, color=EC_GRAY, marker="o", linewidth=2, label="Schedule latency (μs)")
    ax2.set_ylabel("Avg schedule latency (μs)", color=EC_GRAY)
    ax2.tick_params(axis="y", labelcolor=EC_GRAY)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig09_prefetch_dispatch_overhead.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig09_prefetch_dispatch_overhead.pdf/png")


if __name__ == "__main__":
    main()

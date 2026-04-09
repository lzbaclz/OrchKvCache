#!/usr/bin/env python3
"""Fig 13–14: throughput + eviction from realistic_workload.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "realistic_workload.json"

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

EDGE_BY_MODE = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

MODES = [
    ("baseline", "GPU-Only", C_GPU),
    ("naive", "FIFO", C_FIFO),
    ("orchkv", "OrchKvCache", C_ORKV),
]


def main():
    with open(JSON_PATH) as f:
        rows = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    workloads = sorted({r.get("workload", "?") for r in rows})
    models = sorted({r.get("model", "?") for r in rows})

    groups = [(wl, m) for wl in workloads for m in models]
    n = len(groups)
    w = 0.22
    fig1, ax1 = plt.subplots(figsize=(max(7, n * 1.0), 3.8))
    for gi, (wl, m) in enumerate(groups):
        center = gi * 1.35
        for mi, (mode_key, _lbl, color) in enumerate(MODES):
            v = 0.0
            for r in rows:
                if r.get("workload") == wl and r.get("model") == m and r.get("mode") == mode_key:
                    v = float(r.get("avg_throughput", 0) or 0)
                    break
            ax1.bar(center + (mi - 1) * w, v, w * 0.95, color=color,
                    edgecolor=EDGE_BY_MODE[mode_key], **BAR_KW,
                    label=_lbl if gi == 0 else None)
    ax1.set_xticks([i * 1.35 for i in range(n)])
    ax1.set_xticklabels([f"{m[:14]}\n{wl[:16]}" for wl, m in groups], fontsize=7, rotation=20, ha="right")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(OUT / f"fig13_realistic_workload_throughput.{ext}")
    plt.close(fig1)
    print(f"Wrote {OUT}/fig13_realistic_workload_throughput.pdf/png")

    fig2, ax2 = plt.subplots(figsize=(max(7, n * 1.0), 3.8))
    for gi, (wl, m) in enumerate(groups):
        center = gi * 1.35
        for mi, (mode_key, _lbl, color) in enumerate(MODES):
            v = 0.0
            for r in rows:
                if r.get("workload") == wl and r.get("model") == m and r.get("mode") == mode_key:
                    v = float(r.get("total_evictions", 0) or 0)
                    break
            ax2.bar(center + (mi - 1) * w, v, w * 0.95, color=color,
                    edgecolor=EDGE_BY_MODE[mode_key], **BAR_KW,
                    label=_lbl if gi == 0 else None)
    ax2.set_xticks([i * 1.35 for i in range(n)])
    ax2.set_xticklabels([f"{m[:14]}\n{wl[:16]}" for wl, m in groups], fontsize=7, rotation=20, ha="right")
    ax2.set_ylabel("Total evictions")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(OUT / f"fig14_realistic_workload_eviction.{ext}")
    plt.close(fig2)
    print(f"Wrote {OUT}/fig14_realistic_workload_eviction.pdf/png")


if __name__ == "__main__":
    main()

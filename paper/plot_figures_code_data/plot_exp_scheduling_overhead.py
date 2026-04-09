#!/usr/bin/env python3
"""Table 10: C vs Python scheduling overhead."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_scheduling_overhead.json"

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

    c_blocks = []
    c_us = []
    for k, v in data.items():
        if not k.startswith("c_overhead_") or not isinstance(v, dict):
            continue
        nb = v.get("n_blocks")
        tot = v.get("total_c_per_step_us", {})
        mu = tot.get("mean") if isinstance(tot, dict) else None
        if mu is None:
            mu = v.get("total_c_per_step_us")
        if nb is not None and mu is not None:
            c_blocks.append(int(nb))
            c_us.append(float(mu))

    order = np.argsort(c_blocks)
    c_blocks = [c_blocks[i] for i in order]
    c_us = [c_us[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    if c_blocks:
        ax1.plot(c_blocks, c_us, marker="o", color=EC_ORKV, label="C path (μs/step)")
        ax1.set_xlabel("n_blocks")
        ax1.set_ylabel("μs per step (C)")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(c_blocks)
        ax1.set_xticklabels([str(b) for b in c_blocks])
        ax1.grid(alpha=0.3)

    # E2E Python-side breakdown for first model key
    e2e_keys = [k for k in data if k.startswith("e2e_")]
    if e2e_keys:
        k0 = e2e_keys[0]
        e2e = data[k0]
        sch = float(e2e.get("python_sched_overhead_ms", {}).get("mean", 0) or 0)
        fwd = float(e2e.get("forward_ms", {}).get("mean", 0) or 0)
        rep = float(e2e.get("report_ms", {}).get("mean", 0) or 0)
        bld = float(e2e.get("build_kv_ms", {}).get("mean", 0) or 0)
        parts = [("Forward", fwd), ("Report", rep), ("Build KV", bld), ("Python sched ovhd", sch)]
        parts = [(a, b) for a, b in parts if b > 0]
        if parts:
            labels, vals = zip(*parts)
            cmap = [C_GPU, C_ORANGE, C_FIFO, C_ORKV]
            ecmap = [EC_GPU, EC_ORANGE, EC_FIFO, EC_ORKV]
            cols = [cmap[i % len(cmap)] for i in range(len(vals))]
            ecs = [ecmap[i % len(ecmap)] for i in range(len(vals))]
            ax2.barh(range(len(vals)), vals, color=cols, edgecolor=ecs, **BAR_KW)
            ax2.set_yticks(range(len(vals)))
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel("Time (ms)")
        ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table10_scheduling_c_vs_python.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table10_scheduling_c_vs_python.pdf/png")


if __name__ == "__main__":
    main()

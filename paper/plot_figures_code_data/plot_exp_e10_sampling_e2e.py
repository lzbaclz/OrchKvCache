#!/usr/bin/env python3
"""Sampling interval E2E throughput from exp_e10_sampling_e2e.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_e10_sampling_e2e.json"

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
        data = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = data.get("results", [])
    ivs = []
    tps = []
    for r in rows:
        iv = r.get("sample_interval")
        if iv is None:
            continue
        ivs.append(int(iv))
        tps.append(float(r.get("avg_throughput_tok_s", 0) or 0))

    order = np.argsort(ivs)
    ivs = [ivs[i] for i in order]
    tps = [tps[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(ivs, tps, marker="o", color=EC_ORKV, linewidth=2)
    for a, b in zip(ivs, tps):
        ax.annotate(f"{b:.0f}", (a, b), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("Sampling interval")
    ax.set_ylabel("Throughput (tok/s)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"e10_sampling_e2e_throughput.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/e10_sampling_e2e_throughput.pdf/png")


if __name__ == "__main__":
    main()

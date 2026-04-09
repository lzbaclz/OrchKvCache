#!/usr/bin/env python3
"""Fig 11: inter-tier bandwidth from benchmark_e8_storage_bw.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "benchmark_e8_storage_bw.json"

C_GPU = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"
C_ORANGE = "#E8C8A0"

EC_GPU = "#6BA882"
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    gd = data.get("gpu_dram", [])
    if gd:
        sizes = [float(r.get("size_mb", r.get("size_kb", 0)) or 0) for r in gd]
        d2h = [float(r.get("d2h_gbps", 0) or 0) for r in gd]
        h2d = [float(r.get("h2d_gbps", 0) or 0) for r in gd]
        x = np.arange(len(sizes))
        w = 0.35
        ax1.bar(x - w / 2, d2h, w, label="Device→host (D2H)", color=C_ORKV,
                edgecolor=EC_ORKV, **BAR_KW)
        ax1.bar(x + w / 2, h2d, w, label="Host→device (H2D)", color=C_GPU,
                edgecolor=EC_GPU, **BAR_KW)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{s:g}" for s in sizes])
        ax1.set_xlabel("Transfer size (MB)")
        ax1.set_ylabel("Bandwidth (GB/s)")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

    ds = data.get("dram_storage", [])
    if ds:
        sizes = [float(r.get("size_mb", 0) or 0) for r in ds]
        wr = [float(r.get("write_gbps", 0) or 0) for r in ds]
        rd = [float(r.get("read_gbps", 0) or 0) for r in ds]
        x = np.arange(len(sizes))
        w = 0.35
        ax2.bar(x - w / 2, wr, w, label="Write (DRAM→storage)", color=C_FIFO,
                edgecolor=EC_FIFO, **BAR_KW)
        ax2.bar(x + w / 2, rd, w, label="Read (storage→DRAM)", color=C_ORANGE,
                edgecolor=EC_ORANGE, **BAR_KW)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{s:g}" for s in sizes])
        ax2.set_xlabel("Transfer size (MB)")
        ax2.set_ylabel("Bandwidth (GB/s)")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig11_inter_tier_bandwidth.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig11_inter_tier_bandwidth.pdf/png")


if __name__ == "__main__":
    main()

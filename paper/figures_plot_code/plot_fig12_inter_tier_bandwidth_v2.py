#!/usr/bin/env python3
"""Redraw fig12: line chart of inter-tier bandwidth vs transfer size."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open(BASE / "benchmark_e8_storage_bw.json") as f:
    data = json.load(f)

gd = data["gpu_dram"]
ds = data["dram_storage"]

sizes = [r["size_mb"] for r in gd]
d2h = [r["d2h_gbps"] for r in gd]
h2d = [r["h2d_gbps"] for r in gd]
ssd_w = [r["write_gbps"] for r in ds]
ssd_r = [r["read_gbps"] for r in ds]

fig, ax = plt.subplots(figsize=(4.5, 3.0))

ax.plot(sizes, h2d, "o-", color="#6BA882", linewidth=1.5, markersize=5, label="DRAM→GPU")
ax.plot(sizes, d2h, "s--", color="#5A90B0", linewidth=1.5, markersize=5, label="GPU→DRAM")
ax.plot(sizes, ssd_r, "^-", color="#C09060", linewidth=1.5, markersize=5, label="SSD Read")
ax.plot(sizes, ssd_w, "v--", color="#C07868", linewidth=1.5, markersize=5, label="SSD Write")

ax.set_xscale("log", base=2)
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s:g}" for s in sizes], fontsize=8)
ax.set_xlabel("Transfer size (MB)")
ax.set_ylabel("Bandwidth (GB/s)")
ax.set_ylim(0, 28)
ax.legend(loc="center right", frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.3)

ax.axhline(y=23, color="#5A90B0", alpha=0.3, linestyle=":", linewidth=0.8)
ax.text(0.6, 23.8, "PCIe Gen4 ≈ 23 GB/s", fontsize=7, color="#5A90B0", alpha=0.6)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig12_inter_tier_bandwidth.{ext}")
plt.close(fig)
print(f"Saved fig12_inter_tier_bandwidth.pdf/png to {OUT}")

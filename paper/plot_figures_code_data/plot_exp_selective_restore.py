#!/usr/bin/env python3
"""Fig 17: EMA vs random coverage from exp_selective_restore.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_selective_restore.json"

C_ORKV = "#9CC0D8"
C_ORANGE = "#E8C8A0"

EC_ORKV = "#5A90B0"
EC_ORANGE = "#C09060"
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

PCTS = [1, 3, 5, 10, 20, 30, 50, 70, 90]


def _top_legend(ax, ncol=3):
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, frameon=False, fontsize=9, columnspacing=1.5,
    )


def main():
    with open(JSON_PATH) as f:
        rows = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = rows[:2] if rows else []
    if not rows:
        OUT.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"fig17_selective_restore_coverage.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/fig17_selective_restore_coverage.pdf/png (empty)")
        return

    n = len(rows)
    fig, axes = plt.subplots(1, min(n, 2), figsize=(6.5, 3.2))
    if n == 1:
        axes = np.array([axes])
    for idx, r in enumerate(rows):
        ax = axes[idx]
        ema = r.get("avg_ema_coverage") or {}
        rnd = r.get("avg_random_coverage") or {}
        ema_vals = [float(ema.get(f"top{p}pct", 0) or 0) for p in PCTS]
        rand_vals = [float(rnd.get(f"top{p}pct", 0) or 0) for p in PCTS]
        x = np.arange(len(PCTS))
        w = 0.35
        ax.bar(x - w / 2, ema_vals, w, label="EMA (ours)", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
        ax.bar(x + w / 2, rand_vals, w, label="Random", color=C_ORANGE, edgecolor=EC_ORANGE, **BAR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}%" for p in PCTS], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Top-K% blocks restored")
        ax.set_ylabel("Attention weight captured (%)")
        ax.set_ylim(0, 115)
        ax.axhline(100, color=EC_GRAY, linestyle=":", linewidth=0.8)
        _top_legend(ax, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig17_selective_restore_coverage.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig17_selective_restore_coverage.pdf/png")


if __name__ == "__main__":
    main()

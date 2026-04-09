#!/usr/bin/env python3
"""Sampling interval — classification accuracy (simulation) from exp_e10_sampling_sim.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_e10_sampling_sim.json"

C_ORKV = "#9CC0D8"
C_GRAY = "#C8C8C8"

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

    summ = data.get("summary", [])
    if not summ:
        raw = data.get("raw", [])
        by_iv = {}
        for r in raw:
            iv = r.get("sample_interval")
            if iv is None:
                continue
            by_iv.setdefault(iv, []).append(float(r.get("gt_accuracy", 0) or 0))
        summ = [{"sample_interval": k, "gt_accuracy": np.mean(v)} for k, v in sorted(by_iv.items())]

    ivs = [int(s.get("sample_interval", 0) or 0) for s in summ]
    acc = [float(s.get("gt_accuracy", 0) or 0) for s in summ]
    agree = [float(s.get("baseline_agreement", 0) or 0) for s in summ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(ivs))
    w = 0.35
    ax.bar(x - w / 2, acc, w, label="GT accuracy", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax.bar(x + w / 2, agree, w, label="Baseline agreement", color=C_GRAY, edgecolor=EC_GRAY, **BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in ivs])
    ax.set_xlabel("Sampling interval (tokens)")
    ax.set_ylabel("Accuracy / agreement")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"e10_sampling_sim_accuracy.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/e10_sampling_sim_accuracy.pdf/png")


if __name__ == "__main__":
    main()

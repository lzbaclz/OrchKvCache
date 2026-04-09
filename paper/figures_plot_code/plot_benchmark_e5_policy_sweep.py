#!/usr/bin/env python3
"""Fig 8: policy (α,β,γ) heatmap from benchmark_e5_policy_sweep.json."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "benchmark_e5_policy_sweep.json"

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

    by_ab = defaultdict(list)
    for r in rows:
        a, b = r.get("alpha"), r.get("beta")
        if a is None or b is None:
            continue
        val = r.get("accuracy")
        if val is None:
            val = r.get("hot_ratio", 0)
        by_ab[(float(a), float(b))].append(float(val or 0))

    alphas = sorted({k[0] for k in by_ab.keys()})
    betas = sorted({k[1] for k in by_ab.keys()})
    if not alphas or not betas:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"fig08_policy_sweep_heatmap.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/fig08_policy_sweep_heatmap.pdf/png (empty)")
        return

    mat = np.full((len(alphas), len(betas)), np.nan)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            key = (a, b)
            if key in by_ab:
                mat[i, j] = np.mean(by_ab[key])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    plt.colorbar(im, ax=ax, label="Value")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig08_policy_sweep_heatmap.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig08_policy_sweep_heatmap.pdf/png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fig 12: hyperparameter λ sweep from exp_p2p3_extended.json (p3_hyperparam_e2e)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_p2p3_extended.json"

C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"

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


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    hp = data.get("p3_hyperparam_e2e", [])
    if not hp:
        hp = data if isinstance(data, list) else []

    lambdas = []
    tok_s = []
    evictions = []
    for r in hp:
        lam = r.get("ema_lambda")
        if lam is None:
            continue
        if r.get("tok_s") is None and r.get("throughput") is None:
            continue
        lambdas.append(float(lam))
        tok_s.append(float(r.get("tok_s", r.get("throughput", 0)) or 0))
        evictions.append(float(r.get("evictions", r.get("total_evictions", 0)) or 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    x = np.arange(len(lambdas))
    ax1.bar(x, tok_s, color=C_ORKV, edgecolor=EC_ORKV, width=0.6, **BAR_KW)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l:.2f}" for l in lambdas])
    ax1.set_xlabel(r"EMA decay $\lambda$")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, evictions, color=C_FIFO, edgecolor=EC_FIFO, width=0.6, **BAR_KW)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l:.2f}" for l in lambdas])
    ax2.set_xlabel(r"EMA decay $\lambda$")
    ax2.set_ylabel("Evictions")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig12_hyperparam_lambda_sweep.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig12_hyperparam_lambda_sweep.pdf/png")


if __name__ == "__main__":
    main()

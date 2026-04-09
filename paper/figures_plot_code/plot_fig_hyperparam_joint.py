#!/usr/bin/env python3
"""Single-panel joint λ×τ heatmap for hyperparameter sensitivity."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE / "exp_e11_hyperparam.json"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open(DATA) as f:
    data = json.load(f)

combined = data["combined"]
lambdas = sorted(set(r["ema_lambda"] for r in combined))
taus = sorted(set(r["recency_tau"] for r in combined))
mat = np.zeros((len(lambdas), len(taus)))
for r in combined:
    i = lambdas.index(r["ema_lambda"])
    j = taus.index(r["recency_tau"])
    mat[i, j] = r["accuracy"]

cmap = LinearSegmentedColormap.from_list(
    "soft", ["#F5E6D8", "#E8C8A0", "#D4A878", "#B88858", "#966840"])

fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(mat, cmap=cmap, aspect="auto",
               vmin=mat.min() * 0.95, vmax=min(mat.max() * 1.02, 1.0))
ax.set_xticks(range(len(taus)))
ax.set_xticklabels([str(t) for t in taus])
ax.set_yticks(range(len(lambdas)))
ax.set_yticklabels([f"{l:.1f}" for l in lambdas])
ax.set_xlabel(r"Recency $\tau$ (steps)")
ax.set_ylabel(r"EMA decay $\lambda$")

for i in range(len(lambdas)):
    for j in range(len(taus)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="#333333")

plt.colorbar(im, ax=ax, shrink=0.85, label="Classification accuracy")

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_hyperparam_joint.{ext}")
plt.close(fig)
print(f"Saved fig_hyperparam_joint.pdf/png to {HERE}")

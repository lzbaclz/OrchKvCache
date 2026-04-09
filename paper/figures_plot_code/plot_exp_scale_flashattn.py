#!/usr/bin/env python3
"""Fig 16: 8K context scaling from exp_scale_flashattn.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_scale_flashattn.json"

EC_GPU  = "#6BA882"
EC_FIFO = "#C07868"
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
        rows = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    by_model = {}
    for r in rows:
        m = r.get("model", "?")
        by_model.setdefault(m, []).append(r)

    fig, axes = plt.subplots(1, len(by_model), figsize=(4.2 * len(by_model), 3.2), squeeze=False)
    axes = axes[0]
    for ax, (_, mrows) in zip(axes, sorted(by_model.items(), key=lambda x: x[0])):
        mrows = sorted(mrows, key=lambda x: int(x.get("seq_len", 0) or 0))
        ctx = [int(x.get("seq_len", 0) or 0) for x in mrows]
        labels = [f"{c // 1024}K" if c >= 1024 else str(c) for c in ctx]
        gpu = [float(x.get("gpu_only_tok_s", 0) or 0) for x in mrows]
        fifo = [float(x.get("fifo_tok_s", 0) or 0) for x in mrows]
        orch = []
        for x in mrows:
            v = x.get("orchkv_tok_s", 0)
            orch.append(float(v) if isinstance(v, (int, float)) else 0.0)
        xi = np.arange(len(ctx))
        ax.plot(xi, gpu, marker="s", label="GPU-Only", color=EC_GPU, linewidth=1.5)
        ax.plot(xi, fifo, marker="^", label="FIFO", color=EC_FIFO, linewidth=1.5)
        ax.plot(xi, orch, marker="o", label="OrchKvCache", color=EC_ORKV, linewidth=1.5)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Throughput (tok/s)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig16_scale_context_8k.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig16_scale_context_8k.pdf/png")


if __name__ == "__main__":
    main()

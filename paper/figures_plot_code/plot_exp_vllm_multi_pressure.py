#!/usr/bin/env python3
"""Table 7: vLLM strategies under pressure — bar chart."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_vllm_multi_pressure.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"

EC_GPU  = "#6BA882"
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
        rows = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    # Pick a representative high-pressure point: lowest gpu_util, max prompts
    candidates = []
    for r in rows:
        gu = float(r.get("gpu_util", 0) or 0)
        np_ = int(r.get("num_prompts", 0) or 0)
        candidates.append((gu, np_, r))
    if not candidates:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"table7_vllm_strategies.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/table7_vllm_strategies.pdf/png (empty)")
        return

    min_gu = min(c[0] for c in candidates)
    max_np = max(c[1] for c in candidates)
    slice_rows = [c[2] for c in candidates if abs(c[0] - min_gu) < 1e-6 and c[1] == max_np]
    if not slice_rows:
        slice_rows = [c[2] for c in candidates if c[0] == min_gu]

    strat_order = ["fifo", "progress", "block_score"]
    labels_map = {"fifo": "FIFO", "progress": "Progress", "block_score": "Block-score"}
    colors = {"fifo": C_FIFO, "progress": C_GPU, "block_score": C_ORKV}
    edge_colors = {"fifo": EC_FIFO, "progress": EC_GPU, "block_score": EC_ORKV}

    by_s = {s: 0.0 for s in strat_order}
    for r in slice_rows:
        s = str(r.get("strategy", "")).lower()
        if s in by_s:
            by_s[s] = float(r.get("avg_throughput", 0) or 0)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    xs = np.arange(len(strat_order))
    ax.bar(
        xs,
        [by_s[s] for s in strat_order],
        color=[colors[s] for s in strat_order],
        edgecolor=[edge_colors[s] for s in strat_order],
        **BAR_KW,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([labels_map[s] for s in strat_order])
    ax.set_ylabel("Avg throughput (tok/s)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table7_vllm_strategies.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table7_vllm_strategies.pdf/png")


if __name__ == "__main__":
    main()

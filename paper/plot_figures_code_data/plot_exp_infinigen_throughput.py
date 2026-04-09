#!/usr/bin/env python3
"""Fig 18: InfiniGen comparison from exp_infinigen_throughput.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_infinigen_throughput.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_GRAY = "#C8C8C8"

EC_GPU  = "#6BA882"
EC_FIFO = "#C07868"
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

    labels = []
    orig = []
    infin = []
    h2o = []

    for cfg in data:
        cfg_s = str(cfg.get("config", ""))
        short = cfg_s
        if "prompt=" in cfg_s:
            try:
                short = "p=" + cfg_s.split("prompt=")[1].split(",")[0]
            except Exception:
                pass
        o = ig = h = None
        for res in cfg.get("results", []):
            scheme = str(res.get("scheme", ""))
            tp = float(res.get("gen_throughput_tok_s", 0) or 0)
            if "FlexGen Original" in scheme:
                o = tp
            elif scheme == "InfiniGen":
                ig = tp
            elif "H2O" in scheme:
                h = tp
        if o is not None and ig is not None and h is not None:
            labels.append(short)
            orig.append(o)
            infin.append(ig)
            h2o.append(h)

    n = len(labels)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No matching results", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"fig18_infinigen_throughput.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/fig18_infinigen_throughput.pdf/png (empty)")
        return

    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.bar(x - w, orig, w, label="FlexGen Original", color=C_GRAY, edgecolor=EC_GRAY, **BAR_KW)
    ax.bar(x, infin, w, label="InfiniGen", color=C_GPU, edgecolor=EC_GPU, **BAR_KW)
    ax.bar(x + w, h2o, w, label="H2O (lossy)", color=C_FIFO, edgecolor=EC_FIFO, **BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Gen throughput (tok/s)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig18_infinigen_throughput.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig18_infinigen_throughput.pdf/png")


if __name__ == "__main__":
    main()

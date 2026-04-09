#!/usr/bin/env python3
"""Table 8: per-step overhead — stacked bar / pie from exp_overhead_breakdown.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "exp_overhead_breakdown.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"
C_ORANGE = "#E8C8A0"
C_PURPLE = "#C8B8D8"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PIE_COLORS = [C_GPU, C_FIFO, C_ORKV, C_ORANGE, C_PURPLE]


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    results = data.get("results", [])
    target = None
    for r in results:
        lbl = str(r.get("label", ""))
        if "N=10" in lbl or "orchkv" in lbl.lower():
            target = r
            if "N=10" in lbl:
                break
    if target is None and results:
        target = results[-1]

    if not target:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"table8_overhead_breakdown.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/table8_overhead_breakdown.pdf/png (empty)")
        return

    parts = []
    for key, lab in [
        ("forward_avg_ms", "Forward"),
        ("build_past_kv_avg_ms", "Build past KV"),
        ("append_token_avg_ms", "Append token"),
        ("report_attn_avg_ms", "Report attn"),
        ("step_schedule_avg_ms", "Schedule"),
    ]:
        v = target.get(key)
        if v is not None and float(v) > 0:
            parts.append((lab, float(v)))

    if not parts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No timing fields", ha="center")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"table8_overhead_breakdown.{ext}")
        plt.close(fig)
        print(f"Wrote {OUT}/table8_overhead_breakdown.pdf/png (empty)")
        return

    labels, sizes = zip(*parts)
    fig, ax = plt.subplots(figsize=(5, 5))
    n = len(sizes)
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=PIE_COLORS[:n],
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"table8_overhead_breakdown.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/table8_overhead_breakdown.pdf/png")


if __name__ == "__main__":
    main()

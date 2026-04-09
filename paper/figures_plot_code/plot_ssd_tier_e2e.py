#!/usr/bin/env python3
"""Fig 15: SSD validation table from ssd_tier_e2e.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "ssd_tier_e2e.json"

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

    headers = ["Model", "Prompt", "Match %", "GPU→DRAM", "DRAM→SSD", "SSD→DRAM", "SSD files"]
    cells = []
    for r in rows:
        cells.append([
            str(r.get("model", "")),
            str(r.get("prompt", "")),
            str(r.get("match_rate", "")),
            str(r.get("gpu_to_dram", "")),
            str(r.get("dram_to_ssd", "")),
            str(r.get("ssd_to_dram", "")),
            str(r.get("ssd_files_created", r.get("ssd_files", ""))),
        ])

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.35 * max(len(cells), 1)))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.35)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig15_ssd_tier_validation_table.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig15_ssd_tier_validation_table.pdf/png")


if __name__ == "__main__":
    main()

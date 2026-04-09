#!/usr/bin/env python3
"""Fig 6: quality table from multimodel_quality.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "multimodel_quality.json"

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

    headers = ["Model", "Prompt", "Prompt len", "Match rate (%)", "Gen tokens"]
    cells = []
    for r in rows:
        cells.append([
            str(r.get("model", "")),
            str(r.get("prompt", r.get("prompt_type", ""))),
            str(r.get("prompt_len", "")),
            str(r.get("match_rate", r.get("match_rate_pct", ""))),
            str(r.get("generated_tokens", r.get("generated", ""))),
        ])

    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.35 * max(len(cells), 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig06_multimodel_quality_table.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/fig06_multimodel_quality_table.pdf/png")


if __name__ == "__main__":
    main()

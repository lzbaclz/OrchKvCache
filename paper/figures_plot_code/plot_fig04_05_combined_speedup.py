#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"
C_GRAY = "#C8C8C8"; EC_GRAY = "#909090"

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

def _load(name: str):
    with open(BASE / name) as f:
        return json.load(f)

def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/{stem}.pdf/png")

MODEL_ORDER = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]

DISPLAY = {
    "Qwen2.5-7B": "Qwen 7B",
    "Mistral-7B": "Mistral 7B",
    "LLaMA-2-7B": "LLaMA-2 7B",
    "LLaMA-2-13B": "LLaMA-2 13B",
}

def _short(m: str) -> str:
    return DISPLAY.get(m, m)

def fig04_05_combined():
    data = _load("update_figures/multimodel_e2e_4models.json")

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(10, 3.8),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35},
    )

    # --- (a) Heatmap: models × context lengths ---
    seq_lens = sorted({r["seq_len"] for r in data})[:4]
    models = [m for m in MODEL_ORDER if m in {r["model"] for r in data}]
    mat = np.full((len(models), len(seq_lens)), np.nan)
    for i, m in enumerate(models):
        for j, sl in enumerate(seq_lens):
            orch = next((r["avg_throughput"] for r in data
                         if r["model"] == m and r["seq_len"] == sl and r["n_requests"] == 8
                         and r["gpu_budget_mb"] == 50 and r["mode"] == "orchkv"), None)
            naive = next((r["avg_throughput"] for r in data
                          if r["model"] == m and r["seq_len"] == sl and r["n_requests"] == 8
                          and r["gpu_budget_mb"] == 50 and r["mode"] == "naive"), None)
            if orch and naive and naive > 0:
                mat[i, j] = orch / naive

    im = ax_heat.imshow(mat, aspect="auto", cmap="YlGnBu")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax_heat.text(j, i, f"{mat[i, j]:.2f}",
                             ha="center", va="center", fontsize=10)
    ax_heat.set_xticks(range(len(seq_lens)))
    ax_heat.set_xticklabels([f"{s // 1024}K" for s in seq_lens])
    ax_heat.set_yticks(range(len(models)))
    ax_heat.set_yticklabels([_short(m) for m in models])
    ax_heat.set_xlabel("Context length")
    ax_heat.set_ylabel("Model")
    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label("Speedup", fontsize=9)
    ax_heat.set_title("(a) Speedup heatmap", fontsize=10, pad=8)

    # --- (b) Per-model speedup bar (same model order) ---
    speedups = []
    for m in models:
        orch = next((r["avg_throughput"] for r in data
                     if r["model"] == m and r["seq_len"] == 2048 and r["n_requests"] == 8
                     and r["gpu_budget_mb"] == 50 and r["mode"] == "orchkv"), 0)
        naive = next((r["avg_throughput"] for r in data
                      if r["model"] == m and r["seq_len"] == 2048 and r["n_requests"] == 8
                      and r["gpu_budget_mb"] == 50 and r["mode"] == "naive"), 1)
        speedups.append(orch / naive if naive > 0 else 0)

    bars = ax_bar.bar(range(len(models)), speedups,
                      color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    for b, s in zip(bars, speedups):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{s:.2f}x", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks(range(len(models)))
    ax_bar.set_xticklabels([_short(m) for m in models], rotation=15, ha="right")
    ax_bar.set_ylabel("OrchKV / FIFO throughput")
    ax_bar.axhline(1.0, color=EC_GRAY, linestyle="--", linewidth=1)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_title("(b) Per-model speedup", fontsize=10, pad=8)

    _save(fig, "fig04_05_combined_speedup")


if __name__ == "__main__":
    fig04_05_combined()

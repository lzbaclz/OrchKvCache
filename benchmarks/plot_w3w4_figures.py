#!/usr/bin/env python3
"""Generate figures for W3 (realistic workload) and W4 (SSD tier) experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGDIR = Path(__file__).parent / "paper_figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})

MODE_COLORS = {"baseline": "#4e79a7", "naive": "#e15759", "orchkv": "#59a14f"}
MODE_LABELS = {"baseline": "GPU-Only", "naive": "FIFO Offload", "orchkv": "OrchKvCache"}

def save(fig, name):
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGDIR / f"{name}.{fmt}", format=fmt)
    print(f"  {name}")
    plt.close(fig)


def fig_w3_throughput():
    data = json.loads((RESULTS / "realistic_workload.json").read_text())
    models = ["Qwen2.5-7B", "LLaMA-2-7B"]
    workloads = ["sharegpt-like", "longcontext-mix"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for wi, wl in enumerate(workloads):
        ax = axes[wi]
        x = np.arange(len(models))
        w = 0.25
        for mi, mode in enumerate(["baseline", "naive", "orchkv"]):
            vals = []
            for m in models:
                r = next((r for r in data if r["model"] == m and r["workload"] == wl
                          and r["mode"] == mode), None)
                vals.append(r["avg_throughput"] if r else 0)
            ax.bar(x + mi * w, vals, w, label=MODE_LABELS[mode],
                   color=MODE_COLORS[mode], edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(x + w)
        ax.set_xticklabels(models, fontsize=7)
        title = "(a) ShareGPT-like" if wi == 0 else "(b) LongContext-mix"
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0)
        if wi == 0:
            ax.legend(frameon=False, ncol=3, loc="upper center",
                      bbox_to_anchor=(1.1, 1.22), fontsize=7)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_w3_throughput")


def fig_w3_eviction():
    data = json.loads((RESULTS / "realistic_workload.json").read_text())
    models = ["Qwen2.5-7B", "LLaMA-2-7B"]
    workloads = ["sharegpt-like", "longcontext-mix"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for wi, wl in enumerate(workloads):
        ax = axes[wi]
        x = np.arange(len(models))
        w = 0.35
        for mi, (mode, label) in enumerate([("naive", "FIFO"), ("orchkv", "OrchKvCache")]):
            vals = []
            for m in models:
                r = next((r for r in data if r["model"] == m and r["workload"] == wl
                          and r["mode"] == mode), None)
                vals.append(max(r["total_evictions"], 1) if r else 1)
            bars = ax.bar(x + mi * w, vals, w, label=label,
                          color=MODE_COLORS[mode], edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_ylabel("Total Evictions (log)")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(models, fontsize=7)
        title = "(a) ShareGPT-like" if wi == 0 else "(b) LongContext-mix"
        ax.set_title(title, fontsize=8)
        ax.legend(frameon=False, fontsize=7)

        for mi_m, m in enumerate(models):
            naive_r = next((r for r in data if r["model"] == m and r["workload"] == wl
                            and r["mode"] == "naive"), None)
            orch_r = next((r for r in data if r["model"] == m and r["workload"] == wl
                           and r["mode"] == "orchkv"), None)
            if naive_r and orch_r and orch_r["total_evictions"] > 0:
                ratio = naive_r["total_evictions"] / orch_r["total_evictions"]
                ax.annotate(f"{ratio:.0f}x", xy=(x[mi_m] + w / 2, orch_r["total_evictions"]),
                            fontsize=6.5, ha="center", va="bottom", color="#59a14f", fontweight="bold")

    fig.tight_layout(w_pad=2)
    save(fig, "fig_w3_eviction")


def fig_w4_ssd():
    data = json.loads((RESULTS / "ssd_tier_e2e.json").read_text())

    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    ax.axis("off")

    headers = ["Model", "Prompt", "Match", "GPU→DRAM", "DRAM→SSD", "SSD→DRAM", "SSD Files"]
    rows = []
    for r in data:
        rows.append([
            r["model"], r["prompt"], f'{r["match_rate"]:.2f}%',
            str(r["gpu_to_dram"]), str(r["dram_to_ssd"]),
            str(r["ssd_to_dram"]), str(r["ssd_files_created"]),
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#d4e6f1")
        table[0, j].set_text_props(fontweight="bold", fontsize=7)
    for i in range(1, len(rows) + 1):
        table[i, 2].set_text_props(fontweight="bold", color="#1a9641")
    ax.set_title("SSD Tier End-to-End Validation (GPU→DRAM→SSD→DRAM→GPU, Lossless)", fontsize=9, pad=15)
    save(fig, "fig_w4_ssd_tier")


if __name__ == "__main__":
    print("Generating W3+W4 figures...")
    fig_w3_throughput()
    fig_w3_eviction()
    fig_w4_ssd()
    print("Done.")

#!/usr/bin/env python3
"""
Generate all paper-ready figures for OrchKvCache (4-model evaluation).
Style: OSDI/SC/FAST — clean serif, two-column IEEE friendly, grayscale-safe.
"""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGDIR = Path(__file__).parent / "paper_figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

MODEL_COLORS = {
    "Qwen2.5-7B":  "#4e79a7",
    "Mistral-7B":  "#f28e2b",
    "LLaMA-2-7B":  "#e15759",
    "LLaMA-2-13B": "#76b7b2",
}
MODE_COLORS = {"baseline": "#4e79a7", "naive": "#e15759", "orchkv": "#59a14f"}
MODE_LABELS = {"baseline": "GPU-Only", "naive": "FIFO Offload", "orchkv": "OrchKvCache"}
MODE_HATCHES = {"baseline": "", "naive": "//", "orchkv": ""}


def save(fig, name):
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGDIR / f"{name}.{fmt}", format=fmt)
    print(f"  {name}.pdf + .png")
    plt.close(fig)


def load_all_e2e():
    all_data = []
    for f in ["multimodel_e2e.json", "llama7b_e2e.json", "llama13b_e2e.json"]:
        path = RESULTS / f
        if path.exists():
            all_data.extend(json.loads(path.read_text()))
    return all_data


def load_all_quality():
    all_data = []
    for f in ["multimodel_quality.json", "llama_quality.json"]:
        path = RESULTS / f
        if path.exists():
            all_data.extend(json.loads(path.read_text()))
    return all_data


def load_all_ablation():
    all_data = []
    for f in ["multimodel_ablation.json", "llama_ablation.json"]:
        path = RESULTS / f
        if path.exists():
            all_data.extend(json.loads(path.read_text()))
    return all_data


# ======================================================================
# Fig 1: Throughput comparison across 4 models (grouped bar)
# ======================================================================
def fig1():
    data = load_all_e2e()
    models = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]
    budget, seq, nreq = 50, 2048, 4

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(models))
    w = 0.25
    for i, mode in enumerate(["baseline", "naive", "orchkv"]):
        vals = []
        for m in models:
            r = next((r for r in data if r["model"] == m and r["mode"] == mode
                       and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                       and r["n_requests"] == nreq and r.get("completed", 0) > 0), None)
            vals.append(r["avg_throughput"] if r else 0)
        ax.bar(x + i * w, vals, w, label=MODE_LABELS[mode],
               color=MODE_COLORS[mode], hatch=MODE_HATCHES[mode],
               edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Model")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xticks(x + w)
    ax.set_xticklabels([m.replace("2.5-", "2.5\n") for m in models], fontsize=7)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
    ax.set_ylim(0)
    ax.set_title(f"budget={budget}MB, seq={seq}, nreq={nreq}", fontsize=8)
    save(fig, "fig1_throughput_4models")


# ======================================================================
# Fig 2: Eviction reduction across 4 models
# ======================================================================
def fig2():
    data = load_all_e2e()
    models = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]
    budget, seq = 50, 2048
    nreqs = [1, 4, 8, 16]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))

    # Left: eviction count (log scale)
    ax = axes[0]
    x = np.arange(len(nreqs))
    w = 0.35
    for i, (mode, label) in enumerate([("naive", "FIFO"), ("orchkv", "OrchKvCache")]):
        vals = []
        for nr in nreqs:
            rs = [r for r in data if r["mode"] == mode and r["gpu_budget_mb"] == budget
                  and r["seq_len"] == seq and r["n_requests"] == nr
                  and r["model"] == "Qwen2.5-7B" and r.get("completed", 0) > 0]
            vals.append(rs[0]["total_evictions"] if rs and rs[0]["total_evictions"] > 0 else 1)
        ax.bar(x + i * w, vals, w, label=label, color=MODE_COLORS[mode if mode != "naive" else mode],
               edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Total Evictions (log)")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(nreqs)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(a) Qwen2.5-7B, seq=2048", fontsize=8)

    # Right: reduction ratio across models
    ax = axes[1]
    for model in models:
        ratios = []
        valid_nreqs = []
        for nr in nreqs:
            naive_r = next((r for r in data if r["model"] == model and r["mode"] == "naive"
                            and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                            and r["n_requests"] == nr and r.get("total_evictions", 0) > 0), None)
            orch_r = next((r for r in data if r["model"] == model and r["mode"] == "orchkv"
                           and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                           and r["n_requests"] == nr and r.get("total_evictions", 0) > 0), None)
            if naive_r and orch_r and orch_r["total_evictions"] > 0:
                ratios.append(naive_r["total_evictions"] / orch_r["total_evictions"])
                valid_nreqs.append(nr)
        if ratios:
            ax.plot(valid_nreqs, ratios, "o-", label=model, color=MODEL_COLORS[model],
                    linewidth=1.2, markersize=4)
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Eviction Reduction (Naive/OrchKv)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(b) Reduction ratio, budget=50MB", fontsize=8)
    ax.set_xticks(nreqs)

    fig.tight_layout(w_pad=2)
    save(fig, "fig2_eviction_comparison")


# ======================================================================
# Fig 3: TPOT stability across request count
# ======================================================================
def fig3():
    data = load_all_e2e()
    budget, seq = 50, 2048
    nreqs = [1, 4, 8, 16]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for mode in ["naive", "orchkv"]:
        vals = []
        for nr in nreqs:
            r = next((r for r in data if r["model"] == "Qwen2.5-7B" and r["mode"] == mode
                       and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                       and r["n_requests"] == nr and r.get("completed", 0) > 0), None)
            vals.append(r["avg_tpot_ms"] if r else 0)
        marker = "s" if mode == "naive" else "o"
        ax.plot(nreqs, vals, f"{marker}-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=1.5, markersize=5)

    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title("Qwen2.5-7B, budget=50MB, seq=2048", fontsize=8)
    ax.legend(frameon=False)
    ax.set_xticks(nreqs)
    save(fig, "fig3_tpot_stability")


# ======================================================================
# Fig 4: Speedup heatmap (budget x nreq) for Qwen
# ======================================================================
def fig4():
    data = load_all_e2e()
    seq = 2048
    budgets = [50, 100, 200]
    nreqs = [1, 4, 8, 16]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
    for ax_idx, model in enumerate(["Qwen2.5-7B", "LLaMA-2-7B"]):
        ax = axes[ax_idx]
        grid = np.ones((len(budgets), len(nreqs)))
        for bi, b in enumerate(budgets):
            for ni, nr in enumerate(nreqs):
                naive_r = next((r for r in data if r["model"] == model and r["mode"] == "naive"
                                and r["gpu_budget_mb"] == b and r["seq_len"] == seq
                                and r["n_requests"] == nr and r.get("avg_throughput", 0) > 0), None)
                orch_r = next((r for r in data if r["model"] == model and r["mode"] == "orchkv"
                               and r["gpu_budget_mb"] == b and r["seq_len"] == seq
                               and r["n_requests"] == nr and r.get("avg_throughput", 0) > 0), None)
                if naive_r and orch_r and naive_r["avg_throughput"] > 0:
                    grid[bi, ni] = orch_r["avg_throughput"] / naive_r["avg_throughput"]

        im = ax.imshow(grid, cmap="RdYlGn", vmin=0.8, vmax=2.0, aspect="auto")
        ax.set_xticks(range(len(nreqs)))
        ax.set_xticklabels(nreqs)
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels([f"{b}MB" for b in budgets])
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel("GPU KV Budget")
        ax.set_title(f"({chr(97+ax_idx)}) {model}", fontsize=8)
        for bi in range(len(budgets)):
            for ni in range(len(nreqs)):
                v = grid[bi, ni]
                ax.text(ni, bi, f"{v:.2f}x", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 1.4 else "black", fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.8, label="OrchKv / Naive Speedup", pad=0.02)
    fig.tight_layout(w_pad=1)
    save(fig, "fig4_speedup_heatmap")


# ======================================================================
# Fig 5: Ablation across 4 models
# ======================================================================
def fig5():
    data = load_all_ablation()
    models = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]
    configs = ["gpu-only", "naive-fifo", "orchkv"]
    config_colors = {"gpu-only": "#4e79a7", "naive-fifo": "#e15759", "orchkv": "#59a14f"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3))

    x = np.arange(len(models))
    w = 0.25
    for i, cfg in enumerate(configs):
        vals_tps = []
        vals_evict = []
        for m in models:
            r = next((r for r in data if r.get("model") == m and r.get("config") == cfg
                       and r.get("status") == "OK"), None)
            vals_tps.append(r["throughput"] if r else 0)
            vals_evict.append(max(r.get("evictions", 0), 0.5) if r else 0.5)
        ax1.bar(x + i * w, vals_tps, w, label=cfg, color=config_colors[cfg],
                edgecolor="white", linewidth=0.4)
        if cfg != "gpu-only":
            ax2.bar(x + (i - 1) * 0.35, vals_evict, 0.35, label=cfg,
                    color=config_colors[cfg], edgecolor="white", linewidth=0.4)

    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_xticks(x + w)
    ax1.set_xticklabels([m.split("-")[0] for m in models], fontsize=7)
    ax1.legend(frameon=False, fontsize=6.5, ncol=3)
    ax1.set_title("(a) Throughput", fontsize=8)

    ax2.set_yscale("log")
    ax2.set_ylabel("Evictions (log)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split("-")[0] for m in models], fontsize=7)
    ax2.legend(frameon=False, fontsize=6.5)
    ax2.set_title("(b) Eviction Count", fontsize=8)

    fig.tight_layout(w_pad=2)
    save(fig, "fig5_ablation_4models")


# ======================================================================
# Fig 6: Quality verification table
# ======================================================================
def fig6():
    data = load_all_quality()

    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    ax.axis("off")

    headers = ["Model", "Prompt", "Length", "Generated", "Match Rate", "Evictions"]
    rows = []
    for r in data:
        if r.get("match_rate", -1) < 0:
            continue
        rows.append([
            r.get("model", "?"), r.get("prompt", "?"),
            str(r.get("prompt_len", "?")), str(r.get("generated", "?")),
            f"{r['match_rate']:.2f}%", str(r.get("evictions", 0)),
        ])

    if not rows:
        plt.close(fig)
        return

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#d4e6f1")
        table[0, j].set_text_props(fontweight="bold", fontsize=7)
    for i in range(1, len(rows) + 1):
        table[i, 4].set_text_props(fontweight="bold", color="#1a9641")

    ax.set_title("Lossless Quality Verification (Greedy Decoding, All Models)", fontsize=9, pad=15)
    save(fig, "fig6_quality_all_models")


# ======================================================================
# Fig 7: Throughput scaling with request count (4 models × orchkv vs naive)
# ======================================================================
def fig7():
    data = load_all_e2e()
    budget, seq = 50, 2048
    nreqs = [1, 4, 8, 16]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3))

    for model in ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B"]:
        for ax, mode in [(ax1, "naive"), (ax2, "orchkv")]:
            vals = []
            valid_nr = []
            for nr in nreqs:
                r = next((r for r in data if r["model"] == model and r["mode"] == mode
                           and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                           and r["n_requests"] == nr and r.get("avg_throughput", 0) > 0), None)
                if r:
                    vals.append(r["avg_throughput"])
                    valid_nr.append(nr)
            if vals:
                ax.plot(valid_nr, vals, "o-", label=model, color=MODEL_COLORS[model],
                        linewidth=1.2, markersize=4)

    ax1.set_title("(a) FIFO Offload", fontsize=8)
    ax2.set_title("(b) OrchKvCache", fontsize=8)
    for ax in [ax1, ax2]:
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(nreqs)
        ax.legend(frameon=False, fontsize=6.5)

    fig.tight_layout(w_pad=2)
    save(fig, "fig7_throughput_scaling")


# ======================================================================
# Fig 8: Speedup by model (bar chart showing OrchKv benefit per model)
# ======================================================================
def fig8():
    data = load_all_e2e()
    models = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]
    budget, seq = 50, 2048

    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    speedups = []
    evict_reductions = []
    for model in models:
        model_speedups = []
        model_reductions = []
        for nr in [1, 4, 8, 16]:
            naive_r = next((r for r in data if r["model"] == model and r["mode"] == "naive"
                            and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                            and r["n_requests"] == nr and r.get("avg_throughput", 0) > 0), None)
            orch_r = next((r for r in data if r["model"] == model and r["mode"] == "orchkv"
                           and r["gpu_budget_mb"] == budget and r["seq_len"] == seq
                           and r["n_requests"] == nr and r.get("avg_throughput", 0) > 0), None)
            if naive_r and orch_r:
                model_speedups.append(orch_r["avg_throughput"] / naive_r["avg_throughput"])
            if naive_r and orch_r and orch_r.get("total_evictions", 0) > 0:
                model_reductions.append(naive_r["total_evictions"] / orch_r["total_evictions"])
        speedups.append(np.mean(model_speedups) if model_speedups else 1.0)
        evict_reductions.append(np.mean(model_reductions) if model_reductions else 1.0)

    x = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax.bar(x, speedups, 0.6, color=colors, edgecolor="white", linewidth=0.4)

    for b, v in zip(bars, speedups):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}x",
                ha="center", fontsize=7, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Avg Speedup over FIFO")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("2.5-", "2.5\n").replace("2-", "2\n") for m in models], fontsize=7)
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.set_title(f"OrchKvCache vs FIFO, budget={budget}MB, seq={seq}", fontsize=8)
    save(fig, "fig8_speedup_per_model")


# ======================================================================
# Fig 9-12: Kernel-level experiments (reuse from old data)
# ======================================================================
def fig9_policy():
    path = RESULTS / "benchmark_e5_policy_sweep.json"
    if not path.exists(): return
    data = json.loads(path.read_text())
    alphas = sorted(set(r["alpha"] for r in data))
    patterns = sorted(set(r["pattern"] for r in data))
    grid = np.zeros((len(patterns), len(alphas)))
    for r in data:
        pi = patterns.index(r["pattern"])
        ai = alphas.index(r["alpha"])
        grid[pi, ai] = r.get("hot_ratio", r.get("n_hot", 0) / max(r.get("total_blocks", 64), 1))

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(patterns))); ax.set_yticklabels(patterns)
    ax.set_xlabel(r"$\alpha$ (attention weight)"); ax.set_ylabel("Access Pattern")
    ax.set_title("Hot Ratio by Policy Weight (E5)", fontsize=8)
    for pi in range(len(patterns)):
        for ai in range(len(alphas)):
            ax.text(ai, pi, f"{grid[pi,ai]:.0%}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "fig9_policy_heatmap")


def fig10_scalability():
    path = RESULTS / "benchmark_e9_scalability.json"
    if not path.exists(): return
    data = json.loads(path.read_text())
    blocks = [r.get("n_blocks", r.get("num_blocks", 0)) for r in data]
    avg = [r.get("avg_latency_us", r.get("avg_us", 0)) for r in data]
    p99 = [r.get("p99_latency_us", r.get("p99_us", 0)) for r in data]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(blocks, avg, "o-", label="Avg", color="#4e79a7", linewidth=1.5, markersize=5)
    ax.plot(blocks, p99, "s--", label="P99", color="#e15759", linewidth=1.5, markersize=5)
    ax.set_xlabel("Number of KV Blocks"); ax.set_ylabel("Scheduling Latency ($\\mu$s)")
    ax.set_title("Scheduler Scalability (E9)", fontsize=8)
    ax.legend(frameon=False)
    ax.axhline(y=60, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    if len(blocks) > 2 and blocks[0] > 0:
        exp = np.log(avg[-1] / max(avg[0], 0.01)) / np.log(blocks[-1] / max(blocks[0], 1))
        ax.text(0.05, 0.92, f"Exponent: {exp:.2f}", transform=ax.transAxes, fontsize=7, va="top")
    save(fig, "fig10_scalability")


def fig11_bandwidth():
    path = RESULTS / "benchmark_e8_storage_bw.json"
    if not path.exists(): return
    data = json.loads(path.read_text())
    if not isinstance(data, dict): return
    colors = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    gpu_dram = data.get("gpu_dram", [])
    dram_stor = data.get("dram_storage", [])
    if gpu_dram:
        sizes = [r["size_mb"] for r in gpu_dram]
        ax.plot(sizes, [r["d2h_gbps"] for r in gpu_dram], "o-", label="GPU→DRAM", color=colors[0], linewidth=1.2, markersize=4)
        ax.plot(sizes, [r["h2d_gbps"] for r in gpu_dram], "s-", label="DRAM→GPU", color=colors[1], linewidth=1.2, markersize=4)
    if dram_stor:
        sizes = [r["size_mb"] for r in dram_stor]
        ax.plot(sizes, [r["write_gbps"] for r in dram_stor], "^-", label="DRAM→SSD (write)", color=colors[2], linewidth=1.2, markersize=4)
        ax.plot(sizes, [r["read_gbps"] for r in dram_stor], "v-", label="SSD→DRAM (read)", color=colors[3], linewidth=1.2, markersize=4)
    ax.set_xlabel("Transfer Size (MB)"); ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Inter-Tier Bandwidth (E8)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig11_bandwidth")


def fig12_prefetch():
    path = RESULTS / "benchmark_e7_prefetch.json"
    if not path.exists(): return
    data = json.loads(path.read_text())
    budgets = sorted(set(r.get("prefetch_budget", r.get("budget", 0)) for r in data))
    dispatched = []
    latencies = []
    for b in budgets:
        rs = [r for r in data if r.get("prefetch_budget", r.get("budget", 0)) == b]
        if rs:
            dispatched.append(rs[0].get("total_dispatched", rs[0].get("dispatched_per_100", 0)))
            latencies.append(rs[0].get("avg_latency_us", rs[0].get("avg_schedule_us", 0)))
    if not dispatched: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.2))
    ax1.plot(budgets, dispatched, "o-", color="#4e79a7", linewidth=1.5, markersize=5)
    ax1.set_xlabel("Prefetch Budget ($K$)"); ax1.set_ylabel("Dispatched / 100 steps")
    ax1.set_title("(a) Dispatch Saturation (E7)", fontsize=8)
    ax2.plot(budgets, latencies, "s-", color="#e15759", linewidth=1.5, markersize=5)
    ax2.set_xlabel("Prefetch Budget ($K$)"); ax2.set_ylabel("Latency ($\\mu$s)")
    ax2.set_title("(b) Per-Step Overhead (E7)", fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig, "fig12_prefetch")


def main():
    print("Generating final paper figures (4 models)...\n")
    figs = [
        ("Fig 1: Throughput 4 models", fig1),
        ("Fig 2: Eviction comparison", fig2),
        ("Fig 3: TPOT stability", fig3),
        ("Fig 4: Speedup heatmap", fig4),
        ("Fig 5: Ablation 4 models", fig5),
        ("Fig 6: Quality table", fig6),
        ("Fig 7: Throughput scaling", fig7),
        ("Fig 8: Speedup per model", fig8),
        ("Fig 9: Policy heatmap (E5)", fig9_policy),
        ("Fig 10: Scalability (E9)", fig10_scalability),
        ("Fig 11: Bandwidth (E8)", fig11_bandwidth),
        ("Fig 12: Prefetch (E7)", fig12_prefetch),
    ]
    for name, func in figs:
        print(f"[{name}]")
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll figures in {FIGDIR}/")
    pdfs = sorted(f for f in os.listdir(FIGDIR) if f.endswith(".pdf"))
    print(f"Total: {len(pdfs)} PDFs")
    for f in pdfs:
        print(f"  {f}")


if __name__ == "__main__":
    main()

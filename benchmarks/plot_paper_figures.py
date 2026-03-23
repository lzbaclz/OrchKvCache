#!/usr/bin/env python3
"""
Generate publication-quality figures for OrchKvCache paper.

Reads JSON data from benchmarks/results/ and produces PDF + PNG figures
in benchmarks/figures/.

Usage:
    python benchmarks/plot_paper_figures.py            # all available
    python benchmarks/plot_paper_figures.py --only e5   # specific experiment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ──── Style ─────────────────────────────────────────────────────────
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "lines.markersize": 7,
}
plt.rcParams.update(STYLE)

COLORS = {
    "orchkv": "#2563EB",
    "baseline": "#DC2626",
    "gpu_hbm": "#7C3AED",
    "dram": "#059669",
    "tmpfs": "#D97706",
    "ssd": "#6B7280",
    "hot": "#DC2626",
    "warm": "#F59E0B",
    "cold": "#3B82F6",
    "accent1": "#8B5CF6",
    "accent2": "#10B981",
    "accent3": "#F43F5E",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "*"]


def load_json(name: str):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name: str):
    for ext in ["pdf", "png"]:
        p = FIGURES / f"{name}.{ext}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  [fig] {name}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════
#  E5: Hot/Cold Policy Sweep — Fig.8 + Fig.9
# ═══════════════════════════════════════════════════════════════════

def plot_e5():
    data = load_json("benchmark_e5_policy_sweep")
    if not data:
        print("  [skip] E5: no data")
        return

    # --- Fig.8: Heatmap (α vs pattern) → hot_ratio for fixed pattern ---
    fixed = [r for r in data if r["pattern"] == "fixed"]
    alphas = sorted(set(r["alpha"] for r in fixed))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax_idx, metric in enumerate(["hot_ratio", "warm_ratio", "cold_ratio"]):
        ax = axes[ax_idx]
        label_map = {"hot_ratio": "Hot", "warm_ratio": "Warm", "cold_ratio": "Cold"}
        color_map = {"hot_ratio": "Reds", "warm_ratio": "YlOrBr", "cold_ratio": "Blues"}

        patterns = ["fixed", "shift", "zipf"]
        grid = np.zeros((len(alphas), len(patterns)))

        for r in data:
            if r["alpha"] in alphas and r["pattern"] in patterns:
                ai = alphas.index(r["alpha"])
                pi = patterns.index(r["pattern"])
                grid[ai, pi] = r[metric]

        im = ax.imshow(grid, cmap=color_map[metric], aspect="auto",
                       vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels(patterns)
        ax.set_yticks(range(len(alphas)))
        beta_gamma = []
        for a in alphas:
            row = [r for r in data if r["alpha"] == a and r["pattern"] == "fixed"]
            if row:
                beta_gamma.append(f"α={a:.1f}\nβ={row[0]['beta']:.1f} γ={row[0]['gamma']:.1f}")
            else:
                beta_gamma.append(f"α={a:.1f}")
        ax.set_yticklabels(beta_gamma, fontsize=8)
        ax.set_title(f"{label_map[metric]} Ratio")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Fig.8: Hot/Cold Classification vs. Policy Weights (α, β, γ)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig08_policy_heatmap")

    # --- Fig.9: Stacked bar — distribution per (α, pattern) ---
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = []
    hot_vals, warm_vals, cold_vals = [], [], []
    for r in data:
        labels.append(f"α={r['alpha']:.1f}\n{r['pattern']}")
        hot_vals.append(r["avg_n_hot"])
        warm_vals.append(r["avg_n_warm"])
        cold_vals.append(r["avg_n_cold"])

    x = np.arange(len(labels))
    width = 0.7

    ax.bar(x, hot_vals, width, label="Hot", color=COLORS["hot"], alpha=0.85)
    ax.bar(x, warm_vals, width, bottom=hot_vals, label="Warm",
           color=COLORS["warm"], alpha=0.85)
    bottoms = [h + w for h, w in zip(hot_vals, warm_vals)]
    ax.bar(x, cold_vals, width, bottom=bottoms, label="Cold",
           color=COLORS["cold"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Number of Blocks")
    ax.set_title("Fig.9: Block Classification Distribution (n=256 blocks)")
    ax.legend(loc="upper right")
    ax.axhline(y=256, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    fig.tight_layout()
    save_fig(fig, "fig09_classification_distribution")


# ═══════════════════════════════════════════════════════════════════
#  E7: Prefetch Effectiveness — Fig.11 + Fig.12
# ═══════════════════════════════════════════════════════════════════

def plot_e7():
    data = load_json("benchmark_e7_prefetch")
    if not data:
        print("  [skip] E7: no data")
        return

    ok = [r for r in data if r.get("status") != "skip"]
    if not ok:
        print("  [skip] E7: no valid data")
        return

    budgets = [r["prefetch_budget"] for r in ok]
    dispatched = [r["avg_prefetches_dispatched"] for r in ok]
    sched_us = [r["avg_schedule_us"] for r in ok]
    n_hot = [r["avg_n_hot"] for r in ok]
    n_warm = [r["avg_n_warm"] for r in ok]

    # --- Fig.11: Prefetch dispatches + classification vs budget ---
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(budgets, dispatched, "o-", color=COLORS["orchkv"],
             label="Prefetches Dispatched", markersize=8)
    ax1.set_xlabel("Prefetch Budget")
    ax1.set_ylabel("Cumulative Dispatches (100 steps)", color=COLORS["orchkv"])
    ax1.tick_params(axis="y", labelcolor=COLORS["orchkv"])

    ax2 = ax1.twinx()
    ax2.plot(budgets, n_hot, "s--", color=COLORS["hot"],
             label="Hot Blocks", markersize=6, alpha=0.8)
    ax2.plot(budgets, n_warm, "^--", color=COLORS["warm"],
             label="Warm Blocks", markersize=6, alpha=0.8)
    ax2.set_ylabel("Block Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Fig.11: Prefetch Dispatches vs. Budget")
    fig.tight_layout()
    save_fig(fig, "fig11_prefetch_dispatches")

    # --- Fig.12: Scheduling latency vs budget ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(budgets)), sched_us, color=COLORS["accent1"], alpha=0.8)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Prefetch Budget")
    ax.set_ylabel("Avg Scheduling Latency (μs)")
    ax.set_title("Fig.12: Scheduling Overhead vs. Prefetch Budget")

    for i, v in enumerate(sched_us):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig12_prefetch_latency")


# ═══════════════════════════════════════════════════════════════════
#  E8: Storage Bandwidth — Fig.13
# ═══════════════════════════════════════════════════════════════════

def plot_e8():
    data = load_json("benchmark_e8_storage_bw")
    if not data:
        print("  [skip] E8: no data")
        return

    gpu_dram = data.get("gpu_dram", [])
    dram_stor = data.get("dram_storage", [])

    if not gpu_dram and not dram_stor:
        print("  [skip] E8: no valid data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if gpu_dram:
        sizes = [r["size_mb"] for r in gpu_dram]
        d2h = [r["d2h_gbps"] for r in gpu_dram]
        h2d = [r["h2d_gbps"] for r in gpu_dram]
        ax.plot(sizes, d2h, "o-", color=COLORS["gpu_hbm"],
                label="GPU→DRAM (D2H)", markersize=6)
        ax.plot(sizes, h2d, "s-", color=COLORS["orchkv"],
                label="DRAM→GPU (H2D)", markersize=6)

    if dram_stor:
        sizes = [r["size_mb"] for r in dram_stor]
        w_bw = [r["write_gbps"] for r in dram_stor]
        r_bw = [r["read_gbps"] for r in dram_stor]
        ax.plot(sizes, w_bw, "^--", color=COLORS["tmpfs"],
                label="DRAM→tmpfs (Write)", markersize=6)
        ax.plot(sizes, r_bw, "D--", color=COLORS["accent2"],
                label="tmpfs→DRAM (Read)", markersize=6)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))
    ax.set_xlabel("Transfer Size (MB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Fig.13: Storage Tier Bandwidth (A100-SXM4 + DDR4 + tmpfs)")
    ax.legend(loc="center right")

    ax.axhspan(20, 25, color=COLORS["gpu_hbm"], alpha=0.05)
    ax.axhspan(5, 15, color=COLORS["tmpfs"], alpha=0.05)

    fig.tight_layout()
    save_fig(fig, "fig13_storage_bandwidth")


# ═══════════════════════════════════════════════════════════════════
#  E9: Scalability — Fig.14
# ═══════════════════════════════════════════════════════════════════

def plot_e9():
    data = load_json("benchmark_e9_scalability")
    if not data:
        print("  [skip] E9: no data")
        return

    ok = [r for r in data if r.get("status") != "skip"]
    if not ok:
        print("  [skip] E9: no valid data")
        return

    n_blocks = [r["n_blocks"] for r in ok]
    avg_us = [r["avg_schedule_us"] for r in ok]
    p50_us = [r["p50_schedule_us"] for r in ok]
    p99_us = [r["p99_schedule_us"] for r in ok]
    std_us = [r.get("std_schedule_us", 0) for r in ok]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(n_blocks, avg_us, yerr=std_us, fmt="o-",
                color=COLORS["orchkv"], label="Avg", capsize=4,
                markersize=7, linewidth=2)
    ax.plot(n_blocks, p50_us, "s--", color=COLORS["accent2"],
            label="P50", markersize=6, linewidth=1.5)
    ax.plot(n_blocks, p99_us, "^--", color=COLORS["accent3"],
            label="P99", markersize=6, linewidth=1.5)

    xs = np.array(n_blocks)
    slope = avg_us[-1] / n_blocks[-1]
    ax.plot(xs, slope * xs, ":", color="gray", alpha=0.5,
            label=f"Linear ref ({slope:.4f}×N)")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Number of Tracked Blocks")
    ax.set_ylabel("Scheduling Latency (μs)")
    ax.set_title("Fig.14: Scheduling Scalability (50 steps, 3 runs)")
    ax.legend(loc="upper left")

    ax.axhline(y=100, color="red", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(n_blocks[0], 105, "100μs threshold", fontsize=8,
            color="red", alpha=0.5)

    fig.tight_layout()
    save_fig(fig, "fig14_scalability")


# ═══════════════════════════════════════════════════════════════════
#  E1: End-to-End Throughput — Fig.1 + Fig.2 + Fig.3
# ═══════════════════════════════════════════════════════════════════

def plot_e1():
    data = load_json("benchmark_e2e")
    if not data:
        print("  [skip] E1: no data")
        return

    ok = [r for r in data if r.get("status") == "ok"]
    if not ok:
        print("  [skip] E1: no valid data")
        return

    backends = sorted(set(r["backend"] for r in ok))
    seq_lens = sorted(set(r["seq_len"] for r in ok))
    batch_sizes = sorted(set(r["batch_size"] for r in ok))

    # --- Fig.1: Throughput vs seq_len (fixed bs=4 or closest) ---
    bs_target = min(batch_sizes, key=lambda x: abs(x - 4))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for be in backends:
        sub = sorted([r for r in ok if r["backend"] == be and r["batch_size"] == bs_target],
                     key=lambda r: r["seq_len"])
        if sub:
            xs = [r["seq_len"] for r in sub]
            ys = [r["throughput_tok_s"] for r in sub]
            color = COLORS.get(be, "#333")
            marker = "o" if be == "baseline" else "s"
            label = "Baseline" if be == "baseline" else "OrchKvCache"
            ax.plot(xs, ys, f"{marker}-", color=color, label=label, markersize=7)

    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Fig.1: Throughput vs. Sequence Length (batch={bs_target})")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()
    save_fig(fig, "fig01_throughput_vs_seqlen")

    # --- Fig.2: Throughput vs batch_size (fixed seq=2048 or closest) ---
    seq_target = min(seq_lens, key=lambda x: abs(x - 2048))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for be in backends:
        sub = sorted([r for r in ok if r["backend"] == be and r["seq_len"] == seq_target],
                     key=lambda r: r["batch_size"])
        if sub:
            xs = [r["batch_size"] for r in sub]
            ys = [r["throughput_tok_s"] for r in sub]
            color = COLORS.get(be, "#333")
            marker = "o" if be == "baseline" else "s"
            label = "Baseline" if be == "baseline" else "OrchKvCache"
            ax.plot(xs, ys, f"{marker}-", color=color, label=label, markersize=7)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Fig.2: Throughput vs. Batch Size (seq={seq_target})")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig02_throughput_vs_batchsize")

    # --- Fig.3: TTFT comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("ttft_est_ms", "TTFT (ms)", "Fig.3a: Time-to-First-Token"),
        ("tpot_est_ms", "TPOT (ms)", "Fig.3b: Time-per-Output-Token"),
    ]):
        ax = axes[ax_idx]
        width = 0.35
        sub = [r for r in ok if r["seq_len"] == seq_target]
        bl_data = sorted([r for r in sub if r["backend"] == "baseline"],
                        key=lambda r: r["batch_size"])
        ok_data = sorted([r for r in sub if r["backend"] == "orchkv"],
                        key=lambda r: r["batch_size"])

        if bl_data and ok_data:
            x = np.arange(len(bl_data))
            bl_vals = [r[metric] for r in bl_data]
            ok_vals = [r[metric] for r in ok_data]
            ax.bar(x - width/2, bl_vals, width, label="Baseline",
                   color=COLORS["baseline"], alpha=0.85)
            ax.bar(x + width/2, ok_vals, width, label="OrchKvCache",
                   color=COLORS["orchkv"], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([f"bs={r['batch_size']}" for r in bl_data])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()

    fig.tight_layout()
    save_fig(fig, "fig03_ttft_tpot")


# ═══════════════════════════════════════════════════════════════════
#  E2: Max Batch Size — Fig.4
# ═══════════════════════════════════════════════════════════════════

def plot_e2():
    data = load_json("test_memory_extension")
    if not data:
        print("  [skip] E2: no data")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    if isinstance(data, dict):
        labels = [f"seq={data['seq_len']}"]
        bl = [data["baseline_max_batch"]]
        ok = [data["orchkv_max_batch"]]
    elif isinstance(data, list):
        labels = [f"seq={r['seq_len']}" for r in data]
        bl = [r["baseline_max_batch"] for r in data]
        ok = [r["orchkv_max_batch"] for r in data]
    else:
        print("  [skip] E2: unexpected format")
        return

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, bl, width, label="Baseline", color=COLORS["baseline"], alpha=0.85)
    ax.bar(x + width/2, ok, width, label="OrchKvCache", color=COLORS["orchkv"], alpha=0.85)

    for i, (b, o) in enumerate(zip(bl, ok)):
        ax.text(i - width/2, b + 0.5, str(b), ha="center", fontsize=9)
        ax.text(i + width/2, o + 0.5, str(o), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Max Batch Size")
    ax.set_title("Fig.4: Maximum Batch Size Before OOM")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig04_max_batch_size")


# ═══════════════════════════════════════════════════════════════════
#  E3: Latency Breakdown — Fig.5
# ═══════════════════════════════════════════════════════════════════

def plot_e3():
    data = load_json("test_latency_breakdown")
    if not data:
        print("  [skip] E3: no data")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    configs = []
    for be in ["baseline", "orchkv"]:
        if be in data and isinstance(data[be], dict):
            info = data[be]
            label = "Baseline" if be == "baseline" else "OrchKvCache"
            ttft = info.get("ttft_est_ms", 0)
            tpot_total = info.get("timing", {}).get("avg_us", 0) / 1000 - ttft
            configs.append({
                "label": label,
                "ttft": ttft,
                "decode": max(tpot_total, 0),
            })

    if not configs:
        print("  [skip] E3: no valid data")
        return

    x = np.arange(len(configs))
    width = 0.5

    ttft_vals = [c["ttft"] for c in configs]
    decode_vals = [c["decode"] for c in configs]
    colors_stack = [COLORS["gpu_hbm"], COLORS["dram"]]
    labels_stack = ["Prefill (TTFT)", "Decode"]

    ax.bar(x, ttft_vals, width, label=labels_stack[0], color=colors_stack[0], alpha=0.85)
    ax.bar(x, decode_vals, width, bottom=ttft_vals, label=labels_stack[1],
           color=colors_stack[1], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in configs])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fig.5: Latency Breakdown (Prefill + Decode)")
    ax.legend()

    for i, c in enumerate(configs):
        total = c["ttft"] + c["decode"]
        ax.text(i, total + 5, f"{total:.0f}ms", ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig05_latency_breakdown")


# ═══════════════════════════════════════════════════════════════════
#  E4: Storage Tier Ablation — Fig.6 + Fig.7
# ═══════════════════════════════════════════════════════════════════

def plot_e4():
    data = load_json("benchmark_e4_tier_ablation")
    if not data:
        print("  [skip] E4: no data")
        return

    ok = [r for r in data if r.get("status") == "ok"]
    if not ok:
        print("  [skip] E4: no valid data")
        return

    labels = [r["tier"] for r in ok]
    throughput = [r["throughput_tok_s"] for r in ok]
    gpu_peak = [r["gpu_peak_mb"] / 1024 for r in ok]
    tier_colors = [COLORS["baseline"], COLORS["dram"], COLORS["tmpfs"], COLORS["orchkv"]]

    # --- Fig.6: Throughput vs tier config ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(labels)), throughput,
                  color=tier_colors[:len(ok)], alpha=0.85)
    for i, v in enumerate(throughput):
        ax.text(i, v + 20, f"{v:.0f}", ha="center", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Fig.6: Throughput vs. Storage Tier Configuration")
    fig.tight_layout()
    save_fig(fig, "fig06_tier_throughput")

    # --- Fig.7: GPU peak memory vs tier config ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(labels)), gpu_peak,
                  color=tier_colors[:len(ok)], alpha=0.85)
    for i, v in enumerate(gpu_peak):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("GPU Peak Memory (GB)")
    ax.set_title("Fig.7: GPU Peak Memory vs. Storage Tier Configuration")
    fig.tight_layout()
    save_fig(fig, "fig07_tier_gpu_memory")


# ═══════════════════════════════════════════════════════════════════
#  E6: Block Size Ablation — Fig.10
# ═══════════════════════════════════════════════════════════════════

def plot_e6():
    data = load_json("benchmark_e6_block_size")
    if not data:
        print("  [skip] E6: no data")
        return

    ok = [r for r in data if r.get("status") == "ok"]
    if not ok:
        return

    blk_sizes = [r["block_size"] for r in ok]
    throughput = [r["throughput_tok_s"] for r in ok]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(blk_sizes, throughput, "o-", color=COLORS["orchkv"], markersize=8, linewidth=2)

    for x, y in zip(blk_sizes, throughput):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Block Size (tokens/block)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Fig.10: Throughput vs. Block Size")
    ax.set_xticks(blk_sizes)
    fig.tight_layout()
    save_fig(fig, "fig10_block_size_ablation")


# ═══════════════════════════════════════════════════════════════════
#  E10: Quality Verification — Tab.1
# ═══════════════════════════════════════════════════════════════════

def plot_e10():
    data = load_json("eval_e10_quality")
    if not data:
        print("  [skip] E10: no data")
        return

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")

    rows = []
    tc = data.get("token_consistency", {})
    if tc and tc.get("status") != "skipped":
        rows.append(["Token Match Rate", f"{tc.get('match_rate', 0):.4%}",
                      f"{tc.get('matching_tokens', 0)}/{tc.get('total_tokens', 0)}",
                      "PASS" if tc.get("status") == "pass" else "FAIL"])

    ppl = data.get("perplexity", {})
    if ppl and ppl.get("status") not in ("skipped", "error"):
        rows.append(["Baseline PPL", f"{ppl.get('baseline_avg_ppl', 0):.4f}", "", ""])
        rows.append(["OrchKvCache PPL", f"{ppl.get('orchkv_avg_ppl', 0):.4f}", "", ""])
        rows.append(["PPL Relative Diff", f"{ppl.get('ppl_relative_diff', 0):.4%}", "",
                      "PASS" if ppl.get("status") == "pass" else "MARGINAL"])

    if not rows:
        print("  [skip] E10: no data")
        return

    headers = ["Metric", "Value", "Detail", "Status"]
    colors = [["#f0f4ff" if i % 2 == 0 else "white"] * len(headers)
              for i in range(len(rows))]

    table = ax.table(cellText=rows, colLabels=headers,
                     cellColours=colors, colColours=["#dbeafe"] * len(headers),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 3 and r > 0:
            status = rows[r-1][3]
            if status == "PASS":
                cell.set_facecolor("#dcfce7")
            elif status == "FAIL":
                cell.set_facecolor("#fee2e2")

    ax.set_title("Tab.1: Generation Quality Verification (E10)", fontsize=13, pad=20)
    fig.tight_layout()
    save_fig(fig, "tab01_quality_verification")


# ═══════════════════════════════════════════════════════════════════
#  Summary Table
# ═══════════════════════════════════════════════════════════════════

def plot_summary_table():
    """Generate a summary table of all available experiment results."""
    rows = []

    e5 = load_json("benchmark_e5_policy_sweep")
    if e5:
        fixed_08 = [r for r in e5 if r["alpha"] == 0.8 and r["pattern"] == "fixed"]
        if fixed_08:
            r = fixed_08[0]
            rows.append(["E5", "Policy (α=0.8)",
                          f"H={r['avg_n_hot']:.0f} W={r['avg_n_warm']:.0f} C={r['avg_n_cold']:.0f}",
                          "PASS" if r["avg_n_hot"] > 0 else "FAIL"])

    e7 = load_json("benchmark_e7_prefetch")
    if e7:
        b16 = [r for r in e7 if r.get("prefetch_budget") == 16]
        if b16:
            r = b16[0]
            rows.append(["E7", "Prefetch (budget=16)",
                          f"dispatched={r['avg_prefetches_dispatched']:.0f}",
                          "PASS"])

    e8 = load_json("benchmark_e8_storage_bw")
    if e8 and "gpu_dram" in e8:
        gd = e8["gpu_dram"]
        peak = max(r["h2d_gbps"] for r in gd)
        rows.append(["E8", "GPU↔DRAM BW", f"{peak:.1f} GB/s", "PASS"])
        ds = e8.get("dram_storage", [])
        if ds:
            peak_s = max(r["read_gbps"] for r in ds)
            rows.append(["E8", "DRAM↔tmpfs BW", f"{peak_s:.1f} GB/s", "PASS"])

    e9 = load_json("benchmark_e9_scalability")
    if e9:
        ok = [r for r in e9 if r.get("status") != "skip"]
        if ok:
            max_blk = ok[-1]
            rows.append(["E9", f"Scalability ({max_blk['n_blocks']} blks)",
                          f"avg={max_blk['avg_schedule_us']:.1f}μs "
                          f"p99={max_blk['p99_schedule_us']:.1f}μs",
                          "PASS" if max_blk["p99_schedule_us"] < 500 else "MARGINAL"])

    if not rows:
        print("  [skip] summary table: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 1 + len(rows) * 0.5))
    ax.axis("off")

    headers = ["Exp", "Configuration", "Result", "Status"]
    colors = [["#f0f4ff" if i % 2 == 0 else "white"] * len(headers)
              for i in range(len(rows))]

    table = ax.table(cellText=rows, colLabels=headers,
                     cellColours=colors, colColours=["#dbeafe"] * len(headers),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 3 and r > 0:
            status = rows[r-1][3]
            if status == "PASS":
                cell.set_facecolor("#dcfce7")
            elif status == "FAIL":
                cell.set_facecolor("#fee2e2")

    ax.set_title("Tab.1: Experiment Results Summary", fontsize=13, pad=20)
    fig.tight_layout()
    save_fig(fig, "tab01_summary")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

PLOTTERS = {
    "e1": plot_e1,
    "e2": plot_e2,
    "e3": plot_e3,
    "e4": plot_e4,
    "e5": plot_e5,
    "e6": plot_e6,
    "e7": plot_e7,
    "e8": plot_e8,
    "e9": plot_e9,
    "e10": plot_e10,
    "summary": plot_summary_table,
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--only", default=None, help="Plot only specific experiment")
    args = parser.parse_args()

    print(f"OrchKvCache — Paper Figure Generation")
    print(f"  Results dir: {RESULTS}")
    print(f"  Figures dir: {FIGURES}\n")

    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
    else:
        keys = list(PLOTTERS.keys())

    for key in keys:
        if key in PLOTTERS:
            print(f"  Plotting {key}...")
            PLOTTERS[key]()
        else:
            print(f"  [warn] unknown: {key}")

    print(f"\nDone. Figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate all paper-ready figures for OrchKvCache.
Style: SOSP/OSDI/SC/FAST — clean, two-column friendly, grayscale-safe.

Outputs PDF + PNG for each figure to benchmarks/figures/.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ── CCF-A style: clean, serif-friendly, compact ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "baseline": "#2c7bb6",
    "naive":    "#d7191c",
    "orchkv":   "#1a9641",
}
HATCHES = {"baseline": "", "naive": "//", "orchkv": ""}


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", format="pdf")
    fig.savefig(FIGDIR / f"{name}.png", format="png")
    print(f"  Saved {name}.pdf + {name}.png")
    plt.close(fig)


def load_json(name):
    path = RESULTS / name
    if not path.exists():
        print(f"  WARN: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


# ======================================================================
# Fig 1: E2E Throughput — OrchKv vs Naive vs Baseline (grouped bar)
# ======================================================================
def fig1_throughput():
    data = load_json("exp_e2e_real.json")
    if not data:
        return

    budget = 50
    seq = 4096
    nreqs = [1, 4, 8, 16]
    modes = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO Offload", "orchkv": "OrchKvCache"}

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(nreqs))
    w = 0.25

    for i, mode in enumerate(modes):
        vals = []
        for nr in nreqs:
            r = next((r for r in data if r["mode"] == mode and r["gpu_budget_mb"] == budget
                       and r["seq_len"] == seq and r["n_requests"] == nr), None)
            vals.append(r["avg_throughput_tok_s"] if r else 0)
        bars = ax.bar(x + i * w, vals, w, label=labels[mode],
                      color=COLORS[mode], hatch=HATCHES[mode], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xticks(x + w)
    ax.set_xticklabels(nreqs)
    ax.set_title(f"seq={seq}, GPU budget={budget}MB", fontsize=9)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_ylim(0)
    save(fig, "fig01_throughput_bar")


# ======================================================================
# Fig 2: Eviction Count Comparison (log-scale bar)
# ======================================================================
def fig2_eviction():
    data = load_json("exp_e2e_real.json")
    if not data:
        return

    budget = 50
    seq = 4096
    nreqs = [1, 4, 8, 16]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(nreqs))
    w = 0.35

    for i, (mode, label) in enumerate([("naive", "FIFO Offload"), ("orchkv", "OrchKvCache")]):
        vals = []
        for nr in nreqs:
            r = next((r for r in data if r["mode"] == mode and r["gpu_budget_mb"] == budget
                       and r["seq_len"] == seq and r["n_requests"] == nr), None)
            vals.append(max(r["total_evictions"], 1) if r else 1)
        ax.bar(x + i * w, vals, w, label=label,
               color=COLORS[mode], hatch=HATCHES[mode], edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Total Evictions (log scale)")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(nreqs)
    ax.set_title(f"seq={seq}, GPU budget={budget}MB", fontsize=9)
    ax.legend(frameon=False)

    for i, nr in enumerate(nreqs):
        naive_r = next((r for r in data if r["mode"] == "naive" and r["gpu_budget_mb"] == budget
                         and r["seq_len"] == seq and r["n_requests"] == nr), None)
        orch_r = next((r for r in data if r["mode"] == "orchkv" and r["gpu_budget_mb"] == budget
                        and r["seq_len"] == seq and r["n_requests"] == nr), None)
        if naive_r and orch_r and orch_r["total_evictions"] > 0:
            ratio = naive_r["total_evictions"] / orch_r["total_evictions"]
            ax.annotate(f"{ratio:.0f}x", xy=(x[i] + w / 2, orch_r["total_evictions"]),
                        fontsize=7, ha="center", va="bottom", color=COLORS["orchkv"], fontweight="bold")

    save(fig, "fig02_eviction_bar")


# ======================================================================
# Fig 3: TPOT Latency vs #Requests
# ======================================================================
def fig3_tpot():
    data = load_json("exp_e2e_real.json")
    if not data:
        return

    budget = 50
    seq = 4096
    nreqs = [1, 4, 8, 16]
    modes = ["naive", "orchkv"]
    labels = {"naive": "FIFO Offload", "orchkv": "OrchKvCache"}
    markers = {"naive": "s", "orchkv": "o"}

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    for mode in modes:
        vals = []
        for nr in nreqs:
            r = next((r for r in data if r["mode"] == mode and r["gpu_budget_mb"] == budget
                       and r["seq_len"] == seq and r["n_requests"] == nr), None)
            vals.append(r["avg_tpot_ms"] if r else 0)
        ax.plot(nreqs, vals, marker=markers[mode], label=labels[mode],
                color=COLORS[mode], linewidth=1.5, markersize=5)

    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title(f"Per-Token Latency, seq={seq}, budget={budget}MB", fontsize=9)
    ax.legend(frameon=False)
    ax.set_xticks(nreqs)
    save(fig, "fig03_tpot_line")


# ======================================================================
# Fig 4: Speedup Heatmap (budget × nreq)
# ======================================================================
def fig4_speedup_heatmap():
    data = load_json("exp_e2e_real.json")
    if not data:
        return

    seq = 4096
    budgets = [50, 100, 200, 500]
    nreqs = [1, 4, 8, 16]

    grid = np.zeros((len(budgets), len(nreqs)))
    for bi, b in enumerate(budgets):
        for ni, nr in enumerate(nreqs):
            naive_r = next((r for r in data if r["mode"] == "naive" and r["gpu_budget_mb"] == b
                             and r["seq_len"] == seq and r["n_requests"] == nr), None)
            orch_r = next((r for r in data if r["mode"] == "orchkv" and r["gpu_budget_mb"] == b
                            and r["seq_len"] == seq and r["n_requests"] == nr), None)
            if naive_r and orch_r and naive_r["avg_throughput_tok_s"] > 0:
                grid[bi, ni] = orch_r["avg_throughput_tok_s"] / naive_r["avg_throughput_tok_s"]
            else:
                grid[bi, ni] = 1.0

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.85, vmax=1.40, aspect="auto")
    ax.set_xticks(range(len(nreqs)))
    ax.set_xticklabels(nreqs)
    ax.set_yticks(range(len(budgets)))
    ax.set_yticklabels([f"{b}MB" for b in budgets])
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("GPU KV Budget")
    ax.set_title(f"OrchKv/Naive Speedup, seq={seq}", fontsize=9)

    for bi in range(len(budgets)):
        for ni in range(len(nreqs)):
            ax.text(ni, bi, f"{grid[bi, ni]:.2f}x", ha="center", va="center", fontsize=7,
                    color="white" if grid[bi, ni] > 1.25 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Speedup", fontsize=8)
    save(fig, "fig04_speedup_heatmap")


# ======================================================================
# Fig 5: Ablation — Throughput & Evictions
# ======================================================================
def fig5_ablation():
    data = load_json("exp_ablation.json")
    if not data:
        return

    ok_data = [r for r in data if r.get("status") == "OK"]
    configs = [r["config"] for r in ok_data]
    throughputs = [r["throughput_tok_s"] for r in ok_data]
    evictions = [r["evictions"] for r in ok_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.2))

    colors_abl = ["#2c7bb6", "#d7191c", "#1a9641"]
    bars1 = ax1.bar(configs, throughputs, color=colors_abl[:len(configs)], edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title("(a) Throughput Comparison", fontsize=9)
    for b, v in zip(bars1, throughputs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}", ha="center", fontsize=7)

    bars2 = ax2.bar(configs, [max(e, 0.5) for e in evictions],
                    color=colors_abl[:len(configs)], edgecolor="white", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Evictions (log scale)")
    ax2.set_title("(b) Eviction Count", fontsize=9)
    for b, v in zip(bars2, evictions):
        if v > 0:
            ax2.text(b.get_x() + b.get_width() / 2, v * 1.5, f"{v:,}", ha="center", fontsize=7)

    fig.tight_layout(w_pad=2)
    save(fig, "fig05_ablation")


# ======================================================================
# Fig 6: Quality — Token Match Table (visual)
# ======================================================================
def fig6_quality():
    data = load_json("exp_quality.json")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(4.5, 1.5))
    ax.axis("off")

    headers = ["Prompt", "Length", "Generated", "Match Rate", "Evictions", "Promotions"]
    rows = []
    for r in data:
        rows.append([
            r["prompt"], str(r["prompt_len"]), str(r["generated_len"]),
            f'{r["token_match_rate"]:.2f}%',
            str(r["evictions"]), str(r["promotions"]),
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#d4e6f1")
        table[0, j].set_text_props(fontweight="bold")

    for i in range(1, len(rows) + 1):
        table[i, 3].set_text_props(fontweight="bold", color="#1a9641")

    ax.set_title("Lossless Quality Verification (Greedy Decoding)", fontsize=10, pad=20)
    save(fig, "fig06_quality_table")


# ======================================================================
# Fig 7: E5 — Hot-Cold Policy Heatmap (from old experiment)
# ======================================================================
def fig7_policy():
    data = load_json("benchmark_e5_policy_sweep.json")
    if not data:
        return

    alphas = sorted(set(r["alpha"] for r in data))
    patterns = sorted(set(r["pattern"] for r in data))

    grid = np.zeros((len(patterns), len(alphas)))
    for r in data:
        pi = patterns.index(r["pattern"])
        ai = alphas.index(r["alpha"])
        grid[pi, ai] = r.get("hot_ratio", r.get("n_hot", 0) / max(r.get("total_blocks", 64), 1))

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels(patterns)
    ax.set_xlabel(r"$\alpha$ (attention weight)")
    ax.set_ylabel("Access Pattern")
    ax.set_title("Hot Ratio by Policy Weight", fontsize=9)

    for pi in range(len(patterns)):
        for ai in range(len(alphas)):
            ax.text(ai, pi, f"{grid[pi, ai]:.0%}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Hot Ratio")
    save(fig, "fig07_policy_heatmap")


# ======================================================================
# Fig 8: E7 — Prefetch Dispatch vs Budget
# ======================================================================
def fig8_prefetch():
    data = load_json("benchmark_e7_prefetch.json")
    if not data:
        return

    budgets = sorted(set(r.get("prefetch_budget", r.get("budget", 0)) for r in data))
    dispatched = []
    latencies = []

    for b in budgets:
        rs = [r for r in data if r.get("prefetch_budget", r.get("budget", 0)) == b]
        if rs:
            dispatched.append(rs[0].get("total_dispatched", rs[0].get("dispatched_per_100", 0)))
            latencies.append(rs[0].get("avg_latency_us", rs[0].get("avg_schedule_us", 0)))

    if not dispatched:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.2))

    ax1.plot(budgets, dispatched, "o-", color="#2c7bb6", linewidth=1.5, markersize=5)
    ax1.set_xlabel("Prefetch Budget (K)")
    ax1.set_ylabel("Dispatched / 100 steps")
    ax1.set_title("(a) Dispatch Saturation", fontsize=9)
    ax1.axhline(y=max(dispatched), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax2.plot(budgets, latencies, "s-", color="#d7191c", linewidth=1.5, markersize=5)
    ax2.set_xlabel("Prefetch Budget (K)")
    ax2.set_ylabel("Scheduling Latency (us)")
    ax2.set_title("(b) Per-Step Overhead", fontsize=9)

    fig.tight_layout(w_pad=2)
    save(fig, "fig08_prefetch")


# ======================================================================
# Fig 9: E8 — Storage Bandwidth
# ======================================================================
def fig9_bandwidth():
    data = load_json("benchmark_e8_storage_bw.json")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    colors_bw = ["#2c7bb6", "#1a9641", "#d7191c", "#fdae61"]
    lines = []

    if isinstance(data, dict):
        gpu_dram = data.get("gpu_dram", [])
        dram_stor = data.get("dram_storage", [])

        if gpu_dram:
            sizes = [r["size_mb"] for r in gpu_dram]
            ax.plot(sizes, [r["d2h_gbps"] for r in gpu_dram], "o-",
                    label="GPU→DRAM", color=colors_bw[0], linewidth=1.5, markersize=4)
            ax.plot(sizes, [r["h2d_gbps"] for r in gpu_dram], "s-",
                    label="DRAM→GPU", color=colors_bw[1], linewidth=1.5, markersize=4)
        if dram_stor:
            sizes = [r["size_mb"] for r in dram_stor]
            ax.plot(sizes, [r["write_gbps"] for r in dram_stor], "^-",
                    label="DRAM→Storage (write)", color=colors_bw[2], linewidth=1.5, markersize=4)
            ax.plot(sizes, [r["read_gbps"] for r in dram_stor], "v-",
                    label="Storage→DRAM (read)", color=colors_bw[3], linewidth=1.5, markersize=4)

    ax.set_xlabel("Transfer Size (MB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Inter-Tier Bandwidth", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig09_bandwidth")


# ======================================================================
# Fig 10: E9 — Scheduling Scalability
# ======================================================================
def fig10_scalability():
    data = load_json("benchmark_e9_scalability.json")
    if not data:
        return

    blocks = [r.get("n_blocks", r.get("num_blocks", 0)) for r in data]
    latencies = [r.get("avg_latency_us", r.get("avg_us", 0)) for r in data]
    p99s = [r.get("p99_latency_us", r.get("p99_us", 0)) for r in data]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(blocks, latencies, "o-", label="Avg", color="#2c7bb6", linewidth=1.5, markersize=5)
    ax.plot(blocks, p99s, "s--", label="P99", color="#d7191c", linewidth=1.5, markersize=5)
    ax.set_xlabel("Number of KV Blocks")
    ax.set_ylabel("Scheduling Latency (us)")
    ax.set_title("Scheduler Scalability", fontsize=9)
    ax.legend(frameon=False)
    ax.axhline(y=60, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="60us target")

    if len(blocks) > 2 and blocks[0] > 0 and blocks[-1] > 0:
        ratio = latencies[-1] / max(latencies[0], 0.01)
        block_ratio = blocks[-1] / max(blocks[0], 1)
        if ratio > 0 and block_ratio > 0:
            exp = np.log(ratio) / np.log(block_ratio)
            ax.text(0.05, 0.95, f"Scaling exponent: {exp:.2f}",
                    transform=ax.transAxes, fontsize=7, va="top")

    save(fig, "fig10_scalability")


# ======================================================================
# Fig 11: Throughput across budgets (line plot)
# ======================================================================
def fig11_throughput_vs_budget():
    data = load_json("exp_e2e_real.json")
    if not data:
        return

    seq = 4096
    nreq = 8
    budgets = [50, 100, 200, 500]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    for mode, label, marker in [("naive", "FIFO Offload", "s"), ("orchkv", "OrchKvCache", "o")]:
        vals = []
        for b in budgets:
            r = next((r for r in data if r["mode"] == mode and r["gpu_budget_mb"] == b
                       and r["seq_len"] == seq and r["n_requests"] == nreq), None)
            vals.append(r["avg_throughput_tok_s"] if r else 0)
        ax.plot(budgets, vals, f"{marker}-", label=label, color=COLORS[mode], linewidth=1.5, markersize=5)

    ax.set_xlabel("GPU KV Budget (MB)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"seq={seq}, nreq={nreq}", fontsize=9)
    ax.legend(frameon=False)
    ax.set_xticks(budgets)
    save(fig, "fig11_throughput_vs_budget")


# ======================================================================
# Fig 12: Eviction reduction ratio across configs
# ======================================================================
def fig12_eviction_reduction():
    data = load_json("exp_e2e_real.json")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    seq_configs = [(2048, "#2c7bb6", "o"), (4096, "#1a9641", "s")]
    budget = 50

    for seq, color, marker in seq_configs:
        nreqs = [1, 4, 8, 16]
        ratios = []
        for nr in nreqs:
            naive_r = next((r for r in data if r["mode"] == "naive" and r["gpu_budget_mb"] == budget
                             and r["seq_len"] == seq and r["n_requests"] == nr), None)
            orch_r = next((r for r in data if r["mode"] == "orchkv" and r["gpu_budget_mb"] == budget
                            and r["seq_len"] == seq and r["n_requests"] == nr), None)
            if naive_r and orch_r and orch_r["total_evictions"] > 0:
                ratios.append(naive_r["total_evictions"] / orch_r["total_evictions"])
            else:
                ratios.append(0)
        ax.plot(nreqs, ratios, f"{marker}-", label=f"seq={seq}", color=color, linewidth=1.5, markersize=5)

    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Eviction Reduction (Naive / OrchKv)")
    ax.set_title(f"Eviction Efficiency, budget={budget}MB", fontsize=9)
    ax.legend(frameon=False)
    ax.set_xticks([1, 4, 8, 16])
    save(fig, "fig12_eviction_reduction")


# ======================================================================
# Main
# ======================================================================
def main():
    print("Generating paper figures...")
    print()

    generators = [
        ("Fig 1: E2E Throughput Bar", fig1_throughput),
        ("Fig 2: Eviction Count", fig2_eviction),
        ("Fig 3: TPOT Latency", fig3_tpot),
        ("Fig 4: Speedup Heatmap", fig4_speedup_heatmap),
        ("Fig 5: Ablation", fig5_ablation),
        ("Fig 6: Quality Table", fig6_quality),
        ("Fig 7: Policy Heatmap (E5)", fig7_policy),
        ("Fig 8: Prefetch (E7)", fig8_prefetch),
        ("Fig 9: Bandwidth (E8)", fig9_bandwidth),
        ("Fig 10: Scalability (E9)", fig10_scalability),
        ("Fig 11: Throughput vs Budget", fig11_throughput_vs_budget),
        ("Fig 12: Eviction Reduction", fig12_eviction_reduction),
    ]

    for name, func in generators:
        print(f"[{name}]")
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll figures saved to {FIGDIR}/")
    print(f"Files: {sorted(os.listdir(FIGDIR))}")


if __name__ == "__main__":
    main()

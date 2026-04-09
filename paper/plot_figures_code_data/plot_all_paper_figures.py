#!/usr/bin/env python3
"""
Generate ALL figures for the OrchKvCache paper.

Usage:
    cd paper/plot_figures_code_data
    python plot_all_paper_figures.py

Output: all PDFs and PNGs in ./out_figures_1/

Data files (JSON) should be in the same directory as this script.

Figure list:
  Fig 1:  fig1_throughput_4models      — 4-model E2E throughput bar
  Fig 2:  fig2_eviction_comparison     — eviction count + reduction ratio
  Fig 3:  fig3_tpot_stability          — TPOT vs request count
  Fig 4:  fig4_speedup_heatmap         — budget × nreq heatmap
  Fig 5:  fig5_speedup_per_model       — per-model speedup bar
  Fig 6:  fig6_quality_all_models      — quality verification table
  Tab LM: fig_lm_eval_accuracy         — LM-Eval downstream task accuracy
  Fig 7:  fig7_ablation_4models        — ablation throughput + eviction
  Fig 8:  fig8_policy_heatmap          — classifier α,β,γ accuracy
  Fig 9:  fig9_prefetch                — prefetch dispatch + overhead
  Fig 10: fig10_scalability            — scheduling latency vs blocks
  Fig 11: fig11_bandwidth              — inter-tier bandwidth
  Fig 12: fig12_hyperparam             — hyperparameter sensitivity (E2E)
  Fig 13: fig13_realistic_throughput   — realistic workload throughput
  Fig 14: fig14_realistic_eviction     — realistic workload eviction
  Fig 15: fig15_ssd_tier               — SSD 3-tier validation
  Fig 16: fig16_scale_context          — 8K context scaling
  Fig 17: fig17_selective_restore      — EMA vs Random coverage
  Fig 18: fig18_infinigen_throughput   — InfiniGen FlexGen comparison
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path(__file__).parent
OUT = DATA / "out_figures_1"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    'gpu_only': '#A8D5BA',
    'fifo': '#E8B4A8',
    'orchkv': '#9CC0D8',
    'gray': '#C8C8C8',
}
EC = {
    'gpu_only': '#6BA882',
    'fifo': '#C07868',
    'orchkv': '#5A90B0',
    'gray': '#909090',
}
BAR_KW = dict(linewidth=0.7)


def save(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT / f"{name}.{fmt}", format=fmt)
    print(f"  {name}")
    plt.close(fig)


def load(fname):
    with open(DATA / fname) as f:
        return json.load(f)


# ======================================================================
# Fig 1: 4-model E2E throughput (budget=50, seq=2048, nreq=4)
# ======================================================================
def fig1_throughput_4models():
    data = load("multimodel_e2e.json")
    models = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]
    modes = ["baseline", "naive", "orchkv"]
    labels = ["GPU-Only", "FIFO", "OrchKvCache"]
    ckeys = ['gpu_only', 'fifo', 'orchkv']

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(models))
    w = 0.25

    for i, (mode, label, ck) in enumerate(zip(modes, labels, ckeys)):
        vals = []
        for m in models:
            r = next((r for r in data if r["model"] == m and r.get("seq_len") == 2048
                      and r.get("num_requests") == 4 and r.get("budget_mb") == 50
                      and r["mode"] == mode), None)
            vals.append(r["avg_throughput"] if r else 0)
        ax.bar(x + i * w, vals, w, label=label, color=COLORS[ck], edgecolor=EC[ck], **BAR_KW)

    ax.set_xticks(x + w)
    ax.set_xticklabels(models)
    ax.set_ylabel('Throughput (tok/s)')
    ax.legend()
    save(fig, "fig1_throughput_4models")


# ======================================================================
# Fig 2: Eviction comparison
# ======================================================================
def fig2_eviction_comparison():
    data = load("multimodel_e2e.json")
    models = ["Qwen2.5-7B", "Mistral-7B", "LLaMA-2-7B", "LLaMA-2-13B"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # (a) Qwen eviction counts
    qwen_data = [r for r in data if r["model"] == "Qwen2.5-7B" and r.get("seq_len") == 2048 and r.get("budget_mb") == 50]
    nreqs = sorted(set(r["num_requests"] for r in qwen_data))
    for mode, label, color, ek in [
        ("naive", "FIFO", COLORS['fifo'], 'fifo'),
        ("orchkv", "OrchKvCache", COLORS['orchkv'], 'orchkv'),
    ]:
        vals = []
        for nr in nreqs:
            r = next((r for r in qwen_data if r["num_requests"] == nr and r["mode"] == mode), None)
            vals.append(max(r.get("total_evictions", 1), 1) if r else 1)
        ax1.bar([str(n) for n in nreqs], vals, label=label, color=color, alpha=0.8,
                edgecolor=EC[ek], **BAR_KW)
    ax1.set_yscale('log')
    ax1.set_xlabel('Request Count')
    ax1.set_ylabel('Total Evictions (log)')
    ax1.legend(fontsize=7)

    # (b) Reduction ratio across models
    ratios = []
    for m in models:
        naive_evicts = [r.get("total_evictions", 0) for r in data if r["model"] == m and r["mode"] == "naive" and r.get("seq_len") == 2048 and r.get("budget_mb") == 50]
        orch_evicts = [r.get("total_evictions", 0) for r in data if r["model"] == m and r["mode"] == "orchkv" and r.get("seq_len") == 2048 and r.get("budget_mb") == 50]
        if orch_evicts and naive_evicts and sum(orch_evicts) > 0:
            ratios.append(sum(naive_evicts) / sum(orch_evicts))
        else:
            ratios.append(0)
    ax2.bar(models, ratios, color=COLORS['orchkv'], edgecolor=EC['orchkv'], **BAR_KW)
    ax2.set_ylabel('Eviction Reduction (×)')
    for i, v in enumerate(ratios):
        if v > 0:
            ax2.text(i, v + 10, f'{v:.0f}×', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save(fig, "fig2_eviction_comparison")


# ======================================================================
# Fig 12: Hyperparameter E2E (lambda sweep)
# ======================================================================
def fig12_hyperparam():
    data = load("exp_p2p3_extended.json")
    hp = data["p3_hyperparam_e2e"]

    lambdas = [r["ema_lambda"] for r in hp if "tok_s" in r]
    tok_s = [r["tok_s"] for r in hp if "tok_s" in r]
    evictions = [r["evictions"] for r in hp if "tok_s" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax1.bar(range(len(lambdas)), tok_s, color=COLORS['orchkv'], edgecolor=EC['orchkv'], width=0.6, **BAR_KW)
    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels([f'{l:.2f}' for l in lambdas])
    ax1.set_xlabel(r'EMA decay $\lambda$')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_ylim(min(tok_s) * 0.85, max(tok_s) * 1.1)
    ax1.axhline(y=np.mean(tok_s), color='red', linestyle='--', linewidth=0.8, label=f'mean={np.mean(tok_s):.1f}')
    ax1.legend()

    ax2.bar(range(len(lambdas)), evictions, color='#E8C8A0', edgecolor='#B89870', width=0.6, **BAR_KW)
    ax2.set_xticks(range(len(lambdas)))
    ax2.set_xticklabels([f'{l:.2f}' for l in lambdas])
    ax2.set_xlabel(r'EMA decay $\lambda$')
    ax2.set_ylabel('Eviction Count')

    plt.tight_layout()
    save(fig, "fig12_hyperparam_sensitivity")


# ======================================================================
# Fig 13-14: Realistic workload throughput + eviction
# ======================================================================
def fig13_realistic_throughput():
    data = load("realistic_workload.json")
    models = ["Qwen2.5-7B", "LLaMA-2-7B"]
    workloads = ["sharegpt-like", "longcontext-mix"]
    mode_colors = {"baseline": COLORS['orchkv'], "naive": COLORS['fifo'], "orchkv": COLORS['gpu_only']}
    mode_ec = {"baseline": EC['orchkv'], "naive": EC['fifo'], "orchkv": EC['gpu_only']}
    mode_labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for wi, wl in enumerate(workloads):
        ax = axes[wi]
        x = np.arange(len(models))
        w = 0.25
        for mi, mode in enumerate(["baseline", "naive", "orchkv"]):
            vals = []
            for m in models:
                r = next((r for r in data if r["model"] == m and r["workload"] == wl and r["mode"] == mode), None)
                vals.append(r["avg_throughput"] if r else 0)
            ax.bar(x + mi * w, vals, w, label=mode_labels[mode], color=mode_colors[mode],
                   edgecolor=mode_ec[mode], **BAR_KW)
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(x + w)
        ax.set_xticklabels(models, fontsize=8)
        if wi == 0:
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.22), fontsize=7)
    plt.tight_layout()
    save(fig, "fig13_realistic_throughput")


def fig14_realistic_eviction():
    data = load("realistic_workload.json")
    models = ["Qwen2.5-7B", "LLaMA-2-7B"]
    workloads = ["sharegpt-like", "longcontext-mix"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for wi, wl in enumerate(workloads):
        ax = axes[wi]
        x = np.arange(len(models))
        w = 0.35
        for mi, (mode, label, ck) in enumerate([("naive", "FIFO", "fifo"), ("orchkv", "OrchKvCache", "orchkv")]):
            vals = []
            for m in models:
                r = next((r for r in data if r["model"] == m and r["workload"] == wl and r["mode"] == mode), None)
                vals.append(max(r["total_evictions"], 1) if r else 1)
            ax.bar(x + mi * w, vals, w, label=label, color=COLORS[ck], edgecolor=EC[ck], **BAR_KW)
        ax.set_yscale("log")
        ax.set_ylabel("Total Evictions (log)")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(models, fontsize=8)
        ax.legend(frameon=False, fontsize=7)

        for mi_m, m in enumerate(models):
            naive_r = next((r for r in data if r["model"] == m and r["workload"] == wl and r["mode"] == "naive"), None)
            orch_r = next((r for r in data if r["model"] == m and r["workload"] == wl and r["mode"] == "orchkv"), None)
            if naive_r and orch_r and orch_r["total_evictions"] > 0:
                ratio = naive_r["total_evictions"] / orch_r["total_evictions"]
                ax.annotate(f"{ratio:.0f}×", xy=(x[mi_m] + w, orch_r["total_evictions"] * 2.5),
                           fontsize=8, ha="center", va="bottom", color=EC['gpu_only'], fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    plt.tight_layout()
    save(fig, "fig14_realistic_eviction")


# ======================================================================
# Fig 16: 8K Scale — context vs throughput
# ======================================================================
def fig16_scale_context():
    data = load("exp_scale_flashattn.json")
    models = {}
    for r in data:
        m = r["model"]
        if m not in models:
            models[m] = {"ctx": [], "fifo": [], "orchkv": []}
        models[m]["ctx"].append(r["seq_len"])
        models[m]["fifo"].append(r.get("fifo_tok_s", 0) if isinstance(r.get("fifo_tok_s"), (int, float)) else 0)
        models[m]["orchkv"].append(r.get("orchkv_tok_s", 0) if isinstance(r.get("orchkv_tok_s"), (int, float)) else 0)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for idx, (model, d) in enumerate(models.items()):
        ax = axes[idx]
        ctx_labels = [f'{c//1024}K' for c in d["ctx"]]
        ax.plot(range(len(d["ctx"])), d["fifo"], marker='^', color=EC['fifo'], label='FIFO', linewidth=1.5, markersize=6)
        ax.plot(range(len(d["ctx"])), d["orchkv"], marker='o', color=EC['orchkv'], label='OrchKvCache', linewidth=1.5, markersize=6)
        for i in range(len(d["ctx"])):
            if d["fifo"][i] > 0 and d["orchkv"][i] > 0:
                ratio = d["orchkv"][i] / d["fifo"][i]
                ax.annotate(f'{ratio:.2f}×', xy=(i, d["orchkv"][i]), xytext=(0, 8), textcoords='offset points',
                           fontsize=8, ha='center', color=EC['orchkv'], fontweight='bold')
        ax.set_xticks(range(len(d["ctx"])))
        ax.set_xticklabels(ctx_labels)
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Throughput (tok/s)')
        ax.legend(loc='upper left')
    plt.tight_layout()
    save(fig, "fig16_scale_context")


# ======================================================================
# Fig 17: Selective Restore — EMA vs Random
# ======================================================================
def fig17_selective_restore():
    data = load("exp_selective_restore.json")
    pcts = [1, 3, 5, 10, 20, 30, 50, 70, 90]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for idx, r in enumerate(data):
        ax = axes[idx]
        ema_vals = [r["avg_ema_coverage"].get(f"top{p}pct", 0) for p in pcts]
        rand_vals = [r["avg_random_coverage"].get(f"top{p}pct", 0) for p in pcts]
        x = np.arange(len(pcts))
        w = 0.35
        ax.bar(x - w/2, ema_vals, w, label='EMA (ours)', color=COLORS['orchkv'], edgecolor=EC['orchkv'], **BAR_KW)
        ax.bar(x + w/2, rand_vals, w, label='Random', color='#E8C8A0', edgecolor='#B89870', alpha=0.7, **BAR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}%' for p in pcts], rotation=45, ha='right')
        ax.set_xlabel('Top-K% blocks restored')
        ax.set_ylabel('Attention coverage (%)')
        ax.set_ylim(0, 115)
        ax.axhline(y=100, color=EC['gray'], linestyle=':', linewidth=0.5)
        ax.legend(loc='lower right')
    plt.tight_layout()
    save(fig, "fig17_selective_restore")


# ======================================================================
# Fig 18: InfiniGen throughput
# ======================================================================
def fig18_infinigen_throughput():
    data = load("exp_infinigen_throughput.json")

    fig, ax = plt.subplots(figsize=(5, 2.8))
    prompts, original, infinigen, h2o = [], [], [], []
    for cfg in data:
        for r in cfg["results"]:
            if r["scheme"] == "FlexGen Original":
                prompts.append(cfg["config"].split("prompt=")[1].split(",")[0])
                original.append(r["gen_throughput_tok_s"])
            elif r["scheme"] == "InfiniGen":
                infinigen.append(r["gen_throughput_tok_s"])
            elif "H2O" in r["scheme"]:
                h2o.append(r["gen_throughput_tok_s"])

    x = np.arange(len(prompts))
    w = 0.25
    ax.bar(x - w, original, w, label='FlexGen Original', color=COLORS['gray'], edgecolor=EC['gray'], **BAR_KW)
    ax.bar(x, infinigen, w, label='InfiniGen', color=COLORS['gpu_only'], edgecolor=EC['gpu_only'], **BAR_KW)
    ax.bar(x + w, h2o, w, label='H2O (lossy)', color=COLORS['fifo'], edgecolor=EC['fifo'], alpha=0.7, **BAR_KW)
    for i in range(len(prompts)):
        if original[i] > 0:
            ax.annotate(f'{infinigen[i]/original[i]:.1f}×', xy=(x[i], infinigen[i]),
                       xytext=(0, 5), textcoords='offset points', fontsize=7.5, ha='center', color=EC['gpu_only'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'p={p}' for p in prompts])
    ax.set_xlabel('Prompt Length')
    ax.set_ylabel('Throughput (tok/s)')
    ax.legend(fontsize=7.5)
    plt.tight_layout()
    save(fig, "fig18_infinigen_throughput")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("Generating paper figures from data...")
    print(f"  Data dir:   {DATA}")
    print(f"  Output dir: {OUT}")
    print()

    fig1_throughput_4models()
    fig2_eviction_comparison()
    fig12_hyperparam()
    fig13_realistic_throughput()
    fig14_realistic_eviction()
    fig16_scale_context()
    fig17_selective_restore()
    fig18_infinigen_throughput()

    print(f"\nDone! {len(list(OUT.glob('*.pdf')))} PDFs generated in {OUT}")
    print("\nNote: Figures 3-11, 15 (TPOT, heatmap, ablation, quality,")
    print("policy, prefetch, scalability, bandwidth, SSD tier) use")
    print("the original plot scripts in benchmarks/. Their data is")
    print("also included here for reference.")

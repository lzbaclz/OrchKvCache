#!/usr/bin/env python3
"""
Generate all missing/new figures for orchkvcache6:
  P0: fig17_hyperparam_sensitivity.pdf (from P3 E2E data)
  P1: fig18_scale_context.pdf (8K scale experiment)
  P1: fig19_selective_restore.pdf (EMA vs Random coverage)
  P2: fig20_infinigen_throughput.pdf (InfiniGen comparison)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


def plot_p0_hyperparam():
    """P0: Hyperparameter E2E sensitivity (lambda sweep)."""
    with open(RESULTS / "exp_p2p3_extended.json") as f:
        data = json.load(f)
    hp = data["p3_hyperparam_e2e"]

    lambdas = [r["ema_lambda"] for r in hp if "tok_s" in r]
    tok_s = [r["tok_s"] for r in hp if "tok_s" in r]
    evictions = [r["evictions"] for r in hp if "tok_s" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax1.bar(range(len(lambdas)), tok_s, color='#4C72B0', width=0.6)
    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels([f'{l:.2f}' for l in lambdas])
    ax1.set_xlabel(r'EMA decay $\lambda$')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_title('(a) E2E Throughput vs λ')
    ymin = min(tok_s) * 0.85
    ymax = max(tok_s) * 1.1
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(y=np.mean(tok_s), color='red', linestyle='--', linewidth=0.8, label=f'mean={np.mean(tok_s):.1f}')
    ax1.legend()

    ax2.bar(range(len(lambdas)), evictions, color='#DD8452', width=0.6)
    ax2.set_xticks(range(len(lambdas)))
    ax2.set_xticklabels([f'{l:.2f}' for l in lambdas])
    ax2.set_xlabel(r'EMA decay $\lambda$')
    ax2.set_ylabel('Eviction Count')
    ax2.set_title('(b) Evictions vs λ')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES / f"fig17_hyperparam_sensitivity.{ext}")
    plt.close()
    print(f"  P0: {FIGURES}/fig17_hyperparam_sensitivity.pdf")


def plot_p1_scale():
    """P1: 8K Scale — context length vs throughput."""
    with open(RESULTS / "exp_scale_flashattn.json") as f:
        data = json.load(f)

    models = {}
    for r in data:
        m = r["model"]
        if m not in models:
            models[m] = {"ctx": [], "gpu": [], "fifo": [], "orchkv": []}
        models[m]["ctx"].append(r["seq_len"])
        models[m]["gpu"].append(r.get("gpu_only_tok_s", 0))
        models[m]["fifo"].append(r.get("fifo_tok_s", 0))
        orch = r.get("orchkv_tok_s", 0)
        models[m]["orchkv"].append(orch if isinstance(orch, (int, float)) else 0)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

    colors = {'GPU-Only': '#2ca02c', 'FIFO': '#d62728', 'OrchKvCache': '#1f77b4'}
    markers = {'GPU-Only': 's', 'FIFO': '^', 'OrchKvCache': 'o'}

    for idx, (model, d) in enumerate(models.items()):
        ax = axes[idx]
        ctx = d["ctx"]
        ctx_labels = [f'{c//1024}K' for c in ctx]

        ax.plot(range(len(ctx)), d["fifo"], marker=markers['FIFO'], color=colors['FIFO'],
                label='FIFO', linewidth=1.5, markersize=6)
        ax.plot(range(len(ctx)), d["orchkv"], marker=markers['OrchKvCache'], color=colors['OrchKvCache'],
                label='OrchKvCache', linewidth=1.5, markersize=6)

        for i in range(len(ctx)):
            if d["fifo"][i] > 0 and d["orchkv"][i] > 0:
                ratio = d["orchkv"][i] / d["fifo"][i]
                ax.annotate(f'{ratio:.2f}×', xy=(i, d["orchkv"][i]),
                           xytext=(0, 8), textcoords='offset points',
                           fontsize=7.5, ha='center', color=colors['OrchKvCache'], fontweight='bold')

        ax.set_xticks(range(len(ctx)))
        ax.set_xticklabels(ctx_labels)
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Throughput (tok/s)')
        ax.set_title(f'({chr(97+idx)}) {model}')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f"fig18_scale_context.{ext}")
    plt.close()
    print(f"  P1: {OUT_DIR}/fig18_scale_context.pdf")


def plot_p1_selective():
    """P1: Selective Restore — EMA vs Random coverage."""
    with open(RESULTS / "exp_selective_restore.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    pcts = [1, 3, 5, 10, 20, 30, 50, 70, 90]
    pct_labels = [f'{p}%' for p in pcts]

    for idx, r in enumerate(data):
        ax = axes[idx]
        ema_vals = [r["avg_ema_coverage"].get(f"top{p}pct", 0) for p in pcts]
        rand_vals = [r["avg_random_coverage"].get(f"top{p}pct", 0) for p in pcts]

        x = np.arange(len(pcts))
        w = 0.35
        bars1 = ax.bar(x - w/2, ema_vals, w, label='EMA (ours)', color='#1f77b4')
        bars2 = ax.bar(x + w/2, rand_vals, w, label='Random', color='#ff7f0e', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(pct_labels, rotation=45, ha='right')
        ax.set_xlabel('Top-K% blocks restored')
        ax.set_ylabel('Attention weight captured (%)')
        ax.set_title(f'({chr(97+idx)}) {r["model"]}')
        ax.set_ylim(0, 115)
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.5)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f"fig19_selective_restore.{ext}")
    plt.close()
    print(f"  P1: {OUT_DIR}/fig19_selective_restore.pdf")


def plot_p2_infinigen():
    """P2: InfiniGen throughput comparison."""
    with open(RESULTS / "exp_infinigen_throughput.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.8))

    prompts = []
    original = []
    infinigen = []
    h2o = []

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

    ax.bar(x - w, original, w, label='FlexGen Original', color='#7f7f7f')
    ax.bar(x, infinigen, w, label='InfiniGen', color='#2ca02c')
    ax.bar(x + w, h2o, w, label='H2O (lossy)', color='#d62728', alpha=0.7)

    for i in range(len(prompts)):
        if original[i] > 0:
            speedup_ig = infinigen[i] / original[i]
            ax.annotate(f'{speedup_ig:.1f}×', xy=(x[i], infinigen[i]),
                       xytext=(0, 5), textcoords='offset points',
                       fontsize=7, ha='center', color='#2ca02c', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'p={p}' for p in prompts])
    ax.set_xlabel('Prompt Length')
    ax.set_ylabel('Generation Throughput (tok/s)')
    ax.set_title('OPT-1.3B (FlexGen offloading, bs=4, gen=128)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f"fig20_infinigen_throughput.{ext}")
    plt.close()
    print(f"  P2: {OUT_DIR}/fig20_infinigen_throughput.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    plot_p0_hyperparam()
    plot_p1_scale()
    plot_p1_selective()
    plot_p2_infinigen()
    print("Done!")

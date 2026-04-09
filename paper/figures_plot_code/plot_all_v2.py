#!/usr/bin/env python3
"""Generate ALL paper figures from JSON data — unified IMPRESS style.

Output: out_figures_code_2/  (PDF + PNG for every figure)
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_code_2"

# ---------------------------------------------------------------------------
# Muted IMPRESS-style palette
# ---------------------------------------------------------------------------
C_GPU = "#A8D5BA";  EC_GPU = "#6BA882"
C_FIFO = "#E8B4A8"; EC_FIFO = "#C07868"
C_ORKV = "#9CC0D8"; EC_ORKV = "#5A90B0"
C_GRAY = "#C8C8C8"; EC_GRAY = "#909090"
C_ORANGE = "#E8C8A0"; EC_ORANGE = "#C09060"
C_PURPLE = "#C8B8D8"; EC_PURPLE = "#9080A8"

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(name: str):
    with open(BASE / name) as f:
        return json.load(f)


def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem}")


def _top_legend(ax, ncol=3):
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, frameon=False, fontsize=9, columnspacing=1.5,
    )


def _short(m: str) -> str:
    return (m.replace("Qwen2.5-", "Qwen ").replace("Llama-2-7b-hf", "LLaMA-2-7B")
             .replace("LLaMA-2-", "LLaMA ").replace("meta-llama/", "")
             .replace("-7B", " 7B").replace("-13B", " 13B").strip())


# ===================================================================
# fig01 — multimodel throughput bar
# ===================================================================
def fig01():
    data = _load("multimodel_e2e.json")
    sel = [r for r in data if r["seq_len"] == 2048 and r["n_requests"] == 8 and r["gpu_budget_mb"] == 50]
    models = sorted({r["model"] for r in sel})
    modes = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    fills = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    edges = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(models)); w = 0.25
    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            v = next((r["avg_throughput"] for r in sel if r["model"] == m and r["mode"] == mode), 0)
            vals.append(v)
        ax.bar(x + (i - 1) * w, vals, w, label=labels[mode],
               color=fills[mode], edgecolor=edges[mode], **BAR_KW)
    ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("Throughput (tok/s)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig01_multimodel_throughput")


# ===================================================================
# fig02 — multimodel eviction comparison
# ===================================================================
def fig02():
    data = _load("multimodel_e2e.json")
    sel = [r for r in data if r["seq_len"] == 2048 and r["n_requests"] == 8 and r["gpu_budget_mb"] == 50]
    models = sorted({r["model"] for r in sel})
    modes = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    fills = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    edges = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(models)); w = 0.25
    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            v = next((r["total_evictions"] for r in sel if r["model"] == m and r["mode"] == mode), 0)
            vals.append(v)
        ax.bar(x + (i - 1) * w, vals, w, label=labels[mode],
               color=fills[mode], edgecolor=edges[mode], **BAR_KW)
    ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("Total evictions")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig02_multimodel_eviction")


# ===================================================================
# fig03 — TPOT stability (Mistral, varying n_requests)
# ===================================================================
def fig03():
    data = _load("multimodel_e2e.json")
    m0 = "Mistral-7B"
    sub = [r for r in data if r["model"] == m0 and r["seq_len"] == 2048 and r["gpu_budget_mb"] == 50]
    reqs = sorted({r["n_requests"] for r in sub})
    modes = ["baseline", "naive", "orchkv"]
    labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}
    colors = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for mode in modes:
        xs, ys = [], []
        for nr in reqs:
            r = next((r for r in sub if r["mode"] == mode and r["n_requests"] == nr), None)
            if r:
                xs.append(nr); ys.append(r["avg_tpot_ms"])
        ax.plot(xs, ys, marker="o", markersize=5, label=labels[mode], color=colors[mode])
    ax.set_xlabel("Concurrent requests"); ax.set_ylabel("Avg TPOT (ms)")
    _top_legend(ax); ax.grid(alpha=0.3)
    _save(fig, "fig03_tpot_stability")


# ===================================================================
# fig04 — speedup heatmap (orchkv / naive)
# ===================================================================
def fig04():
    data = _load("multimodel_e2e.json")
    seq_lens = sorted({r["seq_len"] for r in data})[:4]
    models = sorted({r["model"] for r in data})
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

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(len(seq_lens))); ax.set_xticklabels([f"{s//1024}K" for s in seq_lens])
    ax.set_yticks(range(len(models))); ax.set_yticklabels([_short(m) for m in models])
    ax.set_xlabel("Context length"); ax.set_ylabel("Model")
    plt.colorbar(im, ax=ax, label="Speedup")
    _save(fig, "fig04_speedup_heatmap")


# ===================================================================
# fig05 — per-model speedup bar
# ===================================================================
def fig05():
    data = _load("multimodel_e2e.json")
    models = sorted({r["model"] for r in data})
    speedups = []
    for m in models:
        orch = next((r["avg_throughput"] for r in data
                     if r["model"] == m and r["seq_len"] == 2048 and r["n_requests"] == 8
                     and r["gpu_budget_mb"] == 50 and r["mode"] == "orchkv"), 0)
        naive = next((r["avg_throughput"] for r in data
                      if r["model"] == m and r["seq_len"] == 2048 and r["n_requests"] == 8
                      and r["gpu_budget_mb"] == 50 and r["mode"] == "naive"), 1)
        speedups.append(orch / naive if naive > 0 else 0)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(models)), speedups, color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("OrchKv / FIFO throughput")
    ax.axhline(1.0, color=EC_GRAY, linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig05_per_model_speedup")


# ===================================================================
# fig06 — quality verification table
# ===================================================================
def fig06():
    rows = _load("multimodel_quality.json")
    headers = ["Model", "Prompt", "Prompt len", "Match %", "Evictions"]
    cells = [[r["model"], r["prompt"], str(r["prompt_len"]),
              f'{r["match_rate"]:.0f}', str(r["evictions"])] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.35 * max(len(cells), 1)))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.4)
    _save(fig, "fig06_quality_table")


# ===================================================================
# fig07 — ablation throughput + eviction
# ===================================================================
def fig07():
    rows = _load("multimodel_ablation.json")
    models = sorted({r["model"] for r in rows})
    configs = [("gpu-only", "GPU-Only", C_GPU, EC_GPU),
               ("naive-fifo", "FIFO", C_FIFO, EC_FIFO),
               ("orchkv", "OrchKvCache", C_ORKV, EC_ORKV)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    x = np.arange(len(models)); w = 0.25
    for i, (key, label, fc, ec) in enumerate(configs):
        tp = [next((r["throughput"] for r in rows if r["model"] == m and r["config"] == key), 0) for m in models]
        ev = [next((r["evictions"] for r in rows if r["model"] == m and r["config"] == key), 0) for m in models]
        ax1.bar(x + (i-1)*w, tp, w, label=label, color=fc, edgecolor=ec, **BAR_KW)
        ax2.bar(x + (i-1)*w, ev, w, label=label, color=fc, edgecolor=ec, **BAR_KW)
    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models])
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Throughput (tok/s)")
    ax2.set_ylabel("Evictions")
    _top_legend(ax1); _top_legend(ax2)
    plt.tight_layout()
    _save(fig, "fig07_ablation")


# ===================================================================
# fig08 — LM-Eval accuracy
# ===================================================================
def fig08():
    data = _load("exp_lm_eval.json")
    tasks = ["piqa", "rte", "copa", "openbookqa"]
    task_labels = ["PIQA", "RTE", "COPA", "OBQA"]

    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 3.2), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for idx, r in enumerate(data):
        ax = axes[idx]
        xi = np.arange(len(tasks)); w = 0.35
        gpu_vals = [r["gpu_only"].get(t, 0) for t in tasks]
        orch_vals = [r["orchkv"].get(t, 0) for t in tasks]
        ax.bar(xi - w/2, gpu_vals, w, label="GPU-Only", color=C_GPU, edgecolor=EC_GPU, **BAR_KW)
        ax.bar(xi + w/2, orch_vals, w, label="OrchKvCache", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
        for i in range(len(tasks)):
            if gpu_vals[i] == orch_vals[i] and gpu_vals[i] > 0:
                ax.annotate("=", xy=(xi[i], max(gpu_vals[i], orch_vals[i]) + 0.5),
                            ha="center", fontsize=9, fontweight="bold", color="#333")
        ax.set_xticks(xi); ax.set_xticklabels(task_labels, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_ylim(0, 105)
        _top_legend(ax, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig08_lm_eval_accuracy")


# ===================================================================
# fig09 — competitive ratio
# ===================================================================
def fig09():
    data = _load("exp_competitive_ratio.json")
    rows = data["results"]
    blocks = [r["n_blocks"] for r in rows]
    strats = ["OPT", "EMA", "LFU", "LRU", "FIFO"]
    fill_map = {"OPT": C_ORKV, "EMA": C_GPU, "LFU": C_ORANGE, "LRU": C_FIFO, "FIFO": C_PURPLE}
    edge_map = {"OPT": EC_ORKV, "EMA": EC_GPU, "LFU": EC_ORANGE, "LRU": EC_FIFO, "FIFO": EC_PURPLE}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xi = np.arange(len(blocks)); w = 0.15
    for si, s in enumerate(strats):
        vals = [r[f"{s}_evictions"] for r in rows]
        ax.bar(xi + (si - 2) * w, vals, w, label=s,
               color=fill_map[s], edgecolor=edge_map[s], **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("n_blocks"); ax.set_ylabel("Eviction count")
    _top_legend(ax, ncol=5); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig09_competitive_ratio")


# ===================================================================
# fig10 — policy sweep heatmap (alpha x beta, averaged over pattern)
# ===================================================================
def fig10():
    rows = _load("benchmark_e5_policy_sweep.json")
    by_ab = defaultdict(list)
    for r in rows:
        a, b = r.get("alpha"), r.get("beta")
        if a is not None and b is not None:
            by_ab[(float(a), float(b))].append(float(r.get("hot_ratio", 0)))
    alphas = sorted({k[0] for k in by_ab}); betas = sorted({k[1] for k in by_ab})
    mat = np.full((len(alphas), len(betas)), np.nan)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            if (a, b) in by_ab:
                mat[i, j] = np.mean(by_ab[(a, b)])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\alpha$")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Hot ratio")
    _save(fig, "fig10_policy_sweep_heatmap")


# ===================================================================
# fig11 — prefetch dispatch + schedule latency
# ===================================================================
def fig11():
    rows = _load("benchmark_e7_prefetch.json")
    rows = sorted(rows, key=lambda r: r["prefetch_budget"])
    budgets = [r["prefetch_budget"] for r in rows]
    disp = [r["avg_prefetches_dispatched"] for r in rows]
    lat = [r["avg_schedule_us"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    xi = np.arange(len(budgets))
    ax1.bar(xi - 0.2, disp, 0.4, label="Prefetches dispatched",
            color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax1.set_xlabel("Prefetch budget"); ax1.set_ylabel("Dispatches", color=EC_ORKV)
    ax1.set_xticks(xi); ax1.set_xticklabels([str(int(b)) for b in budgets])
    ax1.tick_params(axis="y", labelcolor=EC_ORKV)

    ax2 = ax1.twinx()
    ax2.plot(xi, lat, color=EC_GRAY, marker="o", markersize=5, linewidth=2, label="Schedule latency (us)")
    ax2.set_ylabel("Avg schedule latency (us)", color=EC_GRAY)
    ax2.tick_params(axis="y", labelcolor=EC_GRAY)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2,
               loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=9)
    _save(fig, "fig11_prefetch_dispatch")


# ===================================================================
# fig12 — inter-tier bandwidth
# ===================================================================
def fig12():
    data = _load("benchmark_e8_storage_bw.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    gd = data["gpu_dram"]
    sizes = [r["size_mb"] for r in gd]
    d2h = [r["d2h_gbps"] for r in gd]; h2d = [r["h2d_gbps"] for r in gd]
    xi = np.arange(len(sizes)); w = 0.35
    ax1.bar(xi - w/2, d2h, w, label="D2H", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax1.bar(xi + w/2, h2d, w, label="H2D", color=C_GPU, edgecolor=EC_GPU, **BAR_KW)
    ax1.set_xticks(xi); ax1.set_xticklabels([f"{s:g}" for s in sizes])
    ax1.set_xlabel("Transfer size (MB)"); ax1.set_ylabel("Bandwidth (GB/s)")
    _top_legend(ax1, ncol=2); ax1.grid(axis="y", alpha=0.3)

    ds = data["dram_storage"]
    sizes2 = [r["size_mb"] for r in ds]
    wr = [r["write_gbps"] for r in ds]; rd = [r["read_gbps"] for r in ds]
    xi2 = np.arange(len(sizes2))
    ax2.bar(xi2 - w/2, wr, w, label="Write", color=C_FIFO, edgecolor=EC_FIFO, **BAR_KW)
    ax2.bar(xi2 + w/2, rd, w, label="Read", color=C_ORANGE, edgecolor=EC_ORANGE, **BAR_KW)
    ax2.set_xticks(xi2); ax2.set_xticklabels([f"{s:g}" for s in sizes2])
    ax2.set_xlabel("Transfer size (MB)"); ax2.set_ylabel("Bandwidth (GB/s)")
    _top_legend(ax2, ncol=2); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig12_inter_tier_bandwidth")


# ===================================================================
# fig13 — scheduling latency scalability
# ===================================================================
def fig13():
    rows = _load("benchmark_e9_scalability.json")
    rows = sorted(rows, key=lambda r: r["n_blocks"])
    xs = [r["n_blocks"] for r in rows]
    p50 = [r["p50_schedule_us"] for r in rows]
    p99 = [r["p99_schedule_us"] for r in rows]
    mean = [r["avg_schedule_us"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, mean, marker="o", markersize=5, label="mean", color=EC_ORKV)
    ax.plot(xs, p50, marker="s", markersize=5, label="p50", color=EC_GPU)
    ax.plot(xs, p99, marker="^", markersize=5, label="p99", color=EC_FIFO)
    ax.set_xlabel("Number of blocks"); ax.set_ylabel("Schedule latency (us)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    _top_legend(ax); ax.grid(alpha=0.3)
    _save(fig, "fig13_scheduling_scalability")


# ===================================================================
# fig14 — hyperparameter sensitivity (lambda sweep)
# ===================================================================
def fig14():
    data = _load("exp_p2p3_extended.json")
    hp = data["p3_hyperparam_e2e"]
    lambdas = [r["ema_lambda"] for r in hp if r.get("tok_s") is not None]
    tok_s = [r["tok_s"] for r in hp if r.get("tok_s") is not None]
    evictions = [r["evictions"] for r in hp if r.get("tok_s") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    xi = np.arange(len(lambdas))
    ax1.bar(xi, tok_s, 0.6, color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax1.set_xticks(xi); ax1.set_xticklabels([f"{l:.2f}" for l in lambdas])
    ax1.set_xlabel(r"EMA $\lambda$"); ax1.set_ylabel("Throughput (tok/s)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(xi, evictions, 0.6, color=C_FIFO, edgecolor=EC_FIFO, **BAR_KW)
    ax2.set_xticks(xi); ax2.set_xticklabels([f"{l:.2f}" for l in lambdas])
    ax2.set_xlabel(r"EMA $\lambda$"); ax2.set_ylabel("Evictions")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig14_hyperparam_sensitivity")


# ===================================================================
# fig15 — per-step overhead breakdown (stacked bar)
# ===================================================================
def fig15():
    data = _load("exp_overhead_breakdown.json")
    results = data["results"]
    part_keys = [
        ("forward_avg_ms", "Forward", C_GPU, EC_GPU),
        ("build_past_kv_avg_ms", "Build KV", C_ORKV, EC_ORKV),
        ("append_token_avg_ms", "Append token", C_ORANGE, EC_ORANGE),
        ("report_attn_avg_ms", "Report attn", C_FIFO, EC_FIFO),
        ("step_schedule_avg_ms", "Schedule", C_PURPLE, EC_PURPLE),
    ]
    labels_x = [r["label"].replace("OrchKvCache ", "OrchKv ") for r in results]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    xi = np.arange(len(results))
    bottoms = np.zeros(len(results))
    for key, label, fc, ec in part_keys:
        vals = np.array([float(r.get(key, 0) or 0) for r in results])
        if vals.sum() > 0:
            ax.bar(xi, vals, 0.6, bottom=bottoms, label=label,
                   color=fc, edgecolor=ec, **BAR_KW)
            bottoms += vals
    ax.set_xticks(xi); ax.set_xticklabels(labels_x, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Time per step (ms)")
    _top_legend(ax, ncol=5); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig15_overhead_breakdown")


# ===================================================================
# fig16 — context length scaling (FlashAttention/SDPA)
# ===================================================================
def fig16():
    rows = _load("exp_scale_flashattn.json")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    n = len(by_model)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
    axes = axes[0]
    for ax, (model, mrows) in zip(axes, sorted(by_model.items())):
        mrows = sorted(mrows, key=lambda r: r["seq_len"])
        ctx = [r["seq_len"] for r in mrows]
        labels_c = [f"{c//1024}K" if c >= 1024 else str(c) for c in ctx]
        gpu = [r.get("gpu_only_tok_s", 0) for r in mrows]
        fifo = [r.get("fifo_tok_s", 0) for r in mrows]
        orch = [float(r.get("orchkv_tok_s", 0) or 0) for r in mrows]
        xi = np.arange(len(ctx))
        ax.plot(xi, gpu, marker="s", markersize=5, label="GPU-Only", color=EC_GPU, linewidth=1.5)
        ax.plot(xi, fifo, marker="^", markersize=5, label="FIFO", color=EC_FIFO, linewidth=1.5)
        ax.plot(xi, orch, marker="o", markersize=5, label="OrchKvCache", color=EC_ORKV, linewidth=1.5)
        ax.set_xticks(xi); ax.set_xticklabels(labels_c)
        ax.set_xlabel("Context length"); ax.set_ylabel("Throughput (tok/s)")
        _top_legend(ax)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig16_scale_context")


# ===================================================================
# fig17 — selective restore EMA vs Random
# ===================================================================
def fig17():
    rows = _load("exp_selective_restore.json")
    pcts = [1, 3, 5, 10, 20, 30, 50, 70, 90]
    rows = rows[:2]

    fig, axes = plt.subplots(1, len(rows), figsize=(6.5, 3.2), squeeze=False)
    axes = axes[0]
    for idx, r in enumerate(rows):
        ax = axes[idx]
        ema = r.get("avg_ema_coverage", {})
        rnd = r.get("avg_random_coverage", {})
        ema_vals = [float(ema.get(f"top{p}pct", 0)) for p in pcts]
        rand_vals = [float(rnd.get(f"top{p}pct", 0)) for p in pcts]
        xi = np.arange(len(pcts)); w = 0.35
        ax.bar(xi - w/2, ema_vals, w, label="EMA (ours)", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
        ax.bar(xi + w/2, rand_vals, w, label="Random", color=C_ORANGE, edgecolor=EC_ORANGE, **BAR_KW)
        ax.set_xticks(xi); ax.set_xticklabels([f"{p}%" for p in pcts], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Top-K% restored"); ax.set_ylabel("Attn coverage (%)")
        ax.set_ylim(0, 115); ax.axhline(100, color=EC_GRAY, linestyle=":", linewidth=0.8)
        _top_legend(ax, ncol=2); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig17_selective_restore")


# ===================================================================
# fig18 — InfiniGen throughput comparison
# ===================================================================
def fig18():
    data = _load("exp_infinigen_throughput.json")
    labels, orig, infin, h2o = [], [], [], []
    for cfg in data:
        cfg_s = cfg.get("config", "")
        short = cfg_s
        if "prompt=" in cfg_s:
            try: short = "p=" + cfg_s.split("prompt=")[1].split(",")[0]
            except: pass
        o = ig = h = None
        for res in cfg.get("results", []):
            scheme = res.get("scheme", "")
            tp = float(res.get("gen_throughput_tok_s", 0))
            if "FlexGen Original" in scheme: o = tp
            elif scheme == "InfiniGen": ig = tp
            elif "H2O" in scheme: h = tp
        if o is not None and ig is not None and h is not None:
            labels.append(short); orig.append(o); infin.append(ig); h2o.append(h)

    xi = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.bar(xi - w, orig, w, label="FlexGen Original", color=C_GRAY, edgecolor=EC_GRAY, **BAR_KW)
    ax.bar(xi, infin, w, label="InfiniGen", color=C_GPU, edgecolor=EC_GPU, **BAR_KW)
    ax.bar(xi + w, h2o, w, label="H2O (lossy)", color=C_FIFO, edgecolor=EC_FIFO, **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Gen throughput (tok/s)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig18_infinigen_throughput")


# ===================================================================
# fig19 — realistic workload throughput
# ===================================================================
def fig19():
    rows = _load("realistic_workload.json")
    workloads = sorted({r["workload"] for r in rows})
    models = sorted({r["model"] for r in rows})
    modes = [("baseline", "GPU-Only", C_GPU, EC_GPU),
             ("naive", "FIFO", C_FIFO, EC_FIFO),
             ("orchkv", "OrchKvCache", C_ORKV, EC_ORKV)]
    groups = [(wl, m) for wl in workloads for m in models]
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 3.8))
    for gi, (wl, m) in enumerate(groups):
        center = gi * 1.35
        for mi, (mk, label, fc, ec) in enumerate(modes):
            v = next((r["avg_throughput"] for r in rows
                      if r["workload"] == wl and r["model"] == m and r["mode"] == mk), 0)
            ax.bar(center + (mi - 1) * w, v, w * 0.95,
                   color=fc, edgecolor=ec, label=label if gi == 0 else None, **BAR_KW)
    ax.set_xticks([i * 1.35 for i in range(len(groups))])
    ax.set_xticklabels([f"{_short(m)}\n{wl[:16]}" for wl, m in groups], fontsize=7)
    ax.set_ylabel("Throughput (tok/s)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig19_realistic_throughput")


# ===================================================================
# fig20 — realistic workload eviction
# ===================================================================
def fig20():
    rows = _load("realistic_workload.json")
    workloads = sorted({r["workload"] for r in rows})
    models = sorted({r["model"] for r in rows})
    modes = [("baseline", "GPU-Only", C_GPU, EC_GPU),
             ("naive", "FIFO", C_FIFO, EC_FIFO),
             ("orchkv", "OrchKvCache", C_ORKV, EC_ORKV)]
    groups = [(wl, m) for wl in workloads for m in models]
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 3.8))
    for gi, (wl, m) in enumerate(groups):
        center = gi * 1.35
        for mi, (mk, label, fc, ec) in enumerate(modes):
            v = next((r["total_evictions"] for r in rows
                      if r["workload"] == wl and r["model"] == m and r["mode"] == mk), 0)
            ax.bar(center + (mi - 1) * w, v, w * 0.95,
                   color=fc, edgecolor=ec, label=label if gi == 0 else None, **BAR_KW)
    ax.set_xticks([i * 1.35 for i in range(len(groups))])
    ax.set_xticklabels([f"{_short(m)}\n{wl[:16]}" for wl, m in groups], fontsize=7)
    ax.set_ylabel("Total evictions")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig20_realistic_eviction")


# ===================================================================
# fig21 — three-tier throughput
# ===================================================================
def fig21():
    rows = _load("w4_tier_throughput.json")
    models = sorted({r["model"] for r in rows})
    configs = [("GPU-Only", C_GPU, EC_GPU), ("GPU+DRAM", C_FIFO, EC_FIFO), ("GPU+DRAM+SSD", C_ORKV, EC_ORKV)]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xi = np.arange(len(models)); w = 0.25
    for i, (cfg, fc, ec) in enumerate(configs):
        vals = [next((r["avg_throughput_tok_s"] for r in rows if r["model"] == m and r["config"] == cfg), 0) for m in models]
        ax.bar(xi + (i - 1) * w, vals, w, label=cfg, color=fc, edgecolor=ec, **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("Throughput (tok/s)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig21_three_tier_throughput")


# ===================================================================
# fig22 — sampling interval vs throughput (E2E)
# ===================================================================
def fig22():
    data = _load("exp_e10_sampling_e2e.json")
    results = sorted(data["results"], key=lambda r: r["sample_interval"])
    results = [r for r in results if r["sample_interval"] > 0]
    ivs = [r["sample_interval"] for r in results]
    tps = [r["avg_throughput_tok_s"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(ivs, tps, marker="o", markersize=5, color=EC_ORKV, linewidth=2)
    for a, b in zip(ivs, tps):
        ax.annotate(f"{b:.0f}", (a, b), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("Sampling interval (N)"); ax.set_ylabel("Throughput (tok/s)")
    ax.grid(alpha=0.3)
    _save(fig, "fig22_sampling_e2e_throughput")


# ===================================================================
# fig23 — sampling interval vs classification accuracy (sim)
# ===================================================================
def fig23():
    data = _load("exp_e10_sampling_sim.json")
    summ = data.get("summary", [])
    ivs = [s["sample_interval"] for s in summ]
    acc = [s["gt_accuracy"] for s in summ]
    agree = [s["baseline_agreement"] for s in summ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    xi = np.arange(len(ivs)); w = 0.35
    ax.bar(xi - w/2, acc, w, label="GT accuracy", color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax.bar(xi + w/2, agree, w, label="Baseline agreement", color=C_GRAY, edgecolor=EC_GRAY, **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels([str(v) for v in ivs])
    ax.set_xlabel("Sampling interval"); ax.set_ylabel("Accuracy / agreement")
    ax.set_ylim(0, 1.1)
    _top_legend(ax, ncol=2); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig23_sampling_sim_accuracy")


# ===================================================================
# fig24 — fair baseline landscape
# ===================================================================
def fig24():
    data = _load("exp_fair_baseline.json")
    rows = data["results"]
    metrics = [
        ("gpu_only_eager", "GPU-Only (eager)", C_GPU, EC_GPU),
        ("fast_fifo", "Fast FIFO", C_FIFO, EC_FIFO),
        ("fast_orchkv", "Fast OrchKv", C_ORKV, EC_ORKV),
        ("gpu_only_sdpa", "GPU-Only (SDPA)", C_GRAY, EC_GRAY),
    ]
    models = [r["model"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xi = np.arange(len(models)); w = 0.2
    for i, (key, label, fc, ec) in enumerate(metrics):
        vals = [float(r.get(key, 0)) for r in rows]
        ax.bar(xi + (i - 1.5) * w, vals, w, label=label, color=fc, edgecolor=ec, **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels([_short(m) for m in models])
    ax.set_ylabel("Throughput (tok/s)")
    _top_legend(ax, ncol=4); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig24_fair_baseline")


# ===================================================================
# fig25 — PPL comparison (InfiniGen table)
# ===================================================================
def fig25():
    data = _load("exp_ppl_infinigen_comparison.json")
    table = data.get("infinigen_table2", {})
    models = list(table.keys())
    configs = [
        ("full_cache", "Full cache", C_GPU, EC_GPU),
        ("80pct_fifo", "80% FIFO", C_FIFO, EC_FIFO),
        ("80pct_lru", "80% LRU", C_ORANGE, EC_ORANGE),
        ("80pct_counter", "80% counter", C_ORKV, EC_ORKV),
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xi = np.arange(len(models)); w = 0.2
    for i, (key, label, fc, ec) in enumerate(configs):
        vals = [float(table.get(m, {}).get(key, 0)) for m in models]
        ax.bar(xi + (i - 1.5) * w, vals, w, label=label, color=fc, edgecolor=ec, **BAR_KW)
    ax.set_xticks(xi); ax.set_xticklabels(models)
    ax.set_ylabel("Perplexity")
    _top_legend(ax, ncol=4); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig25_ppl_comparison")


# ===================================================================
# fig26 — vLLM strategies (heatmap-like grouped bar by gpu_util)
# ===================================================================
def fig26():
    rows = _load("exp_vllm_multi_pressure.json")
    gpu_utils = sorted({r["gpu_util"] for r in rows})
    prompts = sorted({r["num_prompts"] for r in rows})
    strats = ["fifo", "progress", "block_score"]
    strat_labels = {"fifo": "FIFO", "progress": "Progress", "block_score": "Block-score"}
    strat_fill = {"fifo": C_FIFO, "progress": C_GPU, "block_score": C_ORKV}
    strat_edge = {"fifo": EC_FIFO, "progress": EC_GPU, "block_score": EC_ORKV}

    groups = [(gu, np_) for gu in gpu_utils for np_ in prompts]
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.9), 3.5))
    w = 0.25
    for gi, (gu, np_) in enumerate(groups):
        center = gi * 1.2
        for si, s in enumerate(strats):
            v = next((r["avg_throughput"] for r in rows
                      if r["gpu_util"] == gu and r["num_prompts"] == np_ and r["strategy"] == s), 0)
            ax.bar(center + (si - 1) * w, v, w * 0.9,
                   color=strat_fill[s], edgecolor=strat_edge[s],
                   label=strat_labels[s] if gi == 0 else None, **BAR_KW)
    ax.set_xticks([i * 1.2 for i in range(len(groups))])
    ax.set_xticklabels([f"u={gu}\nn={np_}" for gu, np_ in groups], fontsize=6)
    ax.set_ylabel("Avg throughput (tok/s)")
    _top_legend(ax); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig26_vllm_strategies")


# ===================================================================
# fig27 — C scheduling overhead (n_blocks vs us/step)
# ===================================================================
def fig27():
    data = _load("exp_scheduling_overhead.json")
    blocks, us = [], []
    for k, v in sorted(data.items()):
        if not k.startswith("c_overhead_") or not isinstance(v, dict):
            continue
        nb = v.get("n_blocks")
        tot = v.get("total_c_per_step_us", {})
        mu = tot.get("mean") if isinstance(tot, dict) else tot
        if nb is not None and mu is not None:
            blocks.append(int(nb)); us.append(float(mu))
    order = np.argsort(blocks)
    blocks = [blocks[i] for i in order]; us = [us[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(blocks, us, marker="o", markersize=5, color=EC_ORKV, linewidth=2)
    ax.set_xlabel("n_blocks"); ax.set_ylabel("us per step (C)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks); ax.set_xticklabels([str(b) for b in blocks])
    ax.grid(alpha=0.3)
    _save(fig, "fig27_c_scheduling_overhead")


# ===================================================================
# fig28 — SSD tier validation table
# ===================================================================
def fig28():
    rows = _load("ssd_tier_e2e.json")
    headers = ["Model", "Prompt", "Match %", "GPU->DRAM", "DRAM->SSD", "SSD->DRAM", "SSD files"]
    cells = [[r["model"], r["prompt"], f'{r["match_rate"]:.0f}',
              str(r["gpu_to_dram"]), str(r["dram_to_ssd"]),
              str(r["ssd_to_dram"]), str(r["ssd_files_created"])] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.35 * max(len(cells), 1)))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.05, 1.35)
    _save(fig, "fig28_ssd_tier_table")


# ===================================================================
# Main
# ===================================================================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT}\n")

    fig01(); fig02(); fig03(); fig04(); fig05()
    fig06(); fig07(); fig08(); fig09(); fig10()
    fig11(); fig12(); fig13(); fig14(); fig15()
    fig16(); fig17(); fig18(); fig19(); fig20()
    fig21(); fig22(); fig23(); fig24(); fig25()
    fig26(); fig27(); fig28()

    n = len(list(OUT.glob("*.pdf")))
    print(f"\nDone — {n} PDFs in {OUT}")


if __name__ == "__main__":
    main()

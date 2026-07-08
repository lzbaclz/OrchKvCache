#!/usr/bin/env python3
"""
Generate all SIGMETRICS 2027 paper figures from experiment results.

Figures:
  Fig 2  — Workload characterization (reuse distance CDF, Gini, Jaccard)
  Fig 3  — Promotion latency breakdown
  Fig 5  — Signal overhead comparison (full attn vs sampling vs QK proxy)
  Fig 6  — Policy simulation comparison
  Table 2 — End-to-end system comparison
  Fig 7  — Goodput under SLO at varying load
  Fig 8  — Memory pressure sweep
  Fig 9  — SSD backend ablation

Style: ACM SIGMETRICS — clean serif, two-column friendly, grayscale-safe.

Usage:
    python -m benchmarks.sigmetrics.plot_figures
    python -m benchmarks.sigmetrics.plot_figures --results_dir path/to/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
FIGDIR = Path(__file__).parent / "figures"

# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

BASELINE_COLORS = {
    "gpu_only":         "#4e79a7",
    "fifo_offload":     "#e15759",
    "orchkv":           "#59a14f",
    "orchkv_sampling":  "#f28e2b",
    "orchkv_qk_proxy":  "#76b7b2",
}
BASELINE_LABELS = {
    "gpu_only":         "GPU-Only",
    "fifo_offload":     "FIFO Offload",
    "orchkv":           "OrchKv (full)",
    "orchkv_sampling":  "OrchKv (N=50)",
    "orchkv_qk_proxy":  "OrchKv (QK proxy)",
}
BASELINE_HATCHES = {
    "gpu_only": "",
    "fifo_offload": "//",
    "orchkv": "",
    "orchkv_sampling": "\\\\",
    "orchkv_qk_proxy": "..",
}
BASELINE_MARKERS = {
    "gpu_only": "D",
    "fifo_offload": "s",
    "orchkv": "o",
    "orchkv_sampling": "^",
    "orchkv_qk_proxy": "v",
}

MODEL_COLORS = {
    "qwen2.5-7b":   "#4e79a7",
    "llama-3.1-8b": "#f28e2b",
    "llama-2-7b":   "#e15759",
    "mistral-7b":   "#76b7b2",
}
MODEL_LABELS = {
    "qwen2.5-7b":   "Qwen2.5-7B",
    "llama-3.1-8b": "LLaMA-3.1-8B",
    "llama-2-7b":   "LLaMA-2-7B",
    "mistral-7b":   "Mistral-7B",
}

WORKLOAD_LABELS = {
    "sharegpt": "ShareGPT",
    "longbench": "LongBench",
    "ruler": "RULER",
    "rag": "RAG",
    "agentic": "Agentic",
}

COL_WIDTH = 3.4    # inches, single-column ACM
FULL_WIDTH = 7.0   # inches, full-width ACM


# ── Helpers ───────────────────────────────────────────────────────────

def save(fig, name: str):
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGDIR / f"{name}.{fmt}", format=fmt)
    print(f"  {name}.pdf + .png")
    plt.close(fig)


def load_json(name: str) -> Any:
    """Try multiple naming conventions for result files."""
    for pattern in [name, f"sigmetrics_{name}", f"sigmetrics_full_{name}"]:
        path = RESULTS_DIR / f"{pattern}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def load_sweep_results() -> list[dict]:
    """Load all sweep result files and merge."""
    all_results = []
    for path in sorted(RESULTS_DIR.glob("sigmetrics_*.json")):
        if "partial" in path.name:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict) and "config" in data:
            all_results.append(data)
    return all_results


def load_measurement_results() -> list[dict]:
    """Load all measurement/characterization result files."""
    results = []
    for path in sorted(RESULTS_DIR.glob("measurement_*.json")):
        results.append(json.loads(path.read_text()))
    return results


def _find(results, **filters):
    """Find result entries matching filters on config fields."""
    matches = []
    for r in results:
        cfg = r.get("config", {})
        if all(cfg.get(k) == v for k, v in filters.items()):
            matches.append(r)
    return matches


def _get_metric(result, metric_path, default=0):
    """Navigate nested metric path like 'throughput.mean'."""
    parts = metric_path.split(".")
    val = result
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p, default)
        else:
            return default
    return val if isinstance(val, (int, float)) else default


# =====================================================================
#  Fig 2: Workload Characterization
# =====================================================================

def fig2_workload_characterization():
    """Reuse distance CDF, Gini coefficient, and Jaccard stability."""
    measurements = load_measurement_results()
    if not measurements:
        print("  SKIP: no measurement data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.3))

    # (a) Reuse distance CDF by workload
    ax = axes[0]
    cdf_keys = ["p25", "p50", "p75", "p90", "p95", "p99"]
    cdf_x = [25, 50, 75, 90, 95, 99]
    workload_reuse: dict[str, list[float]] = defaultdict(list)

    for m in measurements:
        agg = m.get("aggregate", {})
        rd = agg.get("reuse_distance", {})
        model = m.get("model", "?")
        for prompt in m.get("per_prompt", []):
            src = prompt.get("prompt_tokens", 0)  # use as proxy
            prd = prompt.get("reuse_distance", {})
            if prd:
                for mk in measurements:
                    if mk.get("model") == model:
                        wl = _infer_workload(mk)
                        vals = [prd.get(k, 0) for k in cdf_keys]
                        if any(v > 0 for v in vals):
                            workload_reuse[wl].append(vals)
                        break

    if not workload_reuse:
        for m in measurements:
            agg = m.get("aggregate", {})
            rd = agg.get("reuse_distance", {})
            wl = _infer_workload(m)
            vals = [rd.get(k, 0) for k in cdf_keys]
            if any(v > 0 for v in vals):
                ax.plot(cdf_x, vals, "o-", label=WORKLOAD_LABELS.get(wl, wl),
                        linewidth=1.2, markersize=3)
    else:
        for wl, all_vals in workload_reuse.items():
            avg = np.mean(all_vals, axis=0)
            ax.plot(cdf_x, avg, "o-", label=WORKLOAD_LABELS.get(wl, wl),
                    linewidth=1.2, markersize=3)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Reuse Distance (blocks)")
    ax.set_title("(a) Reuse Distance CDF", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)

    # (b) Gini coefficient per model × workload
    ax = axes[1]
    gini_data: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        agg = m.get("aggregate", {})
        gini = agg.get("gini", {}).get("mean", 0)
        if gini > 0:
            model = m.get("model", "?")
            gini_data[MODEL_LABELS.get(model, model)].append(gini)

    if gini_data:
        labels = list(gini_data.keys())
        means = [np.mean(v) for v in gini_data.values()]
        stds = [np.std(v) for v in gini_data.values()]
        x = np.arange(len(labels))
        colors = [MODEL_COLORS.get(k.lower().replace(" ", "-"), "#999")
                  for k in gini_data.keys()]
        ax.bar(x, means, 0.6, yerr=stds, color=colors, edgecolor="white",
               linewidth=0.4, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")

    ax.set_ylabel("Gini Coefficient")
    ax.set_title("(b) Attention Inequality", fontsize=9)
    ax.set_ylim(0, 1)

    # (c) Jaccard stability over decode steps
    ax = axes[2]
    for m in measurements:
        model = m.get("model", "?")
        for pi, prompt in enumerate(m.get("per_prompt", [])[:3]):
            jvals = prompt.get("jaccard_stability", {}).get("values", [])
            if jvals:
                ax.plot(range(len(jvals)), jvals, alpha=0.5, linewidth=0.8,
                        color=MODEL_COLORS.get(model, "#999"),
                        label=MODEL_LABELS.get(model, model) if pi == 0 else None)

    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("(c) Hot-Set Stability", fontsize=9)
    ax.set_ylim(0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(dict(zip(labels, handles)).values(),
                  dict(zip(labels, handles)).keys(),
                  frameon=False, fontsize=6.5)

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_workload_characterization")


def _infer_workload(measurement: dict) -> str:
    """Infer workload name from measurement file metadata."""
    for prompt in measurement.get("per_prompt", []):
        src = prompt.get("metadata", prompt).get("source", "")
        if src:
            return src
    return "unknown"


# =====================================================================
#  Fig 3: Promotion Latency Breakdown
# =====================================================================

def fig3_promotion_latency():
    """Per-baseline promotion latency breakdown."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3))
    baselines_with_promo = ["fifo_offload", "orchkv", "orchkv_sampling", "orchkv_qk_proxy"]

    # (a) P50 / P99 promotion stall latency by baseline
    for bl in baselines_with_promo:
        matches = _find(results, baseline=bl, budget_fraction=0.25)
        if not matches:
            continue
        p50_vals = [_get_metric(r, "tpot.p50") for r in matches if _get_metric(r, "tpot.p50") > 0]
        p99_vals = [_get_metric(r, "tpot.p99") for r in matches if _get_metric(r, "tpot.p99") > 0]
        if p50_vals:
            label = BASELINE_LABELS.get(bl, bl)
            x_pos = baselines_with_promo.index(bl)
            ax1.bar(x_pos - 0.15, np.mean(p50_vals), 0.3, label="P50" if bl == baselines_with_promo[0] else "",
                    color=BASELINE_COLORS.get(bl, "#999"), alpha=0.7, edgecolor="white")
            if p99_vals:
                ax1.bar(x_pos + 0.15, np.mean(p99_vals), 0.3, label="P99" if bl == baselines_with_promo[0] else "",
                        color=BASELINE_COLORS.get(bl, "#999"), alpha=1.0, edgecolor="white")

    ax1.set_xticks(range(len(baselines_with_promo)))
    ax1.set_xticklabels([BASELINE_LABELS.get(b, b) for b in baselines_with_promo],
                         fontsize=6.5, rotation=15, ha="right")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title("(a) Decode Latency by Baseline", fontsize=9)
    ax1.legend(frameon=False, fontsize=7)

    # (b) TTFT breakdown
    for bl in ["gpu_only"] + baselines_with_promo:
        matches = _find(results, baseline=bl, budget_fraction=0.25)
        ttft_vals = [_get_metric(r, "ttft.mean") for r in matches if _get_metric(r, "ttft.mean") > 0]
        if ttft_vals:
            ax2.bar(list(BASELINES).index(bl), np.mean(ttft_vals), 0.6,
                    color=BASELINE_COLORS.get(bl, "#999"), edgecolor="white",
                    linewidth=0.4)

    ax2.set_xticks(range(len(BASELINES)))
    ax2.set_xticklabels([BASELINE_LABELS.get(b, b) for b in BASELINES],
                         fontsize=6.5, rotation=15, ha="right")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title("(b) Time to First Token", fontsize=9)

    fig.tight_layout(w_pad=2)
    save(fig, "fig3_promotion_latency")


# =====================================================================
#  Fig 5: Signal Overhead Comparison
# =====================================================================

def fig5_signal_overhead():
    """Compare throughput overhead of full attention, sparse sampling, and QK proxy."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    signal_baselines = ["gpu_only", "orchkv", "orchkv_sampling", "orchkv_qk_proxy"]
    models = list(MODEL_LABELS.keys())

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
    x = np.arange(len(models))
    w = 0.8 / len(signal_baselines)

    for i, bl in enumerate(signal_baselines):
        vals = []
        for m in models:
            matches = _find(results, model=m, baseline=bl, budget_fraction=0.25)
            thrs = [_get_metric(r, "throughput.mean") for r in matches
                    if _get_metric(r, "throughput.mean") > 0]
            vals.append(np.mean(thrs) if thrs else 0)
        ax.bar(x + i * w, vals, w, label=BASELINE_LABELS.get(bl, bl),
               color=BASELINE_COLORS.get(bl, "#999"),
               hatch=BASELINE_HATCHES.get(bl, ""),
               edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Model")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_xticks(x + w * (len(signal_baselines) - 1) / 2)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=7)
    ax.legend(frameon=False, fontsize=6.5, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_ylim(0)
    ax.set_title("Signal Overhead (budget=25%)", fontsize=9)
    save(fig, "fig5_signal_overhead")


# =====================================================================
#  Fig 6: Policy Simulation Comparison
# =====================================================================

def fig6_policy_comparison():
    """Eviction count and throughput across baselines for each workload."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    workloads = list(WORKLOAD_LABELS.keys())
    core_baselines = ["fifo_offload", "orchkv"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3))

    x = np.arange(len(workloads))
    w = 0.35
    for i, bl in enumerate(core_baselines):
        evict_vals = []
        thr_vals = []
        for wl in workloads:
            matches = _find(results, baseline=bl, workload=wl, budget_fraction=0.25)
            evicts = [_get_metric(r, "evictions.mean") for r in matches
                      if _get_metric(r, "evictions.mean") > 0]
            thrs = [_get_metric(r, "throughput.mean") for r in matches
                    if _get_metric(r, "throughput.mean") > 0]
            evict_vals.append(max(np.mean(evicts), 0.5) if evicts else 0.5)
            thr_vals.append(np.mean(thrs) if thrs else 0)

        ax1.bar(x + i * w, evict_vals, w, label=BASELINE_LABELS[bl],
                color=BASELINE_COLORS[bl], edgecolor="white", linewidth=0.4)
        ax2.bar(x + i * w, thr_vals, w, label=BASELINE_LABELS[bl],
                color=BASELINE_COLORS[bl], edgecolor="white", linewidth=0.4)

    ax1.set_yscale("log")
    ax1.set_ylabel("Evictions (log)")
    ax1.set_xticks(x + w / 2)
    ax1.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads], fontsize=7)
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title("(a) Eviction Count (budget=25%)", fontsize=9)

    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_xticks(x + w / 2)
    ax2.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads], fontsize=7)
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_title("(b) Throughput (budget=25%)", fontsize=9)

    fig.tight_layout(w_pad=2)
    save(fig, "fig6_policy_comparison")


# =====================================================================
#  Table 2: End-to-End System Comparison
# =====================================================================

def table2_e2e_comparison():
    """Render a publication-quality table comparing all baselines."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.5))
    ax.axis("off")

    headers = ["Model", "Baseline", "Thr (tok/s)", "TTFT (ms)",
               "TPOT (ms)", "ITL P99", "Evictions", "Match"]
    rows = []

    models = list(MODEL_LABELS.keys())
    baselines = ["gpu_only", "fifo_offload", "orchkv"]

    for m in models:
        for bl in baselines:
            matches = _find(results, model=m, baseline=bl, budget_fraction=0.25)
            if not matches:
                continue
            r = matches[0]
            thr = _get_metric(r, "throughput.mean")
            ttft = _get_metric(r, "ttft.mean")
            tpot = _get_metric(r, "tpot.mean")
            p99 = _get_metric(r, "itl_p99.mean")
            evict = _get_metric(r, "evictions.mean")
            match = _get_metric(r, "bit_exact_match.mean", -1)
            match_str = f"{match:.2%}" if match >= 0 else "N/A"

            rows.append([
                MODEL_LABELS.get(m, m), BASELINE_LABELS.get(bl, bl),
                f"{thr:.0f}", f"{ttft:.1f}", f"{tpot:.1f}",
                f"{p99:.1f}", f"{evict:.0f}", match_str,
            ])

    if not rows:
        plt.close(fig)
        print("  SKIP: no matching data for Table 2")
        return

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#d4e6f1")
        table[0, j].set_text_props(fontweight="bold", fontsize=7)
    for i in range(1, len(rows) + 1):
        bl_name = rows[i - 1][1]
        if "OrchKv" in bl_name:
            for j in range(len(headers)):
                table[i, j].set_facecolor("#eafaea")

    ax.set_title("Table 2: End-to-End Comparison (budget=25%, all workloads avg)",
                 fontsize=9, pad=15)
    save(fig, "table2_e2e_comparison")


# =====================================================================
#  Fig 7: Goodput Under SLO
# =====================================================================

def fig7_goodput_slo():
    """Goodput under SLO constraints at varying budget pressure."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    budgets = [0.05, 0.10, 0.25, 0.50, 0.75]
    core = ["fifo_offload", "orchkv"]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    for bl in core:
        goodputs = []
        valid_budgets = []
        for bf in budgets:
            matches = _find(results, baseline=bl, budget_fraction=bf)
            gvals = [_get_metric(r, "goodput_under_slo.mean")
                     for r in matches if _get_metric(r, "goodput_under_slo.mean") > 0]
            if gvals:
                goodputs.append(np.mean(gvals))
                valid_budgets.append(bf * 100)
        if goodputs:
            ax.plot(valid_budgets, goodputs,
                    f"{BASELINE_MARKERS.get(bl, 'o')}-",
                    label=BASELINE_LABELS[bl],
                    color=BASELINE_COLORS[bl],
                    linewidth=1.5, markersize=5)

    ax.set_xlabel("GPU KV Budget (%)")
    ax.set_ylabel("Goodput (fraction meeting SLO)")
    ax.set_title("Goodput Under SLO", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    save(fig, "fig7_goodput_slo")


# =====================================================================
#  Fig 8: Memory Pressure Sweep
# =====================================================================

def fig8_memory_pressure():
    """Throughput and eviction rate vs. memory budget fraction."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    budgets = [0.05, 0.10, 0.25, 0.50, 0.75]
    core = ["fifo_offload", "orchkv"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3))

    for bl in core:
        thrs = []
        evicts = []
        valid_budgets = []
        for bf in budgets:
            matches = _find(results, baseline=bl, budget_fraction=bf)
            t = [_get_metric(r, "throughput.mean") for r in matches
                 if _get_metric(r, "throughput.mean") > 0]
            e = [_get_metric(r, "evictions.mean") for r in matches]
            if t:
                thrs.append(np.mean(t))
                evicts.append(max(np.mean(e), 0.1) if e else 0.1)
                valid_budgets.append(bf * 100)

        marker = BASELINE_MARKERS.get(bl, "o")
        color = BASELINE_COLORS[bl]
        label = BASELINE_LABELS[bl]
        if thrs:
            ax1.plot(valid_budgets, thrs, f"{marker}-", label=label,
                     color=color, linewidth=1.5, markersize=5)
            ax2.plot(valid_budgets, evicts, f"{marker}-", label=label,
                     color=color, linewidth=1.5, markersize=5)

    ax1.set_xlabel("GPU KV Budget (%)")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_title("(a) Throughput vs. Budget", fontsize=9)
    ax1.legend(frameon=False, fontsize=7)

    ax2.set_xlabel("GPU KV Budget (%)")
    ax2.set_ylabel("Evictions (log)")
    ax2.set_yscale("log")
    ax2.set_title("(b) Evictions vs. Budget", fontsize=9)
    ax2.legend(frameon=False, fontsize=7)

    fig.tight_layout(w_pad=2)
    save(fig, "fig8_memory_pressure")


# =====================================================================
#  Fig 9: SSD Backend Ablation
# =====================================================================

def fig9_ssd_ablation():
    """Compare performance with and without SSD tier."""
    results = load_sweep_results()
    if not results:
        print("  SKIP: no sweep data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3))
    models = list(MODEL_LABELS.keys())

    # (a) SSD traffic by budget
    budgets = [0.05, 0.10, 0.25]
    x = np.arange(len(budgets))
    w = 0.8 / max(len(models), 1)

    for mi, m in enumerate(models):
        traffic = []
        for bf in budgets:
            matches = _find(results, model=m, baseline="orchkv", budget_fraction=bf)
            ssd_vals = [_get_metric(r, "ssd_traffic_mb", 0) for r in matches]
            traffic.append(np.mean(ssd_vals) if ssd_vals else 0)
        ax1.bar(x + mi * w, traffic, w, label=MODEL_LABELS[m],
                color=MODEL_COLORS[m], edgecolor="white", linewidth=0.4)

    ax1.set_xlabel("GPU KV Budget (%)")
    ax1.set_ylabel("SSD Traffic (MB)")
    ax1.set_xticks(x + w * (len(models) - 1) / 2)
    ax1.set_xticklabels([f"{int(b*100)}%" for b in budgets])
    ax1.legend(frameon=False, fontsize=6.5, ncol=2)
    ax1.set_title("(a) SSD I/O by Model", fontsize=9)

    # (b) DRAM usage vs GPU budget
    for bl in ["fifo_offload", "orchkv"]:
        dram_vals = []
        valid_budgets = []
        for bf in [0.05, 0.10, 0.25, 0.50, 0.75]:
            matches = _find(results, baseline=bl, budget_fraction=bf)
            d = [_get_metric(r, "gpu_mem.mean") for r in matches
                 if _get_metric(r, "gpu_mem.mean") > 0]
            if d:
                dram_vals.append(np.mean(d))
                valid_budgets.append(bf * 100)
        if dram_vals:
            ax2.plot(valid_budgets, dram_vals,
                     f"{BASELINE_MARKERS.get(bl, 'o')}-",
                     label=BASELINE_LABELS[bl],
                     color=BASELINE_COLORS[bl],
                     linewidth=1.5, markersize=5)

    ax2.set_xlabel("GPU KV Budget (%)")
    ax2.set_ylabel("Peak GPU Memory (MB)")
    ax2.set_title("(b) GPU Memory Usage", fontsize=9)
    ax2.legend(frameon=False, fontsize=7)

    fig.tight_layout(w_pad=2)
    save(fig, "fig9_ssd_ablation")


# =====================================================================
#  Main
# =====================================================================

FIGURES = [
    ("Fig 2: Workload Characterization",    fig2_workload_characterization),
    ("Fig 3: Promotion Latency Breakdown",  fig3_promotion_latency),
    ("Fig 5: Signal Overhead Comparison",   fig5_signal_overhead),
    ("Fig 6: Policy Simulation",            fig6_policy_comparison),
    ("Table 2: E2E Comparison",             table2_e2e_comparison),
    ("Fig 7: Goodput Under SLO",            fig7_goodput_slo),
    ("Fig 8: Memory Pressure Sweep",        fig8_memory_pressure),
    ("Fig 9: SSD Backend Ablation",         fig9_ssd_ablation),
]


def main():
    global RESULTS_DIR, FIGDIR

    parser = argparse.ArgumentParser(description="Generate SIGMETRICS paper figures")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output figure directory")
    parser.add_argument("--figures", nargs="+", type=int, default=None,
                        help="Generate specific figures only (e.g., 2 3 7)")
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    if args.output_dir:
        FIGDIR = Path(args.output_dir)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating SIGMETRICS 2027 figures...")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Output:  {FIGDIR}\n")

    for name, func in FIGURES:
        fig_num = name.split(":")[0].split()[-1]
        if args.figures and int(fig_num) not in args.figures:
            continue
        print(f"[{name}]")
        try:
            func()
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nAll figures in {FIGDIR}/")
    pdfs = sorted(f for f in os.listdir(FIGDIR) if f.endswith(".pdf"))
    print(f"Total: {len(pdfs)} PDFs")
    for f in pdfs:
        print(f"  {f}")


if __name__ == "__main__":
    main()

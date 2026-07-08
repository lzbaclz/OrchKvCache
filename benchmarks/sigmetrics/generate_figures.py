#!/usr/bin/env python3
"""Generate all SIGMETRICS paper figures from experiment JSON results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUTDIR  = Path(__file__).resolve().parent.parent.parent / "paper" / "sigmetrics" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

def _load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

def _setup():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.labelsize":    10,
        "axes.titlesize":    10,
        "legend.fontsize":   8,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.color":        "#cccccc",
        "axes.axisbelow":    True,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })

def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  [OK] {name}.pdf / .png")


# ── Fig 1: Workload Characterization ────────────────────────────────
def fig1_workload_characterization():
    qwen  = _load("workload_char_2048.json")
    llama = _load("cross_model_llama3.json")
    mistr = _load("cross_model_mistral.json")

    models = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B"]

    gini = [
        qwen["gini_mean"],
        llama["workload_characterization"]["gini_coefficient"]["mean"],
        mistr["workload_characterization"]["gini_coefficient"]["mean"],
    ]
    gini_std = [
        qwen["gini_std"],
        llama["workload_characterization"]["gini_coefficient"]["std"],
        mistr["workload_characterization"]["gini_coefficient"]["std"],
    ]

    jacc = [
        qwen["jaccard_mean"],
        llama["workload_characterization"]["jaccard_stability"]["mean"],
        mistr["workload_characterization"]["jaccard_stability"]["mean"],
    ]
    jacc_std = [
        0.0,
        llama["workload_characterization"]["jaccard_stability"]["std"],
        mistr["workload_characterization"]["jaccard_stability"]["std"],
    ]

    top10 = [
        qwen["top10_concentration"],
        llama["workload_characterization"]["top10_concentration"]["mean"],
        mistr["workload_characterization"]["top10_concentration"]["mean"],
    ]
    top10_std = [
        0.0,
        llama["workload_characterization"]["top10_concentration"]["std"],
        mistr["workload_characterization"]["top10_concentration"]["std"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    x = np.arange(len(models))
    w = 0.55

    titles = [
        "(a) Gini Coefficient",
        "(b) Jaccard Stability",
        "(c) Top-10% Concentration",
    ]
    vals  = [gini, jacc, top10]
    stds  = [gini_std, jacc_std, top10_std]

    for ax, title, v, s in zip(axes, titles, vals, stds):
        bars = ax.bar(x, v, w, yerr=s, capsize=3, color=COLORS[:3],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.6, 1.0)
        for bar, val in zip(bars, v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    axes[0].set_ylabel("Score")
    fig.tight_layout()
    _save(fig, "fig1_workload_characterization")


# ── Fig 2: Predictor Comparison ─────────────────────────────────────
def fig2_predictor_comparison():
    inf   = _load("infinigen_2048.json")
    llama = _load("cross_model_llama3.json")
    mistr = _load("cross_model_mistral.json")

    models  = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B"]
    methods = ["EMA", "InfiniGen", "Oracle"]

    ema_j = [
        inf["ema_jaccard_mean"],
        llama["predictor_comparison"]["ema_predictor"]["jaccard_mean"],
        mistr["predictor_comparison"]["ema_predictor"]["jaccard_mean"],
    ]
    ig_j = [
        inf["infinigen_jaccard_mean"],
        llama["predictor_comparison"]["infinigen_predictor"]["jaccard_mean"],
        mistr["predictor_comparison"]["infinigen_predictor"]["jaccard_mean"],
    ]
    oracle_j = [
        inf["oracle_jaccard_mean"],
        1.0,
        1.0,
    ]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(models))
    w = 0.22

    for i, (label, vals) in enumerate(zip(methods, [ema_j, ig_j, oracle_j])):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Predictor Accuracy Comparison")
    fig.tight_layout()
    _save(fig, "fig2_predictor_comparison")


# ── Fig 3: Memory Pressure Sweep ────────────────────────────────────
def fig3_memory_pressure():
    data = _load("memory_pressure_sweep.json")
    results = data["results"]

    budgets    = [r["budget_pct"] for r in results]
    throughput = [r["throughput_tok_s"] for r in results]
    evictions  = [r["evictions"] for r in results]

    fig, ax1 = plt.subplots(figsize=(3.3, 2.4))
    c1, c2 = COLORS[0], COLORS[1]

    ax1.plot(budgets, throughput, "o-", color=c1, linewidth=1.5, markersize=5,
             label="Throughput")
    ax1.set_xlabel("Budget (%)")
    ax1.set_ylabel("Throughput (tok/s)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(budgets)

    ax2 = ax1.twinx()
    ax2.plot(budgets, evictions, "s--", color=c2, linewidth=1.5, markersize=5,
             label="Evictions")
    ax2.set_ylabel("Evictions", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
               framealpha=0.9, fontsize=7)
    ax1.set_title("Memory Pressure Sweep")
    fig.tight_layout()
    _save(fig, "fig3_memory_pressure")


# ── Fig 4: SSD Ablation ─────────────────────────────────────────────
def fig4_ssd_ablation():
    data    = _load("ssd_ablation.json")
    configs = data["configs"]

    labels = [c["config_name"].split("(")[0].strip() for c in configs]
    thput  = [c["throughput_tok_s"] for c in configs]
    p50    = [c["promotion_p50_us"] for c in configs]
    p99    = [c["promotion_p99_us"] for c in configs]

    metrics  = ["Throughput\n(tok/s)", "Promo P50\n(µs)", "Promo P99\n(µs)"]
    all_vals = [thput, p50, p99]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    x = np.arange(len(labels))
    w = 0.55

    for ax, metric, vals in zip(axes, metrics, all_vals):
        bars = ax.bar(x, vals, w, color=COLORS[:3], edgecolor="white",
                      linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("SSD Tier Ablation", fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_ssd_ablation")


# ── Fig 5: Signal Overhead ──────────────────────────────────────────
def fig5_signal_overhead():
    data   = _load("attn_proxy_overhead.json")
    attn   = data["attention_overhead"]
    nstep  = data["nstep_sampling"]
    qk     = data["qk_proxy"]

    labels = ["No Attn.", "Full Attn.", "N=10\nSampling", "QK Proxy"]
    tpots  = [
        attn["tpot_no_attn_ms"],
        attn["tpot_with_attn_ms"],
        nstep["N=10"]["tpot_ms"],
        qk["tpot_qk_proxy_ms"],
    ]
    accs = [None, 1.0, nstep["N=10"]["classification_accuracy"], None]

    fig, ax1 = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(labels))
    bars = ax1.bar(x, tpots, 0.55, color=COLORS[:4], edgecolor="white",
                   linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title("Attention Signal Overhead")

    for bar, t in zip(bars, tpots):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{t:.2f}", ha="center", va="bottom", fontsize=7)

    for i, acc in enumerate(accs):
        if acc is not None:
            ax1.annotate(f"acc={acc:.2f}",
                         xy=(i, tpots[i]),
                         xytext=(0, 14), textcoords="offset points",
                         ha="center", fontsize=6.5, color="#333333",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc="#f0f0f0", ec="#aaaaaa", lw=0.5))

    fig.tight_layout()
    _save(fig, "fig5_signal_overhead")


# ── Fig 6: vLLM Serving Goodput ─────────────────────────────────────
def fig6_vllm_goodput():
    data = _load("vllm_serving_goodput.json")
    runs = data["runs"]

    bs   = [r["batch_size"] for r in runs]
    thpt = [r["throughput_tok_s"] for r in runs]
    tpot = [r["tpot_per_req_ms"] for r in runs]

    fig, ax1 = plt.subplots(figsize=(3.3, 2.4))
    c1, c2 = COLORS[0], COLORS[1]

    ax1.plot(bs, thpt, "o-", color=c1, linewidth=1.5, markersize=5,
             label="Throughput")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (tok/s)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(bs)
    ax1.set_xticklabels([str(b) for b in bs])

    ax2 = ax1.twinx()
    ax2.plot(bs, tpot, "s--", color=c2, linewidth=1.5, markersize=5,
             label="TPOT/req")
    ax2.set_ylabel("TPOT per Request (ms)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               framealpha=0.9, fontsize=7)
    ax1.set_title("vLLM Native Serving Goodput")
    fig.tight_layout()
    _save(fig, "fig6_vllm_goodput")


# ── Fig 7: Concurrency Analysis ─────────────────────────────────────
def fig7_concurrency():
    data = _load("vllm_connector_experiment.json")
    cap  = data["experiment_3_kv_capacity"]
    cold = data["experiment_2_cold_block_analysis"]

    gmu_vals   = []
    vllm_conc  = []
    orchkv_50  = []
    orchkv_70  = []
    orchkv_90  = []

    for gmu_str in ["0.3", "0.5", "0.7"]:
        entry = cap.get(gmu_str)
        if entry is None or entry.get("status") != "OK":
            continue
        gmu_vals.append(float(gmu_str))
        vllm_conc.append(entry["concurrency_analysis"]["max_concurrent_seqs_vllm"])
        for oc in entry["orchkv_comparison"]:
            if oc["offload_ratio"] == 0.5:
                orchkv_50.append(oc["max_concurrent_orchkv"])
            elif oc["offload_ratio"] == 0.7:
                orchkv_70.append(oc["max_concurrent_orchkv"])
            elif oc["offload_ratio"] == 0.9:
                orchkv_90.append(oc["max_concurrent_orchkv"])

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(gmu_vals))
    w = 0.18

    ax.bar(x - 1.5*w, vllm_conc, w, label="vLLM native",
           color=COLORS[0], edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5*w, orchkv_50, w, label="OrchKV 50% offload",
           color=COLORS[2], edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5*w, orchkv_70, w, label="OrchKV 70% offload",
           color=COLORS[3], edgecolor="white", linewidth=0.5)
    ax.bar(x + 1.5*w, orchkv_90, w, label="OrchKV 90% offload",
           color=COLORS[1], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g:.0%}" for g in gmu_vals])
    ax.set_xlabel("GPU Memory Utilization")
    ax.set_ylabel("Max Concurrent Sequences")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
    ax.set_title("Concurrency: vLLM vs OrchKvCache")
    fig.tight_layout()
    _save(fig, "fig7_concurrency")


# ── Fig 8: Long Context Scaling ─────────────────────────────────────
def fig8_long_context():
    data = _load("long_context_scaling.json")

    bl_lens = [r["prompt_len"] for r in data["baseline"]]
    bl_thpt = [r["throughput_tok_s"] for r in data["baseline"]]
    bl_tpot = [r["tpot_ms"] for r in data["baseline"]]

    ok_lens = [r["prompt_len"] for r in data["orchkv"]]
    ok_thpt = [r["throughput_tok_s"] for r in data["orchkv"]]
    ok_tpot = [r["tpot_ms"] for r in data["orchkv"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    ax1.plot(bl_lens, bl_thpt, "o-", color=COLORS[0], linewidth=1.5,
             markersize=5, label="Baseline")
    ax1.plot(ok_lens, ok_thpt, "s--", color=COLORS[1], linewidth=1.5,
             markersize=5, label="OrchKvCache")
    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_xticks(bl_lens)
    ax1.set_xticklabels(["1K", "2K", "4K"])
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.set_title("(a) Throughput vs Context Length")

    ax2.plot(bl_lens, bl_tpot, "o-", color=COLORS[0], linewidth=1.5,
             markersize=5, label="Baseline")
    ax2.plot(ok_lens, ok_tpot, "s--", color=COLORS[1], linewidth=1.5,
             markersize=5, label="OrchKvCache")
    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("TPOT (ms)")
    ax2.set_xticks(bl_lens)
    ax2.set_xticklabels(["1K", "2K", "4K"])
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.set_title("(b) TPOT vs Context Length")

    fig.tight_layout()
    _save(fig, "fig8_long_context_scaling")


# ── Table 1: Cross-Model Correctness (LaTeX) ────────────────────────
def table1_cross_model():
    llama = _load("cross_model_llama3.json")
    mistr = _load("cross_model_mistral.json")
    qwen  = _load("workload_char_2048.json")

    print("\n" + "=" * 72)
    print("TABLE 1: Cross-Model Summary (LaTeX)")
    print("=" * 72)
    header = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Cross-model KV cache characterization and correctness.}" "\n"
        r"\label{tab:cross-model}" "\n"
        r"\begin{tabular}{lccccc}" "\n"
        r"\toprule" "\n"
        r"Model & Gini & Jaccard & Top-10\% & Exact Match & EMA Jaccard \\" "\n"
        r"\midrule"
    )

    def _row(name, gini, jacc, t10, em, ema_j):
        return (f"{name} & {gini:.3f} & {jacc:.3f} & {t10:.3f} "
                f"& {em:.1%} & {ema_j:.3f} \\\\")

    rows = [
        _row("Qwen2.5-7B",
             qwen["gini_mean"], qwen["jaccard_mean"],
             qwen["top10_concentration"], 1.0, 0.659),
        _row("Llama-3.1-8B",
             llama["workload_characterization"]["gini_coefficient"]["mean"],
             llama["workload_characterization"]["jaccard_stability"]["mean"],
             llama["workload_characterization"]["top10_concentration"]["mean"],
             llama["correctness"]["exact_match_rate"],
             llama["predictor_comparison"]["ema_predictor"]["jaccard_mean"]),
        _row("Mistral-7B",
             mistr["workload_characterization"]["gini_coefficient"]["mean"],
             mistr["workload_characterization"]["jaccard_stability"]["mean"],
             mistr["workload_characterization"]["top10_concentration"]["mean"],
             mistr["correctness"]["exact_match_rate"],
             mistr["predictor_comparison"]["ema_predictor"]["jaccard_mean"]),
    ]

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )

    table = "\n".join([header] + rows + [footer])
    print(table)
    print("=" * 72 + "\n")

    with open(OUTDIR / "table1_cross_model.tex", "w") as f:
        f.write(table + "\n")
    print("  [OK] table1_cross_model.tex")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    _setup()
    generators = [
        ("Fig 1", fig1_workload_characterization),
        ("Fig 2", fig2_predictor_comparison),
        ("Fig 3", fig3_memory_pressure),
        ("Fig 4", fig4_ssd_ablation),
        ("Fig 5", fig5_signal_overhead),
        ("Fig 6", fig6_vllm_goodput),
        ("Fig 7", fig7_concurrency),
        ("Fig 8", fig8_long_context),
        ("Table 1", table1_cross_model),
    ]

    ok, fail = 0, 0
    for name, fn in generators:
        print(f"\n[{name}]")
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            fail += 1

    print(f"\n{'=' * 40}")
    print(f"Done: {ok} succeeded, {fail} failed")
    print(f"Output directory: {OUTDIR}")


if __name__ == "__main__":
    main()

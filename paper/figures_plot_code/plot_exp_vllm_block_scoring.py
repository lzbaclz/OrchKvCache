#!/usr/bin/env python3
"""
Paper Figure: vLLM block-level scoring strategies under memory pressure.

Generates:
  1. Grouped bar chart: throughput by strategy at each (gpu_util, num_prompts)
  2. LaTeX table for direct inclusion in the paper
  3. Speedup annotation on bars

Reads: exp_vllm_block_scoring.json (from benchmarks/exp_vllm_block_scoring.py)
       OR falls back to exp_vllm_multi_pressure.json (existing data)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output_figures"
OUT.mkdir(parents=True, exist_ok=True)

# -- Style --
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

STRATEGY_META = {
    "fifo":       {"label": "vLLM-default",    "color": "#E8B4A8", "edge": "#C07868", "hatch": ""},
    "progress":   {"label": "Progress-aware",  "color": "#A8D5BA", "edge": "#6BA882", "hatch": ""},
    "block_v1":   {"label": "Block-V1 (orig)", "color": "#D4C5E2", "edge": "#9878B0", "hatch": "//"},
    "block_v2":   {"label": "Block-V2 (fast)", "color": "#9CC0D8", "edge": "#5A90B0", "hatch": ""},
    "hybrid":     {"label": "Hybrid (V3)",     "color": "#F5D6A8", "edge": "#C8A060", "hatch": ""},
    "block_score": {"label": "Block-V1 (orig)", "color": "#D4C5E2", "edge": "#9878B0", "hatch": "//"},
}


def load_data():
    """Load experiment data, preferring the new comprehensive JSON."""
    new_path = BASE / "exp_vllm_block_scoring.json"
    old_path = BASE / "exp_vllm_multi_pressure.json"

    if new_path.exists():
        with open(new_path) as f:
            data = json.load(f)
        print(f"Loaded {new_path} ({len(data)} rows)")
        return data, True

    if old_path.exists():
        with open(old_path) as f:
            data = json.load(f)
        print(f"Loaded {old_path} ({len(data)} rows, old format)")
        return data, False

    print("No data file found", file=sys.stderr)
    sys.exit(1)


def get_configs(data):
    """Extract unique (gpu_util, num_prompts) combinations, sorted."""
    configs = sorted(set(
        (float(r["gpu_util"]), int(r["num_prompts"]))
        for r in data
    ))
    return configs


def get_strategies(data):
    """Extract strategies in desired display order."""
    present = set(r["strategy"] for r in data)
    order = ["fifo", "progress", "block_v1", "block_score",
             "block_v2", "hybrid"]
    return [s for s in order if s in present]


def plot_grouped_bars(data, strategies, configs, out_prefix="fig_vllm_scoring"):
    """Grouped bar chart: one group per config, one bar per strategy."""
    n_groups = len(configs)
    n_bars = len(strategies)
    if n_groups == 0 or n_bars == 0:
        return

    bar_width = 0.8 / n_bars
    fig_width = max(5.5, 1.2 * n_groups * n_bars)
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))

    x = np.arange(n_groups)

    for j, strat in enumerate(strategies):
        meta = STRATEGY_META.get(strat, {
            "label": strat, "color": "#CCC", "edge": "#999", "hatch": ""
        })
        vals = []
        errs = []
        for gu, np_ in configs:
            rows = [r for r in data
                    if abs(float(r["gpu_util"]) - gu) < 1e-6
                    and int(r["num_prompts"]) == np_
                    and r["strategy"] == strat]
            if rows:
                vals.append(float(rows[0]["avg_throughput"]))
                errs.append(float(rows[0].get("std_throughput", 0)))
            else:
                vals.append(0)
                errs.append(0)

        offset = (j - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width * 0.92,
            yerr=errs if any(e > 0 for e in errs) else None,
            capsize=2,
            color=meta["color"],
            edgecolor=meta["edge"],
            hatch=meta["hatch"],
            linewidth=0.6,
            label=meta["label"],
            zorder=3,
        )

        fifo_vals = []
        for gu, np_ in configs:
            frows = [r for r in data
                     if abs(float(r["gpu_util"]) - gu) < 1e-6
                     and int(r["num_prompts"]) == np_
                     and r["strategy"] == "fifo"]
            fifo_vals.append(
                float(frows[0]["avg_throughput"]) if frows else 0)

        if strat != "fifo":
            for k, (bar, v, fv) in enumerate(zip(bars, vals, fifo_vals)):
                if v > 0 and fv > 0:
                    speedup = v / fv
                    if abs(speedup - 1.0) > 0.005:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(errs[k], 0) + 30,
                            f"{speedup:.2f}×",
                            ha="center", va="bottom", fontsize=6.5,
                            fontweight="bold",
                            color=meta["edge"],
                        )

    labels = [f"{gu:.2f}\n({np_} req)" for gu, np_ in configs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(fontsize=7.5, ncol=min(n_bars, 3), loc="upper left",
              framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{out_prefix}.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}/{out_prefix}.pdf/png")


def plot_pressure_focus(data, strategies, out_prefix="fig_vllm_pressure_focus"):
    """Focused bar chart: only the tightest memory pressure configs."""
    min_gu = min(float(r["gpu_util"]) for r in data)
    pressure_data = [r for r in data if abs(float(r["gpu_util"]) - min_gu) < 1e-6]
    if not pressure_data:
        return

    configs = get_configs(pressure_data)
    plot_grouped_bars(pressure_data, strategies, configs, out_prefix)


def generate_latex_table(data, strategies, configs):
    """Generate LaTeX table source for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{vLLM integration: victim-selection strategies "
                 r"under memory pressure. Speedup is relative to "
                 r"vLLM-default at each configuration.}")
    lines.append(r"\label{tab:vllm-block}")
    lines.append(r"\small")

    ncols = 4
    lines.append(r"\begin{tabular}{@{}clrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"gpu\_util & Policy & tok/s & vs Default \\")
    lines.append(r"\midrule")

    for i, (gu, np_) in enumerate(configs):
        rows_here = [r for r in data
                     if abs(float(r["gpu_util"]) - gu) < 1e-6
                     and int(r["num_prompts"]) == np_]
        fifo_thr = 0
        for r in rows_here:
            if r["strategy"] == "fifo":
                fifo_thr = float(r["avg_throughput"])
                break

        config_label = f"{gu:.2f} ({np_} req)"
        strats_here = [s for s in strategies if any(
            r["strategy"] == s for r in rows_here)]

        for j, strat in enumerate(strats_here):
            r = next((r for r in rows_here if r["strategy"] == strat), None)
            if r is None:
                continue
            thr = float(r["avg_throughput"])
            sp = thr / fifo_thr if fifo_thr > 0 else 0
            meta = STRATEGY_META.get(strat, {"label": strat})

            thr_str = f"{thr:,.0f}"
            sp_str = f"{sp:.2f}$\\times$"
            if sp > 1.03:
                sp_str = r"\textbf{" + sp_str + "}"

            if j == 0:
                lines.append(
                    f"\\multirow{{{len(strats_here)}}}{{*}}"
                    f"{{{config_label}}}")
            lines.append(
                f"& {meta['label']:<20s} & {thr_str:>6s} "
                f"& {sp_str} \\\\")

        if i < len(configs) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)

    tex_path = OUT / "tab_vllm_block_scoring.tex"
    with open(tex_path, "w") as f:
        f.write(table_str)
    print(f"Wrote {tex_path}")
    print("\n--- LaTeX Table ---")
    print(table_str)
    print("--- End ---\n")


def main():
    data, is_new = load_data()
    configs = get_configs(data)
    strategies = get_strategies(data)

    print(f"Configs: {configs}")
    print(f"Strategies: {strategies}")

    plot_grouped_bars(data, strategies, configs)

    pressure_configs = [c for c in configs if c[0] <= 0.20]
    if pressure_configs:
        pressure_data = [r for r in data if float(r["gpu_util"]) <= 0.20]
        plot_grouped_bars(pressure_data, strategies,
                          get_configs(pressure_data),
                          "fig_vllm_pressure_focus")

    generate_latex_table(data, strategies, configs)


if __name__ == "__main__":
    main()

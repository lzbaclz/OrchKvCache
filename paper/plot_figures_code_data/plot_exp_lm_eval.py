#!/usr/bin/env python3
"""
Plot LM-Eval benchmark results: GPU-Only vs OrchKvCache accuracy.
Since OrchKvCache is lossless, both bars should be identical.

Input:  exp_lm_eval.json (same directory)
Output: output_figures/fig_lm_eval_accuracy.pdf/png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path(__file__).parent
OUT = DATA / "output_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def main():
    with open(DATA / "exp_lm_eval.json") as f:
        data = json.load(f)

    tasks = ["piqa", "rte", "copa", "openbookqa"]
    task_labels = ["PIQA", "RTE", "COPA", "OBQA"]
    models = [r["model"] for r in data]

    fig, axes = plt.subplots(1, len(models), figsize=(6.5, 2.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    colors = {'GPU-Only': '#2ca02c', 'OrchKvCache': '#1f77b4'}

    for idx, r in enumerate(data):
        ax = axes[idx]
        x = np.arange(len(tasks))
        w = 0.35

        gpu_vals = [r["gpu_only"].get(t, 0) for t in tasks]
        orch_vals = [r["orchkv"].get(t, 0) for t in tasks]

        bars1 = ax.bar(x - w/2, gpu_vals, w, label='GPU-Only',
                       color=colors['GPU-Only'], edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + w/2, orch_vals, w, label='OrchKvCache',
                       color=colors['OrchKvCache'], edgecolor='white', linewidth=0.5)

        for i in range(len(tasks)):
            if gpu_vals[i] == orch_vals[i] and gpu_vals[i] > 0:
                ax.annotate('=', xy=(x[i], max(gpu_vals[i], orch_vals[i]) + 0.5),
                           ha='center', fontsize=9, fontweight='bold', color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy (%)' if idx == 0 else '')
        ax.set_title(f'({chr(97+idx)}) {r["model"]}')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUT / f"fig_lm_eval_accuracy.{fmt}", format=fmt)
    print(f"Saved to {OUT}/fig_lm_eval_accuracy.pdf")
    plt.close()


if __name__ == "__main__":
    main()

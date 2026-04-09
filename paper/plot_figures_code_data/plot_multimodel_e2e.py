#!/usr/bin/env python3
"""Fig 1–5 from multimodel_e2e.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_figures_1"
JSON_PATH = BASE / "multimodel_e2e.json"

C_GPU  = "#A8D5BA"
C_FIFO = "#E8B4A8"
C_ORKV = "#9CC0D8"
C_GRAY = "#999999"

EC_GPU  = "#6BA882"
EC_FIFO = "#C07868"
EC_ORKV = "#5A90B0"

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


def _load():
    with open(JSON_PATH) as f:
        return json.load(f)


def _short_model(m: str) -> str:
    return m.replace("Qwen2.5-", "Qwen ").replace("-7B", "").strip() or m


def _filter_config(rows, seq_len=2048, n_requests=8, budget_mb=50):
    out = []
    for r in rows:
        if r.get("seq_len") != seq_len:
            continue
        if r.get("n_requests", r.get("num_requests")) != n_requests:
            continue
        b = r.get("gpu_budget_mb", r.get("budget_mb"))
        if b is not None and b != budget_mb:
            continue
        out.append(r)
    return out


def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)


def _top_legend(ax, ncol=3):
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, frameon=False, fontsize=9, columnspacing=1.5,
    )


def main():
    data = _load()
    modes = ["baseline", "naive", "orchkv"]
    mode_colors = {"baseline": C_GPU, "naive": C_FIFO, "orchkv": C_ORKV}
    mode_ec     = {"baseline": EC_GPU, "naive": EC_FIFO, "orchkv": EC_ORKV}
    mode_labels = {"baseline": "GPU-Only", "naive": "FIFO", "orchkv": "OrchKvCache"}

    # Fig 1: throughput bar (one representative config)
    sel = _filter_config(data)
    if not sel:
        sel = data[: min(9, len(data))]
    models = sorted({r.get("model", "?") for r in sel})
    fig1, ax1 = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(models))
    w = 0.25
    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            v = None
            for r in sel:
                if r.get("model") == m and r.get("mode") == mode:
                    v = float(r.get("avg_throughput", 0) or 0)
                    break
            vals.append(v if v is not None else 0.0)
        ax1.bar(x + (i - 1) * w, vals, w,
                label=mode_labels.get(mode, mode),
                color=mode_colors[mode], edgecolor=mode_ec[mode], **BAR_KW)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_short_model(m) for m in models])
    ax1.set_ylabel("Throughput (tok/s)")
    
    _top_legend(ax1)
    ax1.grid(axis="y", alpha=0.3)
    _save(fig1, "fig01_multimodel_throughput_bar")
    print(f"Wrote {OUT}/fig01_multimodel_throughput_bar.pdf/png")

    # Fig 2: eviction comparison (same selection)
    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            v = 0.0
            for r in sel:
                if r.get("model") == m and r.get("mode") == mode:
                    v = float(r.get("total_evictions", 0) or 0)
                    break
            vals.append(v)
        ax2.bar(x + (i - 1) * w, vals, w,
                label=mode_labels.get(mode, mode),
                color=mode_colors[mode], edgecolor=mode_ec[mode], **BAR_KW)
    ax2.set_xticks(x)
    ax2.set_xticklabels([_short_model(m) for m in models])
    ax2.set_ylabel("Total evictions")
    
    _top_legend(ax2)
    ax2.grid(axis="y", alpha=0.3)
    _save(fig2, "fig02_multimodel_eviction")
    print(f"Wrote {OUT}/fig02_multimodel_eviction.pdf/png")

    # Fig 3: TPOT stability vs num_requests (seq_len=2048, budget=50, first model)
    fig3, ax3 = plt.subplots(figsize=(6.5, 3.2))
    m0 = models[0] if models else None
    if m0:
        reqs = sorted({int(r.get("n_requests", r.get("num_requests", 0)) or 0)
                       for r in data if r.get("model") == m0 and r.get("seq_len") == 2048})
        for mode in modes:
            ys = []
            xs = []
            for nr in reqs:
                for r in data:
                    if (r.get("model") == m0 and r.get("mode") == mode
                            and r.get("seq_len") == 2048
                            and int(r.get("n_requests", r.get("num_requests", 0)) or 0) == nr):
                        b = r.get("gpu_budget_mb", r.get("budget_mb"))
                        if b is not None and b != 50:
                            continue
                        xs.append(nr)
                        ys.append(float(r.get("avg_tpot_ms", 0) or 0))
                        break
            if xs:
                ax3.plot(xs, ys, marker="o", markersize=5,
                         label=mode_labels.get(mode, mode), color=mode_ec[mode])
    ax3.set_xlabel("Concurrent requests")
    ax3.set_ylabel("Avg TPOT (ms)")
    
    _top_legend(ax3)
    ax3.grid(alpha=0.3)
    _save(fig3, "fig03_multimodel_tpot_stability")
    print(f"Wrote {OUT}/fig03_multimodel_tpot_stability.pdf/png")

    # Fig 4: speedup heatmap orchkv / naive (rows=models, cols=seq_len)
    seq_lens = sorted({int(r.get("seq_len", 0) or 0) for r in data if r.get("seq_len")})[:8]
    if not seq_lens:
        seq_lens = [2048, 4096]
    heat_models = sorted({r.get("model", "?") for r in data})
    mat = np.zeros((len(heat_models), len(seq_lens)))
    for i, m in enumerate(heat_models):
        for j, sl in enumerate(seq_lens):
            orch = naive = None
            for r in data:
                if r.get("model") != m or int(r.get("seq_len", 0) or 0) != sl:
                    continue
                if int(r.get("n_requests", r.get("num_requests", -1)) or -1) != 8:
                    continue
                b = r.get("gpu_budget_mb", r.get("budget_mb"))
                if b is not None and b != 50:
                    continue
                mode = r.get("mode")
                tp = float(r.get("avg_throughput", 0) or 0)
                if mode == "orchkv":
                    orch = tp
                elif mode == "naive":
                    naive = tp
            if orch is not None and naive and naive > 1e-6:
                mat[i, j] = orch / naive
            else:
                mat[i, j] = np.nan
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    im = ax4.imshow(mat, aspect="auto", cmap="YlGnBu")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax4.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax4.set_xticks(range(len(seq_lens)))
    ax4.set_xticklabels([f"{s // 1024}K" for s in seq_lens])
    ax4.set_yticks(range(len(heat_models)))
    ax4.set_yticklabels([_short_model(m) for m in heat_models])
    ax4.set_xlabel("Context length")
    ax4.set_ylabel("Model")
    
    plt.colorbar(im, ax=ax4, label="Speedup")
    _save(fig4, "fig04_multimodel_speedup_heatmap")
    print(f"Wrote {OUT}/fig04_multimodel_speedup_heatmap.pdf/png")

    # Fig 5: per-model speedup (orchkv vs naive), same slice as heatmap
    speedups = []
    for m in heat_models:
        orch = naive = None
        for r in data:
            if r.get("model") != m:
                continue
            if int(r.get("seq_len", 0) or 0) != 2048:
                continue
            if int(r.get("n_requests", r.get("num_requests", -1)) or -1) != 8:
                continue
            b = r.get("gpu_budget_mb", r.get("budget_mb"))
            if b is not None and b != 50:
                continue
            tp = float(r.get("avg_throughput", 0) or 0)
            if r.get("mode") == "orchkv":
                orch = tp
            elif r.get("mode") == "naive":
                naive = tp
        if orch is not None and naive and naive > 1e-6:
            speedups.append(orch / naive)
        else:
            speedups.append(0.0)
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    ax5.bar(range(len(heat_models)), speedups,
            color=C_ORKV, edgecolor=EC_ORKV, **BAR_KW)
    ax5.set_xticks(range(len(heat_models)))
    ax5.set_xticklabels([_short_model(m) for m in heat_models])
    ax5.set_ylabel("OrchKv / naive throughput")
    
    ax5.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1)
    ax5.grid(axis="y", alpha=0.3)
    _save(fig5, "fig05_multimodel_per_model_speedup")
    print(f"Wrote {OUT}/fig05_multimodel_per_model_speedup.pdf/png")


if __name__ == "__main__":
    main()

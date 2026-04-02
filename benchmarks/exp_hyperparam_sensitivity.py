#!/usr/bin/env python3
"""
E11: Hyperparameter Sensitivity Analysis

Sweeps classifier parameters (λ, τ, cooldown) to quantify their effect on
classification quality and scheduling stability.  Uses orchkv_core trace
simulation — no GPU required.

Parameters studied:
  1. λ (ema_lambda)     — EMA decay factor for attention scores
  2. τ (recency_tau)    — time constant for recency score decay
  3. cooldown_sec       — min interval between adaptive threshold adjustments
  4. Combined λ × τ     — joint sensitivity heatmap

Usage:
    python benchmarks/exp_hyperparam_sensitivity.py
    python benchmarks/exp_hyperparam_sensitivity.py --n-blocks 512 --n-steps 300
    python benchmarks/exp_hyperparam_sensitivity.py --sweep lambda
    python benchmarks/exp_hyperparam_sensitivity.py --sweep combined
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, save_csv, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None

from exp_attn_sampling import generate_attention_trace

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

BASE_PARAMS = dict(
    alpha=0.5, beta=0.3, gamma=0.2,
    prefetch_budget=16,
    schedule_interval_us=500,
    gpu_hwm=0.80, gpu_lwm=0.60,
    dram_hwm=0.80, dram_lwm=0.60,
    threshold_to_gpu=0.4,
    threshold_to_dram=0.15,
    ema_lambda=0.9,
    recency_tau=50.0,
    cooldown_sec=0.0,
    adjust_step=0.02,
)

LAMBDA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
TAU_VALUES = [5, 10, 25, 50, 100, 200]
COOLDOWN_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]


# ═══════════════════════════════════════════════════════════════════
#  Ground Truth
# ═══════════════════════════════════════════════════════════════════

def _ground_truth_counts(trace, n_blocks,
                         hot_thresh=0.50, warm_thresh=0.08):
    gt = []
    for step_data in trace:
        weights = {bid: w for bid, w in step_data}
        n_hot = sum(1 for b in range(n_blocks) if weights.get(b, 0) >= hot_thresh)
        n_warm = sum(1 for b in range(n_blocks)
                     if warm_thresh <= weights.get(b, 0) < hot_thresh)
        n_cold = n_blocks - n_hot - n_warm
        gt.append({"n_hot": n_hot, "n_warm": n_warm, "n_cold": n_cold})
    return gt


# ═══════════════════════════════════════════════════════════════════
#  Simulation Core
# ═══════════════════════════════════════════════════════════════════

def _create_tm(n_blocks: int, params: dict):
    base_kwargs = dict(
        tracker_cap=n_blocks * 2,
        max_blocks=n_blocks + 64,
        ema_lambda=params["ema_lambda"],
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        prefetch_budget=params["prefetch_budget"],
        schedule_interval_us=params["schedule_interval_us"],
        gpu_hwm=params["gpu_hwm"],
        gpu_lwm=params["gpu_lwm"],
        dram_hwm=params["dram_hwm"],
        dram_lwm=params["dram_lwm"],
        threshold_to_gpu=params["threshold_to_gpu"],
        threshold_to_dram=params["threshold_to_dram"],
    )
    try:
        return _C.tm_create(
            **base_kwargs,
            recency_tau=params.get("recency_tau", 50.0),
            cooldown_sec=params.get("cooldown_sec", 0.0),
            adjust_step=params.get("adjust_step", 0.02),
        )
    except TypeError:
        return _C.tm_create(**base_kwargs)


def _run_simulation(trace, n_blocks, params, report_threshold=0.05):
    """Run trace through tiered_manager; return per-step records and final stats."""
    tm = _create_tm(n_blocks, params)

    for bid in range(n_blocks):
        _C.tm_register_block_id(tm, bid, int(_C.GPU_HBM), 0)

    records = []
    for step, step_data in enumerate(trace):
        for bid, w in step_data:
            if w >= report_threshold:
                _C.tm_report_attn(tm, bid, w)

        _C.tm_step_done(tm)

        if step % 5 == 0:
            _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.55)
            _C.tm_schedule_once(tm)

        s = _C.tm_get_stats(tm)
        records.append({
            "step": step,
            "n_hot": s["n_hot"],
            "n_warm": s["n_warm"],
            "n_cold": s["n_cold"],
        })

    final = _C.tm_get_stats(tm)
    _C.tm_destroy(tm)
    return records, final


def _compute_metrics(records, gt, n_blocks, final_stats):
    n = min(len(records), len(gt))
    acc_total = 0.0
    for rec, ref in zip(records[:n], gt[:n]):
        diff = (abs(rec["n_hot"] - ref["n_hot"])
                + abs(rec["n_warm"] - ref["n_warm"])
                + abs(rec["n_cold"] - ref["n_cold"]))
        acc_total += max(0.0, 1.0 - diff / (2 * n_blocks))
    accuracy = acc_total / max(n, 1)

    hot_series = [r["n_hot"] for r in records]
    hot_std = statistics.stdev(hot_series) if len(hot_series) > 1 else 0.0

    oscillations = 0
    for i in range(2, len(hot_series)):
        d1 = hot_series[i - 1] - hot_series[i - 2]
        d2 = hot_series[i] - hot_series[i - 1]
        if d1 * d2 < 0 and abs(d1) >= 2 and abs(d2) >= 2:
            oscillations += 1

    gpu_demotes = final_stats.get("gpu_demotes", 0)
    dram_demotes = final_stats.get("dram_demotes", 0)
    adj_up = final_stats.get("threshold_adj_up", 0)
    adj_down = final_stats.get("threshold_adj_down", 0)

    return {
        "accuracy": round(accuracy, 4),
        "hot_std": round(hot_std, 2),
        "oscillations": oscillations,
        "gpu_demotes": gpu_demotes,
        "total_migrations": gpu_demotes + dram_demotes,
        "threshold_adj_total": adj_up + adj_down,
    }


# ═══════════════════════════════════════════════════════════════════
#  Sweep Functions
# ═══════════════════════════════════════════════════════════════════

def _sweep_param(name, values, trace, gt, n_blocks, n_runs, seed):
    """Sweep one parameter while holding others at defaults."""
    results = []
    for val in values:
        run_accs, run_stds, run_oscs, run_migs = [], [], [], []
        for rid in range(n_runs):
            run_seed = seed + rid * 1000
            tr = generate_attention_trace(n_blocks, len(trace[0]) if trace else 200,
                                          seed=run_seed)
            gt_r = _ground_truth_counts(tr, n_blocks)

            params = dict(BASE_PARAMS)
            params[name] = val
            records, final = _run_simulation(tr, n_blocks, params)
            m = _compute_metrics(records, gt_r, n_blocks, final)
            run_accs.append(m["accuracy"])
            run_stds.append(m["hot_std"])
            run_oscs.append(m["oscillations"])
            run_migs.append(m["total_migrations"])

        results.append({
            "param": name,
            "value": val,
            "accuracy": round(statistics.mean(run_accs), 4),
            "accuracy_std": round(statistics.stdev(run_accs), 4) if n_runs > 1 else 0,
            "hot_std": round(statistics.mean(run_stds), 2),
            "oscillations": round(statistics.mean(run_oscs), 1),
            "total_migrations": round(statistics.mean(run_migs), 1),
        })
        print(f"    {name}={val:<8}  acc={results[-1]['accuracy']:.3f}  "
              f"hot_std={results[-1]['hot_std']:.1f}  "
              f"osc={results[-1]['oscillations']:.0f}  "
              f"mig={results[-1]['total_migrations']:.0f}")

    return results


def sweep_ema_lambda(trace, gt, n_blocks, n_runs=3, seed=42):
    print(f"\n  --- Sweep: ema_lambda (λ) ---")
    print(f"  Default: {BASE_PARAMS['ema_lambda']}, range: {LAMBDA_VALUES}")
    return _sweep_param("ema_lambda", LAMBDA_VALUES, trace, gt, n_blocks, n_runs, seed)


def sweep_recency_tau(trace, gt, n_blocks, n_runs=3, seed=42):
    print(f"\n  --- Sweep: recency_tau (τ) ---")
    print(f"  Default: {BASE_PARAMS['recency_tau']}, range: {TAU_VALUES}")
    return _sweep_param("recency_tau", TAU_VALUES, trace, gt, n_blocks, n_runs, seed)


def sweep_cooldown(trace, gt, n_blocks, n_runs=3, seed=42):
    print(f"\n  --- Sweep: cooldown_sec ---")
    print(f"  Default: {BASE_PARAMS['cooldown_sec']}, range: {COOLDOWN_VALUES}")
    return _sweep_param("cooldown_sec", COOLDOWN_VALUES, trace, gt, n_blocks, n_runs, seed)


def sweep_combined(n_blocks, n_steps, n_runs=3, seed=42):
    """Joint λ × τ sweep → heatmap data."""
    print(f"\n  --- Sweep: combined λ × τ ---")
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    taus = [5, 10, 25, 50, 100]
    results = []

    for lam in lambdas:
        for tau in taus:
            run_accs = []
            for rid in range(n_runs):
                run_seed = seed + rid * 1000
                tr = generate_attention_trace(n_blocks, n_steps, seed=run_seed)
                gt_r = _ground_truth_counts(tr, n_blocks)
                params = dict(BASE_PARAMS)
                params["ema_lambda"] = lam
                params["recency_tau"] = tau
                records, final = _run_simulation(tr, n_blocks, params)
                m = _compute_metrics(records, gt_r, n_blocks, final)
                run_accs.append(m["accuracy"])

            avg_acc = statistics.mean(run_accs)
            results.append({
                "ema_lambda": lam,
                "recency_tau": tau,
                "accuracy": round(avg_acc, 4),
            })
            print(f"    λ={lam:.2f} τ={tau:>4}  acc={avg_acc:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="E11: Hyperparameter Sensitivity")
    ap.add_argument("--n-blocks", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", default="all",
                    choices=["all", "lambda", "tau", "cooldown", "combined"],
                    help="Which parameter to sweep")
    args = ap.parse_args()

    print("=" * 70)
    print("  E11: Hyperparameter Sensitivity Analysis")
    print(f"  n_blocks={args.n_blocks}  n_steps={args.n_steps}  n_runs={args.n_runs}")
    print("=" * 70)

    if _C is None:
        print("  [SKIP] orchkv_core not available — build first")
        return

    trace = generate_attention_trace(args.n_blocks, args.n_steps, seed=args.seed)
    gt = _ground_truth_counts(trace, args.n_blocks)

    all_results = {}

    if args.sweep in ("all", "lambda"):
        all_results["ema_lambda"] = sweep_ema_lambda(
            trace, gt, args.n_blocks, args.n_runs, args.seed)

    if args.sweep in ("all", "tau"):
        all_results["recency_tau"] = sweep_recency_tau(
            trace, gt, args.n_blocks, args.n_runs, args.seed)

    if args.sweep in ("all", "cooldown"):
        all_results["cooldown"] = sweep_cooldown(
            trace, gt, args.n_blocks, args.n_runs, args.seed)

    if args.sweep in ("all", "combined"):
        all_results["combined"] = sweep_combined(
            args.n_blocks, args.n_steps, args.n_runs, args.seed)

    output = {
        "config": {
            "n_blocks": args.n_blocks,
            "n_steps": args.n_steps,
            "n_runs": args.n_runs,
            "seed": args.seed,
            "base_params": BASE_PARAMS,
        },
        **all_results,
    }
    save_json(output, "exp_e11_hyperparam")

    flat_rows = []
    for key in ("ema_lambda", "recency_tau", "cooldown"):
        if key in all_results:
            flat_rows.extend(all_results[key])
    if flat_rows:
        save_csv(flat_rows, "exp_e11_hyperparam")

    print(f"\n{'=' * 70}")
    print("  E11 Summary")
    print(f"{'=' * 70}")
    for key in ("ema_lambda", "recency_tau", "cooldown"):
        rows = all_results.get(key, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["accuracy"])
        worst = min(rows, key=lambda r: r["accuracy"])
        spread = best["accuracy"] - worst["accuracy"]
        print(f"  {key:>14s}:  best={best['value']} (acc={best['accuracy']:.3f})  "
              f"worst={worst['value']} (acc={worst['accuracy']:.3f})  "
              f"spread={spread:.3f}")

    if "combined" in all_results:
        best = max(all_results["combined"], key=lambda r: r["accuracy"])
        print(f"  {'combined':>14s}:  best=(λ={best['ema_lambda']}, "
              f"τ={best['recency_tau']}) acc={best['accuracy']:.3f}")


if __name__ == "__main__":
    main()

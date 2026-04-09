#!/usr/bin/env python3
"""
W6 Response: Signal Ablation — Attention vs Recency+Frequency

Compares classification accuracy with different signal weight configurations:
  1. Full EMA   (α=0.7, β=0.2, γ=0.1) — default, attention-dominant
  2. No-attn    (α=0.0, β=0.6, γ=0.4) — simulates SDPA fallback
  3. Recency    (α=0.0, β=1.0, γ=0.0) — pure LRU equivalent
  4. Frequency  (α=0.0, β=0.0, γ=1.0) — pure LFU equivalent

Uses orchkv_core trace simulation (no GPU needed).

Usage:
    python benchmarks/exp_signal_ablation.py
    python benchmarks/exp_signal_ablation.py --n-blocks 256 --n-steps 200 --n-runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    print("ERROR: orchkv_core not found. Build with: cd build && cmake .. && make -j")
    sys.exit(1)


CONFIGS = {
    "full_ema":       {"alpha": 0.7, "beta": 0.2, "gamma": 0.1, "desc": "Full EMA (default)"},
    "no_attn":        {"alpha": 0.0, "beta": 0.6, "gamma": 0.4, "desc": "No attention (SDPA fallback)"},
    "recency_only":   {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "desc": "Recency only (LRU-like)"},
    "frequency_only": {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "desc": "Frequency only (LFU-like)"},
}

BASE_TM_PARAMS = dict(
    prefetch_budget=16,
    schedule_interval_us=500,
    ema_lambda=0.9,
    recency_tau=50.0,
    cooldown_sec=0.5,
    threshold_to_gpu=0.6,
    threshold_to_dram=0.15,
)


def generate_trace(n_blocks, n_steps, hot_frac=0.10, shift_every=5, seed=42):
    rng = random.Random(seed)
    n_hot = max(1, int(n_blocks * hot_frac))
    n_warm = max(1, int(n_blocks * 0.15 * 2))
    hot_set = set(range(n_hot))
    warm_pool = set(range(n_hot, n_hot + n_warm))

    trace, gt_labels = [], []
    for step in range(n_steps):
        if step > 0 and step % shift_every == 0:
            n_shift = max(1, int(n_hot * 0.15))
            removable = list(hot_set)
            to_remove = set(rng.sample(removable, min(n_shift, len(removable))))
            cold_cands = [b for b in range(n_blocks) if b not in hot_set]
            to_add = set(rng.sample(cold_cands, min(n_shift, len(cold_cands))))
            hot_set = (hot_set - to_remove) | to_add
            warm_pool = (warm_pool | to_remove) - to_add

        step_obs = []
        for b in range(n_blocks):
            if b in hot_set:
                w = rng.uniform(0.5, 1.0)
            elif b in warm_pool:
                w = rng.uniform(0.01, 0.15)
            else:
                w = rng.uniform(0.0, 0.01)
            step_obs.append((b, w))
        trace.append(step_obs)

        labels = {}
        for b in range(n_blocks):
            if b in hot_set:
                labels[b] = "hot"
            elif b in warm_pool:
                labels[b] = "warm"
            else:
                labels[b] = "cold"
        gt_labels.append(labels)

    return trace, gt_labels


def run_config(name, cfg, trace, gt_labels, n_blocks, n_runs, seed):
    results = []
    for run in range(n_runs):
        params = dict(BASE_TM_PARAMS)
        params["alpha"] = cfg["alpha"]
        params["beta"] = cfg["beta"]
        params["gamma"] = cfg["gamma"]

        handle = _C.tm_create(
            max_blocks=n_blocks + 64,
            ema_lambda=params["ema_lambda"],
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            prefetch_budget=params["prefetch_budget"],
            schedule_interval_us=params["schedule_interval_us"],
            recency_tau=params["recency_tau"],
            cooldown_sec=params["cooldown_sec"],
            threshold_to_gpu=params["threshold_to_gpu"],
            threshold_to_dram=params["threshold_to_dram"],
        )

        for b in range(n_blocks):
            _C.tm_register_block_id(handle, b)

        correct, total = 0, 0
        gpu_demotes = 0

        for step, (obs, gt) in enumerate(zip(trace, gt_labels)):
            for block_id, weight in obs:
                _C.tm_report_attn(handle, block_id, weight)
            _C.tm_step_done(handle)
            result = _C.tm_schedule_once(handle)
            gpu_demotes += result.get("gpu_demotes", 0) if isinstance(result, dict) else 0

            if step == len(trace) - 1:
                for b in range(n_blocks):
                    score = _C.tm_get_block_score(handle, b)
                    if score is not None:
                        pred_heat = "hot" if score >= params["threshold_to_gpu"] else (
                            "cold" if score < params["threshold_to_dram"] else "warm")
                        if gt[b] == pred_heat:
                            correct += 1
                        total += 1

        acc = correct / total if total > 0 else 0.0
        results.append({
            "run": run,
            "config": name,
            "desc": cfg["desc"],
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "gamma": cfg["gamma"],
            "accuracy": round(acc, 4),
            "gpu_demotes": gpu_demotes,
        })
        print(f"  {name} run {run}: acc={acc:.4f}, demotes={gpu_demotes}")

        _C.tm_destroy(handle)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-blocks", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Signal Ablation: {args.n_blocks} blocks, {args.n_steps} steps, {args.n_runs} runs")
    trace, gt = generate_trace(args.n_blocks, args.n_steps, seed=args.seed)

    all_results = []
    for name, cfg in CONFIGS.items():
        print(f"\n--- {cfg['desc']} (α={cfg['alpha']}, β={cfg['beta']}, γ={cfg['gamma']}) ---")
        results = run_config(name, cfg, trace, gt, args.n_blocks, args.n_runs, args.seed)
        all_results.extend(results)

    summary = {}
    for name, cfg in CONFIGS.items():
        runs = [r for r in all_results if r["config"] == name]
        avg_acc = sum(r["accuracy"] for r in runs) / len(runs)
        avg_dem = sum(r["gpu_demotes"] for r in runs) / len(runs)
        summary[name] = {"desc": cfg["desc"], "avg_accuracy": round(avg_acc, 4),
                         "avg_demotes": round(avg_dem, 1)}

    print("\n=== Summary ===")
    for name, s in summary.items():
        print(f"  {s['desc']:35s}  acc={s['avg_accuracy']:.4f}  demotes={s['avg_demotes']:.0f}")

    out = {"args": vars(args), "raw": all_results, "summary": summary}
    save_json(out, RESULTS_DIR / "exp_signal_ablation.json")
    print(f"\nSaved to {RESULTS_DIR / 'exp_signal_ablation.json'}")


if __name__ == "__main__":
    main()

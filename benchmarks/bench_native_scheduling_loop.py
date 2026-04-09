#!/usr/bin/env python3
"""
P0.1: Measure the FULL C native scheduling loop latency.

Unlike the existing classifier-only microbenchmark (<40µs), this measures
the complete per-step scheduling pipeline:
  1. tm_report_attn() for active blocks (simulated attention)
  2. tm_step_done() (EMA decay)
  3. tm_set_usage() (memory pressure)
  4. tm_schedule_once() (classify + evict + prefetch + threshold adapt)

This directly answers Reviewer Q2: "how much gap disappears when Python
orchestration is removed?"

Usage:
    conda run -n orchkv env PYTHONPATH=build/bindings \
        python benchmarks/bench_native_scheduling_loop.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
import orchkv_core as _C

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_benchmark(n_blocks: int, n_steps: int, hot_frac: float = 0.25,
                  report_frac: float = 0.3, seed: int = 42) -> dict:
    """Run full scheduling loop and measure per-step latency."""
    rng = random.Random(seed)

    tm = _C.tm_create(
        tracker_cap=n_blocks,
        max_blocks=n_blocks,
        ema_lambda=0.9,
        alpha=0.7, beta=0.2, gamma=0.1,
        prefetch_budget=min(16, n_blocks // 4),
        gpu_hwm=0.9, gpu_lwm=0.7,
        dram_hwm=0.9, dram_lwm=0.7,
    )

    for bid in range(n_blocks):
        tier = 0 if rng.random() < 0.7 else 1
        _C.tm_register_block_id(tm, bid, tier=tier, flags=0)

    # Mark first 4 blocks as attention sinks
    for bid in range(min(4, n_blocks)):
        _C.tm_register_block_id(tm, bid, tier=0, flags=1)

    n_report = max(1, int(n_blocks * report_frac))
    hot_blocks = list(range(int(n_blocks * hot_frac)))

    # Warmup
    for _ in range(5):
        for bid in rng.sample(range(n_blocks), n_report):
            w = rng.uniform(0.5, 1.0) if bid in hot_blocks else rng.uniform(0.0, 0.1)
            _C.tm_report_attn(tm, bid, w)
        _C.tm_step_done(tm)
        _C.tm_set_usage(tm, 0.85, 0.5)
        _C.tm_schedule_once(tm)

    # Timed run
    step_times = []
    for step in range(n_steps):
        active = rng.sample(range(n_blocks), n_report)

        t0 = time.perf_counter_ns()

        for bid in active:
            w = rng.uniform(0.5, 1.0) if bid in hot_blocks else rng.uniform(0.0, 0.1)
            _C.tm_report_attn(tm, bid, w)
        _C.tm_step_done(tm)
        gpu_usage = 0.80 + 0.15 * (step % 10) / 10
        _C.tm_set_usage(tm, gpu_usage, 0.5)
        _C.tm_schedule_once(tm)

        t1 = time.perf_counter_ns()
        step_times.append((t1 - t0) / 1000.0)  # µs

    _C.tm_destroy(tm)

    step_times.sort()
    n = len(step_times)
    return {
        "n_blocks": n_blocks,
        "n_steps": n_steps,
        "n_report_per_step": n_report,
        "mean_us": round(sum(step_times) / n, 2),
        "median_us": round(step_times[n // 2], 2),
        "p95_us": round(step_times[int(n * 0.95)], 2),
        "p99_us": round(step_times[int(n * 0.99)], 2),
        "min_us": round(step_times[0], 2),
        "max_us": round(step_times[-1], 2),
    }


def main():
    block_counts = [64, 128, 256, 512, 1024, 2048, 4096]
    n_steps = 200

    print(f"{'='*70}")
    print(f"  Native C Scheduling Loop Benchmark")
    print(f"  Full loop: report_attn + step_done + set_usage + schedule_once")
    print(f"  Steps per config: {n_steps}")
    print(f"{'='*70}")
    print(f"  {'Blocks':>8s} {'Mean(µs)':>10s} {'Median':>10s} "
          f"{'P95':>10s} {'P99':>10s} {'Max':>10s}")
    print(f"  {'-'*58}")

    results = []
    for nb in block_counts:
        r = run_benchmark(nb, n_steps)
        results.append(r)
        print(f"  {r['n_blocks']:>8d} {r['mean_us']:>10.1f} "
              f"{r['median_us']:>10.1f} {r['p95_us']:>10.1f} "
              f"{r['p99_us']:>10.1f} {r['max_us']:>10.1f}")

    # Compare with Python scheduling loop (from paper: 9.4ms = 9400µs)
    python_loop_us = 9400
    print(f"\n  Python scheduling loop (from overhead table): {python_loop_us} µs")
    for r in results:
        speedup = python_loop_us / r["mean_us"]
        print(f"  {r['n_blocks']:>6d} blocks: C loop = {r['mean_us']:.1f} µs"
              f"  → {speedup:.0f}× faster than Python")

    out = RESULTS_DIR / "bench_native_scheduling_loop.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

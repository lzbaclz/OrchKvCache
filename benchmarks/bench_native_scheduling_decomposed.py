#!/usr/bin/env python3
"""
P0.1 (refined): Decomposed C scheduling loop latency measurement.

Separates:
  Phase A: tm_report_attn × N_active  (in native: called from CUDA kernel, zero Python overhead)
  Phase B: tm_step_done + tm_set_usage + tm_schedule_once  (the "scheduler core")

In a native integration, Phase A would be a single kernel callback (no Python FFI).
Phase B is the true scheduling overhead that persists even with native integration.

The paper claims "<40µs" for classifier-only; this measures the full Phase B.
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


def run_benchmark(n_blocks: int, n_steps: int = 200,
                  hot_frac: float = 0.25, report_frac: float = 0.3,
                  seed: int = 42) -> dict:
    rng = random.Random(seed)

    tm = _C.tm_create(
        tracker_cap=n_blocks, max_blocks=n_blocks,
        ema_lambda=0.9, alpha=0.7, beta=0.2, gamma=0.1,
        prefetch_budget=min(16, n_blocks // 4),
        gpu_hwm=0.9, gpu_lwm=0.7, dram_hwm=0.9, dram_lwm=0.7,
    )

    for bid in range(n_blocks):
        tier = 0 if rng.random() < 0.7 else 1
        _C.tm_register_block_id(tm, bid, tier=tier, flags=0)
    for bid in range(min(4, n_blocks)):
        _C.tm_register_block_id(tm, bid, tier=0, flags=1)

    n_report = max(1, int(n_blocks * report_frac))
    hot_set = set(range(int(n_blocks * hot_frac)))

    for _ in range(10):
        for bid in rng.sample(range(n_blocks), n_report):
            _C.tm_report_attn(tm, bid, rng.uniform(0.0, 1.0))
        _C.tm_step_done(tm)
        _C.tm_set_usage(tm, 0.85, 0.5)
        _C.tm_schedule_once(tm)

    report_times = []
    sched_times = []
    total_times = []

    for step in range(n_steps):
        active = rng.sample(range(n_blocks), n_report)
        gpu_usage = 0.80 + 0.15 * (step % 10) / 10

        t0 = time.perf_counter_ns()
        for bid in active:
            w = rng.uniform(0.5, 1.0) if bid in hot_set else rng.uniform(0.0, 0.1)
            _C.tm_report_attn(tm, bid, w)
        t1 = time.perf_counter_ns()

        _C.tm_step_done(tm)
        _C.tm_set_usage(tm, gpu_usage, 0.5)
        _C.tm_schedule_once(tm)
        t2 = time.perf_counter_ns()

        report_times.append((t1 - t0) / 1000.0)
        sched_times.append((t2 - t1) / 1000.0)
        total_times.append((t2 - t0) / 1000.0)

    _C.tm_destroy(tm)

    def stats(times):
        times = sorted(times)
        n = len(times)
        return {
            "mean": round(sum(times) / n, 1),
            "median": round(times[n // 2], 1),
            "p99": round(times[int(n * 0.99)], 1),
        }

    return {
        "n_blocks": n_blocks,
        "n_report": n_report,
        "phase_a_report_us": stats(report_times),
        "phase_b_sched_us": stats(sched_times),
        "total_us": stats(total_times),
    }


def main():
    block_counts = [64, 128, 256, 512, 1024, 2048, 4096]
    print(f"{'='*80}")
    print(f"  Decomposed C Scheduling Loop Benchmark")
    print(f"  Phase A = report_attn × N (Python→C FFI per call)")
    print(f"  Phase B = step_done + set_usage + schedule_once (scheduler core)")
    print(f"  In native integration: Phase A = 0 (kernel callback), Phase B = real cost")
    print(f"{'='*80}")
    print(f"  {'Blocks':>6s} {'Report':>6s}  "
          f"{'A:mean':>8s} {'B:mean':>8s} {'B:p99':>8s}  "
          f"{'Total':>8s}  {'B vs 9.4ms':>10s}")
    print(f"  {'-'*70}")

    results = []
    python_sched_us = 9400

    for nb in block_counts:
        r = run_benchmark(nb)
        results.append(r)
        b_mean = r["phase_b_sched_us"]["mean"]
        speedup = python_sched_us / b_mean if b_mean > 0 else 0
        print(f"  {r['n_blocks']:>6d} {r['n_report']:>6d}  "
              f"{r['phase_a_report_us']['mean']:>7.1f}µ "
              f"{b_mean:>7.1f}µ "
              f"{r['phase_b_sched_us']['p99']:>7.1f}µ  "
              f"{r['total_us']['mean']:>7.1f}µ  "
              f"{speedup:>8.0f}×")

    # Key result for paper
    print(f"\n  === Key result for paper ===")
    for r in results:
        if r["n_blocks"] in [256, 512, 1024]:
            b = r["phase_b_sched_us"]
            tokens_approx = r["n_blocks"] * 16
            print(f"  {r['n_blocks']} blocks ({tokens_approx:,} tokens): "
                  f"Phase B = {b['mean']:.1f} µs (p99: {b['p99']:.1f} µs)")

    out = RESULTS_DIR / "bench_native_scheduling_decomposed.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

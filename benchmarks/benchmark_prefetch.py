#!/usr/bin/env python3
"""
D4 / E7: Prefetch effectiveness benchmark.

Measures prefetch hit rate and latency hiding ratio across
different prefetch budgets and workloads.

Uses orchkv_core directly — no vLLM dependency required.

Usage:
    python benchmarks/benchmark_prefetch.py
    python benchmarks/benchmark_prefetch.py --n-blocks 512 --n-steps 200
"""
from __future__ import annotations

import argparse
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, save_csv, CPUTimer

try:
    import orchkv_core as _C
except ImportError:
    _C = None


def run_prefetch_sweep(
    n_blocks: int = 256,
    n_steps: int = 100,
    budgets: list[int] | None = None,
) -> list[dict]:
    """
    E7: Sweep prefetch_budget and measure hit rate / scheduling overhead.

    Simulates a decode workload where a subset of blocks have high attention
    scores (hot) and the rest are cold. Measures how well the prefetch
    scheduler predicts upcoming access patterns.
    """
    if _C is None:
        print("[SKIP] orchkv_core not available")
        return [{"status": "skip"}]

    if budgets is None:
        budgets = [0, 4, 8, 16, 32]

    n_hot = n_blocks // 4
    results = []

    for budget in budgets:
        print(f"  [budget={budget:3d}] ", end="", flush=True)

        tm = _C.tm_create(
            tracker_cap=n_blocks * 2,
            prefetch_budget=budget,
            schedule_interval_us=500,
        )

        timer = CPUTimer()

        for step in range(n_steps):
            hot_set = set(random.sample(range(n_blocks), n_hot))
            for bid in range(n_blocks):
                w = 0.9 if bid in hot_set else 0.05
                _C.tm_report_attn(tm, bid, w)
            _C.tm_step_done(tm)

            if step % 5 == 0:
                _C.tm_set_usage(tm, gpu_ratio=0.82, dram_ratio=0.55)
                timer.start()
                _C.tm_schedule_once(tm)
                timer.stop()

        stats = _C.tm_get_stats(tm)
        sched_timing = timer.stats()

        r = {
            "prefetch_budget": budget,
            "n_blocks": n_blocks,
            "n_hot": n_hot,
            "n_steps": n_steps,
            "schedule_cycles": stats["schedule_cycles"],
            "prefetches_dispatched": stats["prefetches_dispatched"],
            "prefetch_hits": stats["prefetch_hits"],
            "prefetch_wasted": stats["prefetch_wasted"],
            "prefetch_hit_rate": stats["prefetch_hit_rate"],
            "gpu_demotes": stats["gpu_demotes"],
            "dram_demotes": stats["dram_demotes"],
            "blocks_migrated": stats["blocks_migrated"],
            "avg_schedule_us": round(sched_timing.get("avg_us", 0), 2),
            "p99_schedule_us": round(sched_timing.get("p99_us", 0), 2),
        }
        results.append(r)
        _C.tm_destroy(tm)

        print(f"dispatched={r['prefetches_dispatched']:4d}  "
              f"hits={r['prefetch_hits']:4d}  "
              f"rate={r['prefetch_hit_rate']:.3f}  "
              f"sched_avg={r['avg_schedule_us']:.1f}μs")

    return results


def main():
    parser = argparse.ArgumentParser(description="E7 Prefetch Benchmark")
    parser.add_argument("--n-blocks", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--budgets", default="0,4,8,16,32")
    args = parser.parse_args()

    budgets = list(map(int, args.budgets.split(",")))

    print(f"\n{'='*60}")
    print(f"  E7: Prefetch Effectiveness (blocks={args.n_blocks}, "
          f"steps={args.n_steps})")
    print(f"{'='*60}\n")

    results = run_prefetch_sweep(args.n_blocks, args.n_steps, budgets)

    save_json(results, "benchmark_e7_prefetch")
    ok = [r for r in results if r.get("status") != "skip"]
    if ok:
        save_csv(ok, "benchmark_e7_prefetch")

    print(f"\nDone. {len(ok)} configs tested.")


if __name__ == "__main__":
    main()

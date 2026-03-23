#!/usr/bin/env python3
"""
E7: Prefetch effectiveness benchmark.

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
    n_runs: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    E7: Sweep prefetch_budget and measure hit rate / scheduling overhead.

    Simulates a decode workload where a subset of blocks have high attention
    scores (hot) and the rest are cold. Blocks start on mixed tiers to
    enable the prefetch scheduler to select promotion candidates.
    """
    if _C is None:
        print("[SKIP] orchkv_core not available")
        return [{"status": "skip"}]

    if budgets is None:
        budgets = [0, 2, 4, 8, 16, 32]

    n_hot = n_blocks // 4
    n_on_dram = n_blocks // 2
    results = []

    for budget in budgets:
        run_data = []

        for run_id in range(n_runs):
            random.seed(seed + run_id)

            tm = _C.tm_create(
                tracker_cap=n_blocks * 2,
                prefetch_budget=budget,
                schedule_interval_us=500,
                gpu_hwm=0.85, gpu_lwm=0.60,
                dram_hwm=0.85, dram_lwm=0.60,
                max_blocks=n_blocks + 64,
                threshold_to_gpu=0.3,
                threshold_to_dram=0.1,
            )

            for bid in range(n_blocks):
                tier = int(_C.HOST_DRAM) if bid >= n_on_dram else int(_C.GPU_HBM)
                _C.tm_register_block_id(tm, bid, tier, 0)

            timer = CPUTimer()

            for step in range(n_steps):
                hot_set = set(random.sample(range(n_blocks), n_hot))
                warm_pool = [b for b in range(n_blocks) if b not in hot_set]
                warm_set = set(random.sample(warm_pool,
                               min(n_blocks // 8, len(warm_pool))))
                for bid in hot_set:
                    _C.tm_report_attn(tm, bid, 0.85)
                for bid in warm_set:
                    _C.tm_report_attn(tm, bid, 0.15)
                _C.tm_step_done(tm)

                if step % 3 == 0:
                    _C.tm_set_usage(tm, gpu_ratio=0.75, dram_ratio=0.50)
                    timer.start()
                    _C.tm_schedule_once(tm)
                    timer.stop()

            stats = _C.tm_get_stats(tm)
            sched_timing = timer.stats()

            run_data.append({
                "prefetches_dispatched": stats["prefetches_dispatched"],
                "prefetch_hits": stats["prefetch_hits"],
                "prefetch_wasted": stats["prefetch_wasted"],
                "prefetch_hit_rate": stats["prefetch_hit_rate"],
                "gpu_demotes": stats["gpu_demotes"],
                "dram_demotes": stats["dram_demotes"],
                "blocks_migrated": stats["blocks_migrated"],
                "n_hot": stats["n_hot"],
                "n_warm": stats["n_warm"],
                "n_cold": stats["n_cold"],
                "avg_schedule_us": round(sched_timing.get("avg_us", 0), 2),
                "p99_schedule_us": round(sched_timing.get("p99_us", 0), 2),
            })
            _C.tm_destroy(tm)

        def avg(key):
            return sum(r[key] for r in run_data) / max(len(run_data), 1)

        r = {
            "prefetch_budget": budget,
            "n_blocks": n_blocks,
            "n_hot_ground_truth": n_hot,
            "n_on_dram": n_on_dram,
            "n_steps": n_steps,
            "n_runs": n_runs,
            "avg_prefetches_dispatched": round(avg("prefetches_dispatched"), 1),
            "avg_prefetch_hits": round(avg("prefetch_hits"), 1),
            "avg_prefetch_wasted": round(avg("prefetch_wasted"), 1),
            "avg_prefetch_hit_rate": round(avg("prefetch_hit_rate"), 4),
            "avg_gpu_demotes": round(avg("gpu_demotes"), 1),
            "avg_n_hot": round(avg("n_hot"), 1),
            "avg_n_warm": round(avg("n_warm"), 1),
            "avg_n_cold": round(avg("n_cold"), 1),
            "avg_schedule_us": round(avg("avg_schedule_us"), 2),
            "p99_schedule_us": round(max(r["p99_schedule_us"] for r in run_data), 2),
        }
        results.append(r)

        print(f"  [budget={budget:3d}] "
              f"dispatched={r['avg_prefetches_dispatched']:6.1f}  "
              f"hits={r['avg_prefetch_hits']:6.1f}  "
              f"rate={r['avg_prefetch_hit_rate']:.4f}  "
              f"hot={r['avg_n_hot']:5.1f} "
              f"warm={r['avg_n_warm']:5.1f} "
              f"cold={r['avg_n_cold']:5.1f}  "
              f"sched_avg={r['avg_schedule_us']:.1f}μs")

    return results


def main():
    parser = argparse.ArgumentParser(description="E7 Prefetch Benchmark")
    parser.add_argument("--n-blocks", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--budgets", default="0,2,4,8,16,32")
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    budgets = list(map(int, args.budgets.split(",")))

    print(f"\n{'='*60}")
    print(f"  E7: Prefetch Effectiveness (blocks={args.n_blocks}, "
          f"steps={args.n_steps})")
    print(f"{'='*60}\n")

    results = run_prefetch_sweep(args.n_blocks, args.n_steps, budgets, args.n_runs)

    save_json(results, "benchmark_e7_prefetch")
    ok = [r for r in results if r.get("status") != "skip"]
    if ok:
        save_csv(ok, "benchmark_e7_prefetch")

    print(f"\nDone. {len(ok)} budget configs × {args.n_runs} runs each.")


if __name__ == "__main__":
    main()

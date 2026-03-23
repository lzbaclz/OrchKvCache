#!/usr/bin/env python3
"""
E9: Scalability benchmark — scheduling latency vs. block count.

Measures how OrchKvCache's tiered_manager scales with increasing
number of concurrent blocks.

Uses orchkv_core directly — no vLLM dependency required.

Usage:
    python benchmarks/benchmark_scalability.py
    python benchmarks/benchmark_scalability.py --max-blocks 8192 --n-runs 5
"""
from __future__ import annotations

import argparse
import sys
import os
import random
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, save_csv, CPUTimer

try:
    import orchkv_core as _C
except ImportError:
    _C = None


def run_scalability(
    block_counts: list[int] | None = None,
    n_steps: int = 50,
    n_runs: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    E9: Measure scheduling latency as block count increases.

    For each block_count, registers blocks, runs attention + scheduling,
    and measures avg/p50/p99 schedule latency across multiple runs.
    """
    if _C is None:
        print("[SKIP] orchkv_core not available")
        return [{"status": "skip"}]

    if block_counts is None:
        block_counts = [64, 128, 256, 512, 1024, 2048, 4096]

    results = []
    for n_blocks in block_counts:
        print(f"  [n_blocks={n_blocks:5d}] ", end="", flush=True)

        all_avgs = []
        all_p50s = []
        all_p99s = []
        all_maxs = []
        all_demotes = []
        all_prefetches = []

        for run_id in range(n_runs):
            random.seed(seed + run_id)

            tm = _C.tm_create(
                tracker_cap=n_blocks * 2,
                prefetch_budget=min(16, max(1, n_blocks // 16)),
                schedule_interval_us=1000,
                gpu_hwm=0.85, gpu_lwm=0.65,
                dram_hwm=0.85, dram_lwm=0.65,
                max_blocks=n_blocks + 64,
                threshold_to_gpu=0.4,
                threshold_to_dram=0.15,
            )

            n_on_gpu = n_blocks * 3 // 4
            for bid in range(n_blocks):
                tier = int(_C.GPU_HBM) if bid < n_on_gpu else int(_C.HOST_DRAM)
                _C.tm_register_block_id(tm, bid, tier, 0)

            n_hot = max(1, n_blocks // 8)
            timer = CPUTimer()

            for step in range(n_steps):
                hot_ids = set(random.sample(range(n_blocks), n_hot))
                n_warm = max(1, n_blocks // 10)
                warm_pool = [b for b in range(n_blocks) if b not in hot_ids]
                warm_ids = set(random.sample(warm_pool,
                               min(n_warm, len(warm_pool))))
                for bid in hot_ids:
                    _C.tm_report_attn(tm, bid, 0.8)
                for bid in warm_ids:
                    _C.tm_report_attn(tm, bid, 0.12)
                _C.tm_step_done(tm)

                _C.tm_set_usage(tm, gpu_ratio=0.88, dram_ratio=0.60)
                timer.start()
                _C.tm_schedule_once(tm)
                timer.stop()

            stats = _C.tm_get_stats(tm)
            timing = timer.stats()

            all_avgs.append(timing.get("avg_us", 0))
            all_p50s.append(timing.get("p50_us", 0))
            all_p99s.append(timing.get("p99_us", 0))
            all_maxs.append(timing.get("max_us", 0))
            all_demotes.append(stats["gpu_demotes"] + stats["dram_demotes"])
            all_prefetches.append(stats["prefetches_dispatched"])

            _C.tm_destroy(tm)

        r = {
            "n_blocks": n_blocks,
            "n_steps": n_steps,
            "n_runs": n_runs,
            "avg_schedule_us": round(statistics.mean(all_avgs), 2),
            "p50_schedule_us": round(statistics.mean(all_p50s), 2),
            "p99_schedule_us": round(statistics.mean(all_p99s), 2),
            "max_schedule_us": round(max(all_maxs), 2),
            "std_schedule_us": round(statistics.stdev(all_avgs), 2) if n_runs > 1 else 0,
            "avg_demotes": round(statistics.mean(all_demotes), 1),
            "avg_prefetches": round(statistics.mean(all_prefetches), 1),
        }
        results.append(r)

        print(f"avg={r['avg_schedule_us']:8.1f}±{r['std_schedule_us']:.1f}μs  "
              f"p99={r['p99_schedule_us']:8.1f}μs  "
              f"demotes={r['avg_demotes']:.0f}  "
              f"prefetches={r['avg_prefetches']:.0f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="E9 Scalability Benchmark")
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    counts = [2**i for i in range(6, 13) if 2**i <= args.max_blocks]

    print(f"\n{'='*60}")
    print(f"  E9: Scalability Benchmark (max_blocks={args.max_blocks}, "
          f"n_runs={args.n_runs})")
    print(f"{'='*60}\n")

    results = run_scalability(counts, args.n_steps, args.n_runs)
    save_json(results, "benchmark_e9_scalability")
    ok = [r for r in results if r.get("status") != "skip"]
    if ok:
        save_csv(ok, "benchmark_e9_scalability")

    print(f"\nDone. {len(ok)} configs × {args.n_runs} runs each.")


if __name__ == "__main__":
    main()

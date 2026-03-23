#!/usr/bin/env python3
"""
D4 / E9: Scalability benchmark — multi-request concurrent scheduling.

Measures how OrchKvCache's tiered_manager scales with increasing
number of concurrent blocks and requests.

Uses orchkv_core directly — no vLLM dependency required.

Usage:
    python benchmarks/benchmark_scalability.py
    python benchmarks/benchmark_scalability.py --max-blocks 4096
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


def run_scalability(
    block_counts: list[int] | None = None,
    n_steps: int = 50,
) -> list[dict]:
    """
    E9: Measure scheduling latency as block count increases.

    For each block_count:
      1. Create a tiered_manager with that many tracked blocks
      2. Run n_steps of attention reporting + scheduling
      3. Measure avg/p50/p99 schedule latency
    """
    if _C is None:
        print("[SKIP] orchkv_core not available")
        return [{"status": "skip"}]

    if block_counts is None:
        block_counts = [64, 128, 256, 512, 1024, 2048, 4096]

    results = []
    for n_blocks in block_counts:
        print(f"  [n_blocks={n_blocks:5d}] ", end="", flush=True)

        tm = _C.tm_create(
            tracker_cap=n_blocks * 2,
            prefetch_budget=min(16, n_blocks // 4),
            schedule_interval_us=1000,
        )

        n_hot = max(1, n_blocks // 8)
        timer = CPUTimer()

        for step in range(n_steps):
            hot_ids = random.sample(range(n_blocks), n_hot)
            for bid in range(n_blocks):
                w = 0.8 if bid in hot_ids else 0.05
                _C.tm_report_attn(tm, bid, w)
            _C.tm_step_done(tm)

            _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.6)
            timer.start()
            _C.tm_schedule_once(tm)
            timer.stop()

        stats = _C.tm_get_stats(tm)
        timing = timer.stats()

        r = {
            "n_blocks": n_blocks,
            "n_steps": n_steps,
            "avg_schedule_us": round(timing.get("avg_us", 0), 2),
            "p50_schedule_us": round(timing.get("p50_us", 0), 2),
            "p99_schedule_us": round(timing.get("p99_us", 0), 2),
            "max_schedule_us": round(timing.get("max_us", 0), 2),
            "schedule_cycles": stats["schedule_cycles"],
            "blocks_migrated": stats["blocks_migrated"],
            "gpu_demotes": stats["gpu_demotes"],
            "dram_demotes": stats["dram_demotes"],
            "prefetches": stats["prefetches_dispatched"],
        }
        results.append(r)
        _C.tm_destroy(tm)

        print(f"avg={r['avg_schedule_us']:8.1f}μs  "
              f"p99={r['p99_schedule_us']:8.1f}μs  "
              f"migrated={r['blocks_migrated']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="E9 Scalability Benchmark")
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=50)
    args = parser.parse_args()

    counts = [2**i for i in range(6, 13) if 2**i <= args.max_blocks]

    print(f"\n{'='*60}")
    print(f"  E9: Scalability Benchmark (max_blocks={args.max_blocks})")
    print(f"{'='*60}\n")

    results = run_scalability(counts, args.n_steps)
    save_json(results, "benchmark_e9_scalability")
    ok = [r for r in results if r.get("status") != "skip"]
    if ok:
        save_csv(ok, "benchmark_e9_scalability")

    print(f"\nDone. {len(ok)} block-count configs tested.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
E4-E6: Ablation benchmarks.

E4: Storage tier ablation   (GPU-only → full 4-tier)    — requires vLLM
E5: Hot/cold policy sweep   (α, β, γ parameter grid)    — orchkv_core only
E6: Block size ablation     (16, 32, 64, 128 tok/blk)   — requires vLLM

Usage:
    python benchmarks/benchmark_ablation.py --exp e5
    python benchmarks/benchmark_ablation.py --exp e5 --n-blocks 512 --n-steps 200
"""
from __future__ import annotations

import argparse
import itertools
import sys
import os
import time
import random

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import (
    save_json, save_csv, CPUTimer, gpu_mem_mb, reset_gpu_peak,
    generate_synthetic_prompts, build_vllm_engine,
)

try:
    import orchkv_core as _C
except ImportError:
    _C = None


# ─── E4: Storage Tier Ablation ───────────────────────────────────────

TIER_CONFIGS = [
    {"name": "GPU-only",             "tiers": 1, "dram_pool_gb": 0},
    {"name": "GPU+DRAM",             "tiers": 2, "dram_pool_gb": 8},
    {"name": "GPU+DRAM+NVM",         "tiers": 3, "dram_pool_gb": 8},
    {"name": "GPU+DRAM+NVM+SSD",     "tiers": 4, "dram_pool_gb": 8},
]


def run_e4(model: str, seq_len: int = 4096, batch_size: int = 4,
           max_new_tokens: int = 64) -> list[dict]:
    """E4: Compare throughput across tier configurations."""
    print(f"\n{'='*60}")
    print(f"  E4: Storage Tier Ablation (seq={seq_len}, bs={batch_size})")
    print(f"{'='*60}")

    results = []

    for tcfg in TIER_CONFIGS:
        label = tcfg["name"]
        print(f"\n  [{label}] ...", end=" ", flush=True)

        orchkv_on = tcfg["tiers"] > 1

        try:
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens)
            prompts = generate_synthetic_prompts(batch_size, seq_len)
            engine = build_vllm_engine(
                model, orchkv_enabled=orchkv_on,
                dram_pool_gb=tcfg["dram_pool_gb"],
                max_model_len=seq_len + max_new_tokens)

            if engine is None:
                print("SKIP")
                results.append({"tier": label, "status": "skip"})
                continue

            engine.generate(prompt_token_ids=prompts, sampling_params=sampling)

            timer = CPUTimer()
            reset_gpu_peak()
            for _ in range(3):
                timer.start()
                outputs = engine.generate(
                    prompt_token_ids=prompts, sampling_params=sampling)
                timer.stop()

            n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
            stats = timer.stats()
            avg_ms = stats.get("avg_us", 0) / 1000
            total_tok = batch_size * seq_len + n_out

            r = {
                "tier": label,
                "n_tiers": tcfg["tiers"],
                "avg_ms": round(avg_ms, 2),
                "throughput_tok_s": round(
                    total_tok / (avg_ms / 1000), 1) if avg_ms > 0 else 0,
                "gpu_peak_mb": round(gpu_mem_mb()["max_allocated_mb"], 1),
                "status": "ok",
            }
            results.append(r)
            print(f"OK  {r['throughput_tok_s']} tok/s")

            del engine
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"tier": label, "status": "error", "error": str(e)})
            torch.cuda.empty_cache()

    save_json(results, "benchmark_e4_tier_ablation")
    return results


# ─── E5: Hot/Cold Policy Sweep ───────────────────────────────────────

def _generate_attention_patterns(n_blocks: int, n_hot: int, step: int,
                                  pattern: str = "fixed") -> list[tuple[int, float]]:
    """
    Generate (block_id, attn_weight) pairs for blocks accessed this step.

    Only returns blocks that are actually "attended to" — cold blocks
    are omitted so their recency/frequency naturally decay, producing
    realistic three-level classification.
    """
    if pattern == "fixed":
        result = [(bid, 0.85) for bid in range(n_hot)]
        n_luke = n_blocks // 8
        for bid in range(n_hot, n_hot + n_luke):
            result.append((bid, 0.15 + random.random() * 0.1))
        return result
    elif pattern == "shift":
        offset = (step // 20) * (n_hot // 2) % n_blocks
        result = []
        for i in range(n_hot):
            bid = (offset + i) % n_blocks
            result.append((bid, 0.85))
        n_luke = n_blocks // 8
        for i in range(n_luke):
            bid = (offset + n_hot + i) % n_blocks
            result.append((bid, 0.15 + random.random() * 0.1))
        return result
    elif pattern == "random":
        hot_set = random.sample(range(n_blocks), n_hot)
        result = [(bid, 0.85) for bid in hot_set]
        warm_set = random.sample(
            [b for b in range(n_blocks) if b not in hot_set],
            min(n_blocks // 8, n_blocks - n_hot))
        for bid in warm_set:
            result.append((bid, 0.15 + random.random() * 0.1))
        return result
    elif pattern == "zipf":
        result = []
        for bid in range(n_blocks):
            w = 1.0 / (bid + 1) ** 1.2
            if w > 0.02:
                result.append((bid, min(w, 1.0)))
        return result
    return [(bid, 0.5) for bid in range(n_hot)]


def run_e5(n_blocks: int = 256, n_steps: int = 100,
           n_runs: int = 3, seed: int = 42) -> list[dict]:
    """
    E5: Sweep α/β/γ weights and measure scheduling behavior.

    Registers blocks with the classifier and tracker for realistic stats.
    """
    print(f"\n{'='*60}")
    print(f"  E5: Hot/Cold Policy Sweep (n_blocks={n_blocks}, steps={n_steps})")
    print(f"{'='*60}")

    if _C is None:
        print("[SKIP] orchkv_core not available")
        return [{"status": "skip"}]

    param_grid = [
        (0.2, 0.5, 0.3),
        (0.3, 0.4, 0.3),
        (0.4, 0.4, 0.2),
        (0.5, 0.3, 0.2),
        (0.5, 0.1, 0.4),
        (0.6, 0.2, 0.2),
        (0.6, 0.3, 0.1),
        (0.7, 0.2, 0.1),
        (0.8, 0.1, 0.1),
    ]

    n_hot = n_blocks // 4
    results = []
    patterns = ["fixed", "shift", "zipf"]

    for alpha, beta, gamma in param_grid:
        for pattern in patterns:
            run_results = []

            for run_id in range(n_runs):
                random.seed(seed + run_id)

                tm = _C.tm_create(
                    tracker_cap=n_blocks * 2,
                    alpha=alpha, beta=beta, gamma=gamma,
                    prefetch_budget=16,
                    schedule_interval_us=500,
                    gpu_hwm=0.80, gpu_lwm=0.60,
                    dram_hwm=0.80, dram_lwm=0.60,
                    max_blocks=n_blocks + 64,
                    threshold_to_gpu=0.4,
                    threshold_to_dram=0.15,
                )

                for bid in range(n_blocks):
                    _C.tm_register_block_id(tm, bid, int(_C.GPU_HBM), 0)

                step_trace = []

                for step in range(n_steps):
                    attn_pairs = _generate_attention_patterns(
                        n_blocks, n_hot, step, pattern)
                    for bid, w in attn_pairs:
                        _C.tm_report_attn(tm, bid, w)
                    _C.tm_step_done(tm)

                    if step % 5 == 0:
                        _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.55)
                        _C.tm_schedule_once(tm)

                    if step % 10 == 0:
                        s = _C.tm_get_stats(tm)
                        step_trace.append({
                            "step": step,
                            "n_hot": s["n_hot"],
                            "n_warm": s["n_warm"],
                            "n_cold": s["n_cold"],
                        })

                stats = _C.tm_get_stats(tm)
                run_results.append({
                    "n_hot": stats["n_hot"],
                    "n_warm": stats["n_warm"],
                    "n_cold": stats["n_cold"],
                    "gpu_demotes": stats["gpu_demotes"],
                    "dram_demotes": stats["dram_demotes"],
                    "prefetches": stats["prefetches_dispatched"],
                    "schedule_cycles": stats["schedule_cycles"],
                })
                _C.tm_destroy(tm)

            avg_hot = sum(r["n_hot"] for r in run_results) / n_runs
            avg_warm = sum(r["n_warm"] for r in run_results) / n_runs
            avg_cold = sum(r["n_cold"] for r in run_results) / n_runs
            avg_demotes = sum(r["gpu_demotes"] + r["dram_demotes"]
                              for r in run_results) / n_runs

            r = {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "pattern": pattern,
                "n_blocks": n_blocks,
                "n_steps": n_steps,
                "n_runs": n_runs,
                "avg_n_hot": round(avg_hot, 1),
                "avg_n_warm": round(avg_warm, 1),
                "avg_n_cold": round(avg_cold, 1),
                "avg_demotes": round(avg_demotes, 1),
                "avg_prefetches": round(
                    sum(r["prefetches"] for r in run_results) / n_runs, 1),
                "hot_ratio": round(avg_hot / max(n_blocks, 1), 4),
                "warm_ratio": round(avg_warm / max(n_blocks, 1), 4),
                "cold_ratio": round(avg_cold / max(n_blocks, 1), 4),
            }
            results.append(r)

            print(f"  α={alpha:.1f} β={beta:.1f} γ={gamma:.1f} "
                  f"[{pattern:5s}]  "
                  f"hot={r['avg_n_hot']:5.1f}  "
                  f"warm={r['avg_n_warm']:5.1f}  "
                  f"cold={r['avg_n_cold']:5.1f}  "
                  f"demotes={r['avg_demotes']:.1f}")

    save_json(results, "benchmark_e5_policy_sweep")
    if results:
        save_csv(results, "benchmark_e5_policy_sweep")
    return results


# ─── E6: Block Size Ablation ────────────────────────────────────────

def run_e6(model: str = "Qwen/Qwen2.5-7B",
           seq_len: int = 4096, batch_size: int = 4,
           max_new_tokens: int = 64) -> list[dict]:
    """E6: Compare throughput across block sizes."""
    print(f"\n{'='*60}")
    print(f"  E6: Block Size Ablation (seq={seq_len}, bs={batch_size})")
    print(f"{'='*60}")

    block_sizes = [16, 32, 64, 128]
    results = []

    for blk in block_sizes:
        print(f"  [block_size={blk}] ...", end=" ", flush=True)
        try:
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens)
            prompts = generate_synthetic_prompts(batch_size, seq_len)
            engine = build_vllm_engine(
                model, orchkv_enabled=True, block_size=blk,
                max_model_len=seq_len + max_new_tokens)

            if engine is None:
                print("SKIP")
                results.append({"block_size": blk, "status": "skip"})
                continue

            engine.generate(prompt_token_ids=prompts, sampling_params=sampling)

            timer = CPUTimer()
            for _ in range(3):
                timer.start()
                outputs = engine.generate(
                    prompt_token_ids=prompts, sampling_params=sampling)
                timer.stop()

            n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
            stats = timer.stats()
            avg_ms = stats.get("avg_us", 0) / 1000
            total_tok = batch_size * seq_len + n_out

            r = {
                "block_size": blk,
                "avg_ms": round(avg_ms, 2),
                "throughput_tok_s": round(
                    total_tok / (avg_ms / 1000), 1) if avg_ms > 0 else 0,
                "status": "ok",
            }
            results.append(r)
            print(f"OK  {r['throughput_tok_s']} tok/s")

            del engine
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"block_size": blk, "status": "error", "error": str(e)})
            torch.cuda.empty_cache()

    save_json(results, "benchmark_e6_block_size")
    return results


def main():
    parser = argparse.ArgumentParser(description="E4-E6 Ablation Benchmarks")
    parser.add_argument("--exp", default="all",
                        choices=["e4", "e5", "e6", "all"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-blocks", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    if args.exp in ("e4", "all"):
        run_e4(args.model, args.seq_len, args.batch_size)
    if args.exp in ("e5", "all"):
        run_e5(args.n_blocks, args.n_steps, args.n_runs)
    if args.exp in ("e6", "all"):
        run_e6(args.model, args.seq_len, args.batch_size)


if __name__ == "__main__":
    main()

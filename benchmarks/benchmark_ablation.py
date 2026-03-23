#!/usr/bin/env python3
"""
D4 / E4-E6: Ablation benchmarks.

E4: Storage tier ablation   (GPU-only, GPU+DRAM, GPU+DRAM+NVM, full 4-tier)
E5: Hot/cold policy sweep   (α, β, γ parameter grid)
E6: Block size ablation     (16, 32, 64, 128 tokens per block)

All experiments output JSON to benchmarks/results/.

Usage:
    python benchmarks/benchmark_ablation.py --exp e4
    python benchmarks/benchmark_ablation.py --exp e5
    python benchmarks/benchmark_ablation.py --exp e6
    python benchmarks/benchmark_ablation.py --exp all
"""
from __future__ import annotations

import argparse
import itertools
import sys
import os
import time

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

        if tcfg["tiers"] == 1:
            orchkv_on = False
        else:
            orchkv_on = True

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

def run_e5(n_blocks: int = 256, n_steps: int = 100) -> list[dict]:
    """
    E5: Sweep α/β/γ weights and measure scheduling behavior.

    Uses orchkv_core directly (no vLLM needed) for fast parameter sweep.
    """
    print(f"\n{'='*60}")
    print(f"  E5: Hot/Cold Policy Sweep (n_blocks={n_blocks}, steps={n_steps})")
    print(f"{'='*60}")

    if _C is None:
        print("[SKIP] orchkv_core not available")
        return [{"status": "skip"}]

    alphas = [0.2, 0.5, 0.8]
    betas = [0.1, 0.3, 0.5]
    gammas = [0.1, 0.2, 0.4]

    results = []
    for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
        s = alpha + beta + gamma
        if abs(s - 1.0) > 0.01:
            continue

        tm = _C.tm_create(
            tracker_cap=n_blocks * 2,
            alpha=alpha, beta=beta, gamma=gamma,
            prefetch_budget=16,
            schedule_interval_us=500,
        )

        for step in range(n_steps):
            for bid in range(n_blocks):
                weight = 0.9 if bid < n_blocks // 4 else 0.1
                _C.tm_report_attn(tm, bid, weight)
            _C.tm_step_done(tm)

            if step % 10 == 0:
                _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.6)
                _C.tm_schedule_once(tm)

        stats = _C.tm_get_stats(tm)
        r = {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "schedule_cycles": stats["schedule_cycles"],
            "gpu_demotes": stats["gpu_demotes"],
            "dram_demotes": stats["dram_demotes"],
            "prefetches_dispatched": stats["prefetches_dispatched"],
            "n_hot": stats["n_hot"],
            "n_warm": stats["n_warm"],
            "n_cold": stats["n_cold"],
        }
        results.append(r)
        _C.tm_destroy(tm)

        print(f"  α={alpha:.1f} β={beta:.1f} γ={gamma:.1f}  "
              f"hot={r['n_hot']} warm={r['n_warm']} cold={r['n_cold']}  "
              f"demotes={r['gpu_demotes']+r['dram_demotes']}")

    save_json(results, "benchmark_e5_policy_sweep")
    if results:
        save_csv(results, "benchmark_e5_policy_sweep")
    return results


# ─── E6: Block Size Ablation ────────────────────────────────────────

def run_e6(model: str = "meta-llama/Llama-2-7b-hf",
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
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.exp in ("e4", "all"):
        run_e4(args.model, args.seq_len, args.batch_size)
    if args.exp in ("e5", "all"):
        run_e5()
    if args.exp in ("e6", "all"):
        run_e6(args.model, args.seq_len, args.batch_size)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark: Block-Level Partial Swap vs Request-Level FIFO in vLLM.

Compares three modes under GPU memory pressure:
  1. FIFO (vLLM default) — request-level preemption
  2. OrchKv-request — attention-aware request-level victim selection
  3. OrchKv-block   — block-level partial swap (cold-block eviction)

Usage:
    conda run -n orchkv python benchmarks/exp_vllm_partial_swap.py \
        --model Qwen/Qwen2.5-7B \
        --gpu-util 0.30 \
        --num-prompts 32 \
        --swap-space 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("exp_partial_swap")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_prompts(num_prompts: int, prompt_len: int = 512) -> list[str]:
    """Generate synthetic prompts of approximately `prompt_len` tokens."""
    base = (
        "Explain the concept of attention mechanisms in transformer models "
        "and how they relate to memory management in large language model "
        "inference systems. Discuss the trade-offs between different "
        "approaches to KV cache management. "
    )
    repeated = base * max(1, prompt_len // len(base.split()))
    return [repeated] * num_prompts


def run_vllm(
    model: str,
    prompts: list[str],
    mode: str,
    gpu_util: float,
    swap_space: int,
    max_tokens: int,
    max_model_len: int,
) -> dict:
    """Run vLLM inference with the specified swap mode and return metrics."""
    if mode == "orchkv-block":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ["ORCHKV_PARTIAL_SWAP"] = "1"
    elif mode == "orchkv-request":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ.pop("ORCHKV_PARTIAL_SWAP", None)
    else:
        os.environ.pop("ORCHKV_SWAP", None)
        os.environ.pop("ORCHKV_PARTIAL_SWAP", None)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_util,
        swap_space=swap_space,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="float16",
        trust_remote_code=True,
    )

    partial_mgr = None
    if mode == "orchkv-block":
        try:
            from orchkv.vllm_integration.engine_patch import (
                install_block_level_swap,
            )
            scheduler = llm.llm_engine.scheduler[0]
            partial_mgr = install_block_level_swap(
                scheduler=scheduler,
                gpu_hwm=0.90,
                max_evict_frac=0.25,
                ema_lambda=0.3,
                sink_blocks=1,
            )
            logger.info("Block-level partial swap installed successfully")
        except Exception as e:
            logger.error("Failed to install partial swap: %s", e)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    logger.info("Starting %s inference (%d prompts, gpu_util=%.2f)",
                mode, len(prompts), gpu_util)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()

    total_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    elapsed = t1 - t0
    throughput = total_generated / elapsed if elapsed > 0 else 0

    result = {
        "mode": mode,
        "model": model,
        "gpu_util": gpu_util,
        "swap_space_gb": swap_space,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "total_tokens_generated": total_generated,
        "elapsed_s": round(elapsed, 3),
        "throughput_tok_s": round(throughput, 2),
    }

    if partial_mgr is not None:
        result["partial_swap_stats"] = partial_mgr.get_stats()
        logger.info("Partial swap stats: %s", partial_mgr.get_stats())

    scheduler = llm.llm_engine.scheduler[0]
    result["num_preemptions"] = scheduler.num_cumulative_preemption

    del llm
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Block-Level Partial Swap Benchmark")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-7B")
    parser.add_argument("--gpu-util", type=float, nargs="+",
                        default=[0.25, 0.30, 0.40])
    parser.add_argument("--num-prompts", type=int, nargs="+",
                        default=[8, 16, 32])
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--swap-space", type=int, default=32)
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["fifo", "orchkv-request", "orchkv-block"])
    parser.add_argument("--output", type=str,
                        default=str(RESULTS_DIR / "partial_swap.json"))
    args = parser.parse_args()

    all_results = []

    for gpu_util in args.gpu_util:
        for num_prompts in args.num_prompts:
            prompts = build_prompts(num_prompts, args.prompt_len)

            for mode in args.modes:
                logger.info(
                    "=== %s | gpu_util=%.2f | nreq=%d ===",
                    mode, gpu_util, num_prompts,
                )
                try:
                    result = run_vllm(
                        model=args.model,
                        prompts=prompts,
                        mode=mode,
                        gpu_util=gpu_util,
                        swap_space=args.swap_space,
                        max_tokens=args.max_tokens,
                        max_model_len=args.max_model_len,
                    )
                    logger.info(
                        "  throughput=%.2f tok/s, preemptions=%d, elapsed=%.1fs",
                        result["throughput_tok_s"],
                        result["num_preemptions"],
                        result["elapsed_s"],
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    all_results.append({
                        "mode": mode,
                        "gpu_util": gpu_util,
                        "num_prompts": num_prompts,
                        "error": str(e),
                    })

    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20} {'GPU%':<8} {'Reqs':<6} {'Tok/s':<10} {'Preempt':<10}")
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"{r['mode']:<20} {r['gpu_util']:<8} {r['num_prompts']:<6} ERROR")
        else:
            print(
                f"{r['mode']:<20} {r['gpu_util']:<8.2f} "
                f"{r['num_prompts']:<6} {r['throughput_tok_s']:<10.2f} "
                f"{r['num_preemptions']:<10}"
            )


if __name__ == "__main__":
    main()

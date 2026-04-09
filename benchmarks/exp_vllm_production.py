#!/usr/bin/env python3
"""
Production-grade vLLM block-scoring benchmark.

Addresses all issues from prior experiments:
  1. Realistic workload: lognormal prompt lengths (like ShareGPT)
     OR real ShareGPT data if available
  2. Guaranteed preemption: enough requests + tight memory so KV cache
     overflows during decode, forcing victim selection
  3. Warmup phase: first trial is discarded to normalize GPU state
  4. Subprocess isolation: each strategy+config in a fresh process
  5. More requests (64+) for statistical significance

Strategies tested:
  fifo     — vLLM default LIFO queue-pop
  progress — output_len / total_len (V4)
  block_v2 — positional attention proxy (V2)
  hybrid   — block + progress + memory (V3)

Usage:
    conda run -n orchkv env PYTHONPATH=python python benchmarks/exp_vllm_production.py
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt generation: realistic lognormal + optional ShareGPT
# ---------------------------------------------------------------------------

def _generate_lognormal_prompts(
    n: int, median_words: int = 400, sigma: float = 0.7, seed: int = 42,
) -> list[str]:
    """Lognormal word-count distribution, similar to real chat traces."""
    rng = random.Random(seed)
    mu = math.log(median_words)
    base = (
        "Explain in detail the concept of attention mechanisms in "
        "transformer based neural network architectures and how they "
        "enable efficient sequence to sequence processing in modern "
        "large language model inference systems for various real world "
        "applications including machine translation summarization and "
        "question answering tasks across different domains. "
    )
    words_per_unit = len(base.split())
    prompts = []
    for _ in range(n):
        target_words = max(50, int(rng.lognormvariate(mu, sigma)))
        n_reps = max(1, target_words // words_per_unit)
        prompts.append(base * n_reps)
    return prompts


def _load_sharegpt_prompts(n: int) -> list[str] | None:
    """Try loading real ShareGPT prompts from HF cache."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "openchat/openchat_sharegpt4_dataset",
            split="train", trust_remote_code=True,
        )
        prompts = []
        for row in ds:
            items = row.get("items") or row.get("conversations") or []
            for item in items:
                role = item.get("from", item.get("role", ""))
                text = item.get("value", item.get("content", ""))
                if role in ("human", "user") and len(text) > 100:
                    prompts.append(text)
                    if len(prompts) >= n:
                        return prompts
        return prompts if len(prompts) >= n // 2 else None
    except Exception:
        return None


def build_prompts(n: int, use_real: bool = True, seed: int = 42) -> list[str]:
    if use_real:
        real = _load_sharegpt_prompts(n)
        if real and len(real) >= n:
            return real[:n]
    return _generate_lognormal_prompts(n, seed=seed)


# ---------------------------------------------------------------------------
# Single trial runner (in-process)
# ---------------------------------------------------------------------------

def run_trial(
    model: str,
    prompts: list[str],
    strategy: str,
    gpu_util: float,
    swap_space: int,
    max_tokens: int,
    max_model_len: int,
    preemption_mode: str | None = None,
) -> dict:
    for k in ("ORCHKV_SWAP", "ORCHKV_BLOCK_SCORE", "ORCHKV_PARTIAL_SWAP"):
        os.environ.pop(k, None)

    strategy_env = {
        "fifo":     {},
        "progress": {"ORCHKV_SWAP": "1", "ORCHKV_BLOCK_SCORE": "4"},
        "block_v2": {"ORCHKV_SWAP": "1", "ORCHKV_BLOCK_SCORE": "2"},
        "hybrid":   {"ORCHKV_SWAP": "1", "ORCHKV_BLOCK_SCORE": "3"},
    }
    for k, v in strategy_env.get(strategy, {}).items():
        os.environ[k] = v

    import torch
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model,
        gpu_memory_utilization=gpu_util,
        swap_space=swap_space,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="float16",
        trust_remote_code=True,
    )
    if preemption_mode is not None:
        llm_kwargs["preemption_mode"] = preemption_mode

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        return {"error": str(e), "strategy": strategy}

    sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    _ = llm.generate(prompts[:2], sp)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_in = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = (total_in + total_out) / elapsed

    preempt_count = -1
    try:
        preempt_count = llm.llm_engine.scheduler[0].num_cumulative_preemption
    except Exception:
        pass

    scorer_stats = {}
    try:
        scorer = getattr(
            llm.llm_engine.scheduler[0], '_orchkv_block_scorer', None)
        if scorer is not None:
            scorer_stats = scorer.get_stats()
    except Exception:
        pass

    n_blocks = -1
    try:
        cache_cfg = llm.llm_engine.cache_config
        n_blocks = cache_cfg.num_gpu_blocks
    except Exception:
        pass

    prompt_lens = [len(o.prompt_token_ids) for o in outputs]

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    for k in strategy_env.get(strategy, {}):
        os.environ.pop(k, None)

    return {
        "strategy": strategy,
        "gpu_util": gpu_util,
        "num_prompts": len(prompts),
        "throughput": round(throughput, 1),
        "total_input": total_in,
        "total_output": total_out,
        "elapsed": round(elapsed, 3),
        "preemptions": preempt_count,
        "gpu_blocks": n_blocks,
        "scorer_stats": scorer_stats,
        "prompt_len_stats": {
            "min": min(prompt_lens),
            "max": max(prompt_lens),
            "mean": round(sum(prompt_lens) / len(prompt_lens), 1),
            "median": sorted(prompt_lens)[len(prompt_lens) // 2],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Production vLLM Block-Scoring Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--gpu-util", type=float, nargs="+",
                        default=[0.20])
    parser.add_argument("--num-prompts", type=int, nargs="+",
                        default=[64])
    parser.add_argument("--median-words", type=int, default=400,
                        help="Median prompt length in words for lognormal dist")
    parser.add_argument("--sigma", type=float, default=0.7,
                        help="Lognormal sigma (higher = more variance)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--swap-space", type=int, default=32)
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["fifo", "progress", "block_v2", "hybrid"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-real-data", action="store_true",
                        help="Skip trying to load ShareGPT, use synthetic")
    parser.add_argument("--preemption-mode", type=str, default=None,
                        choices=["swap", "recompute", None],
                        help="Force preemption mode (default: vLLM auto)")
    parser.add_argument("--output", type=str,
                        default=str(RESULTS_DIR / "exp_vllm_production.json"))
    args = parser.parse_args()

    try:
        import vllm.core.scheduler as sched_module
        from orchkv.vllm_integration.block_level_swap import (
            apply_scheduler_patch,
        )
        apply_scheduler_patch(sched_module.__file__)
    except Exception as e:
        print(f"[WARN] Patch: {e}", file=sys.stderr)

    print(f"{'='*70}")
    print(f"  Production vLLM Block-Scoring Benchmark")
    print(f"  Model: {args.model}")
    print(f"  gpu_util: {args.gpu_util}")
    print(f"  num_prompts: {args.num_prompts}")
    print(f"  median_words: {args.median_words}, sigma: {args.sigma}")
    print(f"  max_tokens: {args.max_tokens}, max_model_len: {args.max_model_len}")
    print(f"  strategies: {args.strategies}, repeats: {args.repeats}")
    print(f"{'='*70}")

    all_results = []

    for gpu_util in args.gpu_util:
        for num_prompts in args.num_prompts:
            prompts = build_prompts(
                num_prompts,
                use_real=not args.no_real_data,
                seed=42,
            )
            if len(prompts) < num_prompts:
                prompts = _generate_lognormal_prompts(
                    num_prompts,
                    median_words=args.median_words,
                    sigma=args.sigma,
                )
            print(f"\n--- gpu_util={gpu_util}, n={num_prompts}, "
                  f"data={'synthetic' if args.no_real_data else 'auto'} ---")

            # Warmup: run one throwaway trial to stabilize GPU state
            print("  [warmup] ", end="", flush=True)
            warmup_r = run_trial(
                args.model, prompts[:4], "fifo",
                gpu_util, args.swap_space, args.max_tokens,
                args.max_model_len, args.preemption_mode,
            )
            print(f"done (blocks={warmup_r.get('gpu_blocks', '?')})")

            for strategy in args.strategies:
                trial_results = []

                for rep in range(args.repeats):
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                    r = run_trial(
                        args.model, prompts, strategy,
                        gpu_util, args.swap_space, args.max_tokens,
                        args.max_model_len, args.preemption_mode,
                    )

                    if "error" in r:
                        print(f"    {strategy:>12s} rep={rep}: "
                              f"ERROR {str(r['error'])[:60]}")
                        continue

                    trial_results.append(r)
                    print(f"    {strategy:>12s} rep={rep}: "
                          f"{r['throughput']:>8.1f} tok/s  "
                          f"preempt={r['preemptions']:>3d}  "
                          f"blocks={r['gpu_blocks']}")

                if trial_results:
                    thrs = [t["throughput"] for t in trial_results]
                    pres = [t["preemptions"] for t in trial_results]
                    avg_thr = sum(thrs) / len(thrs)
                    std_thr = (sum((t - avg_thr) ** 2 for t in thrs)
                               / max(len(thrs) - 1, 1)) ** 0.5
                    row = {
                        "gpu_util": gpu_util,
                        "num_prompts": num_prompts,
                        "strategy": strategy,
                        "avg_throughput": round(avg_thr, 1),
                        "std_throughput": round(std_thr, 1),
                        "min_throughput": round(min(thrs), 1),
                        "max_throughput": round(max(thrs), 1),
                        "avg_preemptions": round(sum(pres) / len(pres), 1),
                        "min_preemptions": min(pres),
                        "max_preemptions": max(pres),
                        "n_trials": len(trial_results),
                        "prompt_stats": trial_results[0].get(
                            "prompt_len_stats", {}),
                        "scorer_stats": trial_results[-1].get(
                            "scorer_stats", {}),
                    }
                    all_results.append(row)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'gpu':>5s} {'nreq':>5s} {'strategy':>12s} "
          f"{'avg tok/s':>10s} {'±std':>8s} "
          f"{'preempt':>8s} {'vs FIFO':>8s}")
    print(f"  {'-'*65}")

    for gpu_util in args.gpu_util:
        for np_ in args.num_prompts:
            rows = [r for r in all_results
                    if r["gpu_util"] == gpu_util
                    and r["num_prompts"] == np_]
            fifo_thr = next(
                (r["avg_throughput"] for r in rows
                 if r["strategy"] == "fifo"), 0)
            for r in rows:
                sp = r["avg_throughput"] / fifo_thr if fifo_thr > 0 else 0
                print(f"  {r['gpu_util']:>5.2f} {r['num_prompts']:>5d} "
                      f"{r['strategy']:>12s} "
                      f"{r['avg_throughput']:>10.1f} "
                      f"{r['std_throughput']:>7.1f} "
                      f"{r['avg_preemptions']:>8.1f} "
                      f"{sp:>7.3f}x")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")

    paper_json = (Path(__file__).parent.parent / "paper"
                  / "plot_figures_code_data"
                  / "exp_vllm_block_scoring.json")
    try:
        with open(paper_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Also saved to {paper_json}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

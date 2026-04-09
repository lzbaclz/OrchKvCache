#!/usr/bin/env python3
"""
vLLM block-level scoring experiment — comprehensive evaluation.

Compares five victim-selection strategies across memory-pressure levels:
  1. FIFO        — vLLM default (LIFO queue pop)
  2. Progress    — lowest output_len / total_len
  3. Block-V1    — original per-block EMA (Python dict, O(blocks) loop)
  4. Block-V2    — fast positional scoring (pre-computed, O(1) lookup)
  5. Hybrid      — block hotness + progress + memory efficiency

Sweeps gpu_memory_utilization × num_prompts with multiple repeats for
confidence intervals.  Outputs JSON for paper figure generation.

Usage:
    conda run -n orchkv python benchmarks/exp_vllm_block_scoring.py
    conda run -n orchkv python benchmarks/exp_vllm_block_scoring.py \
        --model Qwen/Qwen2.5-7B --repeats 3 --gpu-util 0.15 0.20 0.25 0.30
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _apply_patch_if_needed():
    """Ensure the scheduler patch is applied before any vLLM import."""
    try:
        import vllm.core.scheduler as sched_module
        from orchkv.vllm_integration.block_level_swap import (
            apply_scheduler_patch,
        )
        apply_scheduler_patch(sched_module.__file__)
    except Exception as e:
        print(f"[WARN] Patch not applied: {e}", file=sys.stderr)


def run_single_trial(
    model: str,
    gpu_util: float,
    num_prompts: int,
    max_tokens: int,
    strategy: str,
    swap_space: int = 32,
    prompt_len: int = 512,
) -> dict:
    """Run one vLLM trial in a subprocess for clean GPU state."""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    for k in ("ORCHKV_SWAP", "ORCHKV_BLOCK_SCORE", "ORCHKV_PARTIAL_SWAP"):
        env.pop(k, None)

    if strategy == "progress":
        env["ORCHKV_SWAP"] = "1"
        env["ORCHKV_BLOCK_SCORE"] = "4"
    elif strategy == "block_v1":
        env["ORCHKV_SWAP"] = "1"
        env["ORCHKV_BLOCK_SCORE"] = "1"
    elif strategy == "block_v2":
        env["ORCHKV_SWAP"] = "1"
        env["ORCHKV_BLOCK_SCORE"] = "2"
    elif strategy == "hybrid":
        env["ORCHKV_SWAP"] = "1"
        env["ORCHKV_BLOCK_SCORE"] = "3"

    worker_script = Path(__file__).parent / "_vllm_trial_worker.py"
    args_json = json.dumps({
        "model": model,
        "gpu_util": gpu_util,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "swap_space": swap_space,
        "prompt_len": prompt_len,
        "strategy": strategy,
    })

    try:
        result = subprocess.run(
            [sys.executable, str(worker_script), args_json],
            capture_output=True, text=True, env=env, timeout=600,
        )
        if result.returncode != 0:
            return {
                "error": result.stderr[-500:] if result.stderr else "unknown",
                "strategy": strategy,
                "gpu_util": gpu_util,
                "num_prompts": num_prompts,
            }
        return json.loads(result.stdout.strip().split("\n")[-1])
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "strategy": strategy,
                "gpu_util": gpu_util, "num_prompts": num_prompts}
    except Exception as e:
        return {"error": str(e), "strategy": strategy,
                "gpu_util": gpu_util, "num_prompts": num_prompts}


def run_inprocess_trial(
    model: str,
    gpu_util: float,
    num_prompts: int,
    max_tokens: int,
    strategy: str,
    swap_space: int = 32,
    prompt_len: int = 512,
) -> dict:
    """Run a trial in-process (fallback when subprocess worker is unavailable)."""
    for k in ("ORCHKV_SWAP", "ORCHKV_BLOCK_SCORE", "ORCHKV_PARTIAL_SWAP"):
        os.environ.pop(k, None)

    if strategy == "progress":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ["ORCHKV_BLOCK_SCORE"] = "4"
    elif strategy == "block_v1":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ["ORCHKV_BLOCK_SCORE"] = "1"
    elif strategy == "block_v2":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ["ORCHKV_BLOCK_SCORE"] = "2"
    elif strategy == "hybrid":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ["ORCHKV_BLOCK_SCORE"] = "3"

    import torch
    from vllm import LLM, SamplingParams

    max_model_len = int(prompt_len * 1.5 * 1.3) + max_tokens + 128
    try:
        llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_util,
            swap_space=swap_space,
            max_model_len=max_model_len,
            enforce_eager=True,
            dtype="float16",
            trust_remote_code=True,
        )
    except Exception as e:
        return {"error": str(e), "strategy": strategy}

    sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    import random
    base_prompt = (
        "Explain the concept of attention mechanisms in transformer models "
        "and how they enable efficient sequence processing in modern large "
        "language model architectures for various applications. "
    )
    rng = random.Random(42)
    prompts = []
    words_per_rep = len(base_prompt.split())
    for _ in range(num_prompts):
        scale = rng.uniform(0.5, 1.5)
        n_reps = max(1, int(prompt_len * scale / words_per_rep))
        prompts.append(base_prompt * n_reps)

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
        scorer = getattr(llm.llm_engine.scheduler[0], '_orchkv_block_scorer', None)
        if scorer is not None:
            scorer_stats = scorer.get_stats()
    except Exception:
        pass

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    for k in ("ORCHKV_SWAP", "ORCHKV_BLOCK_SCORE", "ORCHKV_PARTIAL_SWAP"):
        os.environ.pop(k, None)

    return {
        "strategy": strategy,
        "gpu_util": gpu_util,
        "num_prompts": num_prompts,
        "throughput": round(throughput, 1),
        "total_input": total_in,
        "total_output": total_out,
        "elapsed": round(elapsed, 3),
        "preemptions": preempt_count,
        "scorer_stats": scorer_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Block-Level Scoring Experiment")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--gpu-util", type=float, nargs="+",
                        default=[0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--num-prompts", type=int, nargs="+",
                        default=[16, 32])
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--swap-space", type=int, default=32)
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["fifo", "progress", "block_v1",
                                 "block_v2", "hybrid"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--subprocess", action="store_true",
                        help="Run each trial in a subprocess for isolation")
    parser.add_argument("--output", type=str,
                        default=str(RESULTS_DIR / "exp_vllm_block_scoring.json"))
    args = parser.parse_args()

    _apply_patch_if_needed()

    run_fn = run_single_trial if args.subprocess else run_inprocess_trial

    print(f"{'='*70}")
    print(f"  vLLM Block-Level Scoring Experiment")
    print(f"  Model: {args.model}")
    print(f"  gpu_util: {args.gpu_util}")
    print(f"  num_prompts: {args.num_prompts}")
    print(f"  strategies: {args.strategies}")
    print(f"  repeats: {args.repeats}")
    print(f"{'='*70}")

    all_results = []

    for gpu_util in args.gpu_util:
        for num_prompts in args.num_prompts:
            print(f"\n--- gpu_util={gpu_util}, num_prompts={num_prompts} ---")

            for strategy in args.strategies:
                trial_thrs = []
                trial_preempt = []
                trial_scorer = []

                for rep in range(args.repeats):
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                    r = run_fn(
                        model=args.model,
                        gpu_util=gpu_util,
                        num_prompts=num_prompts,
                        max_tokens=args.max_tokens,
                        strategy=strategy,
                        swap_space=args.swap_space,
                        prompt_len=args.prompt_len,
                    )

                    if "error" in r:
                        print(f"    {strategy:>12s} rep={rep}: "
                              f"ERROR {str(r['error'])[:60]}")
                        continue

                    trial_thrs.append(r["throughput"])
                    trial_preempt.append(r.get("preemptions", -1))
                    trial_scorer.append(r.get("scorer_stats", {}))
                    print(f"    {strategy:>12s} rep={rep}: "
                          f"{r['throughput']:>8.1f} tok/s  "
                          f"preempt={r.get('preemptions', -1)}")

                if trial_thrs:
                    avg_thr = sum(trial_thrs) / len(trial_thrs)
                    std_thr = (sum((t - avg_thr) ** 2 for t in trial_thrs)
                               / max(len(trial_thrs) - 1, 1)) ** 0.5
                    avg_pre = sum(trial_preempt) / len(trial_preempt)

                    row = {
                        "gpu_util": gpu_util,
                        "num_prompts": num_prompts,
                        "strategy": strategy,
                        "avg_throughput": round(avg_thr, 1),
                        "std_throughput": round(std_thr, 1),
                        "min_throughput": round(min(trial_thrs), 1),
                        "max_throughput": round(max(trial_thrs), 1),
                        "avg_preemptions": round(avg_pre, 1),
                        "n_trials": len(trial_thrs),
                        "scorer_stats": trial_scorer[-1] if trial_scorer else {},
                    }
                    all_results.append(row)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'gpu':>5s} {'nreq':>5s} {'strategy':>12s} {'avg tok/s':>10s} "
          f"{'±std':>8s} {'preempt':>8s} {'vs FIFO':>8s}")
    print(f"  {'-'*65}")

    for gpu_util in args.gpu_util:
        for np_ in args.num_prompts:
            rows = [r for r in all_results
                    if r["gpu_util"] == gpu_util and r["num_prompts"] == np_]
            fifo_thr = next((r["avg_throughput"] for r in rows
                             if r["strategy"] == "fifo"), 0)
            for r in rows:
                speedup = r["avg_throughput"] / fifo_thr if fifo_thr > 0 else 0
                print(f"  {r['gpu_util']:>5.2f} {r['num_prompts']:>5d} "
                      f"{r['strategy']:>12s} {r['avg_throughput']:>10.1f} "
                      f"{r['std_throughput']:>7.1f} "
                      f"{r['avg_preemptions']:>8.1f} {speedup:>7.3f}x")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")

    paper_json = Path(__file__).parent.parent / "paper" / \
        "plot_figures_code_data" / "exp_vllm_block_scoring.json"
    try:
        with open(paper_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Also saved to {paper_json}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

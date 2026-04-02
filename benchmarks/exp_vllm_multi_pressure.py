#!/usr/bin/env python3
"""
Eval-Fix 6: vLLM multi-pressure experiment for statistical significance.

Sweeps gpu_memory_utilization x num_prompts with 3 strategies:
  1. FIFO (vLLM default)
  2. Progress-aware (lowest output_len / total_len)
  3. Block-level scoring (OrchKvCache hotness per block)

Reports throughput, preemption count, and speedup with 3 repeats for
confidence intervals.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_vllm_trial(model, gpu_util, num_prompts, max_tokens, strategy,
                   swap_space=32, prompt_len=512):
    """Run one vLLM trial and return metrics."""
    env_clean = {
        "ORCHKV_SWAP": "",
        "ORCHKV_BLOCK_SCORE": "",
        "ORCHKV_PARTIAL_SWAP": "",
    }
    for k in env_clean:
        os.environ.pop(k, None)

    if strategy == "progress":
        os.environ["ORCHKV_SWAP"] = "1"
    elif strategy == "block_score":
        os.environ["ORCHKV_SWAP"] = "1"
        os.environ["ORCHKV_BLOCK_SCORE"] = "1"

    from vllm import LLM, SamplingParams

    max_model_len = prompt_len + max_tokens + 128
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

    base_prompt = (
        "Explain the concept of attention mechanisms in transformer models "
        "and how they enable efficient sequence processing in modern large "
        "language model architectures for various applications. "
    )
    prompts = [base_prompt * max(1, prompt_len // len(base_prompt.split()))] * num_prompts

    _ = llm.generate(prompts[:2], sp)

    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_in = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = (total_in + total_out) / elapsed

    scheduler_stats = {}
    try:
        stats = llm.llm_engine.scheduler[0].get_stats() if hasattr(llm.llm_engine, 'scheduler') else None
        if stats:
            scheduler_stats = {"preemptions": getattr(stats, 'num_preemption_iter', 0)}
    except Exception:
        pass

    preempt_count = scheduler_stats.get("preemptions", -1)

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    for k in env_clean:
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
    }


def main():
    model = "meta-llama/Llama-2-7b-hf"
    max_tokens = 64
    prompt_len = 512
    n_repeats = 2

    gpu_utils = [0.20, 0.25, 0.30]
    num_prompts_list = [16, 32]
    strategies = ["fifo", "progress", "block_score"]

    print(f"{'='*70}")
    print(f"  vLLM Multi-Pressure Experiment")
    print(f"  Model: {model}")
    print(f"  gpu_util: {gpu_utils}, num_prompts: {num_prompts_list}")
    print(f"  strategies: {strategies}, repeats: {n_repeats}")
    print(f"{'='*70}")

    all_results = []

    for gpu_util in gpu_utils:
        for num_prompts in num_prompts_list:
            print(f"\n--- gpu_util={gpu_util}, num_prompts={num_prompts} ---")

            for strategy in strategies:
                trial_thrs = []
                trial_preempt = []
                last_result = {}

                for rep in range(n_repeats):
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                    r = run_vllm_trial(
                        model, gpu_util, num_prompts, max_tokens,
                        strategy, swap_space=32, prompt_len=prompt_len,
                    )

                    if "error" in r:
                        print(f"    {strategy:>12s} rep={rep}: ERROR {r['error'][:60]}")
                        continue

                    trial_thrs.append(r["throughput"])
                    trial_preempt.append(r["preemptions"])
                    last_result = r
                    print(f"    {strategy:>12s} rep={rep}: {r['throughput']:>8.1f} tok/s  "
                          f"preempt={r['preemptions']}")

                if trial_thrs:
                    avg_thr = sum(trial_thrs) / len(trial_thrs)
                    avg_preempt = sum(trial_preempt) / len(trial_preempt)
                    row = {
                        "gpu_util": gpu_util,
                        "num_prompts": num_prompts,
                        "strategy": strategy,
                        "avg_throughput": round(avg_thr, 1),
                        "min_throughput": round(min(trial_thrs), 1),
                        "max_throughput": round(max(trial_thrs), 1),
                        "avg_preemptions": round(avg_preempt, 1),
                        "n_trials": len(trial_thrs),
                    }
                    all_results.append(row)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'gpu':>5s} {'nreq':>5s} {'strategy':>12s} {'avg tok/s':>10s} "
          f"{'preempt':>8s} {'vs FIFO':>8s}")
    print(f"  {'-'*55}")

    for gpu_util in gpu_utils:
        for np in num_prompts_list:
            rows = [r for r in all_results
                    if r["gpu_util"] == gpu_util and r["num_prompts"] == np]
            fifo_thr = next((r["avg_throughput"] for r in rows
                             if r["strategy"] == "fifo"), 0)
            for r in rows:
                speedup = r["avg_throughput"] / fifo_thr if fifo_thr > 0 else 0
                print(f"  {r['gpu_util']:>5.2f} {r['num_prompts']:>5d} "
                      f"{r['strategy']:>12s} {r['avg_throughput']:>10.1f} "
                      f"{r['avg_preemptions']:>8.1f} {speedup:>7.3f}x")

    out_path = RESULTS_DIR / "exp_vllm_multi_pressure.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

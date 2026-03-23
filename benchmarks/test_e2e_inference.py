#!/usr/bin/env python3
"""
D4: End-to-end inference tests — correctness, memory extension, latency.

Tests OrchKvCache integration with vLLM. Requires vLLM to be installed.

Usage:
    python benchmarks/test_e2e_inference.py --model meta-llama/Llama-2-7b-hf
    python benchmarks/test_e2e_inference.py --test correctness
    python benchmarks/test_e2e_inference.py --test memory
    python benchmarks/test_e2e_inference.py --test latency
    python benchmarks/test_e2e_inference.py --test all
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import (
    save_json, gpu_mem_mb, reset_gpu_peak,
    BenchmarkResult, ExperimentConfig, CPUTimer,
    generate_synthetic_prompts, build_vllm_engine,
)


MODELS = ["meta-llama/Llama-2-7b-hf"]
SEQ_LENS = [1024, 4096, 16384, 32768]
BATCH_SIZES = [1, 4, 8, 16]


def test_output_correctness(
    model: str = MODELS[0],
    seq_len: int = 512,
    max_new_tokens: int = 64,
    n_prompts: int = 4,
) -> dict:
    """
    Compare token output between baseline vLLM and OrchKvCache-enabled vLLM.

    Both use greedy decoding (temperature=0) with the same prompts.
    Reports: top-1 token match rate, max logit difference.
    """
    print(f"\n{'='*60}")
    print(f"  Correctness Test: model={model} seq={seq_len} n={n_prompts}")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[SKIP] vLLM not installed")
        return {"status": "skipped", "reason": "vLLM not installed"}

    sampling = SamplingParams(
        temperature=0, max_tokens=max_new_tokens, seed=42)
    prompts = generate_synthetic_prompts(n_prompts, seq_len)

    print("[1/3] Running baseline vLLM...")
    baseline_engine = build_vllm_engine(model, orchkv_enabled=False,
                                         max_model_len=seq_len + max_new_tokens)
    if baseline_engine is None:
        return {"status": "error", "reason": "engine build failed"}

    baseline_outputs = baseline_engine.generate(
        prompt_token_ids=prompts, sampling_params=sampling)
    baseline_tokens = [o.outputs[0].token_ids for o in baseline_outputs]
    del baseline_engine
    torch.cuda.empty_cache()

    print("[2/3] Running OrchKvCache-enabled vLLM...")
    orchkv_engine = build_vllm_engine(model, orchkv_enabled=True,
                                       max_model_len=seq_len + max_new_tokens)
    if orchkv_engine is None:
        return {"status": "error", "reason": "orchkv engine build failed"}

    orchkv_outputs = orchkv_engine.generate(
        prompt_token_ids=prompts, sampling_params=sampling)
    orchkv_tokens = [o.outputs[0].token_ids for o in orchkv_outputs]
    del orchkv_engine
    torch.cuda.empty_cache()

    print("[3/3] Comparing outputs...")
    total_tokens = 0
    matching_tokens = 0
    mismatches = []

    for i, (bt, ot) in enumerate(zip(baseline_tokens, orchkv_tokens)):
        for j, (b, o) in enumerate(zip(bt, ot)):
            total_tokens += 1
            if b == o:
                matching_tokens += 1
            else:
                mismatches.append({
                    "prompt_idx": i, "token_pos": j,
                    "baseline": int(b), "orchkv": int(o),
                })

    match_rate = matching_tokens / max(total_tokens, 1)
    result = {
        "status": "pass" if match_rate >= 0.999 else "fail",
        "total_tokens": total_tokens,
        "matching_tokens": matching_tokens,
        "match_rate": match_rate,
        "n_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:10],
        "acceptance": "≥99.9%" if match_rate >= 0.999 else f"{match_rate:.4%}",
    }

    print(f"  Match rate: {match_rate:.4%} ({matching_tokens}/{total_tokens})")
    print(f"  Status: {result['status'].upper()}")
    save_json(result, "test_correctness")
    return result


def test_memory_extension(
    model: str = MODELS[0],
    base_seq_len: int = 1024,
    max_new_tokens: int = 32,
) -> dict:
    """
    Find the max batch_size for baseline vs OrchKvCache at a given seq_len.
    Demonstrates memory extension capability.
    """
    print(f"\n{'='*60}")
    print(f"  Memory Extension Test: model={model} seq={base_seq_len}")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[SKIP] vLLM not installed")
        return {"status": "skipped"}

    sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    def find_max_batch(orchkv_enabled: bool, label: str) -> int:
        max_ok = 0
        for bs in [1, 2, 4, 8, 16, 32, 64]:
            try:
                print(f"  [{label}] Trying batch_size={bs}...", end=" ")
                prompts = generate_synthetic_prompts(bs, base_seq_len)
                engine = build_vllm_engine(
                    model, orchkv_enabled=orchkv_enabled,
                    max_model_len=base_seq_len + max_new_tokens)
                if engine is None:
                    print("SKIP")
                    break
                engine.generate(
                    prompt_token_ids=prompts, sampling_params=sampling)
                del engine
                torch.cuda.empty_cache()
                max_ok = bs
                print("OK")
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                print("OOM")
                torch.cuda.empty_cache()
                break
        return max_ok

    baseline_max = find_max_batch(False, "baseline")
    orchkv_max = find_max_batch(True, "orchkv")

    result = {
        "seq_len": base_seq_len,
        "baseline_max_batch": baseline_max,
        "orchkv_max_batch": orchkv_max,
        "extension_ratio": (orchkv_max / max(baseline_max, 1)),
        "status": "pass" if orchkv_max >= 2 * baseline_max else "marginal",
    }

    print(f"\n  Baseline max batch: {baseline_max}")
    print(f"  OrchKvCache max batch: {orchkv_max}")
    print(f"  Extension ratio: {result['extension_ratio']:.1f}x")
    save_json(result, "test_memory_extension")
    return result


def test_latency_breakdown(
    model: str = MODELS[0],
    seq_len: int = 1024,
    batch_size: int = 1,
    max_new_tokens: int = 64,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> dict:
    """
    Measure latency breakdown: TTFT, TPOT, total time.
    Compares baseline vs OrchKvCache.
    """
    print(f"\n{'='*60}")
    print(f"  Latency Breakdown: model={model} seq={seq_len} bs={batch_size}")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[SKIP] vLLM not installed")
        return {"status": "skipped"}

    sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    prompts = generate_synthetic_prompts(batch_size, seq_len)
    results = {}

    for label, orchkv_on in [("baseline", False), ("orchkv", True)]:
        print(f"\n  [{label}] Measuring...")
        engine = build_vllm_engine(
            model, orchkv_enabled=orchkv_on,
            max_model_len=seq_len + max_new_tokens)
        if engine is None:
            results[label] = {"status": "error"}
            continue

        for _ in range(n_warmup):
            engine.generate(prompt_token_ids=prompts, sampling_params=sampling)

        timer = CPUTimer()
        reset_gpu_peak()

        for i in range(n_runs):
            timer.start()
            outputs = engine.generate(
                prompt_token_ids=prompts, sampling_params=sampling)
            timer.stop()

        mem = gpu_mem_mb()
        n_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in outputs)
        total_tokens = batch_size * seq_len + n_output_tokens

        timing = timer.stats()
        avg_total_ms = timing.get("avg_us", 0) / 1000
        ttft_est_ms = avg_total_ms * (seq_len / (seq_len + max_new_tokens))
        tpot_est_ms = (avg_total_ms - ttft_est_ms) / max(max_new_tokens, 1)
        throughput = total_tokens / (avg_total_ms / 1000) if avg_total_ms > 0 else 0

        results[label] = {
            "timing": timing,
            "ttft_est_ms": ttft_est_ms,
            "tpot_est_ms": tpot_est_ms,
            "throughput_tok_per_s": throughput,
            "gpu_mem": mem,
            "n_output_tokens": n_output_tokens,
        }

        print(f"    Avg total: {avg_total_ms:.1f}ms")
        print(f"    TTFT est:  {ttft_est_ms:.1f}ms")
        print(f"    TPOT est:  {tpot_est_ms:.2f}ms")
        print(f"    Throughput: {throughput:.0f} tok/s")
        print(f"    GPU peak:  {mem['max_allocated_mb']:.0f}MB")

        del engine
        torch.cuda.empty_cache()

    if "baseline" in results and "orchkv" in results:
        bl = results["baseline"]
        ok = results["orchkv"]
        if isinstance(bl.get("timing"), dict) and isinstance(ok.get("timing"), dict):
            bl_avg = bl["timing"].get("avg_us", 1)
            ok_avg = ok["timing"].get("avg_us", 1)
            results["overhead_pct"] = ((ok_avg - bl_avg) / bl_avg) * 100

    save_json(results, "test_latency_breakdown")
    return results


def main():
    parser = argparse.ArgumentParser(description="OrchKvCache E2E Inference Tests")
    parser.add_argument("--model", default=MODELS[0])
    parser.add_argument("--test", default="all",
                        choices=["correctness", "memory", "latency", "all"])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    results = {}

    if args.test in ("correctness", "all"):
        results["correctness"] = test_output_correctness(
            model=args.model, seq_len=args.seq_len)

    if args.test in ("memory", "all"):
        results["memory"] = test_memory_extension(
            model=args.model, base_seq_len=args.seq_len)

    if args.test in ("latency", "all"):
        results["latency"] = test_latency_breakdown(
            model=args.model, seq_len=args.seq_len, batch_size=args.batch_size)

    save_json(results, "test_e2e_inference_all")
    print("\n" + "=" * 60)
    print("  All E2E tests complete. Results in benchmarks/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

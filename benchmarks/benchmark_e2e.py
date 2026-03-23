#!/usr/bin/env python3
"""
D4 / E1-E3: End-to-end benchmark — throughput, max batch, latency breakdown.

Sweeps over (model, seq_len, batch_size) matrix and compares baseline vs orchkv.
Outputs JSON + CSV for paper figures.

Usage:
    python benchmarks/benchmark_e2e.py
    python benchmarks/benchmark_e2e.py --seq-lens 1024,4096 --batch-sizes 1,4
    python benchmarks/benchmark_e2e.py --dry-run   # print matrix without running
"""
from __future__ import annotations

import argparse
import time
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import (
    save_json, save_csv, gpu_mem_mb, reset_gpu_peak,
    CPUTimer, generate_synthetic_prompts, build_vllm_engine,
)

DEFAULT_MODELS = ["Qwen/Qwen2.5-7B"]
DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]


def run_one(
    model: str,
    seq_len: int,
    batch_size: int,
    orchkv_enabled: bool,
    max_new_tokens: int = 128,
    n_warmup: int = 2,
    n_runs: int = 3,
) -> dict | None:
    """Run one (model, seq, bs, orchkv) point and return metrics."""
    label = "orchkv" if orchkv_enabled else "baseline"
    print(f"  [{label}] seq={seq_len} bs={batch_size} ...", end=" ", flush=True)

    try:
        from vllm import SamplingParams
    except ImportError:
        print("SKIP (no vLLM)")
        return None

    sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    prompts = generate_synthetic_prompts(batch_size, seq_len)

    try:
        engine = build_vllm_engine(
            model, orchkv_enabled=orchkv_enabled,
            max_model_len=seq_len + max_new_tokens)
        if engine is None:
            print("SKIP (engine None)")
            return None

        for _ in range(n_warmup):
            engine.generate(prompt_token_ids=prompts, sampling_params=sampling)

        timer = CPUTimer()
        reset_gpu_peak()

        for _ in range(n_runs):
            timer.start()
            outputs = engine.generate(
                prompt_token_ids=prompts, sampling_params=sampling)
            timer.stop()

        mem = gpu_mem_mb()
        n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_tok = batch_size * seq_len + n_out
        stats = timer.stats()
        avg_ms = stats.get("avg_us", 0) / 1000

        result = {
            "model": model,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "backend": label,
            "avg_ms": round(avg_ms, 2),
            "p50_ms": round(stats.get("p50_us", 0) / 1000, 2),
            "p99_ms": round(stats.get("p99_us", 0) / 1000, 2),
            "throughput_tok_s": round(total_tok / (avg_ms / 1000), 1) if avg_ms > 0 else 0,
            "ttft_est_ms": round(avg_ms * seq_len / (seq_len + max_new_tokens), 2),
            "tpot_est_ms": round(
                (avg_ms * max_new_tokens / (seq_len + max_new_tokens))
                / max(max_new_tokens, 1), 3),
            "output_tokens": n_out,
            "gpu_peak_mb": round(mem["max_allocated_mb"], 1),
            "status": "ok",
        }

        del engine
        torch.cuda.empty_cache()
        print(f"OK  {result['throughput_tok_s']} tok/s  {avg_ms:.0f}ms")
        return result

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        torch.cuda.empty_cache()
        print(f"OOM/ERROR: {e}")
        return {
            "model": model, "seq_len": seq_len, "batch_size": batch_size,
            "backend": label, "status": "oom", "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="E1-E3 Benchmark")
    parser.add_argument("--model", default=DEFAULT_MODELS[0])
    parser.add_argument("--seq-lens", default=",".join(map(str, DEFAULT_SEQ_LENS)))
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seq_lens = list(map(int, args.seq_lens.split(",")))
    batch_sizes = list(map(int, args.batch_sizes.split(",")))

    matrix = []
    for sl in seq_lens:
        for bs in batch_sizes:
            for orchkv in [False, True]:
                matrix.append((args.model, sl, bs, orchkv))

    print(f"Benchmark matrix: {len(matrix)} points")
    print(f"  seq_lens:    {seq_lens}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  backends:    [baseline, orchkv]")

    if args.dry_run:
        for m, sl, bs, ok in matrix:
            print(f"  {'orchkv' if ok else 'baseline':10s} seq={sl:6d} bs={bs:3d}")
        return

    all_results = []
    for m, sl, bs, ok in matrix:
        r = run_one(m, sl, bs, ok,
                    max_new_tokens=args.max_new_tokens,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs)
        if r:
            all_results.append(r)

    save_json(all_results, "benchmark_e2e")
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    if ok_results:
        save_csv(ok_results, "benchmark_e2e")

    print(f"\nDone. {len(ok_results)}/{len(all_results)} points succeeded.")


if __name__ == "__main__":
    main()

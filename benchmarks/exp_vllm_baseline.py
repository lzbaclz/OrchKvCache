#!/usr/bin/env python3
"""
vLLM Native Baseline: Run the same workloads through vLLM's actual
swap mechanism for fair comparison with OrchKvCache.

This script uses vLLM's LLM API with various swap_space and
gpu_memory_utilization settings to create memory pressure, matching
the exact same models, prompt lengths, request counts, and output
lengths used in exp_multimodel.py and exp_llama.py.

Comparison fairness:
  - Same models (Qwen2.5-7B, Mistral-7B, LLaMA-2-7B, LLaMA-2-13B)
  - Same hardware (A100-80GB)
  - Same prompts (synthetic, same token count)
  - Same output length (64 tokens)
  - Same metrics (throughput tok/s, TPOT ms)
  - Only difference: vLLM uses its native engine; OrchKvCache uses
    HF transformers + orchkv_core manual decode

Usage:
    nohup python -u benchmarks/exp_vllm_baseline.py \
        > benchmarks/results/vllm_baseline_run.log 2>&1 &
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Mistral-7B": "/raid/models/Mistral-7B-v0.1",
    "LLaMA-2-7B": "/raid/models/Llama-2-7b-hf",
    "LLaMA-2-13B": "/raid/models/Llama-2-13b-hf",
}

# Match EXACTLY the same parameters as OrchKvCache experiments
SEQ_LENS = [2048, 4096]
N_REQS = [1, 4, 8, 16]
MAX_NEW = 64

# vLLM configurations to test
# gpu_memory_utilization controls how much GPU vLLM uses for KV cache
# swap_space is CPU swap budget in GB
VLLM_CONFIGS = [
    {"label": "vllm-default",  "gpu_memory_utilization": 0.9, "swap_space": 4},
    {"label": "vllm-swap32",   "gpu_memory_utilization": 0.9, "swap_space": 32},
    {"label": "vllm-limited",  "gpu_memory_utilization": 0.5, "swap_space": 32},
]


def generate_prompts(n: int, seq_len: int, tokenizer) -> list[list[int]]:
    """Generate synthetic prompts as token IDs, truncated to exact seq_len."""
    import random
    base = "Artificial intelligence is transforming every aspect of modern life and technology. "
    text = base * (seq_len // 8 + 2)
    ids = tokenizer.encode(text)[:seq_len]
    return [ids] * n


def run_vllm_point(
    model_path: str, model_name: str,
    seq_len: int, n_req: int, max_new: int,
    vllm_cfg: dict,
) -> dict:
    """Run one (model, seq, nreq, config) point through vLLM."""

    result = {
        "model": model_name,
        "system": vllm_cfg["label"],
        "seq_len": seq_len,
        "n_requests": n_req,
        "max_new": max_new,
    }

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        result["status"] = "NO_VLLM"
        return result

    sampling = SamplingParams(
        temperature=0, max_tokens=max_new,
    )

    try:
        engine = LLM(
            model=model_path,
            max_model_len=seq_len + max_new + 64,
            block_size=16,
            enforce_eager=True,
            gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
            swap_space=vllm_cfg["swap_space"],
            dtype="auto",
            trust_remote_code=True,
        )

        tokenizer = engine.get_tokenizer()
        prompts = generate_prompts(n_req, seq_len, tokenizer)

        # Warmup
        try:
            engine.generate(prompt_token_ids=prompts[:1], sampling_params=sampling)
        except Exception as e:
            result["status"] = f"WARMUP_FAIL: {str(e)[:80]}"
            del engine; gc.collect(); torch.cuda.empty_cache()
            return result

        # Benchmark: run 3 times, take average
        latencies = []
        total_tokens_list = []

        for run_idx in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = engine.generate(prompt_token_ids=prompts, sampling_params=sampling)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            n_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            n_input_tokens = n_req * seq_len
            total_tokens = n_input_tokens + n_output_tokens

            latencies.append(elapsed)
            total_tokens_list.append(total_tokens)

        avg_elapsed = sum(latencies) / len(latencies)
        avg_tokens = sum(total_tokens_list) / len(total_tokens_list)
        throughput = avg_tokens / avg_elapsed
        tpot = (avg_elapsed / max(n_req * max_new, 1)) * 1000

        result.update({
            "status": "OK",
            "completed": n_req,
            "oom": 0,
            "avg_throughput": round(throughput, 1),
            "avg_tpot_ms": round(tpot, 2),
            "avg_elapsed_s": round(avg_elapsed, 3),
            "output_tokens": round(sum(len(o.outputs[0].token_ids) for o in outputs)),
        })

        del engine
        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        result.update({"status": "OOM", "completed": 0, "oom": n_req})
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        result.update({"status": f"ERROR: {str(e)[:100]}", "completed": 0, "oom": 0})
        gc.collect(); torch.cuda.empty_cache()

    return result


def main():
    print("=" * 80)
    print("vLLM Native Baseline Experiment")
    print("=" * 80)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Seq lengths: {SEQ_LENS}")
    print(f"Request counts: {N_REQS}")
    print(f"Max new tokens: {MAX_NEW}")
    print(f"vLLM configs: {[c['label'] for c in VLLM_CONFIGS]}")
    print()

    all_results = []
    total = len(MODELS) * len(SEQ_LENS) * len(N_REQS) * len(VLLM_CONFIGS)
    done = 0

    for model_name, model_path in MODELS.items():
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")

        for vllm_cfg in VLLM_CONFIGS:
            for seq_len in SEQ_LENS:
                for n_req in N_REQS:
                    done += 1
                    tag = (f"[{done}/{total}] {model_name} {vllm_cfg['label']} "
                           f"seq={seq_len} nreq={n_req}")
                    print(f"\n  {tag}", end=" ", flush=True)

                    r = run_vllm_point(
                        model_path, model_name,
                        seq_len, n_req, MAX_NEW, vllm_cfg,
                    )
                    all_results.append(r)

                    if r["status"] == "OK":
                        print(f"→ {r['avg_throughput']:.0f} tok/s, "
                              f"TPOT={r['avg_tpot_ms']:.1f}ms")
                    else:
                        print(f"→ {r['status']}")

            # Free GPU between configs
            gc.collect(); torch.cuda.empty_cache()

        # Free GPU between models
        gc.collect(); torch.cuda.empty_cache()

    # Save results
    out_path = RESULTS_DIR / "vllm_baseline.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_path} ({len(all_results)} points)")

    # Summary table
    print(f"\n{'='*80}")
    print("VLLM BASELINE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<14} {'Config':<14} {'Seq':>6} {'NReq':>5} {'Status':>6} {'Tok/s':>8} {'TPOT':>8}")
    print("-" * 70)
    for r in all_results:
        status = r["status"] if r["status"] != "OK" else "OK"
        tps = r.get("avg_throughput", 0)
        tpot = r.get("avg_tpot_ms", 0)
        print(f"{r['model']:<14} {r['system']:<14} {r['seq_len']:>6} "
              f"{r['n_requests']:>5} {status:>6} {tps:>8.0f} {tpot:>7.1f}ms")


if __name__ == "__main__":
    main()

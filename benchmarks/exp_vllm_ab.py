#!/usr/bin/env python3
"""
vLLM A/B Test: FIFO swap vs OrchKvCache-aware swap within the same vLLM.

This is the fairest possible comparison:
  - Same vLLM codebase, same continuous batching, same FlashAttention
  - Same model, hardware, prompts, output length
  - Only difference: swap victim selection logic (FIFO vs progress-aware)

Controlled by environment variable ORCHKV_SWAP=0|1.

Usage:
    nohup python -u benchmarks/exp_vllm_ab.py \
        > benchmarks/results/vllm_ab_run.log 2>&1 &
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
}

SEQ_LENS = [2048, 4096]
N_REQS = [4, 8, 16, 32]
MAX_NEW = 64

GPU_UTIL_CONFIGS = [
    {"label": "pressure-high", "gpu_memory_utilization": 0.50, "swap_space": 32},
    {"label": "pressure-med",  "gpu_memory_utilization": 0.70, "swap_space": 16},
]


def run_one(model_path, model_name, seq_len, n_req, max_new, gpu_cfg, orchkv_mode):
    """Run one data point through vLLM."""
    mode_label = "orchkv" if orchkv_mode else "fifo"
    os.environ["ORCHKV_SWAP"] = "1" if orchkv_mode else "0"

    result = {
        "model": model_name,
        "mode": mode_label,
        "seq_len": seq_len,
        "n_requests": n_req,
        "gpu_config": gpu_cfg["label"],
        "gpu_util": gpu_cfg["gpu_memory_utilization"],
    }

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        result["status"] = "NO_VLLM"
        return result

    sampling = SamplingParams(temperature=0, max_tokens=max_new)

    try:
        engine = LLM(
            model=model_path,
            max_model_len=seq_len + max_new + 64,
            block_size=16,
            enforce_eager=True,
            gpu_memory_utilization=gpu_cfg["gpu_memory_utilization"],
            swap_space=gpu_cfg["swap_space"],
            dtype="auto",
            trust_remote_code=True,
        )

        tokenizer = engine.get_tokenizer()
        base_text = "Artificial intelligence is transforming every aspect of modern life and technology. "
        text = base_text * (seq_len // 8 + 2)
        ids = tokenizer.encode(text)[:seq_len]
        prompts = [ids] * n_req

        # Warmup
        try:
            engine.generate(prompt_token_ids=prompts[:1], sampling_params=sampling)
        except Exception as e:
            result["status"] = f"WARMUP_FAIL: {str(e)[:80]}"
            del engine; gc.collect(); torch.cuda.empty_cache()
            return result

        # Benchmark: 3 runs
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = engine.generate(prompt_token_ids=prompts, sampling_params=sampling)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_tok = n_req * seq_len + n_out
        throughput = total_tok / avg_time
        tpot = (avg_time / max(n_out, 1)) * 1000

        result.update({
            "status": "OK",
            "completed": n_req,
            "avg_throughput": round(throughput, 1),
            "avg_tpot_ms": round(tpot, 2),
            "avg_elapsed_s": round(avg_time, 3),
            "output_tokens": n_out,
        })

        del engine; gc.collect(); torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        result["status"] = f"ERROR: {str(e)[:100]}"
        gc.collect(); torch.cuda.empty_cache()

    return result


def main():
    print("=" * 80)
    print("vLLM A/B Test: FIFO swap vs OrchKvCache-aware swap")
    print("Same vLLM, same batching, same kernels — only swap policy differs")
    print("=" * 80)

    all_results = []
    total = len(MODELS) * len(GPU_UTIL_CONFIGS) * len(SEQ_LENS) * len(N_REQS) * 2
    done = 0

    for model_name, model_path in MODELS.items():
        print(f"\n{'#'*70}\n# {model_name}\n{'#'*70}")

        for gpu_cfg in GPU_UTIL_CONFIGS:
            for seq_len in SEQ_LENS:
                for n_req in N_REQS:
                    for orchkv_mode in [False, True]:
                        done += 1
                        mode = "orchkv" if orchkv_mode else "fifo"
                        tag = (f"[{done}/{total}] {model_name} {gpu_cfg['label']} "
                               f"seq={seq_len} nreq={n_req} {mode}")
                        print(f"\n  {tag}", end=" ", flush=True)

                        r = run_one(model_path, model_name, seq_len, n_req,
                                    MAX_NEW, gpu_cfg, orchkv_mode)
                        all_results.append(r)

                        if r.get("status") == "OK":
                            print(f"-> {r['avg_throughput']:.0f} tok/s")
                        else:
                            print(f"-> {r.get('status', '?')}")

        gc.collect(); torch.cuda.empty_cache()

    # Save
    out_path = RESULTS_DIR / "vllm_ab_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_path} ({len(all_results)} points)")

    # Summary
    print(f"\n{'='*80}\nSUMMARY: vLLM FIFO vs OrchKvCache swap\n{'='*80}")
    print(f"{'Model':<14} {'GPU cfg':<14} {'Seq':>6} {'NReq':>5} {'FIFO':>8} {'OrchKv':>8} {'Speedup':>8}")
    print("-" * 70)

    for model_name in MODELS:
        for gpu_cfg in GPU_UTIL_CONFIGS:
            for seq_len in SEQ_LENS:
                for n_req in N_REQS:
                    fifo = next((r for r in all_results if r["model"] == model_name
                                 and r["mode"] == "fifo" and r["seq_len"] == seq_len
                                 and r["n_requests"] == n_req
                                 and r["gpu_config"] == gpu_cfg["label"]
                                 and r.get("status") == "OK"), None)
                    orch = next((r for r in all_results if r["model"] == model_name
                                 and r["mode"] == "orchkv" and r["seq_len"] == seq_len
                                 and r["n_requests"] == n_req
                                 and r["gpu_config"] == gpu_cfg["label"]
                                 and r.get("status") == "OK"), None)

                    f_tps = fifo["avg_throughput"] if fifo else 0
                    o_tps = orch["avg_throughput"] if orch else 0
                    speedup = o_tps / f_tps if f_tps > 0 else 0

                    print(f"{model_name:<14} {gpu_cfg['label']:<14} {seq_len:>6} "
                          f"{n_req:>5} {f_tps:>7.0f} {o_tps:>7.0f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()

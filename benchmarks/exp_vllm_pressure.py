#!/usr/bin/env python3
"""
vLLM Under Real Memory Pressure: FIFO vs OrchKvCache swap.

Key insight: previous A/B test used gpu_util=0.5/0.7 where swap rarely triggers.
This experiment uses gpu_util=0.25-0.35 to force frequent swapping, making the
difference between FIFO and OrchKvCache-aware scheduling visible.

Usage:
    CUDA_VISIBLE_DEVICES=0 nohup python -u benchmarks/exp_vllm_pressure.py \
        > benchmarks/results/vllm_pressure_run.log 2>&1 &
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
    "LLaMA-2-7B": "/raid/models/Llama-2-7b-hf",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Mistral-7B": "/raid/models/Mistral-7B-v0.1",
}

SEQ_LENS = [1024, 2048]
N_REQS = [4, 8, 16, 32, 64]
MAX_NEW = 32

PRESSURE_CONFIGS = [
    {"label": "extreme", "gpu_memory_utilization": 0.25, "swap_space": 32},
    {"label": "high",    "gpu_memory_utilization": 0.30, "swap_space": 32},
    {"label": "medium",  "gpu_memory_utilization": 0.40, "swap_space": 32},
]


def run_one(model_path, model_name, seq_len, n_req, max_new, gpu_cfg, orchkv_on):
    mode = "orchkv" if orchkv_on else "fifo"
    os.environ["ORCHKV_SWAP"] = "1" if orchkv_on else "0"

    result = {
        "model": model_name, "mode": mode, "seq_len": seq_len,
        "n_requests": n_req, "pressure": gpu_cfg["label"],
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
        text = "Artificial intelligence is transforming every aspect of modern life. " * (seq_len // 8 + 2)
        ids = tokenizer.encode(text)[:seq_len]
        prompts = [ids] * n_req

        # Warmup
        try:
            engine.generate(prompt_token_ids=prompts[:min(2, n_req)],
                            sampling_params=sampling)
        except Exception as e:
            result["status"] = f"WARMUP_FAIL: {str(e)[:60]}"
            del engine; gc.collect(); torch.cuda.empty_cache()
            return result

        # Benchmark: 3 runs
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = engine.generate(prompt_token_ids=prompts,
                                      sampling_params=sampling)
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
    print("vLLM Under REAL Memory Pressure: FIFO vs OrchKvCache")
    print("gpu_util = 0.25-0.40 → forces frequent swap → policy difference visible")
    print("=" * 80)

    all_results = []
    total = len(MODELS) * len(PRESSURE_CONFIGS) * len(SEQ_LENS) * len(N_REQS) * 2
    done = 0

    for model_name, model_path in MODELS.items():
        print(f"\n{'#'*70}\n# {model_name}\n{'#'*70}")

        for cfg in PRESSURE_CONFIGS:
            for seq_len in SEQ_LENS:
                for n_req in N_REQS:
                    for orchkv_on in [False, True]:
                        done += 1
                        mode = "orchkv" if orchkv_on else "fifo"
                        tag = (f"[{done}/{total}] {model_name} {cfg['label']} "
                               f"seq={seq_len} nreq={n_req} {mode}")
                        print(f"\n  {tag}", end=" ", flush=True)

                        r = run_one(model_path, model_name, seq_len, n_req,
                                    MAX_NEW, cfg, orchkv_on)
                        all_results.append(r)

                        if r.get("status") == "OK":
                            print(f"-> {r['avg_throughput']:.0f} tok/s")
                        else:
                            print(f"-> {r.get('status', '?')}")

            gc.collect(); torch.cuda.empty_cache()

    # Save
    out_path = RESULTS_DIR / "vllm_pressure.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_path} ({len(all_results)} points)")

    # Summary
    print(f"\n{'='*80}")
    print("FIFO vs OrchKvCache under memory pressure (within same vLLM)")
    print(f"{'='*80}")
    print(f"{'Model':<14} {'Pressure':<10} {'Seq':>6} {'NReq':>5} "
          f"{'FIFO':>8} {'OrchKv':>8} {'Speedup':>8} {'Status':>10}")
    print("-" * 80)

    for model_name in MODELS:
        for cfg in PRESSURE_CONFIGS:
            for seq_len in SEQ_LENS:
                for n_req in N_REQS:
                    fifo = next((r for r in all_results if r["model"] == model_name
                                 and r["mode"] == "fifo" and r["seq_len"] == seq_len
                                 and r["n_requests"] == n_req
                                 and r["pressure"] == cfg["label"]), None)
                    orch = next((r for r in all_results if r["model"] == model_name
                                 and r["mode"] == "orchkv" and r["seq_len"] == seq_len
                                 and r["n_requests"] == n_req
                                 and r["pressure"] == cfg["label"]), None)

                    f_ok = fifo and fifo.get("status") == "OK"
                    o_ok = orch and orch.get("status") == "OK"
                    f_tps = fifo["avg_throughput"] if f_ok else 0
                    o_tps = orch["avg_throughput"] if o_ok else 0
                    speedup = o_tps / f_tps if f_tps > 0 else 0

                    f_status = fifo.get("status", "?")[:6] if fifo else "?"
                    o_status = orch.get("status", "?")[:6] if orch else "?"
                    status = f"F:{f_status}/O:{o_status}"

                    print(f"{model_name:<14} {cfg['label']:<10} {seq_len:>6} "
                          f"{n_req:>5} {f_tps:>7.0f} {o_tps:>7.0f} "
                          f"{speedup:>7.2f}x {status:>10}")


if __name__ == "__main__":
    main()

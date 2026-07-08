#!/usr/bin/env python3
"""
ShareGPT-style Online Serving Simulation for SIGMETRICS 2027.

Simulates a realistic serving scenario using vLLM with varying memory
pressure (gpu_memory_utilization). Generates 50 requests with Poisson
arrivals and varying prompt lengths, then measures throughput, latency
percentiles (TPOT), and preemption/rejection counts.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n peerkv-serve \
        python benchmarks/sigmetrics/run_shareGPT_serving.py
"""
from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("VLLM_USE_V1", "0")

from transformers import PreTrainedTokenizerBase  # noqa: E402
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return list(set(self.all_special_tokens))
    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended

MODEL = "/public/model_zoo/Qwen2.5-7B"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "sharegpt_serving.json"

NUM_REQUESTS = 50
GEN_LEN = 128
GPU_MEM_UTILS = [0.4, 0.6, 0.85]
SEED = 42

PROMPT_BASE = (
    "You are a helpful AI assistant. The user has asked a complex question "
    "that requires careful reasoning and detailed explanation. Please provide "
    "a comprehensive response covering all aspects of the topic. "
    "In distributed computing, consensus protocols ensure that a collection "
    "of machines can agree on a single value despite failures. Protocols such "
    "as Raft and Paxos handle leader election, log replication, and safety "
    "across crash-fault and Byzantine-fault models. The CAP theorem limits "
    "the space of possible guarantees. Cloud-native systems now combine "
    "micro-services with container orchestration to achieve horizontal "
    "scalability, resilience, and rapid deployment cycles. Machine-learning "
    "inference serving systems must manage GPU memory carefully, balancing "
    "model weights, KV cache, and intermediate activations under tight "
    "latency constraints. "
)


class PreemptionLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.messages: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "preempt" in msg.lower():
            self.count += 1
            self.messages.append(msg)

    def reset(self):
        self.count = 0
        self.messages.clear()


def generate_request_set(tokenizer, num_requests: int, seed: int) -> list[dict]:
    """Generate requests with Poisson-distributed prompt lengths (200-2000 tokens)."""
    import numpy as np
    rng = np.random.RandomState(seed)

    mean_lengths = rng.uniform(200, 2000, size=num_requests).astype(int)
    actual_lengths = np.clip(
        rng.poisson(lam=mean_lengths), 100, 4000
    )
    inter_arrival_ms = rng.exponential(scale=500, size=num_requests)

    base_tokens = tokenizer.encode(PROMPT_BASE, add_special_tokens=False)
    requests = []
    for i, (target_len, iat) in enumerate(zip(actual_lengths, inter_arrival_ms)):
        target_len = int(target_len)
        n_repeats = max(1, target_len // len(base_tokens) + 1)
        long_ids = (base_tokens * n_repeats)[:target_len]
        prompt_text = tokenizer.decode(long_ids, skip_special_tokens=False)
        actual = len(tokenizer.encode(prompt_text))

        requests.append({
            "request_id": i,
            "prompt": prompt_text,
            "target_len": target_len,
            "actual_len": actual,
            "inter_arrival_ms": round(float(iat), 1),
        })

    return requests


def run_serving_at_gmu(gpu_mem_util: float, output_path: str):
    """Run the serving experiment at a given gpu_memory_utilization."""
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams

    handler = PreemptionLogHandler()
    logging.getLogger("vllm").addHandler(handler)

    max_prompt = 4200
    max_model_len = max_prompt + GEN_LEN + 128

    print(f"\n[serving] Initializing LLM  gpu_memory_utilization={gpu_mem_util}")

    try:
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=max_model_len,
            dtype="float16",
        )
    except Exception as e:
        result = {
            "gpu_mem_util": gpu_mem_util,
            "status": f"INIT_FAILED: {e}",
        }
        Path(output_path).write_text(json.dumps(result, indent=2))
        return

    tokenizer = llm.get_tokenizer()

    print("[serving] Generating request set...")
    requests = generate_request_set(tokenizer, NUM_REQUESTS, SEED)
    prompt_lens = [r["actual_len"] for r in requests]
    print(f"  {len(requests)} requests, prompt lens: "
          f"min={min(prompt_lens)} max={max(prompt_lens)} "
          f"mean={np.mean(prompt_lens):.0f}")

    cache_config = llm.llm_engine.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    total_kv_tokens = num_gpu_blocks * block_size
    print(f"  KV capacity: {num_gpu_blocks} blocks × {block_size} = "
          f"{total_kv_tokens} tokens")

    params = SamplingParams(max_tokens=GEN_LEN, temperature=0, ignore_eos=True)
    prompts = [r["prompt"] for r in requests]

    print(f"[serving] Running {len(prompts)} requests...")
    handler.reset()
    t0 = time.time()

    try:
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0

        per_request = []
        total_gen = 0
        completed = 0
        tpots = []

        for i, out in enumerate(outputs):
            n_gen = len(out.outputs[0].token_ids)
            total_gen += n_gen
            finish = out.outputs[0].finish_reason

            req_elapsed = elapsed / max(len(outputs), 1)
            if n_gen > 0:
                tpot = (req_elapsed * 1000) / n_gen
                tpots.append(tpot)

            if finish in ("length", "stop"):
                completed += 1

            per_request.append({
                "request_id": i,
                "prompt_len": requests[i]["actual_len"],
                "gen_tokens": n_gen,
                "finish_reason": finish,
                "completed": finish in ("length", "stop"),
            })

        throughput = total_gen / max(elapsed, 1e-9)

        tpot_arr = np.array(tpots) if tpots else np.array([0.0])
        latency_stats = {
            "tpot_p50_ms": round(float(np.percentile(tpot_arr, 50)), 2),
            "tpot_p95_ms": round(float(np.percentile(tpot_arr, 95)), 2),
            "tpot_p99_ms": round(float(np.percentile(tpot_arr, 99)), 2),
            "tpot_mean_ms": round(float(np.mean(tpot_arr)), 2),
        }

        result = {
            "gpu_mem_util": gpu_mem_util,
            "status": "OK",
            "num_requests": NUM_REQUESTS,
            "gen_len": GEN_LEN,
            "elapsed_s": round(elapsed, 3),
            "total_gen_tokens": total_gen,
            "total_throughput_tok_s": round(throughput, 1),
            "completed_requests": completed,
            "rejected_requests": NUM_REQUESTS - completed,
            "preemptions": handler.count,
            "preemption_messages": handler.messages[:10],
            "latency": latency_stats,
            "kv_capacity": {
                "num_gpu_blocks": num_gpu_blocks,
                "block_size": block_size,
                "total_kv_tokens": total_kv_tokens,
            },
            "prompt_length_stats": {
                "min": int(min(prompt_lens)),
                "max": int(max(prompt_lens)),
                "mean": round(float(np.mean(prompt_lens)), 1),
                "p50": int(np.percentile(prompt_lens, 50)),
                "p95": int(np.percentile(prompt_lens, 95)),
            },
            "per_request": per_request,
        }

        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.0f} tok/s")
        print(f"  Completed: {completed}/{NUM_REQUESTS}")
        print(f"  Preemptions: {handler.count}")
        print(f"  TPOT P50={latency_stats['tpot_p50_ms']:.1f}ms  "
              f"P95={latency_stats['tpot_p95_ms']:.1f}ms  "
              f"P99={latency_stats['tpot_p99_ms']:.1f}ms")

    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "gpu_mem_util": gpu_mem_util,
            "status": f"ERROR: {str(e)[:500]}",
            "elapsed_s": round(elapsed, 3),
            "preemptions": handler.count,
            "preemption_messages": handler.messages[:10],
        }
        print(f"  ERROR: {str(e)[:300]}")

    Path(output_path).write_text(json.dumps(result, indent=2))

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ShareGPT-style online serving simulation")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["single_gmu"])
    parser.add_argument("--gpu_mem_util", type=float, default=0.6)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.phase == "single_gmu":
        run_serving_at_gmu(args.gpu_mem_util, args.output)
        return

    print("=" * 70)
    print("  ShareGPT-style Online Serving Simulation")
    print("=" * 70)
    print(f"  Model:         {MODEL}")
    print(f"  Requests:      {NUM_REQUESTS}")
    print(f"  Gen length:    {GEN_LEN}")
    print(f"  GMU settings:  {GPU_MEM_UTILS}")
    print(f"  Timestamp:     {datetime.now().isoformat()}")
    print("=" * 70)

    results = {
        "experiment": "sharegpt_serving_simulation",
        "model": MODEL,
        "model_name": "Qwen2.5-7B",
        "num_requests": NUM_REQUESTS,
        "gen_len": GEN_LEN,
        "seed": SEED,
        "gpu_mem_utils": GPU_MEM_UTILS,
        "timestamp": datetime.now().isoformat(),
        "per_gmu": {},
    }

    for gi, gmu in enumerate(GPU_MEM_UTILS):
        print(f"\n{'='*60}")
        print(f"  gpu_memory_utilization = {gmu}")
        print(f"{'='*60}")

        if gi > 0:
            print("  Waiting for GPU memory release...")
            time.sleep(10)

        tmp = str(RESULTS_DIR / f"_tmp_sharegpt_{gmu}_{time.monotonic_ns()}.json")
        cmd = [
            sys.executable, __file__,
            "--phase", "single_gmu",
            "--gpu_mem_util", str(gmu),
            "--output", tmp,
        ]

        try:
            proc = subprocess.run(
                cmd, env=os.environ.copy(), timeout=600,
            )
            with open(tmp) as f:
                gmu_result = json.load(f)
            os.unlink(tmp)
        except subprocess.TimeoutExpired:
            gmu_result = {"gpu_mem_util": gmu, "status": "TIMEOUT"}
            print(f"  TIMEOUT for gmu={gmu}")
        except Exception as e:
            gmu_result = {"gpu_mem_util": gmu, "status": f"SUBPROCESS_FAILED: {e}"}
            print(f"  SUBPROCESS_FAILED: {e}")

        results["per_gmu"][str(gmu)] = gmu_result

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("  SHAREGPT SERVING SUMMARY")
    print("=" * 70)
    for gmu_str, gmu_r in results["per_gmu"].items():
        status = gmu_r.get("status", "unknown")
        if status == "OK":
            lat = gmu_r.get("latency", {})
            print(f"  gmu={gmu_str}:  "
                  f"throughput={gmu_r['total_throughput_tok_s']:.0f} tok/s  "
                  f"completed={gmu_r['completed_requests']}/{gmu_r['num_requests']}  "
                  f"preemptions={gmu_r['preemptions']}  "
                  f"TPOT_p50={lat.get('tpot_p50_ms', 0):.1f}ms  "
                  f"TPOT_p99={lat.get('tpot_p99_ms', 0):.1f}ms")
        else:
            print(f"  gmu={gmu_str}:  {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()

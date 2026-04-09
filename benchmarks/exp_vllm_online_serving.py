#!/usr/bin/env python3
"""
Online serving benchmark: vLLM OpenAI-compatible server + async clients.

Unlike offline LLM.generate(), this creates SUSTAINED memory pressure
with continuous request arrivals, which is the production scenario where
victim selection quality matters most.

Flow:
  1. Start vLLM server (with patched scheduler) in background
  2. Send requests at controlled arrival rate via aiohttp
  3. Measure per-request latency, throughput, and server-side preemptions
  4. Kill server, collect metrics
  5. Repeat for each strategy

Usage:
    conda run -n orchkv env PYTHONPATH=python \
        python benchmarks/exp_vllm_online_serving.py \
        --model meta-llama/Llama-2-7b-hf \
        --gpu-util 0.20 --qps 4.0 --num-requests 100
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SERVER_PORT = 18234


def generate_prompts(n: int, median_words: int = 400, sigma: float = 0.7,
                     seed: int = 42) -> list[dict]:
    """Lognormal prompt lengths, returns list of {prompt, max_tokens}."""
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
    wpunit = len(base.split())
    reqs = []
    for _ in range(n):
        target_words = max(30, int(rng.lognormvariate(mu, sigma)))
        nreps = max(1, target_words // wpunit)
        max_tok = rng.randint(32, 128)
        reqs.append({"prompt": base * nreps, "max_tokens": max_tok})
    return reqs


def start_server(model: str, gpu_util: float, swap_space: int,
                 max_model_len: int, strategy: str,
                 preemption_mode: str) -> subprocess.Popen:
    """Start vLLM OpenAI server with the given strategy."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(
        Path(__file__).resolve().parent.parent / "python")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for k in ("ORCHKV_SWAP", "ORCHKV_BLOCK_SCORE", "ORCHKV_PARTIAL_SWAP"):
        env.pop(k, None)

    strategy_env = {
        "fifo":     {},
        "progress": {"ORCHKV_SWAP": "1", "ORCHKV_BLOCK_SCORE": "4"},
        "block_v2": {"ORCHKV_SWAP": "1", "ORCHKV_BLOCK_SCORE": "2"},
        "hybrid":   {"ORCHKV_SWAP": "1", "ORCHKV_BLOCK_SCORE": "3"},
    }
    for k, v in strategy_env.get(strategy, {}).items():
        env[k] = v

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--gpu-memory-utilization", str(gpu_util),
        "--swap-space", str(swap_space),
        "--max-model-len", str(max_model_len),
        "--enforce-eager",
        "--dtype", "float16",
        "--trust-remote-code",
        "--port", str(SERVER_PORT),
        "--disable-log-requests",
        "--preemption-mode", preemption_mode,
    ]

    log_path = RESULTS_DIR / f"server_{strategy}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server(timeout: float = 120.0) -> bool:
    """Poll until server is ready."""
    import urllib.request
    url = f"http://localhost:{SERVER_PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def kill_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


async def send_requests(reqs: list[dict], qps: float) -> list[dict]:
    """Send requests at target QPS, collect per-request metrics."""
    import aiohttp

    url = f"http://localhost:{SERVER_PORT}/v1/completions"
    interval = 1.0 / qps if qps > 0 else 0
    results = []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        tasks = []

        async def do_request(idx: int, req: dict):
            payload = {
                "model": "meta-llama/Llama-2-7b-hf",
                "prompt": req["prompt"],
                "max_tokens": req["max_tokens"],
                "temperature": 0,
            }
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=payload) as resp:
                    body = await resp.json()
                    t1 = time.perf_counter()
                    usage = body.get("usage", {})
                    return {
                        "idx": idx,
                        "latency": round(t1 - t0, 4),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get(
                            "completion_tokens", 0),
                        "status": resp.status,
                    }
            except Exception as e:
                t1 = time.perf_counter()
                return {
                    "idx": idx,
                    "latency": round(t1 - t0, 4),
                    "error": str(e)[:100],
                }

        for i, req in enumerate(reqs):
            task = asyncio.create_task(do_request(i, req))
            tasks.append(task)
            if interval > 0 and i < len(reqs) - 1:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    return list(results)


def get_server_preemptions(strategy: str) -> int:
    """Parse preemption count from server log."""
    log_path = RESULTS_DIR / f"server_{strategy}.log"
    count = 0
    try:
        with open(log_path) as f:
            for line in f:
                if "preempted by" in line.lower() or \
                   "total_num_cumulative_preemption=" in line:
                    count += 1
    except Exception:
        pass
    return count


def run_online_trial(
    model: str, gpu_util: float, swap_space: int,
    max_model_len: int, strategy: str, preemption_mode: str,
    reqs: list[dict], qps: float,
) -> dict:
    """Full trial: start server → send requests → collect → kill."""
    print(f"      Starting server ({strategy})...", end="", flush=True)
    proc = start_server(
        model, gpu_util, swap_space, max_model_len,
        strategy, preemption_mode,
    )

    if not wait_for_server(timeout=120):
        kill_server(proc)
        return {"strategy": strategy, "error": "server timeout"}
    print(" ready.", flush=True)

    print(f"      Sending {len(reqs)} requests at {qps} QPS...",
          end="", flush=True)
    t0_wall = time.perf_counter()
    results = asyncio.run(send_requests(reqs, qps))
    t1_wall = time.perf_counter()
    wall_time = t1_wall - t0_wall
    print(f" done in {wall_time:.1f}s", flush=True)

    time.sleep(2)
    preemptions = get_server_preemptions(strategy)
    kill_server(proc)
    time.sleep(3)

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]

    if not ok:
        return {"strategy": strategy, "error": "all requests failed",
                "n_errors": len(errs)}

    total_prompt = sum(r["prompt_tokens"] for r in ok)
    total_completion = sum(r["completion_tokens"] for r in ok)
    total_tokens = total_prompt + total_completion
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    latencies = sorted(r["latency"] for r in ok)
    n = len(latencies)

    return {
        "strategy": strategy,
        "gpu_util": gpu_util,
        "qps": qps,
        "num_requests": len(reqs),
        "num_ok": len(ok),
        "num_errors": len(errs),
        "wall_time": round(wall_time, 3),
        "throughput_tok_s": round(throughput, 1),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "preemptions_logged": preemptions,
        "latency_p50": round(latencies[n // 2], 4),
        "latency_p90": round(latencies[int(n * 0.9)], 4),
        "latency_p99": round(latencies[int(n * 0.99)], 4),
        "latency_mean": round(sum(latencies) / n, 4),
        "latency_min": round(latencies[0], 4),
        "latency_max": round(latencies[-1], 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Online Serving vLLM Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--gpu-util", type=float, default=0.20)
    parser.add_argument("--swap-space", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--preemption-mode", default="swap",
                        choices=["swap", "recompute"])
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--qps", type=float, nargs="+", default=[2.0, 4.0],
                        help="Queries per second (arrival rate)")
    parser.add_argument("--median-words", type=int, default=400)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--strategies", nargs="+",
                        default=["fifo", "progress", "block_v2", "hybrid"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default=str(
        RESULTS_DIR / "exp_vllm_online.json"))
    args = parser.parse_args()

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("Installing aiohttp...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "aiohttp"])

    reqs = generate_prompts(
        args.num_requests, args.median_words, args.sigma)

    prompt_words = [len(r["prompt"].split()) for r in reqs]
    print(f"{'='*70}")
    print(f"  Online Serving Benchmark")
    print(f"  Model: {args.model}, gpu_util: {args.gpu_util}")
    print(f"  Requests: {args.num_requests}, QPS: {args.qps}")
    print(f"  Preemption mode: {args.preemption_mode}")
    print(f"  Prompt words: min={min(prompt_words)}, "
          f"max={max(prompt_words)}, "
          f"median={sorted(prompt_words)[len(prompt_words)//2]}")
    print(f"  Strategies: {args.strategies}, repeats: {args.repeats}")
    print(f"{'='*70}")

    all_results = []

    for qps in args.qps:
        print(f"\n--- QPS={qps} ---")
        for strategy in args.strategies:
            trial_results = []
            for rep in range(args.repeats):
                print(f"    {strategy} rep={rep}:")
                r = run_online_trial(
                    args.model, args.gpu_util, args.swap_space,
                    args.max_model_len, strategy,
                    args.preemption_mode, reqs, qps,
                )
                if "error" in r:
                    print(f"      ERROR: {r['error']}")
                else:
                    print(f"      {r['throughput_tok_s']:>7.1f} tok/s  "
                          f"p50={r['latency_p50']:.2f}s  "
                          f"p99={r['latency_p99']:.2f}s  "
                          f"preempt={r['preemptions_logged']}")
                    trial_results.append(r)

            if trial_results:
                thrs = [t["throughput_tok_s"] for t in trial_results]
                p50s = [t["latency_p50"] for t in trial_results]
                p99s = [t["latency_p99"] for t in trial_results]
                pres = [t["preemptions_logged"] for t in trial_results]
                avg = lambda xs: sum(xs) / len(xs)
                row = {
                    "qps": qps,
                    "strategy": strategy,
                    "avg_throughput": round(avg(thrs), 1),
                    "avg_p50": round(avg(p50s), 4),
                    "avg_p99": round(avg(p99s), 4),
                    "avg_preemptions": round(avg(pres), 1),
                    "n_trials": len(trial_results),
                    "trials": trial_results,
                }
                all_results.append(row)

    # Summary
    print(f"\n{'='*80}")
    print(f"  ONLINE SERVING SUMMARY")
    print(f"{'='*80}")
    print(f"  {'QPS':>5s} {'strategy':>12s} {'tok/s':>8s} "
          f"{'p50(s)':>8s} {'p99(s)':>8s} {'preempt':>8s} {'vs FIFO':>8s}")
    print(f"  {'-'*65}")

    for qps in args.qps:
        rows = [r for r in all_results if r["qps"] == qps]
        fifo_thr = next(
            (r["avg_throughput"] for r in rows
             if r["strategy"] == "fifo"), 1)
        for r in rows:
            sp = r["avg_throughput"] / fifo_thr if fifo_thr > 0 else 0
            print(f"  {r['qps']:>5.1f} {r['strategy']:>12s} "
                  f"{r['avg_throughput']:>8.1f} "
                  f"{r['avg_p50']:>8.3f} {r['avg_p99']:>8.3f} "
                  f"{r['avg_preemptions']:>8.1f} {sp:>7.3f}x")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

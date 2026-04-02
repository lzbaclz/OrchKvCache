#!/usr/bin/env python3
"""
Smoke test: verify block-level partial swap works end-to-end in vLLM.

Each mode runs in a SEPARATE subprocess (vLLM can't be reimported).
Uses very tight GPU budget to force preemption.
"""
import gc
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL = "meta-llama/Llama-2-7b-hf"
GPU_UTIL = 0.25          # 874 GPU blocks at this level
NUM_PROMPTS = 16
PROMPT_REPEATS = 30      # ~1500 tokens/prompt → 16*94 = 1504 blocks needed > 874
MAX_TOKENS = 64
SWAP_SPACE = 32
MAX_MODEL_LEN = 4096


WORKER_SCRIPT = r'''
import gc, json, os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mode = sys.argv[1]
model = sys.argv[2]
gpu_util = float(sys.argv[3])
num_prompts = int(sys.argv[4])
prompt_repeats = int(sys.argv[5])
max_tokens = int(sys.argv[6])
swap_space = int(sys.argv[7])
max_model_len = int(sys.argv[8])

if mode == "orchkv-block":
    os.environ["ORCHKV_BLOCK_SCORE"] = "1"
elif mode == "orchkv-request":
    os.environ["ORCHKV_SWAP"] = "1"
    os.environ.pop("ORCHKV_BLOCK_SCORE", None)
else:
    os.environ.pop("ORCHKV_SWAP", None)
    os.environ.pop("ORCHKV_BLOCK_SCORE", None)

from vllm import LLM, SamplingParams

base = (
    "Explain the concept of attention mechanisms in transformer models "
    "and how they relate to memory management in large language model "
    "inference systems. Discuss the trade-offs between different approaches "
    "to KV cache management including paging, offloading and eviction. "
)
prompts = [base * prompt_repeats] * num_prompts

llm = LLM(
    model=model,
    gpu_memory_utilization=gpu_util,
    swap_space=swap_space,
    max_model_len=max_model_len,
    enforce_eager=True,
    dtype="float16",
    trust_remote_code=True,
)

scheduler = llm.llm_engine.scheduler[0]
partial_mgr = getattr(scheduler, "_orchkv_block_scorer", None)

sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

t0 = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.perf_counter() - t0

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed if elapsed > 0 else 0

result = {
    "mode": mode,
    "model": model,
    "gpu_util": gpu_util,
    "num_prompts": num_prompts,
    "total_tokens": total_tokens,
    "elapsed_s": round(elapsed, 2),
    "throughput_tok_s": round(throughput, 2),
    "preemptions": scheduler.num_cumulative_preemption,
}

if partial_mgr is not None:
    result["block_scorer"] = partial_mgr.get_stats()

# Output JSON on a special marker line
print("__RESULT__" + json.dumps(result, default=str))
'''


def run_mode(mode: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running: {mode} | gpu_util={GPU_UTIL} | prompts={NUM_PROMPTS}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-c", WORKER_SCRIPT,
        mode, MODEL, str(GPU_UTIL), str(NUM_PROMPTS),
        str(PROMPT_REPEATS), str(MAX_TOKENS), str(SWAP_SPACE),
        str(MAX_MODEL_LEN),
    ]

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_OFFLINE"] = "1"
    project_python = os.path.join(SCRIPT_DIR, "..", "python")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_python + (":" + existing if existing else "")

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env,
    )

    # Parse result from stdout
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__"):])

    if proc.returncode != 0 and result is None:
        stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
        print(f"  STDERR (last 2000 chars):\n{stderr_tail}")
        return {"mode": mode, "error": f"exit code {proc.returncode}"}

    if result is None:
        print(f"  STDOUT:\n{proc.stdout[-1000:]}")
        return {"mode": mode, "error": "no result found in output"}

    # Print key info from stderr (vLLM logs)
    for line in proc.stderr.splitlines():
        if any(k in line for k in [
            "cuda blocks", "concurrency", "preempted by",
            "partial swap", "ORCHKV",
        ]):
            print(f"  {line.strip()}")

    ps_info = ""
    if "block_scorer" in result:
        s = result["block_scorer"]
        ps_info = (f" | scored={s.get('score_calls', 0)}, "
                   f"victims={s.get('victims_selected', 0)}")

    print(f"  -> {result['throughput_tok_s']:.2f} tok/s, "
          f"{result['preemptions']} preemptions, "
          f"{result['elapsed_s']:.1f}s{ps_info}")

    return result


def main():
    modes = ["fifo", "orchkv-request", "orchkv-block"]
    results = []

    for mode in modes:
        try:
            r = run_mode(mode)
            results.append(r)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 600s")
            results.append({"mode": mode, "error": "timeout"})
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"mode": mode, "error": str(e)})

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'Tok/s':<12} {'Preempt':<10} "
          f"{'Elapsed':<10} {'Extra'}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<20} ERROR: {r['error'][:50]}")
        else:
            extra = ""
            if "block_scorer" in r:
                s = r["block_scorer"]
                extra = (f"scored={s.get('score_calls', 0)} "
                         f"victims={s.get('victims_selected', 0)}")
            print(f"{r['mode']:<20} {r['throughput_tok_s']:<12.2f} "
                  f"{r['preemptions']:<10} "
                  f"{r['elapsed_s']:<10.1f} {extra}")

    fifo = next((r for r in results
                 if r.get("mode") == "fifo" and "error" not in r), None)
    block = next((r for r in results
                  if r.get("mode") == "orchkv-block" and "error" not in r),
                 None)
    if fifo and block:
        speedup = block["throughput_tok_s"] / max(fifo["throughput_tok_s"], 1)
        print(f"\nOrchKv-block vs FIFO speedup: {speedup:.3f}x")

    out_path = os.path.join(RESULTS_DIR, "partial_swap_smoke.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

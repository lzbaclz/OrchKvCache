"""FlexGen baseline benchmark on OPT-1.3B.

Runs FlexGen's offline inference with GPU/CPU/disk offloading on OPT-1.3B
and measures throughput and latency. Saves results to JSON.
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "opt-1.3b"
MODEL_PATH = "/public/model_zoo"
OUTPUT_DIR = "benchmarks/sigmetrics/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_LEN = 512
GEN_LEN = 64
GPU_BATCH_SIZE = 1
NUM_GPU_BATCHES = 1
TOTAL_BATCH = GPU_BATCH_SIZE * NUM_GPU_BATCHES

OFFLOAD_DIR = "/tmp/flexgen_offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)


def run_flexgen_benchmark():
    from flexgen.flex_opt import OptLM, Policy, CompressionConfig
    from flexgen.opt_config import get_opt_config
    from flexgen.utils import ExecutionEnv

    print(f"=== FlexGen Benchmark: {MODEL_NAME} ===")
    print(f"prompt_len={PROMPT_LEN}, gen_len={GEN_LEN}, batch={TOTAL_BATCH}")

    config = get_opt_config(MODEL_NAME)
    print(f"Model config: {config.num_hidden_layers}L, {config.n_head}H, "
          f"d={config.hidden_size}")

    env = ExecutionEnv.create(OFFLOAD_DIR)

    no_comp = CompressionConfig(num_bits=16, group_size=1, group_dim=0, symmetric=True)

    policy = Policy(
        gpu_batch_size=GPU_BATCH_SIZE,
        num_gpu_batches=NUM_GPU_BATCHES,
        w_gpu_percent=1.0,
        w_cpu_percent=0.0,
        cache_gpu_percent=1.0,
        cache_cpu_percent=0.0,
        act_gpu_percent=1.0,
        act_cpu_percent=0.0,
        overlap=False,
        sep_layer=False,
        pin_weight=False,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=no_comp,
        compress_cache=False,
        comp_cache_config=no_comp,
    )

    print("Loading FlexGen OPT model...")
    t_load = time.time()
    model = OptLM(config, env, MODEL_PATH, policy)
    load_time = time.time() - t_load
    print(f"Model loaded in {load_time:.1f}s")

    prompt_ids = np.ones((TOTAL_BATCH, PROMPT_LEN), dtype=np.int32) * 2
    prompt_ids[:, 0] = 0

    print(f"Running warmup generation...")
    try:
        warmup_ids = np.ones((TOTAL_BATCH, PROMPT_LEN), dtype=np.int32) * 2
        warmup_ids[:, 0] = 0
        model.generate(warmup_ids, max_new_tokens=4, verbose=0)
        model.delete_cache()
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    print("Running timed generation...")
    t0 = time.time()
    output_ids = model.generate(prompt_ids, max_new_tokens=GEN_LEN, verbose=1)
    torch_sync()
    elapsed = time.time() - t0

    total_gen_tokens = TOTAL_BATCH * GEN_LEN
    throughput = total_gen_tokens / elapsed
    latency_per_token = (elapsed * 1000) / GEN_LEN

    print(f"\n=== FLEXGEN RESULTS ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Generated: {total_gen_tokens} tokens")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"Latency/token (TPOT): {latency_per_token:.1f} ms")

    results = {
        "model": MODEL_NAME,
        "engine": "FlexGen",
        "prompt_len": PROMPT_LEN,
        "gen_len": GEN_LEN,
        "batch_size": TOTAL_BATCH,
        "total_gen_tokens": total_gen_tokens,
        "elapsed_s": round(elapsed, 3),
        "throughput_tok_s": round(throughput, 2),
        "tpot_ms": round(latency_per_token, 2),
        "load_time_s": round(load_time, 2),
        "policy": {
            "w_gpu_pct": policy.w_gpu_percent,
            "cache_gpu_pct": policy.cache_gpu_percent,
            "act_gpu_pct": policy.act_gpu_percent,
            "overlap": policy.overlap,
        },
    }

    outpath = os.path.join(OUTPUT_DIR, "flexgen_opt1.3b.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {outpath}")

    model.delete_all_weights()
    env.close_copy_threads()

    return results


def torch_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


if __name__ == "__main__":
    run_flexgen_benchmark()

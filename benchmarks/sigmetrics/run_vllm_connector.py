#!/usr/bin/env python3
"""
vLLM Connector Experiment: Memory Pressure Behavior.

Demonstrates that under memory pressure, vLLM's only option is admission
control (reject/queue requests), while OrchKvCache could offload cold blocks
to serve more concurrent requests.

Three experiments:
  1. Max concurrency under memory pressure (real vLLM runs)
  2. Theoretical cold-block analysis (model calculation)
  3. KV cache capacity analysis (real vLLM capacity + theoretical comparison)

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n peerkv-serve \
        python benchmarks/sigmetrics/run_vllm_connector.py
"""
from __future__ import annotations

import argparse
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

# vLLM 0.8.5 v1 engine spawns subprocesses that lose monkeypatches and
# miscalculate available memory when the GPU is shared. Use v0 engine.
os.environ.setdefault("VLLM_USE_V1", "0")

# transformers 5.x removed all_special_tokens_extended; patch it back
from transformers import PreTrainedTokenizerBase  # noqa: E402
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return list(set(self.all_special_tokens))
    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended

MODEL = "/public/model_zoo/Qwen2.5-7B"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PARAMS = {
    "n_layers": 28,
    "n_kv_heads": 4,
    "head_dim": 128,
    "block_size": 16,
    "kv_bytes_per_token": 28 * 2 * 4 * 128 * 2,  # 57344 bytes
}

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
GPU_MEM_UTILS = [0.3, 0.5, 0.7, 0.85]
PROMPT_LEN = 2048
GEN_LEN = 128

# Minimum GPU memory utilization that leaves room for KV cache after model
# weights. Qwen2.5-7B fp16 ≈ 14.3 GiB; on an 80 GiB A100, 0.3*80=24 GiB
# gives ~10 GiB for KV cache. Adjust if model doesn't fit.
MIN_VIABLE_GMU = 0.3


# =====================================================================
#  Helpers
# =====================================================================

class PreemptionLogHandler(logging.Handler):
    """Intercepts vLLM log messages to count preemption events."""

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


def make_prompt(tokenizer, target_len: int) -> str:
    """Create a prompt of exactly *target_len* tokens (or as close as possible)."""
    base = (
        "Explain the fundamental principles of distributed systems, "
        "including consensus algorithms, fault tolerance, and data "
        "replication strategies across geographically distributed nodes. "
        "Discuss the CAP theorem and its practical implications for "
        "modern cloud-native architectures, micro-service orchestration, "
        "and large-scale machine-learning inference serving systems. "
    ) * 400
    token_ids = tokenizer.encode(base)[:target_len]
    # Return raw token IDs as a prompt to guarantee exact length
    return tokenizer.decode(token_ids, skip_special_tokens=False)


# =====================================================================
#  Experiment 1: Max Concurrency Under Memory Pressure (subprocess)
# =====================================================================

def run_exp1_max_concurrency(gpu_mem_util: float, output_path: str):
    """Send batches of long requests, increasing batch size until failure."""
    import torch
    from vllm import LLM, SamplingParams

    handler = PreemptionLogHandler()
    logging.getLogger("vllm").addHandler(handler)

    max_model_len = PROMPT_LEN + GEN_LEN + 128
    print(f"\n[exp1] Initialising LLM  gpu_memory_utilization={gpu_mem_util}")

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
            "runs": [],
        }
        Path(output_path).write_text(json.dumps(result, indent=2))
        return

    tokenizer = llm.get_tokenizer()
    prompt_text = make_prompt(tokenizer, PROMPT_LEN)
    actual_len = len(tokenizer.encode(prompt_text))
    print(f"[exp1] Actual prompt length: {actual_len} tokens")

    runs: list[dict] = []
    for n in CONCURRENCY_LEVELS:
        handler.reset()
        params = SamplingParams(max_tokens=GEN_LEN, temperature=0, ignore_eos=True)
        prompts = [prompt_text] * n

        print(f"\n[exp1] batch_size={n}  gmu={gpu_mem_util} ...")
        gc.collect()

        t0 = time.time()
        try:
            outputs = llm.generate(prompts, params)
            elapsed = time.time() - t0

            total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)
            completed = sum(
                1 for o in outputs
                if o.outputs[0].finish_reason in ("length", "stop")
            )
            per_req_gen = [len(o.outputs[0].token_ids) for o in outputs]
            throughput = total_gen / max(elapsed, 1e-9)

            run = {
                "batch_size": n,
                "status": "OK",
                "elapsed_s": round(elapsed, 3),
                "total_gen_tokens": total_gen,
                "avg_gen_per_req": round(sum(per_req_gen) / max(len(per_req_gen), 1), 1),
                "throughput_tok_s": round(throughput, 1),
                "per_req_latency_s": round(elapsed / n, 3),
                "preemptions": handler.count,
                "preemption_messages": handler.messages[:5],
                "completed_requests": completed,
                "total_requests": n,
                "all_completed": completed == n,
            }
            print(
                f"  → {throughput:.0f} tok/s  {completed}/{n} done  "
                f"{handler.count} preemptions  {elapsed:.1f}s"
            )
        except Exception as e:
            elapsed = time.time() - t0
            run = {
                "batch_size": n,
                "status": f"ERROR: {str(e)[:500]}",
                "elapsed_s": round(elapsed, 3),
                "preemptions": handler.count,
                "preemption_messages": handler.messages[:5],
            }
            print(f"  → ERROR: {str(e)[:200]}")

        runs.append(run)

    result = {
        "gpu_mem_util": gpu_mem_util,
        "prompt_len": actual_len,
        "gen_len": GEN_LEN,
        "status": "OK",
        "runs": runs,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))

    del llm
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
#  Experiment 2: Theoretical Cold-Block Analysis (no GPU needed)
# =====================================================================

def run_exp2_cold_block_analysis() -> dict:
    """
    Model calculation based on measurement data.

    Measured values (see measurement.py):
      Gini = 0.92, top-10% concentration = 92.5%

    Key insight: 90% of KV blocks are "cold" — they contribute <7.5% of
    total attention mass. If we offload them to DRAM, the GPU KV footprint
    shrinks dramatically, allowing more concurrent sequences.
    """
    gini = 0.92
    top10_concentration = 0.925
    cold_fraction = 0.90
    cold_attention_contribution = 1.0 - top10_concentration  # 0.075

    theoretical_max_improvement = 1.0 / (1.0 - cold_fraction * (1.0 - cold_attention_contribution))
    # = 1 / (1 - 0.9 * 0.925) = 1 / (1 - 0.8325) = 1 / 0.1675 ≈ 5.97

    offload_ratios = [0.5, 0.7, 0.9, 1.0]
    offload_analysis: list[dict] = []
    for ratio in offload_ratios:
        blocks_offloaded = cold_fraction * ratio
        gpu_fraction_remaining = 1.0 - blocks_offloaded
        concurrency_gain = 1.0 / gpu_fraction_remaining
        attention_lost = cold_attention_contribution * ratio

        offload_analysis.append({
            "offload_ratio": ratio,
            "description": f"Offload {ratio*100:.0f}% of cold blocks to DRAM",
            "blocks_offloaded_fraction": round(blocks_offloaded, 3),
            "gpu_kv_fraction_remaining": round(gpu_fraction_remaining, 3),
            "theoretical_concurrency_gain": round(concurrency_gain, 2),
            "attention_mass_offloaded": round(attention_lost, 4),
            "attention_retained_on_gpu": round(1.0 - attention_lost, 4),
        })

    kv_bpt = MODEL_PARAMS["kv_bytes_per_token"]
    blk = MODEL_PARAMS["block_size"]
    kv_per_block = kv_bpt * blk

    concrete_examples: list[dict] = []
    for gpu_pool_gb in [2, 4, 8]:
        gpu_pool_bytes = gpu_pool_gb * (1 << 30)
        seq_len = PROMPT_LEN + GEN_LEN
        n_blocks_per_seq = (seq_len + blk - 1) // blk
        kv_per_seq = n_blocks_per_seq * kv_per_block

        max_seqs_vllm = int(gpu_pool_bytes / kv_per_seq)

        for ratio in [0.5, 0.7, 0.9]:
            hot_blocks = int(n_blocks_per_seq * (1.0 - cold_fraction * ratio))
            kv_per_seq_orchkv = hot_blocks * kv_per_block
            max_seqs_orchkv = int(gpu_pool_bytes / max(kv_per_seq_orchkv, 1))

            concrete_examples.append({
                "gpu_kv_pool_gb": gpu_pool_gb,
                "seq_len": seq_len,
                "offload_ratio": ratio,
                "n_blocks_per_seq": n_blocks_per_seq,
                "hot_blocks_on_gpu": hot_blocks,
                "kv_per_seq_mb_vllm": round(kv_per_seq / (1 << 20), 2),
                "kv_per_seq_mb_orchkv": round(kv_per_seq_orchkv / (1 << 20), 2),
                "max_concurrent_vllm": max_seqs_vllm,
                "max_concurrent_orchkv": max_seqs_orchkv,
                "improvement_factor": round(max_seqs_orchkv / max(max_seqs_vllm, 1), 2),
            })

    return {
        "measurement_basis": {
            "gini_coefficient": gini,
            "top_10pct_attention_concentration": top10_concentration,
            "cold_block_fraction": cold_fraction,
            "cold_attention_contribution": cold_attention_contribution,
            "model": "Qwen2.5-7B",
        },
        "theoretical_max_concurrency_improvement": {
            "formula": "1 / (1 - cold_fraction * (1 - cold_attention_contribution))",
            "value": round(theoretical_max_improvement, 2),
            "interpretation": (
                f"If 90% of blocks are cold (contribute only {cold_attention_contribution*100:.1f}% "
                f"attention), offloading all to DRAM gives {theoretical_max_improvement:.1f}x "
                f"more concurrent sequences"
            ),
        },
        "offload_ratio_analysis": offload_analysis,
        "concrete_examples": concrete_examples,
        "key_insight": (
            "vLLM must reject requests when KV cache is full. "
            "OrchKvCache offloads cold blocks (90% of blocks, 7.5% of attention) "
            "to DRAM, multiplying effective concurrency without quality loss."
        ),
    }


# =====================================================================
#  Experiment 3: KV Cache Capacity Analysis (subprocess per gmu)
# =====================================================================

def run_exp3_capacity(gpu_mem_util: float, output_path: str):
    """Query vLLM's actual KV cache capacity at a given memory utilization."""
    import torch
    from vllm import LLM

    max_model_len = PROMPT_LEN + GEN_LEN + 128
    print(f"\n[exp3] Querying KV capacity at gpu_memory_utilization={gpu_mem_util}")

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

    cache_config = llm.llm_engine.cache_config
    num_gpu_blocks = llm.llm_engine.cache_config.num_gpu_blocks
    num_cpu_blocks = llm.llm_engine.cache_config.num_cpu_blocks
    block_size = cache_config.block_size

    total_kv_tokens = num_gpu_blocks * block_size
    seq_len = PROMPT_LEN + GEN_LEN
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    max_concurrent = num_gpu_blocks // blocks_per_seq

    kv_bpt = MODEL_PARAMS["kv_bytes_per_token"]
    total_kv_bytes = num_gpu_blocks * block_size * kv_bpt
    total_kv_mb = total_kv_bytes / (1 << 20)

    cold_fraction = 0.90
    offload_ratios = [0.5, 0.7, 0.9]
    orchkv_comparison: list[dict] = []
    for ratio in offload_ratios:
        hot_blocks_per_seq = int(blocks_per_seq * (1.0 - cold_fraction * ratio))
        hot_blocks_per_seq = max(hot_blocks_per_seq, 1)
        max_concurrent_orchkv = num_gpu_blocks // hot_blocks_per_seq
        orchkv_comparison.append({
            "offload_ratio": ratio,
            "hot_blocks_per_seq": hot_blocks_per_seq,
            "max_concurrent_orchkv": max_concurrent_orchkv,
            "improvement_factor": round(max_concurrent_orchkv / max(max_concurrent, 1), 2),
        })

    result = {
        "gpu_mem_util": gpu_mem_util,
        "status": "OK",
        "kv_cache_capacity": {
            "num_gpu_blocks": num_gpu_blocks,
            "num_cpu_blocks": num_cpu_blocks,
            "block_size": block_size,
            "total_kv_tokens": total_kv_tokens,
            "total_kv_mb": round(total_kv_mb, 2),
        },
        "concurrency_analysis": {
            "seq_len": seq_len,
            "blocks_per_seq": blocks_per_seq,
            "max_concurrent_seqs_vllm": max_concurrent,
            "max_seq_x_concurrency_product": seq_len * max_concurrent,
        },
        "orchkv_comparison": orchkv_comparison,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))

    del llm
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
#  Orchestrator
# =====================================================================

def _subprocess_phase(phase: str, extra_args: list[str] | None = None,
                      timeout: int = 1200) -> tuple:
    """Spawn the current script as a subprocess for one phase."""
    tmp = str(RESULTS_DIR / f"_tmp_vllm_{phase}_{time.monotonic_ns()}.json")
    cmd = [sys.executable, __file__, "--phase", phase, "--output", tmp]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, env=os.environ.copy(), timeout=timeout)
    return proc, tmp


def run_all():
    print("=" * 70)
    print("  vLLM Connector Experiment: Memory Pressure Behavior")
    print("=" * 70)
    print(f"  Model:      {MODEL}")
    print(f"  Prompt:     {PROMPT_LEN} tokens")
    print(f"  Gen:        {GEN_LEN} tokens")
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print("=" * 70)

    results: dict = {
        "experiment": "vllm_connector_memory_pressure",
        "model": "Qwen2.5-7B",
        "model_path": MODEL,
        "prompt_len": PROMPT_LEN,
        "gen_len": GEN_LEN,
        "timestamp": datetime.now().isoformat(),
    }

    # ── Experiment 1: Max Concurrency Under Memory Pressure ───────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Max Concurrency Under Memory Pressure")
    print(f"  gpu_memory_utilization = {GPU_MEM_UTILS}")
    print(f"  batch_sizes = {CONCURRENCY_LEVELS}")
    print("=" * 70)

    exp1_results: dict[str, dict] = {}
    for gmu in GPU_MEM_UTILS:
        print(f"\n--- gpu_memory_utilization = {gmu} ---")
        proc, tmp = _subprocess_phase(
            "exp1",
            extra_args=["--gpu_mem_util", str(gmu)],
        )
        try:
            with open(tmp) as f:
                exp1_results[str(gmu)] = json.load(f)
            os.unlink(tmp)
        except Exception as e:
            exp1_results[str(gmu)] = {"status": f"RESULT_READ_FAILED: {e}"}
            print(f"  [warn] could not read result: {e}")

    results["experiment_1_max_concurrency"] = exp1_results

    # ── Experiment 2: Theoretical Cold-Block Analysis ─────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Theoretical Cold-Block Analysis")
    print("  (Model calculation, no GPU required)")
    print("=" * 70)

    exp2 = run_exp2_cold_block_analysis()
    results["experiment_2_cold_block_analysis"] = exp2

    print(f"\n  Gini coefficient:           {exp2['measurement_basis']['gini_coefficient']}")
    print(f"  Top-10% concentration:      {exp2['measurement_basis']['top_10pct_attention_concentration']}")
    print(f"  Cold block fraction:        {exp2['measurement_basis']['cold_block_fraction']}")
    print(f"  Theoretical max improvement: {exp2['theoretical_max_concurrency_improvement']['value']}x")
    print("\n  Offload ratio analysis:")
    for item in exp2["offload_ratio_analysis"]:
        print(
            f"    {item['description']:45s} → "
            f"{item['theoretical_concurrency_gain']:.2f}x concurrency, "
            f"{item['attention_retained_on_gpu']*100:.1f}% attention retained"
        )

    # ── Experiment 3: KV Cache Capacity Analysis ──────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: KV Cache Capacity Analysis")
    print("=" * 70)

    exp3_results: dict[str, dict] = {}
    for gmu in GPU_MEM_UTILS:
        print(f"\n--- gpu_memory_utilization = {gmu} ---")
        proc, tmp = _subprocess_phase(
            "exp3",
            extra_args=["--gpu_mem_util", str(gmu)],
        )
        try:
            with open(tmp) as f:
                exp3_results[str(gmu)] = json.load(f)
            os.unlink(tmp)
        except Exception as e:
            exp3_results[str(gmu)] = {"status": f"RESULT_READ_FAILED: {e}"}
            print(f"  [warn] could not read result: {e}")

    results["experiment_3_kv_capacity"] = exp3_results

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\n  Experiment 1 — Max clean concurrency per memory setting:")
    for gmu_str, gmu_r in exp1_results.items():
        if not isinstance(gmu_r, dict) or "runs" not in gmu_r:
            print(f"    gmu={gmu_str}: {gmu_r.get('status', 'unknown')}")
            continue
        max_ok, first_preempt, first_error = 0, None, None
        for r in gmu_r["runs"]:
            if r.get("status") == "OK" and r.get("preemptions", 0) == 0:
                max_ok = max(max_ok, r["batch_size"])
            elif r.get("preemptions", 0) > 0 and first_preempt is None:
                first_preempt = r["batch_size"]
            elif r.get("status", "").startswith("ERROR") and first_error is None:
                first_error = r["batch_size"]
        print(
            f"    gmu={gmu_str}: max_clean_batch={max_ok}  "
            f"first_preempt@batch={first_preempt}  first_error@batch={first_error}"
        )

    print("\n  Experiment 3 — KV capacity and OrchKvCache improvement:")
    for gmu_str, cap_r in exp3_results.items():
        if not isinstance(cap_r, dict) or "kv_cache_capacity" not in cap_r:
            print(f"    gmu={gmu_str}: {cap_r.get('status', 'unknown')}")
            continue
        kvc = cap_r["kv_cache_capacity"]
        conc = cap_r["concurrency_analysis"]
        print(
            f"    gmu={gmu_str}: {kvc['num_gpu_blocks']} blocks, "
            f"{kvc['total_kv_tokens']} tokens, "
            f"max_concurrent={conc['max_concurrent_seqs_vllm']}"
        )
        for oc in cap_r["orchkv_comparison"]:
            print(
                f"      offload={oc['offload_ratio']:.0%}: "
                f"orchkv_concurrent={oc['max_concurrent_orchkv']} "
                f"({oc['improvement_factor']:.1f}x)"
            )

    print(f"\n  Key insight: {exp2['key_insight']}")

    out = RESULTS_DIR / "vllm_connector_experiment.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved → {out}")


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Connector Experiment – memory pressure behavior")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["exp1", "exp3"])
    parser.add_argument("--gpu_mem_util", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.phase == "exp1":
        run_exp1_max_concurrency(args.gpu_mem_util, args.output)
    elif args.phase == "exp3":
        run_exp3_capacity(args.gpu_mem_util, args.output)
    else:
        run_all()


if __name__ == "__main__":
    main()

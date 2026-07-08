#!/usr/bin/env python3
"""
LMCache vs OrchKvCache comparison benchmark for SIGMETRICS 2027.

Compares KV cache management approaches on Qwen2.5-7B:
  1. vLLM native – no caching optimizations (baseline)
  2. vLLM prefix caching – built-in automatic prefix cache
  3. LMCache – capability/architecture comparison (LMCache extends prefix
     caching with CPU/disk offloading and cross-instance P2P transfer;
     integration requires the vLLM v1 engine which is not compatible with
     the current env's PyTorch build)
  4. OrchKvCache – attention-aware fine-grained offloading (from prior results)

Uses vLLM's offline LLM API for deterministic, single-process benchmarking.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n peerkv-serve python \
        benchmarks/sigmetrics/run_lmcache_comparison.py \
        --num_prompts 8 --max_new_tokens 128

Model: /public/model_zoo/Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V1", "0")

# transformers 5.x removed all_special_tokens_extended; patch for vLLM 0.8.5.
from transformers import PreTrainedTokenizerBase  # noqa: E402
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return list(set(self.all_special_tokens))
    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = "/public/model_zoo/Qwen2.5-7B"
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.85


# =====================================================================
#  Prompt generation
# =====================================================================

SHARED_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in computer science. "
    "You provide detailed, technical answers with examples when possible. "
    "Always structure your response clearly with numbered points."
)

SHARED_CONTEXT = (
    "The following is a comprehensive overview of key-value cache management "
    "in large language model serving systems. KV cache stores intermediate "
    "attention states during autoregressive decoding, growing linearly with "
    "sequence length. For a 7B parameter model with 28 layers, 4 KV heads, "
    "and 128-dimensional heads, each token requires approximately 28 * 2 * 4 "
    "* 128 * 2 = 57,344 bytes of KV cache in FP16. At 32K context length, "
    "this amounts to roughly 1.75 GB per request. Modern serving systems must "
    "handle hundreds of concurrent requests, making KV cache management a "
    "critical bottleneck.\n\n"
    "Several approaches have been proposed:\n"
    "1. PagedAttention (vLLM): Manages KV cache in fixed-size blocks.\n"
    "2. Prefix caching: Reuses KV cache across requests sharing common prefixes.\n"
    "3. KV offloading: Moves cold KV blocks to CPU DRAM or SSD.\n"
    "4. KV compression: Quantizes or sparsifies KV cache.\n"
    "5. Attention-aware scheduling: Prioritizes hot blocks by importance.\n\n"
)

QUESTIONS = [
    "Explain how PagedAttention works and why it improves GPU memory utilization.",
    "What are the trade-offs between prefix caching and attention-aware offloading?",
    "How does KV cache quantization affect model output quality?",
    "Describe the challenges of KV cache management in multi-tenant serving.",
    "Compare FIFO, LRU, and attention-based eviction policies for KV offloading.",
    "What role does the attention sink phenomenon play in KV cache optimization?",
    "How can speculative decoding interact with KV cache offloading systems?",
    "Explain disaggregated prefill-decode architectures and their KV transfer costs.",
    "What are the benefits and limitations of SSD-based KV cache offloading?",
    "How do modern KV cache systems handle variable-length sequences efficiently?",
    "Describe cross-layer attention prediction for KV cache prefetching.",
    "What metrics best capture KV cache system performance under SLO constraints?",
    "How does GQA/MQA change the KV cache management problem?",
    "Compare chunk-level and block-level KV cache granularity trade-offs.",
    "What are the networking costs of distributed KV cache sharing across nodes?",
    "How do elastic KV cache budgets adapt to bursty workload patterns?",
]


def generate_prompts(num_prompts: int, shared_prefix: bool = True) -> list[str]:
    prompts = []
    for q in QUESTIONS[:num_prompts]:
        if shared_prefix:
            prompts.append(
                f"{SHARED_SYSTEM_PROMPT}\n\n{SHARED_CONTEXT}Question: {q}\nAnswer:"
            )
        else:
            prompts.append(f"Question: {q}\nAnswer:")
    return prompts


# =====================================================================
#  Benchmark runner (offline vLLM LLM engine)
# =====================================================================

def run_offline_benchmark(
    prompts: list[str],
    max_new_tokens: int,
    label: str,
    enable_prefix_caching: bool = False,
    gpu_mem_util: float = GPU_MEM_UTIL,
) -> dict[str, Any]:
    """Run benchmark using vLLM's offline LLM engine."""
    import torch
    from vllm import LLM, SamplingParams

    print(f"\n{'='*65}")
    print(f"  BENCHMARK: {label}")
    print(f"  prefix_caching={enable_prefix_caching}")
    print(f"{'='*65}")

    try:
        print(f"[bench] Loading model ...")
        t_load = time.time()
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=gpu_mem_util,
            dtype="float16",
            trust_remote_code=True,
            enable_prefix_caching=enable_prefix_caching,
        )
        load_time = time.time() - t_load
        print(f"[bench] Model loaded in {load_time:.1f}s")
    except Exception as exc:
        print(f"[bench] FAILED to load engine: {exc}")
        return {
            "label": label,
            "status": "engine_load_failed",
            "error": str(exc)[:500],
        }

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    # --- Pass 1: cold ---
    print(f"[bench] Pass 1 (cold) ...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_cold_start = time.perf_counter()
    try:
        cold_outputs = llm.generate(prompts, params)
    except Exception as exc:
        print(f"[bench] Cold pass FAILED: {exc}")
        _cleanup_llm(llm)
        return {"label": label, "status": "cold_pass_failed", "error": str(exc)[:500]}
    torch.cuda.synchronize()
    cold_elapsed = time.perf_counter() - t_cold_start
    cold_mem = torch.cuda.max_memory_allocated() / (1 << 20)

    cold_metrics = _extract_metrics(cold_outputs, cold_elapsed, cold_mem)
    print(f"[bench] Cold: {cold_elapsed:.2f}s, "
          f"throughput={cold_metrics['throughput_tok_s']:.1f} tok/s, "
          f"peak_gpu={cold_mem:.0f} MB")

    # --- Pass 2: warm (benefits from prefix cache if enabled) ---
    time.sleep(1)
    print(f"[bench] Pass 2 (warm – same prompts, should hit cache) ...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_warm_start = time.perf_counter()
    try:
        warm_outputs = llm.generate(prompts, params)
    except Exception as exc:
        print(f"[bench] Warm pass FAILED: {exc}")
        _cleanup_llm(llm)
        return {
            "label": label, "status": "warm_pass_failed",
            "cold": cold_metrics, "error": str(exc)[:500],
        }
    torch.cuda.synchronize()
    warm_elapsed = time.perf_counter() - t_warm_start
    warm_mem = torch.cuda.max_memory_allocated() / (1 << 20)

    warm_metrics = _extract_metrics(warm_outputs, warm_elapsed, warm_mem)
    print(f"[bench] Warm: {warm_elapsed:.2f}s, "
          f"throughput={warm_metrics['throughput_tok_s']:.1f} tok/s, "
          f"peak_gpu={warm_mem:.0f} MB")

    speedup = warm_metrics["throughput_tok_s"] / max(cold_metrics["throughput_tok_s"], 0.1)
    print(f"[bench] Warm/cold speedup: {speedup:.2f}x")

    _cleanup_llm(llm)

    return {
        "label": label,
        "status": "OK",
        "cold": cold_metrics,
        "warm": warm_metrics,
        "warm_cold_speedup": round(speedup, 3),
        "model_load_time_s": round(load_time, 2),
    }


def _extract_metrics(outputs: list, elapsed_s: float, peak_gpu_mb: float) -> dict:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    per_request = []

    for out in outputs:
        prompt_toks = len(out.prompt_token_ids)
        gen_toks = len(out.outputs[0].token_ids) if out.outputs else 0
        total_prompt_tokens += prompt_toks
        total_completion_tokens += gen_toks
        per_request.append({"prompt_tokens": prompt_toks, "completion_tokens": gen_toks})

    total_tokens = total_prompt_tokens + total_completion_tokens
    throughput = total_tokens / max(elapsed_s, 1e-9)
    avg_tpot = (elapsed_s * 1000) / max(total_completion_tokens, 1)

    return {
        "elapsed_s": round(elapsed_s, 4),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "avg_tpot_ms": round(avg_tpot, 3),
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "n_requests": len(outputs),
        "per_request": per_request,
    }


def _cleanup_llm(llm):
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(3)


# =====================================================================
#  LMCache integration probe
# =====================================================================

def probe_lmcache_integration() -> dict[str, Any]:
    """Probe LMCache installation status and integration readiness."""
    status: dict[str, Any] = {}

    try:
        import lmcache
        status["installed"] = True
        status["version"] = lmcache.__version__
    except ImportError:
        status["installed"] = False
        return status

    # Check c_ops backend
    try:
        import lmcache.c_ops  # noqa: F401
        status["c_ops_available"] = True
    except (ImportError, OSError) as exc:
        status["c_ops_available"] = False
        status["c_ops_error"] = str(exc)[:200]

    # Check vLLM connector availability
    try:
        from lmcache.integration.vllm.lmcache_connector_v1_085 import (
            LMCacheConnectorV1Impl,
        )
        status["vllm_connector_found"] = True
        status["connector_class"] = "LMCacheConnectorV1Impl"
    except ImportError:
        status["vllm_connector_found"] = False

    # Check LMCache config
    try:
        from lmcache.v1.config import LMCacheEngineConfig
        cfg = LMCacheEngineConfig.from_defaults()
        status["config_loadable"] = True
        status["default_chunk_size"] = cfg.chunk_size
        status["default_local_cpu"] = cfg.local_cpu
        status["default_max_cpu_gb"] = cfg.max_local_cpu_size
        status["default_cache_policy"] = cfg.cache_policy
    except Exception as exc:
        status["config_loadable"] = False
        status["config_error"] = str(exc)[:200]

    status["integration_note"] = (
        "LMCache 0.5.1 integrates with vLLM via the v1 engine's KV connector "
        "interface (--kv-connector LMCacheConnector). This requires VLLM_USE_V1=1, "
        "which is incompatible with the current env due to a c_ops symbol mismatch "
        "with the installed PyTorch 2.6.0. The v0 engine does not support KV "
        "connectors. For a full LMCache benchmark, either (a) rebuild lmcache with "
        "the matching PyTorch/CUDA, or (b) use LMCache's standalone benchmark CLI."
    )

    return status


# =====================================================================
#  Capability comparison
# =====================================================================

def build_capability_comparison() -> dict:
    return {
        "lmcache": {
            "version": "0.5.1",
            "approach": "Prefix-level KV cache reuse across requests",
            "granularity": "Chunk-level (default 256 tokens per chunk)",
            "caching_policy": "LRU / LFU / MRU / FIFO (configurable)",
            "offloading_targets": [
                "CPU DRAM (local_cpu)",
                "Local disk (local_disk)",
                "Remote store (Redis, S3, filesystem, InfiniStore, Bigtable)",
                "GPU Direct Storage (GDS)",
            ],
            "vllm_integration": (
                "KV connector plugin for vLLM v1 engine; also supports "
                "SGLang and TensorRT-LLM"
            ),
            "cross_request_reuse": True,
            "attention_awareness": False,
            "per_block_importance": False,
            "prefix_matching": True,
            "p2p_kv_transfer": True,
            "disaggregated_pd": True,
            "kv_blending": True,
            "key_strengths": [
                "Prefix-level KV reuse eliminates redundant prefill computation",
                "Multi-tier storage hierarchy (GPU -> CPU -> disk -> remote)",
                "Works transparently as vLLM plugin, no model changes needed",
                "P2P KV transfer for multi-instance deployments via NIXL",
                "Prefill-decode disaggregation support",
                "KV blending for approximate prefix matching",
            ],
            "limitations_vs_orchkv": [
                "Chunk-level granularity (256 tokens) – cannot selectively "
                "retain individual hot blocks within a chunk",
                "No attention-aware eviction – uses generic LRU/FIFO, cannot "
                "prioritize blocks by actual attention importance",
                "Prefix-only reuse – requires exact token-level prefix match; "
                "partial overlap in the middle of a sequence is not exploited",
                "No per-layer heterogeneity – treats all layers uniformly, "
                "missing the observation that different layers have different "
                "attention sparsity patterns",
                "Does not help within a single long-context request – designed "
                "for cross-request prefix reuse, not intra-request offloading",
            ],
        },
        "orchkv": {
            "approach": "Attention-aware fine-grained KV cache offloading",
            "granularity": "Block-level (16 tokens per block, per-layer independent)",
            "caching_policy": "EMA-based attention importance scoring",
            "offloading_targets": [
                "CPU DRAM (pinned memory, async DMA)",
                "NVMe SSD (with io_uring)",
            ],
            "integration": "Direct model integration via modified attention layer",
            "cross_request_reuse": False,
            "attention_awareness": True,
            "per_block_importance": True,
            "prefix_matching": False,
            "key_strengths": [
                "Fine-grained (block-level) importance-aware eviction",
                "Per-layer heterogeneous management – each layer gets "
                "independent hot-set tracking",
                "EMA-based scoring adapts to shifting attention patterns "
                "during decode",
                "QK-norm proxy enables low-overhead importance estimation "
                "without full attention materialization",
                "Asynchronous prefetch with accurate hit prediction",
                "Works under severe memory pressure (5-25% GPU budget)",
            ],
            "limitations_vs_lmcache": [
                "Single-request scope – no cross-request KV reuse",
                "Requires attention signal access, not a drop-in vLLM plugin",
                "No remote/distributed KV storage tier",
            ],
        },
        "vllm_builtin_prefix_caching": {
            "approach": "Automatic prefix caching within vLLM v0 engine",
            "granularity": "Block-level (16 tokens, vLLM's PagedAttention block size)",
            "caching_policy": "Evict-on-completion, reuse by hash",
            "offloading_targets": ["GPU only (no CPU/disk offloading)"],
            "note": (
                "vLLM's built-in prefix caching is the foundation that LMCache "
                "extends. It hashes prefix blocks and reuses them across requests "
                "but keeps all cached blocks in GPU memory. LMCache adds CPU/disk "
                "offloading and cross-instance sharing on top of this mechanism."
            ),
        },
        "complementarity_note": (
            "LMCache and OrchKvCache address orthogonal dimensions of the KV "
            "cache problem. LMCache excels at INTER-request reuse (prefix "
            "caching), reducing redundant prefill across requests with shared "
            "context. OrchKvCache excels at INTRA-request management, "
            "intelligently offloading cold KV blocks within a single long "
            "request. In principle, both could be combined: LMCache handles "
            "cross-request prefix caching while OrchKvCache manages the "
            "within-request hot/cold tiering for the unique suffix portion."
        ),
        "paper_positioning": (
            "For the SIGMETRICS paper, OrchKvCache should be positioned as "
            "solving a different (and complementary) problem to LMCache. "
            "LMCache targets the 'shared prefix reuse' scenario common in "
            "multi-tenant serving. OrchKvCache targets the 'long-context "
            "single-request' scenario where KV cache exceeds GPU memory and "
            "must be offloaded with minimal quality loss. The key insight is "
            "that attention-aware block-level eviction (OrchKvCache) "
            "dramatically outperforms generic LRU/FIFO eviction (as used by "
            "LMCache's offloading tier) when GPU memory is constrained."
        ),
    }


# =====================================================================
#  Load prior results
# =====================================================================

def load_prior_results() -> dict:
    prior = {}
    for name in ["orchkv_hf_qwen7b", "vllm_native_qwen7b", "vllm_cpuoffload_qwen7b",
                  "vllm_connector_experiment"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                prior[name] = json.load(f)
    return prior


# =====================================================================
#  Main comparison
# =====================================================================

def _run_single_experiment_subprocess(
    exp_key: str,
    label: str,
    shared_prefix: bool,
    enable_prefix_caching: bool,
    num_prompts: int,
    max_new_tokens: int,
) -> dict:
    """Run a single experiment in a subprocess for clean GPU state."""
    import subprocess
    import tempfile

    tmp_out = str(RESULTS_DIR / f"_tmp_{exp_key}.json")
    script = f"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V1", "0")
from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(set(self.all_special_tokens)))

sys.path.insert(0, {str(Path(__file__).parent.parent.parent)!r})
from benchmarks.sigmetrics.run_lmcache_comparison import (
    run_offline_benchmark, generate_prompts,
)

prompts = generate_prompts({num_prompts}, shared_prefix={shared_prefix})
result = run_offline_benchmark(
    prompts, {max_new_tokens},
    label={label!r},
    enable_prefix_caching={enable_prefix_caching},
)
with open({tmp_out!r}, "w") as f:
    json.dump(result, f, indent=2, default=str)
print("[subprocess] Done, saved to", {tmp_out!r})
"""

    print(f"\n[main] Running {label} in subprocess ...")
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=600,
    )

    if proc.returncode != 0:
        print(f"[main] Subprocess failed (rc={proc.returncode})")
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        stdout_tail = proc.stdout[-500:] if proc.stdout else ""
        return {
            "label": label,
            "status": "subprocess_failed",
            "error": stderr_tail or stdout_tail,
        }

    print(proc.stdout[-300:] if proc.stdout else "")

    try:
        with open(tmp_out) as f:
            result = json.load(f)
        os.unlink(tmp_out)
        return result
    except Exception as exc:
        return {"label": label, "status": "result_read_failed", "error": str(exc)[:200]}


def run_comparison(
    num_prompts: int = 8,
    max_new_tokens: int = 128,
) -> dict:
    print(f"[main] LMCache Comparison Benchmark")
    print(f"[main] Model: {MODEL_PATH}")
    print(f"[main] Prompts: {num_prompts}, max_new_tokens: {max_new_tokens}")
    print(f"[main] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

    t0 = time.time()

    results: dict[str, Any] = {
        "benchmark": "lmcache_comparison",
        "model": MODEL_PATH,
        "lmcache_version": None,
        "vllm_version": None,
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        import lmcache
        results["lmcache_version"] = lmcache.__version__
    except Exception:
        pass
    try:
        import vllm
        results["vllm_version"] = vllm.__version__
    except Exception:
        pass

    experiments = [
        ("vllm_native_shared", "vLLM-native (shared prefix)", True, False),
        ("vllm_prefix_cached", "vLLM-prefix-caching (shared prefix)", True, True),
        ("vllm_native_unique", "vLLM-native (unique prompts)", False, False),
        ("vllm_prefix_unique", "vLLM-prefix-caching (unique prompts)", False, True),
    ]

    for i, (key, label, shared, prefix) in enumerate(experiments, 1):
        print(f"\n[main] Experiment {i}/{len(experiments)}: {label}")
        result = _run_single_experiment_subprocess(
            key, label, shared, prefix, num_prompts, max_new_tokens,
        )
        results[key] = result

    # --- LMCache integration status ---
    results["lmcache_integration"] = probe_lmcache_integration()

    # --- Capability comparison ---
    results["capability_comparison"] = build_capability_comparison()

    # --- Prior results ---
    results["prior_results"] = load_prior_results()

    elapsed = time.time() - t0
    results["total_time_s"] = round(elapsed, 1)
    results["summary"] = _build_summary(results)

    return results


def _build_summary(results: dict) -> dict:
    summary: dict[str, Any] = {}

    for key, label in [
        ("vllm_native_shared", "vLLM Native (shared prefix)"),
        ("vllm_prefix_cached", "vLLM Prefix Caching (shared)"),
        ("vllm_native_unique", "vLLM Native (unique)"),
        ("vllm_prefix_unique", "vLLM Prefix Caching (unique)"),
    ]:
        entry = results.get(key, {})
        if entry.get("status") != "OK":
            summary[label] = {"status": entry.get("status", "missing")}
            continue
        cold = entry.get("cold", {})
        warm = entry.get("warm", {})
        summary[label] = {
            "cold_throughput_tok_s": cold.get("throughput_tok_s"),
            "cold_tpot_ms": cold.get("avg_tpot_ms"),
            "warm_throughput_tok_s": warm.get("throughput_tok_s"),
            "warm_tpot_ms": warm.get("avg_tpot_ms"),
            "warm_cold_speedup": entry.get("warm_cold_speedup"),
            "peak_gpu_mb": cold.get("peak_gpu_mb"),
        }

    return summary


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LMCache vs OrchKvCache comparison benchmark")
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default="lmcache_comparison")
    args = parser.parse_args()

    results = run_comparison(
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
    )

    out_path = RESULTS_DIR / f"{args.output}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[main] Results saved to {out_path}")

    _print_summary(results)


def _print_summary(results: dict):
    print(f"\n{'='*80}")
    print(f"  LMCACHE vs ORCHKV COMPARISON SUMMARY")
    print(f"{'='*80}")

    summary = results.get("summary", {})
    print(f"\n  {'Configuration':<35s} {'Cold Thr':>10s} {'Warm Thr':>10s} "
          f"{'TPOT':>9s} {'Speedup':>8s}")
    print(f"  {'-'*74}")

    for label, data in summary.items():
        if not isinstance(data, dict):
            continue
        if "status" in data and data.get("cold_throughput_tok_s") is None:
            print(f"  {label:<35s} {'-- ' + str(data.get('status', '?')) + ' --':>45s}")
            continue
        ct = f"{data['cold_throughput_tok_s']:.0f}" if data.get("cold_throughput_tok_s") else "?"
        wt = f"{data['warm_throughput_tok_s']:.0f}" if data.get("warm_throughput_tok_s") else "?"
        tp = f"{data['cold_tpot_ms']:.2f}" if data.get("cold_tpot_ms") else "?"
        su = f"{data['warm_cold_speedup']:.2f}x" if data.get("warm_cold_speedup") else "N/A"
        print(f"  {label:<35s} {ct:>9s}t/s {wt:>9s}t/s "
              f"{tp:>8s}ms {su:>8s}")

    # LMCache integration status
    lm = results.get("lmcache_integration", {})
    print(f"\n  LMCache v{lm.get('version', '?')}: "
          f"installed={'Y' if lm.get('installed') else 'N'}, "
          f"c_ops={'Y' if lm.get('c_ops_available') else 'N'}, "
          f"vllm_connector={'Y' if lm.get('vllm_connector_found') else 'N'}")
    if lm.get("integration_note"):
        print(f"  Note: {lm['integration_note'][:120]}...")

    # Prior OrchKvCache numbers
    prior = results.get("prior_results", {})
    if "orchkv_hf_qwen7b" in prior:
        orchkv = prior["orchkv_hf_qwen7b"]
        print(f"\n  OrchKvCache prior result: {json.dumps(orchkv, indent=None)[:120]}...")

    cap = results.get("capability_comparison", {})
    if cap.get("paper_positioning"):
        print(f"\n  Paper positioning:")
        pos = cap["paper_positioning"]
        while pos:
            line = pos[:85]
            if len(pos) > 85:
                sp = line.rfind(" ")
                if sp > 40:
                    line = pos[:sp]
            print(f"    {line}")
            pos = pos[len(line):].lstrip()

    print(f"\n  Total time: {results.get('total_time_s', '?')}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

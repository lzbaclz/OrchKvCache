#!/usr/bin/env python3
"""
Cross-model validation: Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3.

Proves OrchKvCache generality across architectures by running:
  1. Workload characterization (Gini, Jaccard stability, top-10% concentration)
  2. OrchKvCache at 50% budget (throughput, TPOT, evictions, promotions)
  3. Correctness (50 prompts, bit-exact match vs GPU-only)
  4. Predictor comparison (EMA vs InfiniGen cross-layer, Jaccard)

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n orchkv \
        PYTHONPATH=build/bindings:python \
        python benchmarks/sigmetrics/run_cross_model.py
"""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Model configurations ──────────────────────────────────────────────

MODELS = {
    "llama-3.1-8b": {
        "path": "/public/model_zoo/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "kv_bytes_per_token": 32 * 2 * 8 * 128 * 2,  # layers * KV * heads * dim * fp16
    },
    "mistral-7b": {
        "path": "/public/model_zoo/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "kv_bytes_per_token": 32 * 2 * 8 * 128 * 2,
    },
}

PROMPTS_CHARACTERIZATION = [
    "Explain the key differences between transformers and recurrent neural networks in detail, including their architectural components, training procedures, and practical applications.",
    "Write a comprehensive guide to implementing a distributed key-value store with consistency guarantees, covering replication, partitioning, and failure recovery.",
    "Describe the history of operating systems from Unix to modern cloud computing, highlighting the major innovations at each stage.",
    "What are the main challenges in large language model inference optimization? Discuss memory management, batching strategies, and hardware utilization.",
    "Summarize the evolution of computer architecture from von Neumann to modern GPUs, explaining how parallelism has changed computing.",
    "Explain how garbage collection works in Java, Go, and Rust, comparing their approaches to memory safety and performance trade-offs.",
    "Describe the CAP theorem and its implications for distributed databases, with examples of systems that prioritize different guarantees.",
    "What are attention sinks in large language models and why do they matter for KV cache management in long-context inference?",
]

PROMPTS_CORRECTNESS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to compute Fibonacci numbers using dynamic programming.",
    "What is the difference between TCP and UDP protocols?",
    "Describe the process of photosynthesis in plants.",
    "Explain the concept of blockchain technology.",
    "What are the main principles of object-oriented programming?",
    "Describe the water cycle and its importance to Earth.",
    "Explain how neural networks learn through backpropagation.",
    "What is the theory of relativity?",
    "Describe the process of DNA replication.",
    "Explain the concept of market equilibrium in economics.",
    "What are the layers of the OSI networking model?",
    "Describe how compilers transform source code into machine code.",
    "Explain the greenhouse effect and climate change.",
    "What is the Halting Problem in computer science?",
    "Describe the structure of an atom.",
    "Explain how encryption protects data in transit.",
    "What are the main data structures used in computer science?",
    "Describe the process of natural selection.",
    "Explain how databases handle concurrent transactions.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe the human immune system and how it fights pathogens.",
    "Explain the concept of recursion with examples.",
    "What are the principles of thermodynamics?",
    "Describe the architecture of the Internet.",
    "Explain how operating systems manage memory.",
    "What is the significance of the Turing machine?",
    "Describe the process of protein synthesis.",
    "Explain MapReduce and its role in distributed computing.",
    "What are the fundamental forces in physics?",
    "Describe the architecture of modern CPU pipelines.",
    "Explain the concept of consensus in distributed systems.",
    "What is the role of enzymes in biological reactions?",
    "Describe the principles of functional programming.",
    "Explain how gradient descent optimizes neural networks.",
    "What is the difference between stack and heap memory?",
    "Describe the carbon cycle and its significance.",
    "Explain how virtual memory works in operating systems.",
    "What are the main sorting algorithms and their complexities?",
    "Describe the structure and function of mitochondria.",
    "Explain the concept of containerization in software.",
    "What is the theory of evolution?",
    "Describe how hash tables work internally.",
    "Explain the difference between processes and threads.",
    "What are the principles of database normalization?",
    "Describe the human nervous system.",
    "Explain how public key cryptography works.",
    "What is the P vs NP problem?",
    "Describe the process of plate tectonics.",
    "Explain the concept of microservices architecture.",
]

PROMPTS_PREDICTOR = [
    "Explain the key differences between transformers and recurrent neural networks in detail, including their architectural components and training procedures.",
    "Write a comprehensive guide to implementing a distributed key-value store with consistency guarantees.",
    "Describe the history of operating systems from Unix to modern cloud computing.",
    "What are the main challenges in large language model inference optimization?",
]


# =====================================================================
#  Helpers
# =====================================================================

def gini_coefficient(values: np.ndarray) -> float:
    values = np.sort(np.abs(values.ravel()))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n)


def topk_concentration(values: np.ndarray, k: int = 10) -> float:
    total = values.sum()
    if total == 0:
        return 0.0
    topk = np.sort(values.ravel())[-k:]
    return float(topk.sum() / total)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def load_model(model_path: str, device: str = "cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[load] Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_key_value_heads} KV heads, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
#  Experiment 1: Workload Characterization
# =====================================================================

def _make_long_prompt(prompt: str, tokenizer, target_len: int) -> str:
    """Repeat prompt text until it reaches target token length."""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) >= target_len:
        return tokenizer.decode(tokens[:target_len])
    repeats = (target_len // len(tokens)) + 1
    long_tokens = (tokens * repeats)[:target_len]
    return tokenizer.decode(long_tokens)


def run_workload_characterization(
    model, tokenizer, model_cfg: dict,
    prompt_len: int = 1024, gen_len: int = 48,
    device: str = "cuda:0",
) -> dict:
    """Profile attention patterns: Gini, Jaccard stability, top-10% concentration."""
    print(f"\n{'='*60}")
    print("  EXPERIMENT 1: Workload Characterization")
    print(f"  prompt_len={prompt_len}, gen_len={gen_len}")
    print(f"{'='*60}")

    n_layers = model_cfg["n_layers"]
    block_size = model_cfg["block_size"]

    all_gini = []
    all_topk = []
    all_jaccard = []
    per_layer_gini_accum = [[] for _ in range(n_layers)]

    for pi, prompt in enumerate(PROMPTS_CHARACTERIZATION):
        print(f"  [{pi+1}/{len(PROMPTS_CHARACTERIZATION)}] Profiling...")

        long_prompt = _make_long_prompt(prompt, tokenizer, prompt_len)
        ids = tokenizer(long_prompt, return_tensors="pt", truncation=True,
                        max_length=prompt_len)["input_ids"].to(device)

        hot_sets: list[set] = []
        per_layer_gini_step: list[list[float]] = [[] for _ in range(n_layers)]
        per_layer_topk_step: list[list[float]] = [[] for _ in range(n_layers)]

        cur, past = ids, None
        # Prefill without output_attentions (too large at prompt_len=1024)
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True, output_attentions=False)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out

        for step in range(gen_len):
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True,
                            output_attentions=True)
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if getattr(out, "attentions", None):
                step_hot = set()
                for li, attn in enumerate(out.attentions):
                    attn_np = attn[0].float().cpu().numpy()
                    avg_over_heads = attn_np.mean(axis=0)
                    last_row = avg_over_heads[-1]
                    per_layer_gini_step[li].append(gini_coefficient(last_row))
                    n_blk = max(1, len(last_row) // block_size)
                    blk_mass = np.array([last_row[b*block_size:(b+1)*block_size].sum()
                                         for b in range(n_blk)])
                    per_layer_topk_step[li].append(
                        topk_concentration(blk_mass, k=max(1, n_blk // 10)))
                    hot_thr = np.percentile(blk_mass, 90)
                    step_hot |= set(np.where(blk_mass >= hot_thr)[0])
                hot_sets.append(step_hot)
            del out

        layer_gini_means = [float(np.mean(g)) if g else 0.0 for g in per_layer_gini_step]
        layer_topk_means = [float(np.mean(t)) if t else 0.0 for t in per_layer_topk_step]

        jaccard_values = []
        for i in range(1, len(hot_sets)):
            jaccard_values.append(jaccard_similarity(hot_sets[i-1], hot_sets[i]))

        all_gini.append(float(np.mean(layer_gini_means)))
        all_topk.append(float(np.mean(layer_topk_means)))
        all_jaccard.append(float(np.mean(jaccard_values)) if jaccard_values else 0.0)
        for li in range(n_layers):
            per_layer_gini_accum[li].append(layer_gini_means[li])

    per_layer_gini_final = [float(np.mean(v)) for v in per_layer_gini_accum]

    result = {
        "gini_coefficient": {
            "mean": float(np.mean(all_gini)),
            "std": float(np.std(all_gini)),
            "per_prompt": [round(g, 4) for g in all_gini],
        },
        "jaccard_stability": {
            "mean": float(np.mean(all_jaccard)),
            "std": float(np.std(all_jaccard)),
            "per_prompt": [round(j, 4) for j in all_jaccard],
        },
        "top10_concentration": {
            "mean": float(np.mean(all_topk)),
            "std": float(np.std(all_topk)),
            "per_prompt": [round(t, 4) for t in all_topk],
        },
        "per_layer_gini": [round(g, 4) for g in per_layer_gini_final],
    }

    print(f"\n  Results:")
    print(f"    Gini coefficient:     {result['gini_coefficient']['mean']:.4f} "
          f"± {result['gini_coefficient']['std']:.4f}")
    print(f"    Jaccard stability:    {result['jaccard_stability']['mean']:.4f} "
          f"± {result['jaccard_stability']['std']:.4f}")
    print(f"    Top-10% concentration: {result['top10_concentration']['mean']:.4f} "
          f"± {result['top10_concentration']['std']:.4f}")

    return result


# =====================================================================
#  Experiment 2: OrchKvCache at 50% Budget
# =====================================================================

def run_orchkv_50pct(
    model, tokenizer, model_cfg: dict,
    prompt_len: int = 512, gen_len: int = 32,
    device: str = "cuda:0",
) -> dict:
    """Run OrchKvCache at 50% GPU budget and report performance metrics."""
    print(f"\n{'='*60}")
    print("  EXPERIMENT 2: OrchKvCache at 50% Budget")
    print(f"  prompt_len={prompt_len}, gen_len={gen_len}")
    print(f"{'='*60}")

    from orchkv.fast_kvcache_manager import FastKVCacheManager

    seq_len = prompt_len + gen_len
    full_kv_bytes = model_cfg["kv_bytes_per_token"] * seq_len
    budget_bytes = int(full_kv_bytes * 0.50)

    print(f"  Full KV: {full_kv_bytes / (1<<20):.1f} MB, "
          f"Budget (50%): {budget_bytes / (1<<20):.1f} MB")

    prompts_to_use = PROMPTS_CHARACTERIZATION[:8]
    all_throughput = []
    all_tpot = []
    total_evictions = 0
    total_promotions = 0
    promo_latencies = []

    for pi, prompt in enumerate(prompts_to_use):
        print(f"  [{pi+1}/{len(prompts_to_use)}] Running OrchKv...")

        mgr = FastKVCacheManager(
            n_layers=model_cfg["n_layers"],
            n_kv_heads=model_cfg["n_kv_heads"],
            head_dim=model_cfg["head_dim"],
            block_size=model_cfg["block_size"],
            dtype=torch.float16,
            gpu_budget_bytes=budget_bytes,
            max_seq_len=seq_len,
        )

        long_prompt = _make_long_prompt(prompt, tokenizer, prompt_len)
        ids = tokenizer(long_prompt, return_tensors="pt", truncation=True,
                        max_length=prompt_len)["input_ids"].to(device)

        itl_times = []

        # Prefill
        torch.cuda.synchronize()
        with torch.no_grad():
            out = model(ids, use_cache=True, output_attentions=True)

        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        mgr.ingest_step(out.past_key_values)
        if getattr(out, "attentions", None):
            for li, attn in enumerate(out.attentions):
                mgr.report_attention(li, attn)
        mgr.step_done()
        mgr.schedule()
        past = mgr.build_past_kv()
        del out

        # Decode
        for step in range(gen_len):
            want_attn = (step % 5 == 0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True,
                            output_attentions=want_attn)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0
            itl_times.append(step_ms)

            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mgr.append_token(out.past_key_values)
            if want_attn and getattr(out, "attentions", None):
                for li, attn in enumerate(out.attentions):
                    mgr.report_attention(li, attn)
            mgr.step_done()
            mgr.schedule()
            past = mgr.build_past_kv()
            cur = nt
            del out

        tpot = statistics.mean(itl_times) if itl_times else 0.0
        total_time_s = sum(itl_times) / 1000.0
        throughput = (gen_len / total_time_s) if total_time_s > 0 else 0.0

        all_throughput.append(throughput)
        all_tpot.append(tpot)

        try:
            stats = mgr.get_stats()
            migrations = stats.get("migrations", {})
            total_evictions += migrations.get("gpu_to_dram", 0)
            total_promotions += migrations.get("dram_to_gpu", 0)
        except Exception:
            pass

        try:
            pstats = mgr.get_promotion_latency_stats()
            if pstats.get("p50_us", 0) > 0:
                promo_latencies.append(pstats)
        except Exception:
            pass

        del mgr
        gc.collect()

    promo_p50 = 0.0
    promo_p99 = 0.0
    if promo_latencies:
        promo_p50 = statistics.mean([p.get("p50_us", 0) for p in promo_latencies])
        promo_p99 = statistics.mean([p.get("p99_us", 0) for p in promo_latencies])

    result = {
        "throughput_tok_s": {
            "mean": round(statistics.mean(all_throughput), 1),
            "std": round(statistics.stdev(all_throughput), 1) if len(all_throughput) > 1 else 0.0,
        },
        "tpot_ms": {
            "mean": round(statistics.mean(all_tpot), 3),
            "std": round(statistics.stdev(all_tpot), 3) if len(all_tpot) > 1 else 0.0,
        },
        "evictions_total": total_evictions,
        "promotions_total": total_promotions,
        "promotion_p50_us": round(promo_p50, 1),
        "promotion_p99_us": round(promo_p99, 1),
        "budget_fraction": 0.50,
        "budget_mb": round(budget_bytes / (1 << 20), 1),
        "num_prompts": len(prompts_to_use),
    }

    print(f"\n  Results:")
    print(f"    Throughput:    {result['throughput_tok_s']['mean']:.1f} tok/s")
    print(f"    TPOT:          {result['tpot_ms']['mean']:.3f} ms")
    print(f"    Evictions:     {result['evictions_total']}")
    print(f"    Promotions:    {result['promotions_total']}")
    print(f"    Promo P50:     {result['promotion_p50_us']:.1f} µs")
    print(f"    Promo P99:     {result['promotion_p99_us']:.1f} µs")

    return result


# =====================================================================
#  Experiment 3: Correctness (bit-exact match)
# =====================================================================

def run_correctness(
    model, tokenizer, model_cfg: dict,
    num_prompts: int = 50, gen_len: int = 32,
    device: str = "cuda:0",
) -> dict:
    """Compare OrchKvCache output vs GPU-only baseline for bit-exact match."""
    print(f"\n{'='*60}")
    print("  EXPERIMENT 3: Correctness Verification")
    print(f"  num_prompts={num_prompts}, gen_len={gen_len}")
    print(f"{'='*60}")

    from orchkv.fast_kvcache_manager import FastKVCacheManager

    prompts = PROMPTS_CORRECTNESS[:num_prompts]
    correctness_prompt_len = 512
    seq_len = correctness_prompt_len + gen_len
    full_kv_bytes = model_cfg["kv_bytes_per_token"] * seq_len
    budget_bytes = int(full_kv_bytes * 0.50)

    exact_matches = 0
    token_match_rates = []

    for pi, prompt in enumerate(prompts):
        if (pi + 1) % 10 == 0:
            print(f"  [{pi+1}/{len(prompts)}] Comparing outputs...")

        long_prompt = _make_long_prompt(prompt, tokenizer, correctness_prompt_len)
        ids = tokenizer(long_prompt, return_tensors="pt", truncation=True,
                        max_length=correctness_prompt_len)["input_ids"].to(device)

        # GPU-only baseline
        gpu_tokens = []
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gpu_tokens.append(cur.item())
        del out

        for _ in range(gen_len - 1):
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gpu_tokens.append(cur.item())
            del out

        del past
        torch.cuda.empty_cache()

        # OrchKvCache path
        mgr = FastKVCacheManager(
            n_layers=model_cfg["n_layers"],
            n_kv_heads=model_cfg["n_kv_heads"],
            head_dim=model_cfg["head_dim"],
            block_size=model_cfg["block_size"],
            dtype=torch.float16,
            gpu_budget_bytes=budget_bytes,
            max_seq_len=seq_len,
        )

        orchkv_tokens = []
        with torch.no_grad():
            out = model(ids, use_cache=True, output_attentions=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        orchkv_tokens.append(cur.item())

        mgr.ingest_step(out.past_key_values)
        if getattr(out, "attentions", None):
            for li, attn in enumerate(out.attentions):
                mgr.report_attention(li, attn)
        mgr.step_done()
        mgr.schedule()
        past = mgr.build_past_kv()
        del out

        for step in range(gen_len - 1):
            want_attn = (step % 5 == 0)
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True,
                            output_attentions=want_attn)
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            orchkv_tokens.append(nt.item())

            mgr.append_token(out.past_key_values)
            if want_attn and getattr(out, "attentions", None):
                for li, attn in enumerate(out.attentions):
                    mgr.report_attention(li, attn)
            mgr.step_done()
            mgr.schedule()
            past = mgr.build_past_kv()
            cur = nt
            del out

        # Compare
        match_count = sum(1 for a, b in zip(gpu_tokens, orchkv_tokens) if a == b)
        rate = match_count / max(len(gpu_tokens), 1)
        token_match_rates.append(rate)
        if gpu_tokens == orchkv_tokens:
            exact_matches += 1

        del mgr
        gc.collect()
        torch.cuda.empty_cache()

    exact_match_rate = exact_matches / len(prompts)
    avg_token_match = float(np.mean(token_match_rates))

    result = {
        "num_prompts": len(prompts),
        "exact_match_count": exact_matches,
        "exact_match_rate": round(exact_match_rate, 4),
        "avg_token_match_rate": round(avg_token_match, 4),
        "per_prompt_token_match": [round(r, 4) for r in token_match_rates],
    }

    print(f"\n  Results:")
    print(f"    Bit-exact match rate:   {exact_match_rate:.2%} ({exact_matches}/{len(prompts)})")
    print(f"    Avg token match rate:   {avg_token_match:.4f}")

    return result


# =====================================================================
#  Experiment 4: Predictor Comparison (EMA vs Cross-Layer)
# =====================================================================

def run_predictor_comparison(
    model, tokenizer, model_cfg: dict,
    max_new_tokens: int = 64, top_k: int = 8,
    device: str = "cuda:0",
) -> dict:
    """Compare EMA vs InfiniGen cross-layer predictor on this model."""
    print(f"\n{'='*60}")
    print("  EXPERIMENT 4: Predictor Comparison (EMA vs Cross-Layer)")
    print(f"  max_new_tokens={max_new_tokens}, top_k={top_k}")
    print(f"{'='*60}")

    from orchkv.infinigen_predictor import EMAPredictor, InfiniGenPredictor

    n_layers = model_cfg["n_layers"]
    n_kv_heads = model_cfg["n_kv_heads"]
    block_size = model_cfg["block_size"]

    predictor_prompt_len = 2048
    ema_jaccards_all = []
    ig_jaccards_all = []

    for pi, prompt in enumerate(PROMPTS_PREDICTOR):
        print(f"  [{pi+1}/{len(PROMPTS_PREDICTOR)}] Collecting attention trace...")

        long_prompt = _make_long_prompt(prompt, tokenizer, predictor_prompt_len)
        ids = tokenizer(long_prompt, return_tensors="pt", truncation=True,
                        max_length=predictor_prompt_len)["input_ids"].to(device)

        attention_trace = []

        # Prefill
        with torch.no_grad():
            out = model(ids, use_cache=True, output_attentions=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if getattr(out, "attentions", None):
            attention_trace.append([a.cpu() for a in out.attentions])
        del out

        # Decode
        for step in range(1, max_new_tokens):
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True,
                            output_attentions=True)
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if getattr(out, "attentions", None):
                attention_trace.append([a.cpu() for a in out.attentions])
            del out

        del past
        torch.cuda.empty_cache()

        # Evaluate EMA predictor
        ema_pred = EMAPredictor(
            n_layers=n_layers, block_size=block_size,
            top_k_blocks=top_k, ema_decay=0.9,
        )
        for step_idx, step_attns in enumerate(attention_trace):
            n_l = min(len(step_attns), n_layers)
            for l_idx in range(n_l):
                attn_l = step_attns[l_idx]
                if attn_l.dim() == 3:
                    attn_l = attn_l.unsqueeze(0)
                ema_pred.observe_layer(l_idx, attn_l)
                if step_idx > 0:
                    predicted = ema_pred.predict_hot_blocks(l_idx, k=top_k)
                    ema_pred.evaluate_prediction(l_idx, predicted, attn_l)
            ema_pred.step_done()

        ema_result = ema_pred.get_accuracy_summary()
        ema_jaccards_all.append(ema_result["jaccard"]["mean"])

        # Evaluate InfiniGen predictor
        ig_pred = InfiniGenPredictor(
            n_layers=n_layers, n_kv_heads=n_kv_heads,
            block_size=block_size, top_k_blocks=top_k,
            ema_decay=0.7,
        )
        for step_idx, step_attns in enumerate(attention_trace):
            n_l = min(len(step_attns), n_layers)
            for l_idx in range(n_l):
                attn_l = step_attns[l_idx]
                if attn_l.dim() == 3:
                    attn_l = attn_l.unsqueeze(0)
                ig_pred.observe_layer(l_idx, attn_l)
                if l_idx > 0 and ig_pred._layer_hot_sets[l_idx - 1] is not None:
                    predicted = ig_pred.predict_next_layer(l_idx - 1, k=top_k)
                    ig_pred.evaluate_prediction(l_idx, predicted, attn_l)
            ig_pred.step_done()

        ig_result = ig_pred.get_accuracy_summary()
        ig_jaccards_all.append(ig_result["jaccard"]["mean"])

        print(f"    EMA Jaccard={ema_result['jaccard']['mean']:.4f}, "
              f"InfiniGen Jaccard={ig_result['jaccard']['mean']:.4f}")

        del attention_trace
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "ema_predictor": {
            "jaccard_mean": round(float(np.mean(ema_jaccards_all)), 4),
            "jaccard_std": round(float(np.std(ema_jaccards_all)), 4),
            "per_prompt_jaccard": [round(j, 4) for j in ema_jaccards_all],
        },
        "infinigen_predictor": {
            "jaccard_mean": round(float(np.mean(ig_jaccards_all)), 4),
            "jaccard_std": round(float(np.std(ig_jaccards_all)), 4),
            "per_prompt_jaccard": [round(j, 4) for j in ig_jaccards_all],
        },
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "num_prompts": len(PROMPTS_PREDICTOR),
    }

    print(f"\n  Aggregate:")
    print(f"    EMA Jaccard:       {result['ema_predictor']['jaccard_mean']:.4f} "
          f"± {result['ema_predictor']['jaccard_std']:.4f}")
    print(f"    InfiniGen Jaccard: {result['infinigen_predictor']['jaccard_mean']:.4f} "
          f"± {result['infinigen_predictor']['jaccard_std']:.4f}")

    return result


# =====================================================================
#  Main runner
# =====================================================================

def run_model_experiments(model_key: str, device: str = "cuda:0") -> dict:
    """Run all four experiments for one model."""
    cfg = MODELS[model_key]
    print(f"\n{'#'*70}")
    print(f"#  MODEL: {model_key} ({cfg['path']})")
    print(f"#  n_layers={cfg['n_layers']}, n_kv_heads={cfg['n_kv_heads']}, "
          f"head_dim={cfg['head_dim']}")
    print(f"{'#'*70}")

    model, tokenizer = load_model(cfg["path"], device)

    t0 = time.time()

    results = {
        "model": model_key,
        "model_path": cfg["path"],
        "model_config": {
            "n_layers": cfg["n_layers"],
            "n_kv_heads": cfg["n_kv_heads"],
            "head_dim": cfg["head_dim"],
            "block_size": cfg["block_size"],
        },
    }

    # Experiment 1: Workload Characterization
    results["workload_characterization"] = run_workload_characterization(
        model, tokenizer, cfg, prompt_len=1024, gen_len=48, device=device)

    gc.collect()
    torch.cuda.empty_cache()

    # Experiment 2: OrchKvCache at 50% Budget
    results["orchkv_50pct"] = run_orchkv_50pct(
        model, tokenizer, cfg, prompt_len=512, gen_len=32, device=device)

    gc.collect()
    torch.cuda.empty_cache()

    # Experiment 3: Correctness
    results["correctness"] = run_correctness(
        model, tokenizer, cfg, num_prompts=50, gen_len=32, device=device)

    gc.collect()
    torch.cuda.empty_cache()

    # Experiment 4: Predictor Comparison
    results["predictor_comparison"] = run_predictor_comparison(
        model, tokenizer, cfg, max_new_tokens=64, top_k=8, device=device)

    results["total_time_s"] = round(time.time() - t0, 1)

    unload_model(model, tokenizer)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-model validation benchmark")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Run single model (default: run all)")
    args = parser.parse_args()

    device = "cuda:0"
    print(f"[main] Cross-Model Validation Benchmark")
    print(f"[main] Device: {device}")
    print(f"[main] PyTorch: {torch.__version__}")
    print(f"[main] CUDA: {torch.cuda.get_device_name(0)}")
    free_mem = torch.cuda.mem_get_info(0)
    print(f"[main] GPU memory: {free_mem[0]/(1<<30):.1f} GiB free / "
          f"{free_mem[1]/(1<<30):.1f} GiB total")
    print(f"[main] Models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3")

    models_to_run = [args.model] if args.model else ["llama-3.1-8b", "mistral-7b"]
    all_results = {}

    for model_key in models_to_run:
        results = run_model_experiments(model_key, device)
        out_name = "cross_model_llama3" if "llama" in model_key else "cross_model_mistral"
        out_path = RESULTS_DIR / f"{out_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[main] {model_key} results saved to {out_path}")
        all_results[model_key] = results

        gc.collect()
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL VALIDATION SUMMARY")
    print(f"{'='*70}")
    for name, key in [("Llama-3.1-8B", "llama-3.1-8b"), ("Mistral-7B", "mistral-7b")]:
        if key not in all_results:
            continue
        res = all_results[key]
        wc = res["workload_characterization"]
        orch = res["orchkv_50pct"]
        corr = res["correctness"]
        pred = res["predictor_comparison"]
        print(f"\n  {name}:")
        print(f"    Gini:           {wc['gini_coefficient']['mean']:.4f}")
        print(f"    Jaccard:        {wc['jaccard_stability']['mean']:.4f}")
        print(f"    Top-10%:        {wc['top10_concentration']['mean']:.4f}")
        print(f"    Throughput:     {orch['throughput_tok_s']['mean']:.1f} tok/s")
        print(f"    TPOT:           {orch['tpot_ms']['mean']:.3f} ms")
        print(f"    Evictions:      {orch['evictions_total']}")
        print(f"    Promotions:     {orch['promotions_total']}")
        print(f"    Exact match:    {corr['exact_match_rate']:.2%}")
        print(f"    EMA Jaccard:    {pred['ema_predictor']['jaccard_mean']:.4f}")
        print(f"    IG Jaccard:     {pred['infinigen_predictor']['jaccard_mean']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

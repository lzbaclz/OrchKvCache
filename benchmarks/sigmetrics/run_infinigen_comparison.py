#!/usr/bin/env python3
"""
InfiniGen vs OrchKvCache predictor comparison benchmark.

Compares three KV importance predictors on Qwen2.5-7B decode:
  1. OrchKvCache (EMA-based, per-layer independent)
  2. InfiniGen-style (cross-layer prediction from layer L -> L+1)
  3. Oracle (Belady-like, future knowledge)

Reports precision@K, recall@K, Jaccard stability for each predictor.

Usage:
    # CUDA_VISIBLE_DEVICES=1
    conda activate orchkv
    PYTHONPATH=build/bindings:python python benchmarks/sigmetrics/run_infinigen_comparison.py \
        --num_prompts 4 --max_new_tokens 64

Model: /public/model_zoo/Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from orchkv.infinigen_predictor import (
    EMAPredictor,
    InfiniGenPredictor,
    OraclePredictor,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = "/public/model_zoo/Qwen2.5-7B"
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 16


# =====================================================================
#  Model loading
# =====================================================================

def load_model(device: str = "cuda:0"):
    """Load Qwen2.5-7B with eager attention for attention output."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[load] Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_key_value_heads} KV heads")
    return model, tokenizer


# =====================================================================
#  Attention collection
# =====================================================================

def collect_attention_trace(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
) -> Tuple[List[List[torch.Tensor]], List[int]]:
    """Run decode with output_attentions=True to collect ground truth.

    Returns:
        attention_trace: list of decode steps, each containing per-layer
            attention tensors [batch, heads, q_len, kv_len]
        generated_tokens: list of generated token IDs
    """
    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=4096)["input_ids"].to(device)

    attention_trace = []
    generated_tokens = []

    # Prefill
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=True)

    past = out.past_key_values
    cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens.append(cur.item())

    # Store prefill attention (per-layer list)
    prefill_attns = [a.cpu() for a in out.attentions]
    attention_trace.append(prefill_attns)
    del out

    # Decode steps
    for step in range(1, max_new_tokens):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=True)

        past = out.past_key_values
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(nt.item())

        step_attns = [a.cpu() for a in out.attentions]
        attention_trace.append(step_attns)
        del out
        cur = nt

    return attention_trace, generated_tokens


# =====================================================================
#  Predictor evaluation
# =====================================================================

def evaluate_ema_predictor(
    attention_trace: List[List[torch.Tensor]],
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate EMA-based predictor (OrchKvCache approach)."""
    predictor = EMAPredictor(
        n_layers=N_LAYERS, block_size=BLOCK_SIZE,
        top_k_blocks=top_k, ema_decay=0.9,
    )

    for step_idx, step_attns in enumerate(attention_trace):
        n_layers = min(len(step_attns), N_LAYERS)
        for l_idx in range(n_layers):
            attn_l = step_attns[l_idx]
            if attn_l.dim() == 3:
                attn_l = attn_l.unsqueeze(0)

            # EMA observes current layer, predicts same layer next step
            predictor.observe_layer(l_idx, attn_l)

            # Evaluate: predict from previous step's EMA for this layer
            if step_idx > 0:
                predicted = predictor.predict_hot_blocks(l_idx, k=top_k)
                predictor.evaluate_prediction(l_idx, predicted, attn_l)

        predictor.step_done()

    return predictor.get_accuracy_summary()


def evaluate_infinigen_predictor(
    attention_trace: List[List[torch.Tensor]],
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate InfiniGen cross-layer predictor."""
    predictor = InfiniGenPredictor(
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,
        block_size=BLOCK_SIZE, top_k_blocks=top_k,
        ema_decay=0.7,
    )

    for step_idx, step_attns in enumerate(attention_trace):
        n_layers = min(len(step_attns), N_LAYERS)
        for l_idx in range(n_layers):
            attn_l = step_attns[l_idx]
            if attn_l.dim() == 3:
                attn_l = attn_l.unsqueeze(0)

            # Observe layer L
            predictor.observe_layer(l_idx, attn_l)

            # Evaluate cross-layer prediction: L-1 predicts L
            if l_idx > 0 and predictor._layer_hot_sets[l_idx - 1] is not None:
                predicted = predictor.predict_next_layer(l_idx - 1, k=top_k)
                predictor.evaluate_prediction(l_idx, predicted, attn_l)

        predictor.step_done()

    return predictor.get_accuracy_summary()


def evaluate_oracle_predictor(
    attention_trace: List[List[torch.Tensor]],
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate Oracle (Belady-like) predictor."""
    oracle = OraclePredictor(
        n_layers=N_LAYERS, block_size=BLOCK_SIZE,
        top_k_blocks=top_k, criticality_threshold=0.01,
    )
    oracle.load_trace(attention_trace)

    # Evaluate: for each (step, layer), compare oracle vs actual
    precisions, recalls, jaccards = [], [], []
    n_steps = len(attention_trace)

    for step_idx in range(n_steps):
        step_attns = attention_trace[step_idx]
        n_layers = min(len(step_attns), N_LAYERS)

        for l_idx in range(n_layers):
            optimal_hot = oracle.get_optimal_hot_set(step_idx, l_idx, k=top_k)
            # Oracle's optimal set IS the actual top-K, so by definition it's perfect
            # But we measure forward-reuse-distance variant vs immediate top-K
            attn_l = step_attns[l_idx]
            if attn_l.dim() == 3:
                attn_l = attn_l.unsqueeze(0)
            B, H, Q, S = attn_l.shape
            n_blocks = math.ceil(S / BLOCK_SIZE)

            # Actual top-K for this step
            avg_attn = attn_l.float().mean(dim=(0, 2))
            padded_len = n_blocks * BLOCK_SIZE
            if S < padded_len:
                avg_attn = torch.nn.functional.pad(avg_attn, (0, padded_len - S))
            avg_attn = avg_attn[:, :padded_len].view(H, n_blocks, BLOCK_SIZE)
            actual_scores = avg_attn.sum(dim=-1).mean(dim=0)
            k_actual = min(top_k, n_blocks)
            actual_hot = set(actual_scores.topk(k_actual).indices.tolist())

            if optimal_hot and actual_hot:
                p = len(optimal_hot & actual_hot) / len(optimal_hot)
                r = len(optimal_hot & actual_hot) / len(actual_hot)
                union = optimal_hot | actual_hot
                j = len(optimal_hot & actual_hot) / len(union) if union else 1.0
                precisions.append(p)
                recalls.append(r)
                jaccards.append(j)

    def _stats(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        t = torch.tensor(vals)
        return {
            "mean": t.mean().item(),
            "std": t.std().item() if len(vals) > 1 else 0.0,
            "count": len(vals),
        }

    return {
        "precision_at_k": _stats(precisions),
        "recall_at_k": _stats(recalls),
        "jaccard": _stats(jaccards),
        "total_predictions": len(precisions),
        "top_k": top_k,
    }


# =====================================================================
#  Main comparison
# =====================================================================

def run_comparison(
    num_prompts: int = 4,
    max_new_tokens: int = 64,
    top_k: int = 8,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Run full predictor comparison and return results."""
    model, tokenizer = load_model(device)

    prompts = [
        "Explain the key differences between transformers and recurrent neural networks in detail.",
        "Write a comprehensive guide to implementing a distributed key-value store.",
        "Describe the history of operating systems from Unix to modern cloud computing.",
        "What are the main challenges in large language model inference optimization?",
        "Summarize the evolution of computer architecture from von Neumann to modern GPUs.",
        "Explain how garbage collection works in Java, Go, and Rust.",
        "Describe the CAP theorem and its implications for distributed databases.",
        "What are attention sinks in large language models and why do they matter?",
    ][:num_prompts]

    all_results = {
        "config": {
            "model": MODEL_PATH,
            "n_layers": N_LAYERS,
            "n_kv_heads": N_KV_HEADS,
            "block_size": BLOCK_SIZE,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
        },
        "per_prompt": [],
        "aggregate": {},
    }

    ema_all = {"precision": [], "recall": [], "jaccard": []}
    ig_all = {"precision": [], "recall": [], "jaccard": []}
    oracle_all = {"precision": [], "recall": [], "jaccard": []}

    for pi, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"  Prompt {pi+1}/{len(prompts)}: {prompt[:60]}...")
        print(f"{'='*60}")

        t0 = time.time()
        print("[trace] Collecting attention trace...")
        attention_trace, tokens = collect_attention_trace(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        trace_time = time.time() - t0
        print(f"[trace] Collected {len(attention_trace)} steps in {trace_time:.1f}s")

        print("[eval] Evaluating EMA predictor (OrchKvCache)...")
        ema_result = evaluate_ema_predictor(attention_trace, top_k)

        print("[eval] Evaluating InfiniGen predictor (cross-layer)...")
        ig_result = evaluate_infinigen_predictor(attention_trace, top_k)

        print("[eval] Evaluating Oracle predictor (Belady)...")
        oracle_result = evaluate_oracle_predictor(attention_trace, top_k)

        prompt_result = {
            "prompt_idx": pi,
            "prompt_len": len(tokenizer.encode(prompt)),
            "generated_tokens": len(tokens),
            "trace_collection_time_s": round(trace_time, 2),
            "ema": ema_result,
            "infinigen": ig_result,
            "oracle": oracle_result,
        }
        all_results["per_prompt"].append(prompt_result)

        # Accumulate for aggregate
        ema_all["precision"].append(ema_result["precision_at_k"]["mean"])
        ema_all["recall"].append(ema_result["recall_at_k"]["mean"])
        ema_all["jaccard"].append(ema_result["jaccard"]["mean"])
        ig_all["precision"].append(ig_result["precision_at_k"]["mean"])
        ig_all["recall"].append(ig_result["recall_at_k"]["mean"])
        ig_all["jaccard"].append(ig_result["jaccard"]["mean"])
        oracle_all["precision"].append(oracle_result["precision_at_k"]["mean"])
        oracle_all["recall"].append(oracle_result["recall_at_k"]["mean"])
        oracle_all["jaccard"].append(oracle_result["jaccard"]["mean"])

        _print_prompt_summary(ema_result, ig_result, oracle_result)

        # Free memory
        del attention_trace
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate results
    def _agg(vals):
        t = torch.tensor(vals)
        return {"mean": t.mean().item(), "std": t.std().item() if len(vals) > 1 else 0.0}

    all_results["aggregate"] = {
        "ema": {
            "precision_at_k": _agg(ema_all["precision"]),
            "recall_at_k": _agg(ema_all["recall"]),
            "jaccard": _agg(ema_all["jaccard"]),
        },
        "infinigen": {
            "precision_at_k": _agg(ig_all["precision"]),
            "recall_at_k": _agg(ig_all["recall"]),
            "jaccard": _agg(ig_all["jaccard"]),
        },
        "oracle": {
            "precision_at_k": _agg(oracle_all["precision"]),
            "recall_at_k": _agg(oracle_all["recall"]),
            "jaccard": _agg(oracle_all["jaccard"]),
        },
    }

    _print_final_summary(all_results["aggregate"])
    return all_results


def _print_prompt_summary(ema: Dict, ig: Dict, oracle: Dict):
    """Print comparison table for one prompt."""
    print(f"\n  {'Predictor':<20s} {'Prec@K':>8s} {'Rec@K':>8s} {'Jaccard':>8s}")
    print(f"  {'-'*46}")
    for name, res in [("OrchKv (EMA)", ema), ("InfiniGen (X-layer)", ig), ("Oracle (Belady)", oracle)]:
        p = res["precision_at_k"]["mean"]
        r = res["recall_at_k"]["mean"]
        j = res["jaccard"]["mean"]
        print(f"  {name:<20s} {p:>8.4f} {r:>8.4f} {j:>8.4f}")


def _print_final_summary(aggregate: Dict):
    """Print final aggregate comparison table."""
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS (across all prompts)")
    print(f"{'='*60}")
    print(f"  {'Predictor':<20s} {'Prec@K':>10s} {'Rec@K':>10s} {'Jaccard':>10s}")
    print(f"  {'-'*52}")
    for name, key in [("OrchKv (EMA)", "ema"),
                      ("InfiniGen (X-layer)", "infinigen"),
                      ("Oracle (Belady)", "oracle")]:
        res = aggregate[key]
        p = res["precision_at_k"]["mean"]
        r = res["recall_at_k"]["mean"]
        j = res["jaccard"]["mean"]
        print(f"  {name:<20s} {p:>10.4f} {r:>10.4f} {j:>10.4f}")
    print(f"{'='*60}")


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="InfiniGen vs OrchKvCache predictor comparison")
    parser.add_argument("--num_prompts", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=8,
                        help="Number of blocks in the predicted hot set")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="infinigen_comparison")
    args = parser.parse_args()

    print(f"[main] InfiniGen Comparison Benchmark")
    print(f"[main] Model: {MODEL_PATH}")
    print(f"[main] Prompts: {args.num_prompts}, tokens: {args.max_new_tokens}, K={args.top_k}")
    print(f"[main] Device: {args.device}")

    t0 = time.time()
    results = run_comparison(
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        device=args.device,
    )
    elapsed = time.time() - t0

    results["total_time_s"] = round(elapsed, 1)

    out_path = RESULTS_DIR / f"{args.output}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[main] Results saved to {out_path}")
    print(f"[main] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

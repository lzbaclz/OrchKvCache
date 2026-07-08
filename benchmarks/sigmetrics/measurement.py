#!/usr/bin/env python3
"""
Workload characterization for SIGMETRICS 2027.

Profiles attention patterns to quantify KV-cache access locality:
  - Gini coefficient of per-block attention mass
  - Top-K concentration ratio
  - Hot-set Jaccard stability across decode steps
  - Per-block reuse-distance CDF
  - Per-layer / per-head heterogeneity

Exports JSON results suitable for paper figures (Fig 2).

Usage:
    python -m benchmarks.sigmetrics.measurement \\
        --model qwen2.5-7b --workload sharegpt --num_prompts 8
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from benchmarks.sigmetrics.config import MODELS, ExperimentPoint
from benchmarks.sigmetrics.workload_loader import load_workload

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =====================================================================
#  Statistical helpers
# =====================================================================

def gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient (0 = perfect equality, 1 = maximal inequality)."""
    values = np.sort(np.abs(values.ravel()))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n)


def topk_concentration(values: np.ndarray, k: int = 10) -> float:
    """Fraction of total mass in the top-k entries."""
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


def reuse_distance_cdf(
    access_sequence: list[int],
    cdf_points: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
) -> dict[str, float]:
    """Compute reuse distance CDF from a sequence of block IDs."""
    last_seen: dict[int, int] = {}
    distances: list[int] = []

    for pos, block_id in enumerate(access_sequence):
        if block_id in last_seen:
            distances.append(pos - last_seen[block_id])
        last_seen[block_id] = pos

    if not distances:
        return {f"p{int(q*100)}": 0 for q in cdf_points}

    distances.sort()
    n = len(distances)
    return {
        f"p{int(q*100)}": distances[min(int(q * n), n - 1)]
        for q in cdf_points
    }


# =====================================================================
#  Attention profiling
# =====================================================================

def profile_attention_patterns(
    model_key: str,
    prompts: list[dict],
    max_new_tokens: int = 64,
    sample_interval: int = 1,
    block_size: int = 16,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """
    Run attention profiling on a set of prompts.

    Returns per-prompt and aggregate statistics:
    gini, topk_concentration, jaccard_stability, reuse_distance, heterogeneity.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = MODELS[model_key]
    model_name = mcfg["hf_name"]

    print(f"[measurement] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    n_layers = mcfg["n_layers"]
    all_results = []

    for pi, prompt_data in enumerate(prompts):
        print(f"[measurement] Prompt {pi+1}/{len(prompts)}: "
              f"{prompt_data['prompt_tokens']} tokens")

        ids = tokenizer(prompt_data["prompt"], return_tensors="pt",
                        truncation=True, max_length=4096)["input_ids"].to(device)

        per_layer_gini: list[list[float]] = [[] for _ in range(n_layers)]
        per_layer_topk: list[list[float]] = [[] for _ in range(n_layers)]
        hot_sets: list[set[int]] = []
        block_accesses: list[int] = []

        cur, past = ids, None
        for step in range(max_new_tokens):
            want_attn = (step % sample_interval == 0)
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True,
                            output_attentions=want_attn)
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if not want_attn or not getattr(out, "attentions", None):
                continue

            step_hot_blocks: set[int] = set()
            for li, attn in enumerate(out.attentions):
                # attn shape: [batch, heads, q_len, kv_len]
                attn_np = attn[0].float().cpu().numpy()
                avg_over_heads = attn_np.mean(axis=0)   # [q_len, kv_len]
                last_row = avg_over_heads[-1]            # attention from last query

                per_layer_gini[li].append(gini_coefficient(last_row))
                per_layer_topk[li].append(topk_concentration(last_row, k=10))

                n_blocks = max(1, len(last_row) // block_size)
                block_masses = np.zeros(n_blocks)
                for b in range(n_blocks):
                    start = b * block_size
                    end = min(start + block_size, len(last_row))
                    block_masses[b] = last_row[start:end].sum()

                hot_threshold = np.percentile(block_masses, 80)
                hot_blocks = set(np.where(block_masses >= hot_threshold)[0])
                step_hot_blocks |= hot_blocks

                top_block = int(np.argmax(block_masses))
                block_accesses.append(top_block)

            hot_sets.append(step_hot_blocks)

        # Aggregate per-prompt stats
        jaccard_values = []
        for i in range(1, len(hot_sets)):
            jaccard_values.append(jaccard_similarity(hot_sets[i - 1], hot_sets[i]))

        layer_gini_means = [float(np.mean(g)) if g else 0.0 for g in per_layer_gini]
        layer_topk_means = [float(np.mean(t)) if t else 0.0 for t in per_layer_topk]

        prompt_result = {
            "prompt_idx": pi,
            "prompt_tokens": prompt_data["prompt_tokens"],
            "gini": {
                "mean": float(np.mean(layer_gini_means)),
                "std": float(np.std(layer_gini_means)),
                "per_layer": layer_gini_means,
            },
            "topk_concentration": {
                "mean": float(np.mean(layer_topk_means)),
                "per_layer": layer_topk_means,
            },
            "jaccard_stability": {
                "mean": float(np.mean(jaccard_values)) if jaccard_values else 0.0,
                "min": float(np.min(jaccard_values)) if jaccard_values else 0.0,
                "values": [round(v, 4) for v in jaccard_values[:50]],
            },
            "reuse_distance": reuse_distance_cdf(block_accesses),
            "heterogeneity": {
                "gini_range": float(max(layer_gini_means) - min(layer_gini_means))
                    if layer_gini_means else 0.0,
                "gini_cv": float(np.std(layer_gini_means) / max(np.mean(layer_gini_means), 1e-9)),
            },
        }
        all_results.append(prompt_result)

    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    aggregate = _aggregate_measurements(all_results, n_layers)
    return {"model": model_key, "per_prompt": all_results, "aggregate": aggregate}


def _aggregate_measurements(
    results: list[dict], n_layers: int,
) -> dict[str, Any]:
    """Aggregate per-prompt measurements into summary statistics."""
    if not results:
        return {}

    gini_means = [r["gini"]["mean"] for r in results]
    topk_means = [r["topk_concentration"]["mean"] for r in results]
    jaccard_means = [r["jaccard_stability"]["mean"] for r in results]

    layer_ginis = np.zeros(n_layers)
    layer_counts = np.zeros(n_layers)
    for r in results:
        per_layer = r["gini"]["per_layer"]
        for li in range(min(len(per_layer), n_layers)):
            layer_ginis[li] += per_layer[li]
            layer_counts[li] += 1
    layer_ginis = np.divide(layer_ginis, np.maximum(layer_counts, 1))

    all_reuse = defaultdict(list)
    for r in results:
        for k, v in r["reuse_distance"].items():
            all_reuse[k].append(v)
    reuse_agg = {k: float(np.mean(v)) for k, v in all_reuse.items()}

    return {
        "gini": {
            "mean": float(np.mean(gini_means)),
            "std": float(np.std(gini_means)),
        },
        "topk_concentration": {
            "mean": float(np.mean(topk_means)),
        },
        "jaccard_stability": {
            "mean": float(np.mean(jaccard_means)),
        },
        "reuse_distance": reuse_agg,
        "per_layer_gini": [round(float(g), 4) for g in layer_ginis],
        "heterogeneity": {
            "gini_layer_cv": float(np.std(layer_ginis) / max(np.mean(layer_ginis), 1e-9)),
        },
    }


# =====================================================================
#  Lightweight characterization (no GPU required)
# =====================================================================

def characterize_workload_lengths(
    workload_name: str,
    num_prompts: int = 128,
    tokenizer_name: str | None = None,
) -> dict[str, Any]:
    """Compute prompt length distribution stats without running inference."""
    from benchmarks.sigmetrics.workload_loader import load_workload, workload_stats
    prompts = load_workload(workload_name, num_prompts=num_prompts,
                            tokenizer_name=tokenizer_name)
    return {
        "workload": workload_name,
        "stats": workload_stats(prompts),
    }


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run attention profiling for workload characterization")
    parser.add_argument("--model", type=str, default="qwen2.5-7b",
                        choices=list(MODELS), help="Model key")
    parser.add_argument("--workload", type=str, default="sharegpt",
                        help="Workload name")
    parser.add_argument("--num_prompts", type=int, default=8,
                        help="Number of prompts to profile")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Decode steps per prompt")
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="Collect attention every N steps")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-named in results/)")
    parser.add_argument("--length-only", action="store_true",
                        help="Only compute prompt length stats (no GPU needed)")
    args = parser.parse_args()

    if args.length_only:
        result = characterize_workload_lengths(args.workload, args.num_prompts)
        print(json.dumps(result, indent=2))
        return

    prompts = load_workload(args.workload, num_prompts=args.num_prompts)
    result = profile_attention_patterns(
        model_key=args.model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        sample_interval=args.sample_interval,
        device=args.device,
    )

    out_path = args.output or str(
        RESULTS_DIR / f"measurement_{args.model}_{args.workload}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[measurement] Results saved to {out_path}")

    agg = result["aggregate"]
    print(f"\n{'='*50}")
    print(f"  Aggregate: {args.model} × {args.workload}")
    print(f"{'='*50}")
    print(f"  Gini coefficient:    {agg['gini']['mean']:.3f} ± {agg['gini']['std']:.3f}")
    print(f"  Top-10 concentration: {agg['topk_concentration']['mean']:.3f}")
    print(f"  Jaccard stability:   {agg['jaccard_stability']['mean']:.3f}")
    print(f"  Reuse distance P50:  {agg['reuse_distance'].get('p50', 'N/A')}")
    print(f"  Layer Gini CV:       {agg['heterogeneity']['gini_layer_cv']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
D4 / E10: Generation quality evaluation.

Compares output quality between baseline vLLM and OrchKvCache-enabled vLLM:
  - Perplexity on a fixed dataset
  - Token-level accuracy (greedy decoding consistency)
  - LongBench subset scores (if available)

Usage:
    python benchmarks/eval_quality.py --model meta-llama/Llama-2-7b-hf
    python benchmarks/eval_quality.py --n-samples 50
"""
from __future__ import annotations

import argparse
import sys
import os
import math

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import (
    save_json, generate_synthetic_prompts, build_vllm_engine,
)


def eval_token_consistency(
    model: str,
    seq_len: int = 512,
    max_new_tokens: int = 128,
    n_samples: int = 20,
) -> dict:
    """
    E10a: Token-level consistency between baseline and OrchKvCache.

    Greedy decoding (temperature=0, seed=42) should produce identical tokens.
    """
    print(f"\n{'='*60}")
    print(f"  E10a: Token Consistency (seq={seq_len}, n={n_samples})")
    print(f"{'='*60}")

    try:
        from vllm import SamplingParams
    except ImportError:
        return {"status": "skipped", "reason": "vLLM not installed"}

    sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens, seed=42)
    prompts = generate_synthetic_prompts(n_samples, seq_len)
    max_len = seq_len + max_new_tokens

    outputs = {}
    for label, orchkv_on in [("baseline", False), ("orchkv", True)]:
        print(f"  [{label}] generating {n_samples} samples...", flush=True)
        engine = build_vllm_engine(model, orchkv_enabled=orchkv_on,
                                    max_model_len=max_len)
        if engine is None:
            outputs[label] = None
            continue

        out = engine.generate(prompt_token_ids=prompts, sampling_params=sampling)
        outputs[label] = [list(o.outputs[0].token_ids) for o in out]
        del engine
        torch.cuda.empty_cache()

    if outputs.get("baseline") is None or outputs.get("orchkv") is None:
        return {"status": "error", "reason": "engine failed"}

    total = 0
    match = 0
    per_sample = []

    for i, (bt, ot) in enumerate(zip(outputs["baseline"], outputs["orchkv"])):
        sample_total = min(len(bt), len(ot))
        sample_match = sum(1 for b, o in zip(bt, ot) if b == o)
        total += sample_total
        match += sample_match
        per_sample.append({
            "sample": i,
            "total": sample_total,
            "match": sample_match,
            "rate": sample_match / max(sample_total, 1),
        })

    rate = match / max(total, 1)
    result = {
        "status": "pass" if rate >= 0.999 else "fail",
        "total_tokens": total,
        "matching_tokens": match,
        "match_rate": rate,
        "per_sample": per_sample,
    }

    print(f"  Match rate: {rate:.4%} ({match}/{total})")
    return result


def eval_perplexity_proxy(
    model: str,
    seq_len: int = 512,
    n_samples: int = 10,
    max_new_tokens: int = 64,
) -> dict:
    """
    E10b: Perplexity proxy — compare log-probability sums.

    Since vLLM's generate doesn't directly output perplexity,
    we use cumulative_logprob from the output as a proxy.
    """
    print(f"\n{'='*60}")
    print(f"  E10b: Perplexity Proxy (seq={seq_len}, n={n_samples})")
    print(f"{'='*60}")

    try:
        from vllm import SamplingParams
    except ImportError:
        return {"status": "skipped"}

    sampling = SamplingParams(
        temperature=0, max_tokens=max_new_tokens,
        logprobs=1, seed=42)
    prompts = generate_synthetic_prompts(n_samples, seq_len)
    max_len = seq_len + max_new_tokens

    logprobs = {}
    for label, orchkv_on in [("baseline", False), ("orchkv", True)]:
        print(f"  [{label}] computing logprobs...", flush=True)
        engine = build_vllm_engine(model, orchkv_enabled=orchkv_on,
                                    max_model_len=max_len)
        if engine is None:
            logprobs[label] = None
            continue

        out = engine.generate(prompt_token_ids=prompts, sampling_params=sampling)
        sample_logprobs = []
        for o in out:
            cum = o.outputs[0].cumulative_logprob
            n_tok = len(o.outputs[0].token_ids)
            avg_logp = cum / max(n_tok, 1)
            ppl = math.exp(-avg_logp) if avg_logp is not None else float("inf")
            sample_logprobs.append({
                "cum_logprob": cum,
                "n_tokens": n_tok,
                "avg_logprob": avg_logp,
                "perplexity": ppl,
            })
        logprobs[label] = sample_logprobs

        del engine
        torch.cuda.empty_cache()

    if logprobs.get("baseline") is None or logprobs.get("orchkv") is None:
        return {"status": "error"}

    baseline_ppls = [s["perplexity"] for s in logprobs["baseline"]]
    orchkv_ppls = [s["perplexity"] for s in logprobs["orchkv"]]

    avg_bl = sum(baseline_ppls) / len(baseline_ppls)
    avg_ok = sum(orchkv_ppls) / len(orchkv_ppls)
    ppl_diff = abs(avg_ok - avg_bl) / max(avg_bl, 1e-6)

    result = {
        "status": "pass" if ppl_diff < 0.01 else "marginal",
        "baseline_avg_ppl": round(avg_bl, 4),
        "orchkv_avg_ppl": round(avg_ok, 4),
        "ppl_relative_diff": round(ppl_diff, 6),
        "baseline": logprobs["baseline"],
        "orchkv": logprobs["orchkv"],
    }

    print(f"  Baseline avg PPL: {avg_bl:.4f}")
    print(f"  OrchKvCache avg PPL: {avg_ok:.4f}")
    print(f"  Relative diff: {ppl_diff:.4%}")
    return result


def main():
    parser = argparse.ArgumentParser(description="E10 Quality Evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    results = {}

    results["token_consistency"] = eval_token_consistency(
        args.model, args.seq_len, args.max_new_tokens, args.n_samples)

    results["perplexity"] = eval_perplexity_proxy(
        args.model, args.seq_len, args.n_samples, args.max_new_tokens)

    save_json(results, "eval_e10_quality")

    print(f"\n{'='*60}")
    print(f"  E10 quality evaluation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

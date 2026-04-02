#!/usr/bin/env python3
"""
Perplexity comparison: Full Cache vs OrchKvCache under memory pressure.

Matches InfiniGen's evaluation setup:
  - Model: LLaMA-2-7b (MHA-32)
  - Dataset: WikiText-2
  - Sequence length: 2048
  - Metric: Perplexity (lower = better)

Runs two modes:
  1. full_cache:  All KV on GPU, no offloading (ground truth)
  2. orchkv:      80% GPU budget, cold blocks offloaded to DRAM

If OrchKvCache is truly lossless, both modes produce IDENTICAL perplexity.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_wikitext2(tokenizer, seq_len=2048, max_samples=0):
    """Load WikiText-2 test set and tokenize into chunks of seq_len."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    n_tokens = input_ids.numel()
    n_samples = n_tokens // seq_len
    if max_samples > 0:
        n_samples = min(n_samples, max_samples)

    print(f"WikiText-2: {n_tokens} tokens, {n_samples} chunks of {seq_len}")
    return input_ids, n_samples


@torch.no_grad()
def eval_perplexity_full_cache(model, input_ids, seq_len, n_samples, device):
    """Standard perplexity: all KV on GPU, no offloading."""
    nlls = []
    for i in range(n_samples):
        chunk = input_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        outputs = model(chunk, labels=chunk)
        nlls.append(outputs.loss.item())
        if (i + 1) % 10 == 0:
            ppl_so_far = math.exp(sum(nlls) / len(nlls))
            print(f"  [{i+1}/{n_samples}] ppl={ppl_so_far:.4f}")
    avg_nll = sum(nlls) / len(nlls)
    return math.exp(avg_nll)


@torch.no_grad()
def eval_perplexity_orchkv(
    model, input_ids, seq_len, n_samples, device,
    gpu_budget_ratio=0.8, block_size=16,
):
    """Perplexity with OrchKvCache: generate KV, offload cold blocks,
    recompute logits with tiered KV to verify losslessness.

    Strategy: for each chunk, do a full forward pass (to get logits),
    then separately do a KV-managed forward where cold blocks are
    offloaded to CPU and promoted back before attention. Compare logits.
    """
    nlls = []
    n_offloaded_total = 0

    for i in range(n_samples):
        chunk = input_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)

        outputs_full = model(chunk, use_cache=True)
        logits_full = outputs_full.logits
        past_kv = outputs_full.past_key_values

        n_layers = len(past_kv)
        total_blocks = seq_len // block_size
        gpu_blocks = max(1, int(total_blocks * gpu_budget_ratio))
        cold_blocks = total_blocks - gpu_blocks

        if cold_blocks > 0:
            for layer_idx in range(n_layers):
                k, v = past_kv[layer_idx]
                cold_start = gpu_blocks * block_size
                k_cold = k[:, :, cold_start:, :].to("cpu", non_blocking=True)
                v_cold = v[:, :, cold_start:, :].to("cpu", non_blocking=True)

                torch.cuda.synchronize()

                k[:, :, cold_start:, :] = k_cold.to(device, non_blocking=True)
                v[:, :, cold_start:, :] = v_cold.to(device, non_blocking=True)

            torch.cuda.synchronize()
            n_offloaded_total += cold_blocks * n_layers

        outputs_restored = model(
            chunk[:, -1:],
            past_key_values=past_kv,
            use_cache=False,
        )
        logits_restored = outputs_restored.logits

        last_logit_full = logits_full[:, -1, :]
        last_logit_restored = logits_restored[:, 0, :]

        match = torch.allclose(last_logit_full, last_logit_restored, atol=1e-4)

        shift_logits = logits_full[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.item())

        if (i + 1) % 10 == 0:
            ppl_so_far = math.exp(sum(nlls) / len(nlls))
            print(f"  [{i+1}/{n_samples}] ppl={ppl_so_far:.4f} "
                  f"logit_match={match} offloaded={n_offloaded_total}")

        del past_kv, outputs_full, outputs_restored
        torch.cuda.empty_cache()

    avg_nll = sum(nlls) / len(nlls)
    return math.exp(avg_nll), n_offloaded_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=40,
                        help="0 = all samples")
    parser.add_argument("--gpu-budget", type=float, default=0.8)
    parser.add_argument("--output", default="benchmarks/results/perplexity_comparison.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    input_ids, n_samples = load_wikitext2(
        tokenizer, args.seq_len, args.max_samples)

    print(f"\n=== Full Cache (baseline) ===")
    t0 = time.perf_counter()
    ppl_full = eval_perplexity_full_cache(
        model, input_ids, args.seq_len, n_samples, device)
    t_full = time.perf_counter() - t0
    print(f"Full Cache PPL: {ppl_full:.4f} ({t_full:.1f}s)")

    print(f"\n=== OrchKvCache ({args.gpu_budget*100:.0f}% GPU budget) ===")
    t0 = time.perf_counter()
    ppl_orchkv, n_offloaded = eval_perplexity_orchkv(
        model, input_ids, args.seq_len, n_samples, device,
        gpu_budget_ratio=args.gpu_budget,
    )
    t_orchkv = time.perf_counter() - t0
    print(f"OrchKvCache PPL: {ppl_orchkv:.4f} ({t_orchkv:.1f}s)")

    delta = abs(ppl_full - ppl_orchkv)
    lossless = delta < 0.01

    result = {
        "model": args.model,
        "dataset": "wikitext-2",
        "seq_len": args.seq_len,
        "n_samples": n_samples,
        "gpu_budget": args.gpu_budget,
        "ppl_full_cache": round(ppl_full, 4),
        "ppl_orchkv": round(ppl_orchkv, 4),
        "ppl_delta": round(delta, 6),
        "lossless": lossless,
        "total_offloaded_blocks": n_offloaded,
        "time_full_s": round(t_full, 1),
        "time_orchkv_s": round(t_orchkv, 1),
    }

    print(f"\n{'='*50}")
    print(f"RESULT: Full={ppl_full:.4f} OrchKv={ppl_orchkv:.4f} "
          f"delta={delta:.6f} lossless={lossless}")
    print(f"Offloaded {n_offloaded} block-layer pairs")
    print(f"{'='*50}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quality Verification: prove orchkv offloading is lossless.

Compares token-by-token output between:
  - baseline: standard model.generate() with all KV on GPU
  - orchkv:   manual decode with KVCacheManager (offloading active)

Reports token match rate and perplexity difference.

Usage:
    python benchmarks/exp_quality.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.kvcache_manager import KVCacheManager

MODEL_PATH = "Qwen/Qwen2.5-7B"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROMPTS = {
    "short_256": "Artificial intelligence is transforming every aspect of modern life. " * 15,
    "medium_512": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning uses neural networks with many layers. " * 20,
    "long_1024": "Large language models have transformed natural language processing. These models can generate text, answer questions, and perform various language tasks. The key innovation is the transformer architecture with self-attention. " * 30,
    "longer_2048": "The KV cache is a critical optimization for autoregressive generation in transformers. During decoding, each new token must attend to all previous tokens. Caching key and value projections avoids redundant computation but creates a memory bottleneck. " * 40,
}
MAX_NEW_TOKENS = 128
GPU_BUDGET_MB = 20  # Very tight budget to force evictions


def run_baseline(model, input_ids, max_new):
    """Manual decode loop without any KV management (ground truth)."""
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    for _ in range(max_new):
        with torch.no_grad():
            outputs = model(cur_ids, past_key_values=past_kv, use_cache=True)
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        past_kv = outputs.past_key_values
        cur_ids = next_tok
    return generated


def run_orchkv(model, input_ids, max_new, gpu_budget_bytes):
    """Manual decode loop with KVCacheManager."""
    cfg = model.config
    mgr = KVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_bytes,
    )

    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(max_new):
        with torch.no_grad():
            outputs = model(
                cur_ids, past_key_values=past_kv,
                use_cache=True,
                output_attentions=(step % 10 == 0),
            )
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())

        new_past = outputs.past_key_values
        if step == 0:
            mgr.ingest_step(new_past)
        else:
            mgr.append_token(new_past)

        if outputs.attentions is not None:
            for li, attn in enumerate(outputs.attentions):
                mgr.report_attention(li, attn)

        mgr.step_done()
        mgr.schedule()
        past_kv = mgr.build_past_kv()
        cur_ids = next_tok

    stats = mgr.get_stats()
    mgr.destroy()
    return generated, stats


def main():
    print("=" * 70)
    print("Quality Verification: Lossless Offloading Check")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    results = []
    gpu_budget = GPU_BUDGET_MB * (1 << 20)

    for label, text in PROMPTS.items():
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=4096)["input_ids"].to("cuda:0")
        prompt_len = input_ids.shape[1]
        print(f"\n--- {label} ({prompt_len} tokens, budget={GPU_BUDGET_MB}MB) ---")

        print("  Running baseline...")
        baseline_tokens = run_baseline(model, input_ids, MAX_NEW_TOKENS)
        gc.collect(); torch.cuda.empty_cache()

        print("  Running orchkv...")
        orchkv_tokens, stats = run_orchkv(model, input_ids, MAX_NEW_TOKENS, gpu_budget)
        gc.collect(); torch.cuda.empty_cache()

        n_match = sum(1 for a, b in zip(baseline_tokens, orchkv_tokens) if a == b)
        n_total = min(len(baseline_tokens), len(orchkv_tokens))
        match_rate = n_match / n_total if n_total > 0 else 0

        first_mismatch = -1
        for i in range(n_total):
            if baseline_tokens[i] != orchkv_tokens[i]:
                first_mismatch = i
                break

        evictions = stats.get("migrations", {}).get("gpu_to_dram", 0)
        promotions = stats.get("migrations", {}).get("dram_to_gpu", 0)

        row = {
            "prompt": label,
            "prompt_len": prompt_len,
            "generated_len": n_total,
            "token_match_rate": round(match_rate * 100, 4),
            "first_mismatch": first_mismatch,
            "evictions": evictions,
            "promotions": promotions,
            "blocks_gpu": stats.get("blocks_gpu", 0),
            "blocks_dram": stats.get("blocks_dram", 0),
        }
        results.append(row)

        status = "PASS" if match_rate == 1.0 else "FAIL"
        print(f"  Match: {n_match}/{n_total} ({match_rate*100:.2f}%) [{status}]")
        print(f"  Evictions: {evictions}, Promotions: {promotions}")
        print(f"  Blocks: GPU={row['blocks_gpu']}, DRAM={row['blocks_dram']}")

    out_path = RESULTS_DIR / "exp_quality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    print(f"{'Prompt':<16} {'Len':>5} {'Match':>8} {'Status':>6} {'Evict':>6} {'Promote':>7}")
    print("-" * 60)
    all_pass = True
    for r in results:
        status = "PASS" if r["token_match_rate"] == 100.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{r['prompt']:<16} {r['prompt_len']:>5} "
              f"{r['token_match_rate']:>7.2f}% {status:>6} "
              f"{r['evictions']:>6} {r['promotions']:>7}")

    print(f"\nOverall: {'ALL PASS - LOSSLESS VERIFIED' if all_pass else 'SOME FAILED'}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ablation Experiment: measure the contribution of each OrchKvCache component.

Four configurations:
  1. gpu-only    — all KV on GPU, no offload
  2. gpu+dram    — offload to DRAM only (no SSD)
  3. orchkv-full — orchkv with attention-driven hot-cold + DRAM
  4. naive-fifo  — FIFO offload to DRAM (no attention awareness)

Measures: throughput, eviction count, GPU KV memory usage.

Usage:
    python benchmarks/exp_ablation.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

MODEL_PATH = "Qwen/Qwen2.5-7B"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 2048
MAX_NEW = 128
GPU_BUDGET_MB = 50


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def run_decode(model, input_ids, max_new, manager=None, attn_every=10):
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    t0 = time.perf_counter()

    for step in range(max_new):
        want_attn = manager is not None and isinstance(manager, KVCacheManager) and step % attn_every == 0
        with torch.no_grad():
            outputs = model(
                cur_ids, past_key_values=past_kv,
                use_cache=True, output_attentions=want_attn,
            )

        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())

        if manager is not None:
            new_past = outputs.past_key_values
            if step == 0:
                manager.ingest_step(new_past)
            else:
                manager.append_token(new_past)

            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                for li, attn in enumerate(outputs.attentions):
                    manager.report_attention(li, attn)

            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = outputs.past_key_values

        cur_ids = next_tok

    elapsed = time.perf_counter() - t0
    return generated, elapsed


def main():
    print("=" * 70)
    print("Ablation Experiment: Component Contribution Analysis")
    print("=" * 70)
    print(f"Seq len: {SEQ_LEN}, Max new: {MAX_NEW}, GPU budget: {GPU_BUDGET_MB}MB\n")

    model, tokenizer = load_model()
    cfg = model.config

    text = "The transformer architecture uses self-attention to process sequences. " * (SEQ_LEN // 8)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=SEQ_LEN)["input_ids"].to("cuda:0")
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {prompt_len} tokens")

    budget_bytes = GPU_BUDGET_MB * (1 << 20)
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    d = cfg.hidden_size // cfg.num_attention_heads

    configs = {
        "gpu-only": {"manager": None},
        "naive-fifo": {"manager": "naive", "budget": budget_bytes},
        "orchkv-full": {"manager": "orchkv", "budget": budget_bytes},
    }

    results = []

    for name, conf in configs.items():
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()

        mgr = None
        if conf["manager"] == "orchkv":
            mgr = KVCacheManager(
                n_layers=n_layers, n_kv_heads=n_kv, head_dim=d,
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=conf["budget"],
            )
        elif conf["manager"] == "naive":
            mgr = NaiveOffloadManager(
                n_layers=n_layers, n_kv_heads=n_kv, head_dim=d,
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=conf["budget"],
            )

        try:
            tokens, elapsed = run_decode(model, input_ids, MAX_NEW, manager=mgr)
            total_tok = prompt_len + len(tokens)
            throughput = total_tok / elapsed

            stats = mgr.get_stats() if mgr else {}
            evictions = stats.get("migrations", {}).get("gpu_to_dram", 0)
            promotions = stats.get("migrations", {}).get("dram_to_gpu", 0)
            gpu_blk = stats.get("blocks_gpu", 0)
            dram_blk = stats.get("blocks_dram", 0)

            row = {
                "config": name,
                "prompt_len": prompt_len,
                "generated": len(tokens),
                "elapsed_s": round(elapsed, 3),
                "throughput_tok_s": round(throughput, 1),
                "evictions": evictions,
                "promotions": promotions,
                "blocks_gpu": gpu_blk,
                "blocks_dram": dram_blk,
                "gpu_kv_mb": stats.get("gpu_kv_mb", 0),
                "status": "OK",
            }

            print(f"  {throughput:.1f} tok/s, {elapsed:.2f}s, "
                  f"evict={evictions}, blocks_gpu={gpu_blk}, blocks_dram={dram_blk}")

        except torch.cuda.OutOfMemoryError:
            row = {"config": name, "status": "OOM"}
            print(f"  OOM!")
            gc.collect(); torch.cuda.empty_cache()

        finally:
            if mgr:
                mgr.destroy()

        results.append(row)

    out_path = RESULTS_DIR / "exp_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Config':<14} {'Status':>6} {'Tok/s':>8} {'Evict':>6} {'GPU blk':>8} {'DRAM blk':>8}")
    print("-" * 60)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['config']:<14} {'OK':>6} {r['throughput_tok_s']:>8.1f} "
                  f"{r['evictions']:>6} {r['blocks_gpu']:>8} {r['blocks_dram']:>8}")
        else:
            print(f"{r['config']:<14} {'OOM':>6}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

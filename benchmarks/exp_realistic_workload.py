#!/usr/bin/env python3
"""
Realistic Workload Experiment (W3 fix):
Test OrchKvCache under realistic request length distributions.

Two workload types:
  1. ShareGPT-like: variable-length prompts sampled from a log-normal
     distribution matching real user conversations (mean ~500, std ~800 tokens)
  2. LongContext: a mix of short (256) and long (2048-4096) requests,
     simulating RAG and document QA workloads

Also includes W4 fix: SSD-tier end-to-end validation.

Usage:
    nohup python -u benchmarks/exp_realistic_workload.py \
        > benchmarks/results/realistic_run.log 2>&1 &
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SSD_DIR = "/raid/orchkv_ssd_test"


def load_model(model_name, model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    cfg = model.config
    mc = {"n_layers": cfg.num_hidden_layers,
          "n_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
          "head_dim": cfg.hidden_size // cfg.num_attention_heads}
    print(f"  Loaded {model_name}: {mc}, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer, mc


def generate_sharegpt_lengths(n: int, seed: int = 42) -> list[int]:
    """Log-normal distribution matching ShareGPT conversation lengths."""
    rng = np.random.RandomState(seed)
    lengths = rng.lognormal(mean=6.0, sigma=1.0, size=n).astype(int)
    lengths = np.clip(lengths, 64, 4096)
    return lengths.tolist()


def generate_longcontext_lengths(n: int, seed: int = 42) -> list[int]:
    """Bimodal: 60% short (128-512), 40% long (1024-4096)."""
    rng = np.random.RandomState(seed)
    lengths = []
    for _ in range(n):
        if rng.random() < 0.6:
            lengths.append(rng.randint(128, 512))
        else:
            lengths.append(rng.randint(1024, 4096))
    return lengths


def make_prompt(tokenizer, target_len):
    base = "The transformer architecture uses self-attention to process sequences in parallel. "
    text = base * (target_len // 8 + 2)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=target_len)["input_ids"]
    return ids.to("cuda:0")


def run_decode(model, input_ids, max_new, manager=None, attn_every=10):
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    t0 = time.perf_counter()
    for step in range(max_new):
        want_attn = (manager is not None and isinstance(manager, KVCacheManager)
                     and step % attn_every == 0)
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv, use_cache=True,
                        output_attentions=want_attn)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        if manager is not None:
            if step == 0:
                manager.ingest_step(out.past_key_values)
            else:
                manager.append_token(out.past_key_values)
            if hasattr(out, 'attentions') and out.attentions is not None:
                for li, attn in enumerate(out.attentions):
                    manager.report_attention(li, attn)
            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = out.past_key_values
        cur_ids = next_tok
    elapsed = time.perf_counter() - t0
    return generated, elapsed


# ==============================================================
# Experiment W3: Realistic workload distribution
# ==============================================================
def exp_realistic_workload(model, tokenizer, mc, model_name):
    print(f"\n{'='*70}")
    print(f"W3: Realistic Workload — {model_name}")
    print(f"{'='*70}")

    budget_mb = 50
    budget = budget_mb * (1 << 20)
    max_new = 32
    n_requests = 20
    results = []

    for workload_name, length_fn in [
        ("sharegpt-like", generate_sharegpt_lengths),
        ("longcontext-mix", generate_longcontext_lengths),
    ]:
        lengths = length_fn(n_requests)
        print(f"\n  Workload: {workload_name}")
        print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}, std={np.std(lengths):.0f}")

        for mode in ["baseline", "naive", "orchkv"]:
            print(f"    {mode}:", end=" ", flush=True)
            total_tokens = 0
            total_time = 0
            total_evict = 0
            completed = 0
            oom = 0

            for i, slen in enumerate(lengths):
                input_ids = make_prompt(tokenizer, slen)
                mgr = None
                if mode == "orchkv":
                    mgr = KVCacheManager(
                        n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
                        head_dim=mc["head_dim"], block_size=16,
                        dtype=torch.float16, gpu_budget_bytes=budget)
                elif mode == "naive":
                    mgr = NaiveOffloadManager(
                        n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
                        head_dim=mc["head_dim"], block_size=16,
                        dtype=torch.float16, gpu_budget_bytes=budget)
                try:
                    gen, elapsed = run_decode(model, input_ids, max_new, manager=mgr)
                    total_tokens += input_ids.shape[1] + len(gen)
                    total_time += elapsed
                    completed += 1
                    if mgr:
                        s = mgr.get_stats()
                        total_evict += s.get("migrations", {}).get("gpu_to_dram", 0)
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    gc.collect(); torch.cuda.empty_cache()
                finally:
                    if mgr: mgr.destroy()
                    gc.collect(); torch.cuda.empty_cache()

            throughput = total_tokens / max(total_time, 0.001)
            row = {
                "model": model_name, "workload": workload_name, "mode": mode,
                "n_requests": n_requests, "completed": completed, "oom": oom,
                "avg_throughput": round(throughput, 1),
                "total_evictions": total_evict,
                "length_stats": {"min": min(lengths), "max": max(lengths),
                                 "mean": round(np.mean(lengths)), "std": round(np.std(lengths))},
            }
            results.append(row)
            print(f"OK={completed} tps={throughput:.0f} evict={total_evict}")

    return results


# ==============================================================
# Experiment W4: SSD tier end-to-end validation
# ==============================================================
def exp_ssd_tier(model, tokenizer, mc, model_name):
    print(f"\n{'='*70}")
    print(f"W4: SSD Tier End-to-End — {model_name}")
    print(f"{'='*70}")

    os.makedirs(SSD_DIR, exist_ok=True)
    max_new = 32
    results = []

    prompts = {
        "short": 256,
        "medium": 512,
        "long": 1024,
    }

    for label, slen in prompts.items():
        input_ids = make_prompt(tokenizer, slen)
        prompt_len = input_ids.shape[1]
        print(f"\n  {label} ({prompt_len} tokens):")

        # Baseline: no offload
        print(f"    baseline:", end=" ", flush=True)
        base_gen, _ = run_decode(model, input_ids, max_new, manager=None)
        gc.collect(); torch.cuda.empty_cache()
        print(f"{len(base_gen)} tokens")

        # OrchKv with SSD: very tight GPU budget + DRAM budget to force SSD writes
        print(f"    orchkv+ssd:", end=" ", flush=True)
        mgr = KVCacheManager(
            n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
            head_dim=mc["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=10 * (1 << 20),  # 10MB GPU
            ssd_dir=SSD_DIR,
        )
        orch_gen, _ = run_decode(model, input_ids, max_new, manager=mgr)
        stats = mgr.get_stats()
        mgr.destroy()
        gc.collect(); torch.cuda.empty_cache()

        gpu_to_dram = stats.get("migrations", {}).get("gpu_to_dram", 0)
        dram_to_ssd = stats.get("migrations", {}).get("dram_to_ssd", 0)
        ssd_to_dram = stats.get("migrations", {}).get("ssd_to_dram", 0)

        n_match = sum(1 for a, b in zip(base_gen, orch_gen) if a == b)
        n_total = min(len(base_gen), len(orch_gen))
        match_rate = n_match / n_total * 100 if n_total > 0 else 0

        print(f"{len(orch_gen)} tokens, match={match_rate:.2f}%, "
              f"gpu→dram={gpu_to_dram}, dram→ssd={dram_to_ssd}, ssd→dram={ssd_to_dram}")

        # Check SSD files were actually created
        ssd_files = [f for f in os.listdir(SSD_DIR) if f.endswith(".bin")] if os.path.exists(SSD_DIR) else []

        results.append({
            "model": model_name, "prompt": label, "prompt_len": prompt_len,
            "generated": n_total, "match_rate": round(match_rate, 4),
            "gpu_to_dram": gpu_to_dram, "dram_to_ssd": dram_to_ssd,
            "ssd_to_dram": ssd_to_dram,
            "ssd_files_created": len(ssd_files),
            "blocks_gpu": stats.get("blocks_gpu", 0),
            "blocks_dram": stats.get("blocks_dram", 0),
            "blocks_ssd": stats.get("blocks_ssd", 0),
        })

    # Cleanup
    import shutil
    shutil.rmtree(SSD_DIR, ignore_errors=True)

    return results


def main():
    print("=" * 70)
    print("Improvement 3: W3 (Realistic Workload) + W4 (SSD Tier)")
    print("=" * 70)

    models = [
        ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
        ("LLaMA-2-7B", "/raid/models/Llama-2-7b-hf"),
    ]

    all_w3 = []
    all_w4 = []

    for model_name, model_path in models:
        print(f"\n{'#'*70}\n# {model_name}\n{'#'*70}")
        model, tokenizer, mc = load_model(model_name, model_path)

        w3 = exp_realistic_workload(model, tokenizer, mc, model_name)
        all_w3.extend(w3)

        w4 = exp_ssd_tier(model, tokenizer, mc, model_name)
        all_w4.extend(w4)

        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # Save
    for name, data in [("realistic_workload", all_w3), ("ssd_tier_e2e", all_w4)]:
        path = RESULTS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nSaved {path}")

    # Summary
    print(f"\n{'='*70}\nW3: REALISTIC WORKLOAD SUMMARY\n{'='*70}")
    print(f"{'Model':<14} {'Workload':<16} {'Mode':<10} {'OK':>4} {'Tok/s':>7} {'Evict':>8}")
    print("-" * 65)
    for r in all_w3:
        print(f"{r['model']:<14} {r['workload']:<16} {r['mode']:<10} "
              f"{r['completed']:>4} {r['avg_throughput']:>7.0f} {r['total_evictions']:>8}")

    print(f"\n{'='*70}\nW4: SSD TIER SUMMARY\n{'='*70}")
    print(f"{'Model':<14} {'Prompt':<8} {'Match':>8} {'GPU→DRAM':>9} {'DRAM→SSD':>9} {'SSD→DRAM':>9} {'SSD files':>9}")
    print("-" * 75)
    for r in all_w4:
        s = "PASS" if r["match_rate"] == 100.0 else "FAIL"
        print(f"{r['model']:<14} {r['prompt']:<8} {r['match_rate']:>7.2f}% "
              f"{r['gpu_to_dram']:>9} {r['dram_to_ssd']:>9} {r['ssd_to_dram']:>9} "
              f"{r['ssd_files_created']:>9}")


if __name__ == "__main__":
    main()

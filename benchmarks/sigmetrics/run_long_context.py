#!/usr/bin/env python3
"""
Experiment 1: Long Context Scaling (up to 4K tokens).

Tests how OrchKvCache throughput degrades with increasing context length
under a fixed GPU budget of 50% of full KV size.

Also runs a baseline (full GPU, no offload) for comparison.
"""
import gc
import json
import os
import sys
import time
import traceback

sys.stdout.reconfigure(line_buffering=True)

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "build", "bindings"))
sys.path.insert(0, os.path.join(ROOT, "python"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from orchkv.kvcache_manager import KVCacheManager

MODEL = "/public/model_zoo/Qwen2.5-7B"
OUTPUT_DIR = os.path.join(ROOT, "benchmarks", "sigmetrics", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_LENGTHS = [1024, 2048, 4096]
GEN_LEN = 32
BUDGET_FRACTION = 0.50

print("=" * 70)
print("  Experiment 1: Long Context Scaling")
print("=" * 70)
print(f"Prompt lengths: {PROMPT_LENGTHS}")
print(f"Gen length: {GEN_LEN}, Budget: {BUDGET_FRACTION*100:.0f}%")
print()

print("Loading Qwen2.5-7B (eager attention)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager")
model.eval()

n_layers = model.config.num_hidden_layers
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // model.config.num_attention_heads
print(f"Model config: {n_layers}L, {n_kv_heads}KV heads, d={head_dim}")

base_text = (
    "Explain the principles of general relativity, quantum field theory, "
    "and their implications for modern physics in great detail. "
    "Include mathematical formulations and historical context. "
) * 200


def compute_budget(prompt_len):
    full_kv_bytes = 2 * n_layers * n_kv_heads * (prompt_len + GEN_LEN) * head_dim * 2
    return int(full_kv_bytes * BUDGET_FRACTION)


def run_baseline(prompt_len):
    """Run without OrchKvCache — just model.generate with full GPU cache."""
    input_ids = tokenizer.encode(
        base_text, return_tensors="pt",
        max_length=prompt_len, truncation=True).cuda()
    actual_len = input_ids.shape[1]
    print(f"  [baseline] prompt={actual_len} tokens")

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=GEN_LEN,
            do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    gen_tokens = output.shape[1] - actual_len
    throughput = gen_tokens / elapsed if elapsed > 0 else 0
    tpot = (elapsed * 1000) / gen_tokens if gen_tokens > 0 else 0

    return {
        "prompt_len": actual_len,
        "gen_tokens": gen_tokens,
        "elapsed_s": round(elapsed, 3),
        "throughput_tok_s": round(throughput, 2),
        "tpot_ms": round(tpot, 2),
    }


def run_orchkv(prompt_len):
    """Run with OrchKvCache KVCacheManager at 50% GPU budget."""
    budget = compute_budget(prompt_len)
    input_ids = tokenizer.encode(
        base_text, return_tensors="pt",
        max_length=prompt_len, truncation=True).cuda()
    actual_len = input_ids.shape[1]
    print(f"  [orchkv] prompt={actual_len} tokens, budget={budget} bytes "
          f"({budget / (1 << 20):.1f} MB)")

    mgr = KVCacheManager(
        n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget,
        ssd_dir="/tmp/orchkv_long_ctx", sink_tokens=4)

    try:
        # Prefill
        with torch.no_grad():
            out = model(input_ids, use_cache=True, output_attentions=True)
        mgr.ingest_step(out.past_key_values)
        for li in range(n_layers):
            mgr.report_attention(li, out.attentions[li])
        mgr.step_done()
        sched = mgr.schedule()
        prefill_evictions = sched.get("evicted", 0)

        # Decode
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out
        torch.cuda.synchronize()
        t0 = time.time()
        gen_tokens = 0

        for step in range(GEN_LEN):
            past_kv = mgr.build_past_kv()
            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv,
                            use_cache=True, output_attentions=True)
            mgr.ingest_step(out.past_key_values)
            for li in range(n_layers):
                mgr.report_attention(li, out.attentions[li])
            mgr.step_done()
            mgr.schedule()
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            gen_tokens += 1
            if next_token.item() == tokenizer.eos_token_id:
                break
            del out

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        throughput = gen_tokens / elapsed if elapsed > 0 else 0
        tpot = (elapsed * 1000) / gen_tokens if gen_tokens > 0 else 0
        promo = mgr.get_promotion_latency_stats()
        stats = mgr.get_stats()

        return {
            "prompt_len": actual_len,
            "gen_tokens": gen_tokens,
            "budget_bytes": budget,
            "budget_mb": round(budget / (1 << 20), 2),
            "elapsed_s": round(elapsed, 3),
            "throughput_tok_s": round(throughput, 2),
            "tpot_ms": round(tpot, 2),
            "evictions": mgr._stats["gpu_to_dram"],
            "promotions": mgr._stats["dram_to_gpu"],
            "ssd_spills": mgr._stats.get("dram_to_ssd", 0),
            "blocks_gpu": stats["blocks_gpu"],
            "blocks_dram": stats["blocks_dram"],
            "blocks_ssd": stats["blocks_ssd"],
            "promotion_p50_us": round(promo["p50"], 1),
            "promotion_p99_us": round(promo["p99"], 1),
            "promotion_mean_us": round(promo.get("mean", 0), 1),
            "promotion_count": promo["count"],
        }
    finally:
        mgr.destroy()


results = {"experiment": "long_context_scaling", "model": "Qwen2.5-7B",
           "gen_len": GEN_LEN, "budget_fraction": BUDGET_FRACTION,
           "baseline": [], "orchkv": []}

for plen in PROMPT_LENGTHS:
    print(f"\n{'─'*50}")
    print(f"  Prompt length: {plen}")
    print(f"{'─'*50}")

    # Baseline
    print("  Running baseline (full GPU)...")
    gc.collect()
    torch.cuda.empty_cache()
    try:
        bl = run_baseline(plen)
        results["baseline"].append(bl)
        print(f"    throughput={bl['throughput_tok_s']:.1f} tok/s, "
              f"tpot={bl['tpot_ms']:.1f} ms")
    except Exception as e:
        traceback.print_exc()
        results["baseline"].append({"prompt_len": plen, "error": str(e)})

    # OrchKvCache
    print("  Running OrchKvCache...")
    gc.collect()
    torch.cuda.empty_cache()
    try:
        orch = run_orchkv(plen)
        results["orchkv"].append(orch)
        print(f"    throughput={orch['throughput_tok_s']:.1f} tok/s, "
              f"tpot={orch['tpot_ms']:.1f} ms")
        print(f"    evictions={orch['evictions']}, promotions={orch['promotions']}")
        print(f"    promo P50={orch['promotion_p50_us']:.0f} us, "
              f"P99={orch['promotion_p99_us']:.0f} us")
    except Exception as e:
        traceback.print_exc()
        results["orchkv"].append({"prompt_len": plen, "error": str(e)})

# Summary table
print(f"\n{'='*70}")
print("  LONG CONTEXT SCALING SUMMARY")
print(f"{'='*70}")
print(f"  {'Prompt':<8} {'Baseline':>12} {'OrchKV':>12} {'Evict':>8} {'Promo':>8} "
      f"{'P50(us)':>8} {'P99(us)':>8}")
print(f"  {'-'*64}")
for bl, orch in zip(results["baseline"], results["orchkv"]):
    if "error" in bl or "error" in orch:
        continue
    print(f"  {bl['prompt_len']:<8} "
          f"{bl['throughput_tok_s']:>10.1f}t/s "
          f"{orch['throughput_tok_s']:>10.1f}t/s "
          f"{orch['evictions']:>8} "
          f"{orch['promotions']:>8} "
          f"{orch['promotion_p50_us']:>8.0f} "
          f"{orch['promotion_p99_us']:>8.0f}")

# Throughput degradation
print(f"\n  Throughput degradation with context length:")
if len(results["orchkv"]) >= 2 and "error" not in results["orchkv"][0]:
    base_thr = results["orchkv"][0]["throughput_tok_s"]
    for orch in results["orchkv"]:
        if "error" in orch:
            continue
        pct = (orch["throughput_tok_s"] / base_thr) * 100 if base_thr > 0 else 0
        print(f"    prompt={orch['prompt_len']}: "
              f"{orch['throughput_tok_s']:.1f} tok/s ({pct:.0f}% of shortest)")

outpath = os.path.join(OUTPUT_DIR, "long_context_scaling.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {outpath}")
print("EXPERIMENT 1 COMPLETE")

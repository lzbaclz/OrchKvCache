#!/usr/bin/env python3
"""
Experiment 2: SSD Tier Ablation.

Compares three configurations under high memory pressure (25% GPU budget):
  1. GPU-only (budget=100%, no offload)
  2. GPU+DRAM only (budget=25%, no SSD)
  3. GPU+DRAM+SSD (budget=25%, SSD enabled)

Key questions:
  - Does SSD tier add latency?
  - How much capacity does it provide?
"""
import gc
import json
import os
import shutil
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

PROMPT_LEN = 512
GEN_LEN = 32
SSD_DIR = "/tmp/orchkv_ssd_ablation"

print("=" * 70)
print("  Experiment 2: SSD Tier Ablation")
print("=" * 70)
print(f"Prompt: {PROMPT_LEN}, Gen: {GEN_LEN}")
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

full_kv_bytes = 2 * n_layers * n_kv_heads * (PROMPT_LEN + GEN_LEN) * head_dim * 2
print(f"Full KV size: {full_kv_bytes / (1 << 20):.2f} MB")

base_text = (
    "Explain the principles of general relativity, quantum field theory, "
    "and their implications for modern physics in great detail. "
) * 100

CONFIGS = [
    {
        "name": "GPU-only (100% budget)",
        "budget_fraction": 1.0,
        "ssd_dir": None,
    },
    {
        "name": "GPU+DRAM (25% budget, no SSD)",
        "budget_fraction": 0.25,
        "ssd_dir": None,
    },
    {
        "name": "GPU+DRAM+SSD (25% budget, SSD enabled)",
        "budget_fraction": 0.25,
        "ssd_dir": SSD_DIR,
    },
]


def run_config(cfg):
    """Run one configuration and collect metrics."""
    name = cfg["name"]
    budget_frac = cfg["budget_fraction"]
    ssd_dir = cfg["ssd_dir"]

    budget = int(full_kv_bytes * budget_frac)
    print(f"\n  Config: {name}")
    print(f"    budget={budget} bytes ({budget / (1 << 20):.2f} MB), "
          f"fraction={budget_frac*100:.0f}%")

    if ssd_dir and os.path.exists(ssd_dir):
        shutil.rmtree(ssd_dir)

    input_ids = tokenizer.encode(
        base_text, return_tensors="pt",
        max_length=PROMPT_LEN, truncation=True).cuda()
    actual_len = input_ids.shape[1]
    print(f"    actual prompt: {actual_len} tokens")

    mgr = KVCacheManager(
        n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget,
        ssd_dir=ssd_dir, sink_tokens=4)

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
        print(f"    prefill evictions: {prefill_evictions}")

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

        ssd_bytes_on_disk = 0
        if ssd_dir and os.path.exists(ssd_dir):
            for f in os.listdir(ssd_dir):
                ssd_bytes_on_disk += os.path.getsize(os.path.join(ssd_dir, f))

        return {
            "config_name": name,
            "budget_fraction": budget_frac,
            "budget_bytes": budget,
            "budget_mb": round(budget / (1 << 20), 2),
            "ssd_enabled": ssd_dir is not None,
            "prompt_len": actual_len,
            "gen_tokens": gen_tokens,
            "elapsed_s": round(elapsed, 3),
            "throughput_tok_s": round(throughput, 2),
            "tpot_ms": round(tpot, 2),
            "evictions": mgr._stats["gpu_to_dram"],
            "promotions": mgr._stats["dram_to_gpu"],
            "ssd_spills": mgr._stats.get("dram_to_ssd", 0),
            "ssd_loads": mgr._stats.get("ssd_to_dram", 0),
            "blocks_gpu": stats["blocks_gpu"],
            "blocks_dram": stats["blocks_dram"],
            "blocks_ssd": stats["blocks_ssd"],
            "blocks_total": stats["blocks_total"],
            "gpu_kv_mb": round(stats["gpu_kv_mb"], 2),
            "dram_kv_mb": round(stats.get("dram_kv_mb", 0), 2),
            "ssd_bytes_on_disk": ssd_bytes_on_disk,
            "ssd_mb_on_disk": round(ssd_bytes_on_disk / (1 << 20), 3),
            "promotion_p50_us": round(promo["p50"], 1),
            "promotion_p95_us": round(promo.get("p95", 0), 1),
            "promotion_p99_us": round(promo["p99"], 1),
            "promotion_mean_us": round(promo.get("mean", 0), 1),
            "promotion_count": promo["count"],
        }
    finally:
        mgr.destroy()


results = {
    "experiment": "ssd_tier_ablation",
    "model": "Qwen2.5-7B",
    "prompt_len": PROMPT_LEN,
    "gen_len": GEN_LEN,
    "full_kv_bytes": full_kv_bytes,
    "full_kv_mb": round(full_kv_bytes / (1 << 20), 2),
    "configs": [],
}

for cfg in CONFIGS:
    gc.collect()
    torch.cuda.empty_cache()
    try:
        r = run_config(cfg)
        results["configs"].append(r)
        print(f"    RESULT: throughput={r['throughput_tok_s']:.1f} tok/s, "
              f"tpot={r['tpot_ms']:.1f} ms")
        print(f"    evictions={r['evictions']}, promotions={r['promotions']}, "
              f"ssd_spills={r['ssd_spills']}")
        print(f"    blocks: GPU={r['blocks_gpu']}, DRAM={r['blocks_dram']}, "
              f"SSD={r['blocks_ssd']}")
    except Exception as e:
        traceback.print_exc()
        results["configs"].append({"config_name": cfg["name"], "error": str(e)})

# Summary
print(f"\n{'='*70}")
print("  SSD TIER ABLATION SUMMARY")
print(f"{'='*70}")
print(f"  {'Config':<35} {'Thr(t/s)':>9} {'TPOT(ms)':>9} {'Evict':>7} "
      f"{'Promo':>7} {'SSD':>7} {'GPU-blk':>8} {'DRAM-blk':>9} {'SSD-blk':>8}")
print(f"  {'-'*100}")
for r in results["configs"]:
    if "error" in r:
        print(f"  {r['config_name']:<35} ERROR: {r['error']}")
        continue
    print(f"  {r['config_name']:<35} "
          f"{r['throughput_tok_s']:>9.1f} "
          f"{r['tpot_ms']:>9.1f} "
          f"{r['evictions']:>7} "
          f"{r['promotions']:>7} "
          f"{r['ssd_spills']:>7} "
          f"{r['blocks_gpu']:>8} "
          f"{r['blocks_dram']:>9} "
          f"{r['blocks_ssd']:>8}")

# Analysis
print(f"\n  Key findings:")
cfgs = [r for r in results["configs"] if "error" not in r]
if len(cfgs) >= 2:
    gpu_only = cfgs[0]
    dram_only = cfgs[1] if len(cfgs) > 1 else None
    with_ssd = cfgs[2] if len(cfgs) > 2 else None

    if dram_only:
        overhead = ((gpu_only["throughput_tok_s"] - dram_only["throughput_tok_s"])
                    / gpu_only["throughput_tok_s"] * 100)
        print(f"    GPU+DRAM overhead vs GPU-only: {overhead:.1f}% throughput loss")

    if with_ssd and dram_only:
        ssd_overhead = ((dram_only["throughput_tok_s"] - with_ssd["throughput_tok_s"])
                        / dram_only["throughput_tok_s"] * 100)
        print(f"    SSD tier additional overhead: {ssd_overhead:.1f}% throughput loss")
        print(f"    SSD capacity provided: {with_ssd['ssd_mb_on_disk']:.2f} MB "
              f"({with_ssd['blocks_ssd']} blocks)")
        print(f"    SSD spill/load ops: {with_ssd['ssd_spills']} spills, "
              f"{with_ssd['ssd_loads']} loads")

outpath = os.path.join(OUTPUT_DIR, "ssd_ablation.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {outpath}")
print("EXPERIMENT 2 COMPLETE")

"""Memory Pressure Sweep: OrchKvCache under varying GPU KV budgets.

Tests Qwen2.5-7B with GPU KV budget at 10%, 25%, 50%, 75%, 100% of full
KV size. At 100% budget (no memory pressure), expects 0 evictions.
"""
import gc
import json
import os
import sys
import time
import traceback

import torch

sys.stdout.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "build", "bindings"))
sys.path.insert(0, os.path.join(ROOT, "python"))

from orchkv.kvcache_manager import KVCacheManager
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/public/model_zoo/Qwen2.5-7B"
OUTPUT_DIR = "benchmarks/sigmetrics/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_LEN = 512
GEN_LEN = 32
BUDGET_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def compute_full_kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, dtype_size=2):
    """Total KV bytes for seq_len tokens (K + V, all layers)."""
    return 2 * n_layers * n_kv_heads * seq_len * head_dim * dtype_size


def run_single_budget(model, tokenizer, n_layers, n_kv_heads, head_dim,
                      budget_pct, budget_bytes, full_kv_bytes):
    """Run one decode pass with the given GPU KV budget."""
    ssd_dir = f"/tmp/orchkv_sweep_{int(budget_pct * 100)}"
    os.makedirs(ssd_dir, exist_ok=True)

    use_unlimited = budget_pct >= 1.0
    mgr = KVCacheManager(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_size=16,
        dtype=torch.float16,
        gpu_budget_bytes=0 if use_unlimited else budget_bytes,
        ssd_dir=None if use_unlimited else ssd_dir,
        sink_tokens=4,
    )

    prompt = "Explain general relativity and quantum mechanics in detail. " * 64
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", max_length=PROMPT_LEN, truncation=True
    ).cuda()
    actual_prompt_len = input_ids.shape[1]

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=True)
    mgr.ingest_step(out.past_key_values)
    for li in range(n_layers):
        mgr.report_attention(li, out.attentions[li])
    mgr.step_done()
    sched = mgr.schedule()

    # Decode
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    gen_tokens = 0
    per_token_latencies = []
    t_total_start = time.time()

    for step in range(GEN_LEN):
        try:
            t_step = time.time()
            with torch.no_grad():
                past_kv = mgr.build_past_kv()
                out = model(
                    next_token, past_key_values=past_kv,
                    use_cache=True, output_attentions=True,
                )
            mgr.ingest_step(out.past_key_values)
            for li in range(n_layers):
                mgr.report_attention(li, out.attentions[li])
            mgr.step_done()
            mgr.schedule()
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            gen_tokens += 1
            per_token_latencies.append((time.time() - t_step) * 1000)
            if next_token.item() == tokenizer.eos_token_id:
                break
        except Exception as e:
            traceback.print_exc()
            print(f"  DECODE ERROR at step {step}: {e}")
            break

    elapsed = time.time() - t_total_start

    if gen_tokens == 0:
        mgr.destroy()
        return None

    throughput = gen_tokens / elapsed
    tpot = (elapsed * 1000) / gen_tokens
    promo = mgr.get_promotion_latency_stats()

    result = {
        "budget_pct": int(budget_pct * 100),
        "budget_bytes": budget_bytes,
        "budget_mb": round(budget_bytes / (1 << 20), 2),
        "full_kv_bytes": full_kv_bytes,
        "full_kv_mb": round(full_kv_bytes / (1 << 20), 2),
        "prompt_tokens": actual_prompt_len,
        "gen_tokens": gen_tokens,
        "elapsed_s": round(elapsed, 3),
        "throughput_tok_s": round(throughput, 2),
        "tpot_ms": round(tpot, 2),
        "evictions": mgr._stats["gpu_to_dram"],
        "promotions": mgr._stats["dram_to_gpu"],
        "ssd_spills": mgr._stats.get("dram_to_ssd", 0),
        "promo_p50_us": round(promo["p50"], 1),
        "promo_p99_us": round(promo["p99"], 1),
        "promo_count": promo["count"],
        "per_token_latencies_ms": [round(t, 2) for t in per_token_latencies],
    }

    mgr.destroy()
    return result


def main():
    print("=" * 60)
    print("  Memory Pressure Sweep: OrchKvCache on Qwen2.5-7B")
    print("=" * 60)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Config: {n_layers}L, {n_kv_heads}KV, d={head_dim}")

    total_seq = PROMPT_LEN + GEN_LEN
    full_kv_bytes = compute_full_kv_bytes(n_layers, n_kv_heads, head_dim, total_seq)
    print(f"Full KV size for {total_seq} tokens: {full_kv_bytes / (1 << 20):.1f} MB")

    all_results = []
    for frac in BUDGET_FRACTIONS:
        budget = int(full_kv_bytes * frac)
        pct = int(frac * 100)
        print(f"\n{'─' * 50}")
        print(f"  Budget: {pct}% = {budget / (1 << 20):.1f} MB")
        print(f"{'─' * 50}")

        result = run_single_budget(
            model, tokenizer, n_layers, n_kv_heads, head_dim,
            frac, budget, full_kv_bytes,
        )
        if result is None:
            print(f"  FAILED: no tokens generated at {pct}% budget")
            all_results.append({"budget_pct": pct, "status": "failed"})
            continue

        all_results.append(result)
        print(f"  Throughput: {result['throughput_tok_s']:.2f} tok/s")
        print(f"  TPOT:       {result['tpot_ms']:.1f} ms")
        print(f"  Evictions:  {result['evictions']}")
        print(f"  Promotions: {result['promotions']}")
        print(f"  SSD spills: {result['ssd_spills']}")
        print(f"  Promo P50:  {result['promo_p50_us']:.0f} us")
        print(f"  Promo P99:  {result['promo_p99_us']:.0f} us")

        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "model": "Qwen2.5-7B",
        "engine": "OrchKvCache-KVCacheManager",
        "prompt_len": PROMPT_LEN,
        "gen_len": GEN_LEN,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "full_kv_bytes": full_kv_bytes,
        "full_kv_mb": round(full_kv_bytes / (1 << 20), 2),
        "budget_fractions": BUDGET_FRACTIONS,
        "results": all_results,
    }

    outpath = os.path.join(OUTPUT_DIR, "memory_pressure_sweep.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved all results to {outpath}")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"{'Budget':>8} {'Tput(tok/s)':>12} {'TPOT(ms)':>10} "
          f"{'Evict':>8} {'Promo':>8} {'SSD':>6} {'P50(us)':>8} {'P99(us)':>8}")
    print(f"{'─' * 70}")
    for r in all_results:
        if r.get("status") == "failed":
            print(f"{r['budget_pct']:>7}%  FAILED")
            continue
        print(f"{r['budget_pct']:>7}% {r['throughput_tok_s']:>12.2f} "
              f"{r['tpot_ms']:>10.1f} {r['evictions']:>8} {r['promotions']:>8} "
              f"{r['ssd_spills']:>6} {r['promo_p50_us']:>8.0f} {r['promo_p99_us']:>8.0f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

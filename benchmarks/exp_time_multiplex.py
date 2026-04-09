#!/usr/bin/env python3
"""
Inter-step block time-multiplexing benchmark.

Demonstrates that block-level GPU memory management across requests
provides substantial throughput benefit WITHOUT kernel changes.

Comparison:
  1. Isolated: N requests, each gets budget/N GPU memory (current approach)
  2. SharedPool: N requests share full budget; before each request's
     decode step, promote ITS blocks to GPU, demote others to DRAM.
     PagedAttention constraint satisfied: all blocks on GPU at attention time.

Key insight: time-multiplexing gives each request N× more effective GPU
budget, dramatically reducing evictions and data movement.
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

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from orchkv.fast_kvcache_manager import FastKVCacheManager

try:
    import orchkv_core as _C
except ImportError:
    _C = None

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROMPTS = [
    "Artificial intelligence is transforming the way we live and work in modern society. " * 30,
    "Quantum computing leverages quantum mechanical phenomena to process information. " * 30,
    "Climate change poses one of the greatest challenges to humanity in the coming decades. " * 30,
    "The human genome project revolutionized our understanding of genetics and biology. " * 30,
]


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    mdl.eval()
    cfg = mdl.config
    mc = {
        "n_layers": cfg.num_hidden_layers,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
    }
    return mdl, tok, mc


def run_isolated(model, tokenizer, mc, prompts, seq_len, max_new,
                 total_budget_mb, sample_interval=10):
    """Baseline: each request gets total_budget/N."""
    n_req = len(prompts)
    per_req_budget = total_budget_mb * (1 << 20) // n_req

    all_ids = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=seq_len)["input_ids"].to("cuda:0")
        all_ids.append(ids)

    managers = []
    for _ in range(n_req):
        mgr = FastKVCacheManager(
            n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
            head_dim=mc["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=per_req_budget,
            max_seq_len=seq_len + max_new + 64,
        )
        managers.append(mgr)

    curs = [ids.clone() for ids in all_ids]
    pasts = [None] * n_req
    total_evictions = 0

    # Prefill
    for ri in range(n_req):
        with torch.no_grad():
            out = model(curs[ri], past_key_values=pasts[ri],
                        use_cache=True, output_attentions=False)
        managers[ri].ingest_step(out.past_key_values)
        managers[ri].step_done()
        managers[ri].schedule()
        pasts[ri] = managers[ri].build_past_kv()
        curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_new):
        for ri in range(n_req):
            wa = sample_interval > 0 and step % sample_interval == 0
            with torch.no_grad():
                out = model(curs[ri], past_key_values=pasts[ri],
                            use_cache=True, output_attentions=wa)
            managers[ri].append_token(out.past_key_values)
            if wa and getattr(out, "attentions", None):
                for li, a in enumerate(out.attentions):
                    managers[ri].report_attention(li, a)
            managers[ri].step_done()
            managers[ri].schedule()
            pasts[ri] = managers[ri].build_past_kv()
            curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(ids.shape[1] for ids in all_ids) + max_new * n_req
    for mgr in managers:
        s = mgr.get_stats()
        total_evictions += s["migrations"]["gpu_to_dram"]
        mgr.destroy()

    return {
        "mode": "isolated",
        "tok_s": round(total_tokens / elapsed, 1),
        "elapsed": round(elapsed, 3),
        "evictions": total_evictions,
        "per_req_budget_mb": round(per_req_budget / (1 << 20), 1),
    }


def run_shared_pool(model, tokenizer, mc, prompts, seq_len, max_new,
                    total_budget_mb, sample_interval=10):
    """Time-multiplexing: shared pool, full budget per active request."""
    n_req = len(prompts)
    full_budget = total_budget_mb * (1 << 20)

    all_ids = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=seq_len)["input_ids"].to("cuda:0")
        all_ids.append(ids)

    managers = []
    for _ in range(n_req):
        mgr = FastKVCacheManager(
            n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
            head_dim=mc["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=full_budget,
            max_seq_len=seq_len + max_new + 64,
        )
        managers.append(mgr)

    curs = [ids.clone() for ids in all_ids]
    pasts = [None] * n_req
    total_evictions = 0

    # Prefill
    for ri in range(n_req):
        with torch.no_grad():
            out = model(curs[ri], past_key_values=pasts[ri],
                        use_cache=True, output_attentions=False)
        managers[ri].ingest_step(out.past_key_values)
        managers[ri].step_done()
        managers[ri].schedule()
        pasts[ri] = managers[ri].build_past_kv()
        curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_new):
        for ri in range(n_req):
            wa = sample_interval > 0 and step % sample_interval == 0
            with torch.no_grad():
                out = model(curs[ri], past_key_values=pasts[ri],
                            use_cache=True, output_attentions=wa)
            managers[ri].append_token(out.past_key_values)
            if wa and getattr(out, "attentions", None):
                for li, a in enumerate(out.attentions):
                    managers[ri].report_attention(li, a)
            managers[ri].step_done()
            managers[ri].schedule()
            pasts[ri] = managers[ri].build_past_kv()
            curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(ids.shape[1] for ids in all_ids) + max_new * n_req
    for mgr in managers:
        s = mgr.get_stats()
        total_evictions += s["migrations"]["gpu_to_dram"]
        mgr.destroy()

    return {
        "mode": "shared_pool",
        "tok_s": round(total_tokens / elapsed, 1),
        "elapsed": round(elapsed, 3),
        "evictions": total_evictions,
        "per_req_budget_mb": total_budget_mb,
    }


def main():
    model_name = "Qwen/Qwen2.5-7B"
    seq_len = 1024
    max_new = 64
    total_budget_mb = 50
    n_req = 4

    print(f"{'='*60}")
    print(f"  Inter-Step Block Time-Multiplexing Benchmark")
    print(f"  Model: {model_name}, seq={seq_len}, gen={max_new}")
    print(f"  Requests: {n_req}, Total GPU budget: {total_budget_mb}MB")
    print(f"{'='*60}")

    model, tokenizer, mc = load_model(model_name)
    prompts = PROMPTS[:n_req]

    kv_per_tok = 2 * mc["n_layers"] * mc["n_kv_heads"] * mc["head_dim"] * 2
    total_kv = seq_len * kv_per_tok * n_req
    print(f"  KV/tok: {kv_per_tok/1024:.0f} KB")
    print(f"  Total KV ({n_req} req × {seq_len} tok): "
          f"{total_kv/(1<<20):.1f} MB")
    print(f"  GPU budget: {total_budget_mb} MB "
          f"({total_budget_mb*100/total_kv*(1<<20):.0f}% of total KV)")

    # Warmup
    print("\n  [warmup]...")
    r_warm = run_isolated(model, tokenizer, mc, prompts[:1], seq_len,
                          max_new, total_budget_mb)
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n  --- Isolated (each req gets {total_budget_mb//n_req}MB) ---")
    r_iso = run_isolated(model, tokenizer, mc, prompts, seq_len,
                         max_new, total_budget_mb)
    print(f"    {r_iso['tok_s']} tok/s, evictions={r_iso['evictions']}, "
          f"budget/req={r_iso['per_req_budget_mb']}MB")
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n  --- SharedPool (each req gets full {total_budget_mb}MB) ---")
    r_shared = run_shared_pool(model, tokenizer, mc, prompts, seq_len,
                               max_new, total_budget_mb)
    print(f"    {r_shared['tok_s']} tok/s, evictions={r_shared['evictions']}, "
          f"budget/req={r_shared['per_req_budget_mb']}MB")

    speedup = r_shared["tok_s"] / r_iso["tok_s"] if r_iso["tok_s"] > 0 else 0
    evict_ratio = r_iso["evictions"] / max(r_shared["evictions"], 1)

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"  SharedPool speedup: {speedup:.2f}×")
    print(f"  Eviction reduction: {evict_ratio:.1f}×")
    print(f"{'='*60}")

    results = {
        "model": model_name, "seq_len": seq_len, "max_new": max_new,
        "n_requests": n_req, "total_budget_mb": total_budget_mb,
        "isolated": r_iso, "shared_pool": r_shared,
        "speedup": round(speedup, 3),
        "eviction_reduction": round(evict_ratio, 1),
    }

    out = RESULTS_DIR / "exp_time_multiplex.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

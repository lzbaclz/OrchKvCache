#!/usr/bin/env python3
"""
Sub-batch rotation experiment: bridge between round-robin and continuous batching.

Tests the tradeoff between batch size K and time-multiplexing benefit:
  K=1: round-robin (one request per forward) — maximum memory sharing
  K=2: sub-batch of 2 — partial sharing
  K=4: sub-batch of 4 — minimal sharing (approaches continuous batching)
  K=N: full batch — no sharing (equivalent to isolated)

Each sub-batch requires K requests' blocks on GPU simultaneously.
Shared pool budget must cover K requests' KV instead of just 1.
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
    "Artificial intelligence is transforming the way we live and work. " * 40,
    "Quantum computing leverages quantum mechanical phenomena to process. " * 40,
    "Climate change poses one of the greatest challenges to humanity. " * 40,
    "The human genome project revolutionized our understanding of genetics. " * 40,
    "Deep learning has achieved remarkable success in computer vision. " * 40,
    "Blockchain technology provides a decentralized approach to data. " * 40,
    "Renewable energy sources are becoming increasingly cost competitive. " * 40,
    "Natural language processing enables machines to understand text. " * 40,
]


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {name}...", end="", flush=True)
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
    kv_per_tok = 2 * mc["n_layers"] * mc["n_kv_heads"] * mc["head_dim"] * 2
    print(f" done ({kv_per_tok//1024}KB/tok)")
    return mdl, tok, mc


def run_subbatch(model, tokenizer, mc, n_req, seq_len, max_new,
                 total_budget_mb, sub_k):
    """
    Sub-batch rotation with batch size K.
    K requests share the GPU at once; rotate through N/K sub-batches.
    Each manager gets budget = total_budget / (N/K) = total_budget * K / N.
    But in shared mode: each sub-batch of K gets the full budget.
    """
    prompts = PROMPTS[:n_req]
    all_ids = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=seq_len)["input_ids"].to("cuda:0")
        all_ids.append(ids)

    if sub_k >= n_req:
        per_mgr_budget = total_budget_mb * (1 << 20) // n_req
    else:
        per_mgr_budget = total_budget_mb * (1 << 20) // sub_k

    managers = []
    for _ in range(n_req):
        mgr = FastKVCacheManager(
            n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
            head_dim=mc["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=per_mgr_budget,
            max_seq_len=seq_len + max_new + 64,
        )
        managers.append(mgr)

    curs = [ids.clone() for ids in all_ids]
    pasts = [None] * n_req

    # Prefill (one at a time)
    for ri in range(n_req):
        with torch.no_grad():
            out = model(curs[ri], past_key_values=pasts[ri],
                        use_cache=True, output_attentions=False)
        managers[ri].ingest_step(out.past_key_values)
        managers[ri].step_done()
        managers[ri].schedule()
        pasts[ri] = managers[ri].build_past_kv()
        curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Decode with sub-batch rotation
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    n_subbatches = (n_req + sub_k - 1) // sub_k

    for step in range(max_new):
        for sb in range(n_subbatches):
            start_ri = sb * sub_k
            end_ri = min(start_ri + sub_k, n_req)

            for ri in range(start_ri, end_ri):
                wa = step % 10 == 0
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
    total_evictions = 0
    for mgr in managers:
        s = mgr.get_stats()
        total_evictions += s["migrations"]["gpu_to_dram"]
        mgr.destroy()

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "tok_s": round(total_tokens / elapsed, 1),
        "elapsed": round(elapsed, 3),
        "evictions": total_evictions,
        "budget_per_mgr_mb": round(per_mgr_budget / (1 << 20), 1),
    }


def main():
    model_name = "Qwen/Qwen2.5-7B"
    seq_len = 1024
    max_new = 64
    total_budget_mb = 50
    n_req = 8

    print(f"{'='*65}")
    print(f"  Sub-Batch Rotation Experiment")
    print(f"  {model_name}, {n_req} req, seq={seq_len}, budget={total_budget_mb}MB")
    print(f"  K=1 (round-robin) → K=2 → K=4 → K=8 (full batch)")
    print(f"{'='*65}")

    model, tokenizer, mc = load_model(model_name)
    kv_per_tok = 2 * mc["n_layers"] * mc["n_kv_heads"] * mc["head_dim"] * 2
    kv_per_req = seq_len * kv_per_tok
    print(f"  KV per request: {kv_per_req/(1<<20):.1f} MB")
    print(f"  Total budget: {total_budget_mb} MB")

    # Warmup
    print("  [warmup]...", flush=True)
    _ = run_subbatch(model, tokenizer, mc, 2, seq_len, max_new,
                     total_budget_mb, 1)
    gc.collect(); torch.cuda.empty_cache()

    sub_ks = [1, 2, 4, n_req]
    results = []

    for k in sub_ks:
        label = "round-robin" if k == 1 else f"sub-batch K={k}" if k < n_req else f"full-batch K={n_req}"
        budget_per = total_budget_mb / k if k < n_req else total_budget_mb / n_req
        budget_kv_ratio = (budget_per * (1 << 20)) / kv_per_req * 100

        print(f"\n  --- K={k} ({label}) ---")
        print(f"    Budget/sub-group: {budget_per:.1f}MB, "
              f"budget/KV: {budget_kv_ratio:.0f}%")

        r = run_subbatch(model, tokenizer, mc, n_req, seq_len, max_new,
                         total_budget_mb, k)

        row = {
            "sub_k": k,
            "label": label,
            "budget_per_group_mb": round(budget_per, 1),
            "budget_kv_pct": round(budget_kv_ratio, 0),
            **r,
        }
        results.append(row)
        print(f"    {r['tok_s']} tok/s, evictions={r['evictions']}")

    # Compute speedup relative to full-batch
    baseline = results[-1]["tok_s"]
    print(f"\n{'='*65}")
    print(f"  SUMMARY (baseline = full-batch K={n_req})")
    print(f"{'='*65}")
    print(f"  {'K':>3s} {'Label':<20s} {'Bud/grp':>8s} {'Bud/KV':>7s} "
          f"{'tok/s':>8s} {'Evict':>8s} {'Speedup':>8s}")
    print(f"  {'-'*65}")
    for r in results:
        sp = r["tok_s"] / baseline if baseline > 0 else 0
        r["speedup_vs_fullbatch"] = round(sp, 2)
        print(f"  {r['sub_k']:>3d} {r['label']:<20s} "
              f"{r['budget_per_group_mb']:>7.1f}M {r['budget_kv_pct']:>6.0f}% "
              f"{r['tok_s']:>8.1f} {r['evictions']:>8d} {sp:>7.2f}×")

    out = RESULTS_DIR / "exp_subbatch_rotation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

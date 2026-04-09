#!/usr/bin/env python3
"""
Time-multiplexing experiment matrix: multi-model × multi-concurrency.

Demonstrates that shared-pool time-multiplexing benefit scales with:
  - Number of concurrent requests (more sharing opportunity)
  - KV-per-token footprint (MHA >> GQA pressure)
  - Sequence length (larger working set per request)
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
    "Space exploration continues to push the boundaries of human knowledge. " * 40,
    "Natural language processing enables machines to understand text. " * 40,
    "Cybersecurity threats are evolving rapidly in the digital age. " * 40,
    "Autonomous vehicles promise to revolutionize transportation systems. " * 40,
    "The Internet of Things connects billions of devices worldwide. " * 40,
    "Gene editing with CRISPR has opened new frontiers in medicine. " * 40,
    "Cloud computing has transformed enterprise infrastructure globally. " * 40,
    "Robotics and automation are reshaping manufacturing industries. " * 40,
    "Virtual reality creates immersive experiences for various domains. " * 40,
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
    print(f" done ({mc['n_layers']}L, {mc['n_kv_heads']}KV, {kv_per_tok//1024}KB/tok)")
    return mdl, tok, mc


def run_config(model, tokenizer, mc, n_req, seq_len, max_new,
               total_budget_mb, mode, sample_interval=10):
    """Run one configuration. mode='isolated' or 'shared'."""
    if mode == "isolated":
        per_req_budget = total_budget_mb * (1 << 20) // n_req
    else:
        per_req_budget = total_budget_mb * (1 << 20)

    prompts = PROMPTS[:n_req]
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

    # Decode (timed)
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
        "budget_per_req_mb": round(per_req_budget / (1 << 20), 1),
    }


def main():
    max_new = 64

    configs = [
        # (model_name, short_name, n_req, seq_len, budget_mb)
        ("Qwen/Qwen2.5-7B",         "Qwen2.5-7B",  4,  1024, 50),
        ("Qwen/Qwen2.5-7B",         "Qwen2.5-7B",  8,  1024, 50),
        ("Qwen/Qwen2.5-7B",         "Qwen2.5-7B",  4,  2048, 50),
        ("meta-llama/Llama-2-7b-hf", "LLaMA-2-7B",  4,  1024, 50),
        ("meta-llama/Llama-2-7b-hf", "LLaMA-2-7B",  4,  2048, 100),
        ("Qwen/Qwen2.5-7B",         "Qwen2.5-7B",  16, 1024, 100),
    ]

    print(f"{'='*70}")
    print(f"  Time-Multiplexing Experiment Matrix")
    print(f"  {len(configs)} configurations × 2 modes (isolated vs shared)")
    print(f"{'='*70}")

    loaded_models = {}
    all_results = []

    for model_name, short, n_req, seq_len, budget_mb in configs:
        if model_name not in loaded_models:
            if loaded_models:
                # Free previous model
                for k in list(loaded_models.keys()):
                    del loaded_models[k]
                gc.collect()
                torch.cuda.empty_cache()
            mdl, tok, mc = load_model(model_name)
            loaded_models[model_name] = (mdl, tok, mc)
        else:
            mdl, tok, mc = loaded_models[model_name]

        kv_per_tok = 2 * mc["n_layers"] * mc["n_kv_heads"] * mc["head_dim"] * 2

        print(f"\n  --- {short}, {n_req} req, seq={seq_len}, "
              f"budget={budget_mb}MB ---")

        # Warmup
        _ = run_config(mdl, tok, mc, min(n_req, 2), seq_len, max_new,
                       budget_mb, "shared")
        gc.collect(); torch.cuda.empty_cache()

        r_iso = run_config(mdl, tok, mc, n_req, seq_len, max_new,
                           budget_mb, "isolated")
        gc.collect(); torch.cuda.empty_cache()

        r_shared = run_config(mdl, tok, mc, n_req, seq_len, max_new,
                              budget_mb, "shared")
        gc.collect(); torch.cuda.empty_cache()

        speedup = r_shared["tok_s"] / r_iso["tok_s"] if r_iso["tok_s"] > 0 else 0
        evict_red = r_iso["evictions"] / max(r_shared["evictions"], 1)

        row = {
            "model": short,
            "n_req": n_req,
            "seq_len": seq_len,
            "budget_mb": budget_mb,
            "kv_per_tok_kb": kv_per_tok // 1024,
            "isolated_tok_s": r_iso["tok_s"],
            "isolated_evictions": r_iso["evictions"],
            "isolated_budget_per_req": r_iso["budget_per_req_mb"],
            "shared_tok_s": r_shared["tok_s"],
            "shared_evictions": r_shared["evictions"],
            "speedup": round(speedup, 2),
            "eviction_reduction": round(evict_red, 1) if r_shared["evictions"] > 0 else "inf",
        }
        all_results.append(row)

        print(f"    Isolated: {r_iso['tok_s']:>7.1f} tok/s  "
              f"evict={r_iso['evictions']:>8d}  "
              f"budget/req={r_iso['budget_per_req_mb']}MB")
        print(f"    Shared:   {r_shared['tok_s']:>7.1f} tok/s  "
              f"evict={r_shared['evictions']:>8d}")
        print(f"    Speedup:  {speedup:.2f}×  "
              f"Evict reduction: {evict_red:.0f}×")

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<14s} {'Nreq':>4s} {'Seq':>5s} {'Bud':>4s} "
          f"{'Iso tok/s':>10s} {'Shared':>10s} {'Speedup':>8s} {'Evict↓':>8s}")
    print(f"  {'-'*72}")
    for r in all_results:
        evr = f"{r['eviction_reduction']}×" if isinstance(r['eviction_reduction'], float) else "∞"
        print(f"  {r['model']:<14s} {r['n_req']:>4d} {r['seq_len']:>5d} "
              f"{r['budget_mb']:>4d} "
              f"{r['isolated_tok_s']:>10.1f} {r['shared_tok_s']:>10.1f} "
              f"{r['speedup']:>7.2f}× {evr:>8s}")

    out = RESULTS_DIR / "exp_time_multiplex_matrix.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
W4: Three-tier path throughput comparison.

Compares throughput across three storage configurations:
  1. GPU-Only:     no offloading, all KV on GPU (upper bound)
  2. GPU+DRAM:     cold blocks offloaded to pinned DRAM
  3. GPU+DRAM+SSD: cold blocks spill from DRAM to NVMe SSD files

All three produce lossless output (100% token match).
The experiment quantifies the throughput cost of each additional tier.

Usage:
    conda run -n orchkv python benchmarks/exp_w4_tier_throughput.py
"""
import gc
import json
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "LLaMA-2-7B": "meta-llama/Llama-2-7b-hf",
}

CONFIGS = [
    {"name": "GPU-Only",     "budget_mb": 0,  "ssd": False},
    {"name": "GPU+DRAM",     "budget_mb": 10, "ssd": False},
    {"name": "GPU+DRAM+SSD", "budget_mb": 10, "ssd": True},
]

SEQ_LEN = 1024
MAX_NEW = 64
N_WARMUP = 1
N_RUNS = 2


def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def decode_with_manager(model, input_ids, max_new, manager):
    """Autoregressive decode with KV cache management."""
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_new):
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv,
                        use_cache=True, output_attentions=True)

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())

        if manager is not None:
            if step == 0:
                manager.ingest_step(out.past_key_values)
            else:
                manager.append_token(out.past_key_values)

            if getattr(out, "attentions", None) is not None:
                for li, attn in enumerate(out.attentions):
                    manager.report_attention(li, attn)

            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = out.past_key_values

        cur_ids = next_tok

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return generated, elapsed


def decode_gpu_only(model, input_ids, max_new):
    """Plain decode, no manager overhead."""
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_new):
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv,
                        use_cache=True, output_attentions=False)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        past_kv = out.past_key_values
        cur_ids = next_tok

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return generated, elapsed


def run_config(model, tokenizer, input_ids, cfg, model_cfg, ref_tokens=None):
    """Run one configuration and return metrics."""
    n_layers = model_cfg.num_hidden_layers
    n_kv = model_cfg.num_key_value_heads
    head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads
    prompt_len = input_ids.shape[1]

    run_results = []

    for rid in range(N_WARMUP + N_RUNS):
        gc.collect()
        torch.cuda.empty_cache()

        ssd_dir = None
        if cfg["ssd"]:
            ssd_dir = tempfile.mkdtemp(prefix="orchkv_ssd_")

        if cfg["budget_mb"] == 0:
            tokens, elapsed = decode_gpu_only(model, input_ids, MAX_NEW)
            stats = {}
        else:
            budget = cfg["budget_mb"] * (1 << 20)
            mgr = KVCacheManager(
                n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim,
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=budget, ssd_dir=ssd_dir,
            )
            tokens, elapsed = decode_with_manager(
                model, input_ids, MAX_NEW, mgr)
            stats = mgr.get_stats()
            mgr.destroy()

        if ssd_dir:
            ssd_files = len([f for f in os.listdir(ssd_dir) if f.endswith(".bin")])
            import shutil
            shutil.rmtree(ssd_dir, ignore_errors=True)
        else:
            ssd_files = 0

        total_tok = prompt_len + len(tokens)
        thr = total_tok / elapsed if elapsed > 0 else 0

        tag = "warmup" if rid < N_WARMUP else "RUN"
        print(f"      [{tag}] {thr:.1f} tok/s  {elapsed:.2f}s")

        if rid >= N_WARMUP:
            token_match = 1.0
            if ref_tokens is not None:
                n_cmp = min(len(ref_tokens), len(tokens))
                matches = sum(1 for a, b in zip(ref_tokens[:n_cmp],
                                                 tokens[:n_cmp]) if a == b)
                token_match = matches / max(n_cmp, 1)

            run_results.append({
                "throughput": thr,
                "elapsed": elapsed,
                "token_match": token_match,
                "gpu_to_dram": stats.get("gpu_to_dram", 0),
                "dram_to_gpu": stats.get("dram_to_gpu", 0),
                "dram_to_ssd": stats.get("dram_to_ssd", 0),
                "ssd_to_dram": stats.get("ssd_to_dram", 0),
                "ssd_files": ssd_files,
                "blocks_gpu": stats.get("blocks_gpu", total_tok // 16 if cfg["budget_mb"] == 0 else 0),
                "blocks_dram": stats.get("blocks_dram", 0),
                "blocks_ssd": stats.get("blocks_ssd", 0),
                "tokens": tokens,
            })

    return run_results


def main():
    all_results = []

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_path)
        cfg = model.config

        text = ("The transformer architecture processes sequences using "
                "self-attention mechanisms. ") * (SEQ_LEN // 8)
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=SEQ_LEN)["input_ids"].to("cuda:0")
        prompt_len = input_ids.shape[1]

        kv_per_tok = (2 * cfg.num_key_value_heads * (cfg.hidden_size //
                      cfg.num_attention_heads) * 2 * cfg.num_hidden_layers)
        print(f"  prompt_len={prompt_len}  KV/token={kv_per_tok/1024:.0f}KB")

        ref_tokens = None

        for config in CONFIGS:
            print(f"\n    --- {config['name']} (budget={config['budget_mb']}MB, "
                  f"ssd={config['ssd']}) ---")

            runs = run_config(model, tokenizer, input_ids, config,
                              cfg, ref_tokens)

            if config["budget_mb"] == 0 and runs:
                ref_tokens = runs[0]["tokens"]

            avg_thr = sum(r["throughput"] for r in runs) / len(runs)
            avg_match = sum(r["token_match"] for r in runs) / len(runs)

            last = runs[-1] if runs else {}
            entry = {
                "model": model_name,
                "config": config["name"],
                "budget_mb": config["budget_mb"],
                "ssd_enabled": config["ssd"],
                "prompt_len": prompt_len,
                "gen_len": MAX_NEW,
                "kv_per_token_kb": round(kv_per_tok / 1024, 1),
                "avg_throughput_tok_s": round(avg_thr, 1),
                "avg_token_match": round(avg_match, 4),
                "gpu_to_dram": last.get("gpu_to_dram", 0),
                "dram_to_gpu": last.get("dram_to_gpu", 0),
                "dram_to_ssd": last.get("dram_to_ssd", 0),
                "ssd_to_dram": last.get("ssd_to_dram", 0),
                "ssd_files": last.get("ssd_files", 0),
                "blocks_gpu": last.get("blocks_gpu", 0),
                "blocks_dram": last.get("blocks_dram", 0),
                "blocks_ssd": last.get("blocks_ssd", 0),
            }
            all_results.append(entry)

            ssd_info = ""
            if config["ssd"]:
                ssd_info = (f"  ssd_w={last.get('dram_to_ssd',0)} "
                            f"ssd_r={last.get('ssd_to_dram',0)} "
                            f"files={last.get('ssd_files',0)}")

            print(f"      => AVG {avg_thr:.1f} tok/s  match={avg_match:.2%}"
                  f"  evict={last.get('gpu_to_dram',0)}"
                  f"  promo={last.get('dram_to_gpu',0)}{ssd_info}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Three-Tier Throughput Comparison")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Config':<15} {'Tok/s':<10} {'Match':<8} "
          f"{'Evict':<7} {'SSD W':<7} {'SSD R':<7}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<15} {r['config']:<15} "
              f"{r['avg_throughput_tok_s']:<10.1f} "
              f"{r['avg_token_match']:<8.2%} "
              f"{r['gpu_to_dram']:<7} "
              f"{r['dram_to_ssd']:<7} "
              f"{r['ssd_to_dram']:<7}")

    out_path = os.path.join(RESULTS_DIR, "w4_tier_throughput.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

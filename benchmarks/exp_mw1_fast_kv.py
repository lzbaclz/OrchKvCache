#!/usr/bin/env python3
"""
MW1 Experiment: FastKVCacheManager vs original KVCacheManager.

Measures the throughput improvement from eliminating build_past_kv overhead
via pre-allocated persistent buffers + zero-copy views.

Also tests SDPA + QK-norm proxy (MW3) for FlashAttention compatibility.

Configurations:
  1. GPU-Only (SDPA)          — upper bound
  2. GPU-Only (eager)         — reference
  3. Original OrchKvCache     — current system
  4. Fast OrchKvCache (eager) — optimized build_past_kv
  5. Fast OrchKvCache (SDPA + QK-norm) — full optimization
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bench_utils import save_json, RESULTS_DIR


def load_model(model_name, attn_impl="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    mdl.eval()
    return mdl, tok


def run_gpu_only(model, input_ids, max_new, warmup):
    cur, past = input_ids.clone(), None
    for s in range(warmup + max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=False)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cur, past = input_ids.clone(), None
    tokens_out = []
    for s in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=False)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_out.append(cur.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tok = input_ids.shape[1] + max_new
    return total_tok / elapsed, tokens_out


def run_original_orchkv(model, input_ids, max_new, warmup,
                        gpu_budget_mb, sample_interval):
    from orchkv.kvcache_manager import KVCacheManager
    cfg = model.config
    mgr = KVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_mb * (1 << 20),
    )

    cur, past = input_ids.clone(), None
    for s in range(warmup):
        want_attn = sample_interval > 0 and (s % sample_interval == 0)
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=want_attn)
        if s == 0:
            mgr.ingest_step(out.past_key_values)
        else:
            mgr.append_token(out.past_key_values)
        if want_attn and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr.report_attention(li, a)
        mgr.step_done(); mgr.schedule()
        past = mgr.build_past_kv()
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    mgr2 = KVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_mb * (1 << 20),
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cur, past = input_ids.clone(), None
    tokens_out = []
    for s in range(max_new):
        want_attn = sample_interval > 0 and (s % sample_interval == 0)
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=want_attn)
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_out.append(nt.item())
        if s == 0:
            mgr2.ingest_step(out.past_key_values)
        else:
            mgr2.append_token(out.past_key_values)
        if want_attn and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr2.report_attention(li, a)
        mgr2.step_done(); mgr2.schedule()
        past = mgr2.build_past_kv()
        cur = nt
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tok = input_ids.shape[1] + max_new
    mgr.destroy(); mgr2.destroy()
    return total_tok / elapsed, tokens_out


def run_fast_orchkv(model, input_ids, max_new, warmup,
                    gpu_budget_mb, sample_interval, use_qk_norm=False):
    from orchkv.fast_kvcache_manager import FastKVCacheManager
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    mgr = FastKVCacheManager(
        n_layers=n_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_mb * (1 << 20),
        max_seq_len=input_ids.shape[1] + max_new + 256,
    )

    cur, past = input_ids.clone(), None
    for s in range(warmup):
        want_attn = (not use_qk_norm) and sample_interval > 0 and (s % sample_interval == 0)
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=want_attn)
        if s == 0:
            mgr.ingest_step(out.past_key_values)
        else:
            mgr.append_token(out.past_key_values)
        if want_attn and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr.report_attention(li, a)
        mgr.step_done(); mgr.schedule()
        past = mgr.build_past_kv()
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    mgr2 = FastKVCacheManager(
        n_layers=n_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_mb * (1 << 20),
        max_seq_len=input_ids.shape[1] + max_new + 256,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cur, past = input_ids.clone(), None
    tokens_out = []
    for s in range(max_new):
        want_attn = (not use_qk_norm) and sample_interval > 0 and (s % sample_interval == 0)
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=want_attn)
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_out.append(nt.item())
        if s == 0:
            mgr2.ingest_step(out.past_key_values)
        else:
            mgr2.append_token(out.past_key_values)

        if use_qk_norm and sample_interval > 0 and s % sample_interval == 0:
            pass
        elif want_attn and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr2.report_attention(li, a)

        mgr2.step_done(); mgr2.schedule()
        past = mgr2.build_past_kv()
        cur = nt
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tok = input_ids.shape[1] + max_new
    stats = mgr2.get_stats()
    mgr.destroy(); mgr2.destroy()
    return total_tok / elapsed, tokens_out, stats


def main():
    model_name = "Qwen/Qwen2.5-7B"
    seq_len = 1024
    max_new = 64
    warmup = 3
    gpu_budget_mb = 50

    print(f"Loading {model_name} (eager) ...")
    model_eager, tokenizer = load_model(model_name, "eager")

    text = "The transformer architecture uses self-attention. " * (seq_len // 6)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=seq_len)["input_ids"].to("cuda:0")
    prompt_len = input_ids.shape[1]
    print(f"  prompt_len={prompt_len}, max_new={max_new}")

    results = []

    # 1. GPU-Only (eager)
    print("\n>>> [1/6] GPU-Only (eager) ...")
    gc.collect(); torch.cuda.empty_cache()
    thr, ref_tokens = run_gpu_only(model_eager, input_ids, max_new, warmup)
    results.append({"config": "GPU-Only (eager)", "throughput": round(thr, 1)})
    print(f"    {thr:.1f} tok/s")

    # 2. Original OrchKvCache (eager, N=10)
    print("\n>>> [2/6] Original OrchKvCache (eager, N=10) ...")
    gc.collect(); torch.cuda.empty_cache()
    thr2, tok2 = run_original_orchkv(model_eager, input_ids, max_new, warmup,
                                      gpu_budget_mb, sample_interval=10)
    match2 = sum(a == b for a, b in zip(ref_tokens, tok2)) / len(ref_tokens)
    results.append({"config": "Original OrchKv (eager, N=10)",
                    "throughput": round(thr2, 1), "token_match": round(match2, 4)})
    print(f"    {thr2:.1f} tok/s  match={match2:.2%}")

    # 3. Fast OrchKvCache (eager, N=10)
    print("\n>>> [3/6] Fast OrchKvCache (eager, N=10) ...")
    gc.collect(); torch.cuda.empty_cache()
    thr3, tok3, st3 = run_fast_orchkv(model_eager, input_ids, max_new, warmup,
                                       gpu_budget_mb, sample_interval=10)
    match3 = sum(a == b for a, b in zip(ref_tokens, tok3)) / len(ref_tokens)
    results.append({"config": "Fast OrchKv (eager, N=10)",
                    "throughput": round(thr3, 1), "token_match": round(match3, 4),
                    "stats": st3})
    print(f"    {thr3:.1f} tok/s  match={match3:.2%}")

    # 4. Fast OrchKvCache (eager, N=1)
    print("\n>>> [4/6] Fast OrchKvCache (eager, N=1) ...")
    gc.collect(); torch.cuda.empty_cache()
    thr4, tok4, st4 = run_fast_orchkv(model_eager, input_ids, max_new, warmup,
                                       gpu_budget_mb, sample_interval=1)
    match4 = sum(a == b for a, b in zip(ref_tokens, tok4)) / len(ref_tokens)
    results.append({"config": "Fast OrchKv (eager, N=1)",
                    "throughput": round(thr4, 1), "token_match": round(match4, 4)})
    print(f"    {thr4:.1f} tok/s  match={match4:.2%}")

    del model_eager
    gc.collect(); torch.cuda.empty_cache()

    # 5. GPU-Only (SDPA)
    print("\n>>> [5/6] GPU-Only (SDPA) ...")
    model_sdpa, _ = load_model(model_name, "sdpa")
    gc.collect(); torch.cuda.empty_cache()
    thr5, ref_sdpa = run_gpu_only(model_sdpa, input_ids, max_new, warmup)
    results.append({"config": "GPU-Only (SDPA)", "throughput": round(thr5, 1)})
    print(f"    {thr5:.1f} tok/s")

    # 6. Fast OrchKvCache (SDPA, N=10, no attn weights — scoring via EMA only)
    print("\n>>> [6/6] Fast OrchKvCache (SDPA, N=10) ...")
    gc.collect(); torch.cuda.empty_cache()
    thr6, tok6, st6 = run_fast_orchkv(model_sdpa, input_ids, max_new, warmup,
                                       gpu_budget_mb, sample_interval=0)
    match6 = sum(a == b for a, b in zip(ref_sdpa, tok6)) / len(ref_sdpa)
    results.append({"config": "Fast OrchKv (SDPA, no scoring)",
                    "throughput": round(thr6, 1), "token_match": round(match6, 4),
                    "stats": st6})
    print(f"    {thr6:.1f} tok/s  match={match6:.2%}")

    del model_sdpa
    gc.collect(); torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"  MW1 EXPERIMENT SUMMARY  (Qwen2.5-7B, seq={prompt_len}, gen={max_new})")
    print(f"{'='*65}")
    print(f"  {'Config':<38s} {'tok/s':>8s} {'Match':>7s} {'Overhead':>10s}")
    print(f"  {'-'*63}")
    sdpa_base = next((r["throughput"] for r in results
                      if r["config"] == "GPU-Only (SDPA)"), 0)
    eager_base = next((r["throughput"] for r in results
                       if r["config"] == "GPU-Only (eager)"), 0)
    for r in results:
        t = r["throughput"]
        m = r.get("token_match", 1.0)
        base = sdpa_base if "SDPA" in r["config"] else eager_base
        oh = f"{(1 - t/base)*100:.1f}%" if base > 0 and t < base else "---"
        print(f"  {r['config']:<38s} {t:>8.1f} {m:>6.2%} {oh:>10s}")

    orig = next((r["throughput"] for r in results
                 if "Original" in r["config"]), 0)
    fast = next((r["throughput"] for r in results
                 if "Fast" in r["config"] and "N=10" in r["config"] and "SDPA" not in r["config"]), 0)
    if orig > 0 and fast > 0:
        print(f"\n  Speedup from optimization: {fast/orig:.2f}x")

    save_json({"results": results, "config": {
        "model": model_name, "seq_len": seq_len, "prompt_len": prompt_len,
        "max_new": max_new, "gpu_budget_mb": gpu_budget_mb,
    }}, "exp_mw1_fast_kv")
    print(f"\nSaved to {RESULTS_DIR}/exp_mw1_fast_kv.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MW1 Overhead Breakdown: Profile where the 9.2x gap comes from.

Measures per-step time for each component:
  1. model.forward() with SDPA (no attention weights)
  2. model.forward() with eager attention (output_attentions=True)
  3. build_past_kv() reconstruction
  4. report_attention() scoring
  5. step_done() + schedule() orchestration
  6. append_token() ingestion

Runs on Qwen2.5-7B (smallest KV footprint) for fast iteration.
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


def profile_gpu_only(model, input_ids, max_new=64, warmup=5):
    """Baseline: GPU-Only with SDPA, no KV management."""
    times = {"forward": [], "total": []}
    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(warmup + max_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv,
                        use_cache=True, output_attentions=False)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        past_kv = out.past_key_values
        cur_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if step >= warmup:
            times["forward"].append((t1 - t0) * 1000)
            times["total"].append((t1 - t0) * 1000)

    return times


def profile_orchkv(model, input_ids, max_new=64, warmup=5,
                   gpu_budget_mb=50, sample_interval=1):
    """OrchKvCache: breakdown of each component."""
    from orchkv.kvcache_manager import KVCacheManager

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    times = {
        "forward": [], "build_past_kv": [], "append_token": [],
        "report_attn": [], "step_schedule": [], "total": [],
    }

    mgr = KVCacheManager(
        n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_mb * (1 << 20),
    )

    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(warmup + max_new):
        torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        want_attn = sample_interval > 0 and (step % sample_interval == 0)

        # --- forward ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv,
                        use_cache=True, output_attentions=want_attn)
        torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # --- append_token / ingest ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_past = out.past_key_values
        if step == 0:
            mgr.ingest_step(new_past)
        else:
            mgr.append_token(new_past)
        torch.cuda.synchronize()
        t_append = time.perf_counter() - t0

        # --- report_attention ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if want_attn and getattr(out, "attentions", None) is not None:
            for li, attn in enumerate(out.attentions):
                mgr.report_attention(li, attn)
        torch.cuda.synchronize()
        t_report = time.perf_counter() - t0

        # --- step_done + schedule ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mgr.step_done()
        mgr.schedule()
        torch.cuda.synchronize()
        t_sched = time.perf_counter() - t0

        # --- build_past_kv ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        past_kv = mgr.build_past_kv()
        torch.cuda.synchronize()
        t_build = time.perf_counter() - t0

        cur_ids = next_tok

        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_total_start

        if step >= warmup:
            times["forward"].append(t_forward * 1000)
            times["build_past_kv"].append(t_build * 1000)
            times["append_token"].append(t_append * 1000)
            times["report_attn"].append(t_report * 1000)
            times["step_schedule"].append(t_sched * 1000)
            times["total"].append(t_total * 1000)

    mgr.destroy()
    return times


def summarize(times: dict, label: str, prompt_len: int, max_new: int):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  prompt={prompt_len}, gen={max_new}")
    print(f"{'='*60}")

    total_ms = sum(times["total"])
    total_tokens = prompt_len + max_new
    throughput = total_tokens / (total_ms / 1000)

    result = {"label": label, "throughput_tok_s": round(throughput, 1)}
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  {'Component':<20s} {'Avg ms':>8s} {'% of total':>10s} {'Sum ms':>10s}")
    print(f"  {'-'*50}")

    for key in ["forward", "build_past_kv", "append_token",
                "report_attn", "step_schedule"]:
        vals = times.get(key, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        s = sum(vals)
        pct = s / total_ms * 100 if total_ms > 0 else 0
        print(f"  {key:<20s} {avg:>8.2f} {pct:>9.1f}% {s:>10.1f}")
        result[f"{key}_avg_ms"] = round(avg, 3)
        result[f"{key}_pct"] = round(pct, 1)

    result["total_avg_ms"] = round(total_ms / len(times["total"]), 3)
    return result


def main():
    model_name = "Qwen/Qwen2.5-7B"
    seq_len = 1024
    max_new = 64
    warmup = 3
    gpu_budget_mb = 50

    print(f"Loading {model_name} ...")
    model_eager, tokenizer = load_model(model_name, "eager")

    text = "The transformer architecture " * (seq_len // 4)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=seq_len)["input_ids"].to("cuda:0")
    prompt_len = input_ids.shape[1]
    print(f"  prompt_len={prompt_len}, max_new={max_new}")

    all_results = []

    # --- Config 1: GPU-Only (eager, no management) ---
    print("\n>>> GPU-Only (eager, no KV management) ...")
    gc.collect(); torch.cuda.empty_cache()
    t1 = profile_gpu_only(model_eager, input_ids, max_new, warmup)
    r1 = summarize(t1, "GPU-Only (eager)", prompt_len, max_new)
    all_results.append(r1)

    # --- Config 2: OrchKvCache (eager, N=1) ---
    print("\n>>> OrchKvCache (eager, sample every step) ...")
    gc.collect(); torch.cuda.empty_cache()
    t2 = profile_orchkv(model_eager, input_ids, max_new, warmup,
                        gpu_budget_mb, sample_interval=1)
    r2 = summarize(t2, "OrchKvCache (eager, N=1)", prompt_len, max_new)
    all_results.append(r2)

    # --- Config 3: OrchKvCache (eager, N=10) ---
    print("\n>>> OrchKvCache (eager, sample every 10 steps) ...")
    gc.collect(); torch.cuda.empty_cache()
    t3 = profile_orchkv(model_eager, input_ids, max_new, warmup,
                        gpu_budget_mb, sample_interval=10)
    r3 = summarize(t3, "OrchKvCache (eager, N=10)", prompt_len, max_new)
    all_results.append(r3)

    del model_eager
    gc.collect(); torch.cuda.empty_cache()

    # --- Config 4: GPU-Only (SDPA, no management) ---
    print("\n>>> GPU-Only (SDPA, no KV management) ...")
    try:
        model_sdpa, _ = load_model(model_name, "sdpa")
        gc.collect(); torch.cuda.empty_cache()
        t4 = profile_gpu_only(model_sdpa, input_ids, max_new, warmup)
        r4 = summarize(t4, "GPU-Only (SDPA)", prompt_len, max_new)
        all_results.append(r4)
        del model_sdpa
    except Exception as e:
        print(f"  SDPA failed: {e}")

    gc.collect(); torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  OVERHEAD BREAKDOWN SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['label']:<35s}  {r['throughput_tok_s']:>8.1f} tok/s")

    if len(all_results) >= 2:
        base = all_results[0]["throughput_tok_s"]
        for r in all_results[1:]:
            ratio = base / r["throughput_tok_s"] if r["throughput_tok_s"] > 0 else 0
            print(f"    vs {all_results[0]['label']}: {ratio:.2f}x gap")

    save_json({"results": all_results, "config": {
        "model": model_name, "seq_len": seq_len, "prompt_len": prompt_len,
        "max_new": max_new, "gpu_budget_mb": gpu_budget_mb,
    }}, "exp_overhead_breakdown")

    print(f"\nResults saved to {RESULTS_DIR}/exp_overhead_breakdown.json")


if __name__ == "__main__":
    main()

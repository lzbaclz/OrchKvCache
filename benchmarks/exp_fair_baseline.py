#!/usr/bin/env python3
"""
Fair Baseline Experiment (Fix 1 + Fix 2 + Fix 4)

Runs all configurations on the SAME framework to isolate pure scheduling benefit:
  1. GPU-Only (SDPA)         — ceiling reference
  2. GPU-Only (eager)        — eager ceiling
  3. Fast-FIFO (eager, N=10) — fair FIFO baseline (same buffer arch, no attn scoring)
  4. Fast-OrchKv (eager, N=10) — OrchKvCache with optimized buffers
  5. Oracle-FIFO (trace sim) — trace-level oracle upper bound

Key metric: Fast-OrchKv / Fast-FIFO = pure policy speedup (no shared overhead bias)

Tests on Qwen2.5-7B (GQA-4) and LLaMA-2-7B (MHA-32) to cover both architectures.
"""
from __future__ import annotations
import gc, os, sys, time, random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None


def load_model(name, attn_impl="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation=attn_impl)
    mdl.eval()
    return mdl, tok


def bench_gpu_only(model, input_ids, max_new, warmup):
    cur, past = input_ids.clone(), None
    for _ in range(warmup):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    tokens_out = []
    cur, past = input_ids.clone(), None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_out.append(cur.item())
    torch.cuda.synchronize()
    thr = (input_ids.shape[1] + max_new) / (time.perf_counter() - t0)
    return thr, tokens_out


def bench_managed(model, input_ids, max_new, warmup, mgr_cls, mgr_kwargs,
                  sample_interval=10, report_attn=True):
    """Benchmark any managed KV system (FastOrchKv or FastFIFO)."""
    cfg = model.config
    base = dict(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        max_seq_len=input_ids.shape[1] + max_new + 256,
    )
    base.update(mgr_kwargs)

    mgr_warm = mgr_cls(**base)
    cur, past = input_ids.clone(), None
    for s in range(warmup):
        wa = report_attn and sample_interval > 0 and s % sample_interval == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=wa)
        if s == 0: mgr_warm.ingest_step(out.past_key_values)
        else: mgr_warm.append_token(out.past_key_values)
        if wa and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr_warm.report_attention(li, a)
        mgr_warm.step_done(); mgr_warm.schedule()
        past = mgr_warm.build_past_kv()
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    mgr_warm.destroy()

    mgr = mgr_cls(**base)
    tokens_out = []
    eviction_total = 0
    cur, past = input_ids.clone(), None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(max_new):
        wa = report_attn and sample_interval > 0 and s % sample_interval == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=wa)
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_out.append(nt.item())
        if s == 0: mgr.ingest_step(out.past_key_values)
        else: mgr.append_token(out.past_key_values)
        if wa and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr.report_attention(li, a)
        mgr.step_done(); mgr.schedule()
        past = mgr.build_past_kv()
        cur = nt
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    thr = (input_ids.shape[1] + max_new) / elapsed
    stats = mgr.get_stats()
    eviction_total = stats.get("migrations", {}).get("gpu_to_dram", 0)
    mgr.destroy()
    return thr, tokens_out, eviction_total


def run_oracle_trace_sim(n_blocks=64, n_steps=200, seed=42):
    """Fix 4: Oracle upper bound via trace simulation."""
    if _C is None:
        return {"oracle_migrations": 0, "ema_migrations": 0, "fifo_migrations": 0}

    from exp_attn_sampling import generate_attention_trace

    trace = generate_attention_trace(n_blocks, n_steps, seed=seed)
    hot_thresh = 0.50

    fifo_evictions = 0
    ema_evictions = 0
    oracle_evictions = 0

    for N_label, sample_N in [("fifo", 0), ("ema", 1), ("oracle", 1)]:
        params = dict(
            tracker_cap=n_blocks * 2, max_blocks=n_blocks + 64,
            alpha=0.5, beta=0.3, gamma=0.2,
            prefetch_budget=16, schedule_interval_us=500,
            gpu_hwm=0.80, gpu_lwm=0.60,
            dram_hwm=0.80, dram_lwm=0.60,
            threshold_to_gpu=0.4, threshold_to_dram=0.15,
        )
        tm = _C.tm_create(**params)
        for bid in range(n_blocks):
            _C.tm_register_block_id(tm, bid, int(_C.GPU_HBM), 0)

        for step, step_data in enumerate(trace):
            if N_label == "oracle":
                future_hot = set()
                for fs in range(step, min(step + 10, n_steps)):
                    for bid, w in trace[fs]:
                        if w >= hot_thresh:
                            future_hot.add(bid)
                for bid, w in step_data:
                    if bid in future_hot:
                        _C.tm_report_attn(tm, bid, max(w, 0.8))
                    else:
                        _C.tm_report_attn(tm, bid, 0.0)
            elif N_label == "ema":
                for bid, w in step_data:
                    if w >= 0.05:
                        _C.tm_report_attn(tm, bid, w)

            _C.tm_step_done(tm)
            if step % 5 == 0:
                _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.55)
                _C.tm_schedule_once(tm)

        s = _C.tm_get_stats(tm)
        mig = s["gpu_demotes"] + s.get("dram_demotes", 0)
        if N_label == "fifo":
            fifo_evictions = mig
        elif N_label == "ema":
            ema_evictions = mig
        else:
            oracle_evictions = mig
        _C.tm_destroy(tm)

    return {
        "fifo_migrations": fifo_evictions,
        "ema_migrations": ema_evictions,
        "oracle_migrations": oracle_evictions,
        "ema_vs_fifo_reduction": round(fifo_evictions / max(ema_evictions, 1), 1),
        "oracle_vs_fifo_reduction": round(fifo_evictions / max(oracle_evictions, 1), 1),
        "ema_optimality": round(oracle_evictions / max(ema_evictions, 1) * 100 if ema_evictions > 0 else 0, 1),
    }


def main():
    models = [
        ("Qwen/Qwen2.5-7B", 1024, 50),
        ("meta-llama/Llama-2-7b-hf", 1024, 50),
    ]
    max_new = 64
    warmup = 3
    all_results = []

    from orchkv.fast_kvcache_manager import FastKVCacheManager
    from orchkv.fast_fifo_manager import FastFIFOManager

    for model_name, seq_len, budget_mb in models:
        short = model_name.split("/")[-1]
        print(f"\n{'='*65}")
        print(f"  Model: {short}  seq={seq_len}  budget={budget_mb}MB  gen={max_new}")
        print(f"{'='*65}")

        model_eager, tokenizer = load_model(model_name, "eager")
        text = "The transformer " * (seq_len // 2)
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=seq_len)["input_ids"].to("cuda:0")
        pl = ids.shape[1]

        row = {"model": short, "prompt_len": pl, "max_new": max_new,
               "budget_mb": budget_mb}

        # 1. GPU-Only (eager)
        gc.collect(); torch.cuda.empty_cache()
        t1, ref_tok = bench_gpu_only(model_eager, ids, max_new, warmup)
        row["gpu_only_eager"] = round(t1, 1)
        print(f"  GPU-Only (eager):       {t1:>8.1f} tok/s")

        # 2. Fast-FIFO (eager, N=10) — no attention scoring
        gc.collect(); torch.cuda.empty_cache()
        t2, tok2, ev2 = bench_managed(
            model_eager, ids, max_new, warmup,
            FastFIFOManager, {"gpu_budget_bytes": budget_mb * (1 << 20)},
            sample_interval=10, report_attn=False)
        m2 = sum(a == b for a, b in zip(ref_tok, tok2)) / len(ref_tok)
        row["fast_fifo"] = round(t2, 1)
        row["fast_fifo_match"] = round(m2, 4)
        row["fast_fifo_evictions"] = ev2
        print(f"  Fast-FIFO (eager):      {t2:>8.1f} tok/s  match={m2:.2%}  evict={ev2}")

        # 3. Fast-OrchKvCache (eager, N=10)
        gc.collect(); torch.cuda.empty_cache()
        t3, tok3, ev3 = bench_managed(
            model_eager, ids, max_new, warmup,
            FastKVCacheManager, {"gpu_budget_bytes": budget_mb * (1 << 20)},
            sample_interval=10, report_attn=True)
        m3 = sum(a == b for a, b in zip(ref_tok, tok3)) / len(ref_tok)
        row["fast_orchkv"] = round(t3, 1)
        row["fast_orchkv_match"] = round(m3, 4)
        row["fast_orchkv_evictions"] = ev3
        print(f"  Fast-OrchKv (eager):    {t3:>8.1f} tok/s  match={m3:.2%}  evict={ev3}")

        policy_speedup = t3 / t2 if t2 > 0 else 0
        row["policy_speedup"] = round(policy_speedup, 3)
        evict_reduction = ev2 / max(ev3, 1) if ev3 > 0 else float("inf")
        row["eviction_reduction"] = round(evict_reduction, 1)
        print(f"  >>> Policy speedup: {policy_speedup:.3f}x  Eviction reduction: {evict_reduction:.1f}x")

        del model_eager; gc.collect(); torch.cuda.empty_cache()

        # 4. GPU-Only (SDPA)
        model_sdpa, _ = load_model(model_name, "sdpa")
        gc.collect(); torch.cuda.empty_cache()
        t4, _ = bench_gpu_only(model_sdpa, ids, max_new, warmup)
        row["gpu_only_sdpa"] = round(t4, 1)
        print(f"  GPU-Only (SDPA):        {t4:>8.1f} tok/s  (ceiling)")

        overhead_fifo = round((1 - t2 / t4) * 100, 1)
        overhead_orchkv = round((1 - t3 / t4) * 100, 1)
        row["overhead_fifo_vs_sdpa"] = overhead_fifo
        row["overhead_orchkv_vs_sdpa"] = overhead_orchkv
        print(f"  Overhead vs SDPA:  FIFO={overhead_fifo}%  OrchKv={overhead_orchkv}%")

        del model_sdpa; gc.collect(); torch.cuda.empty_cache()
        all_results.append(row)

    # Fix 4: Oracle trace simulation
    print(f"\n{'='*65}")
    print(f"  Oracle Upper Bound (trace simulation)")
    print(f"{'='*65}")
    oracle = run_oracle_trace_sim(n_blocks=64, n_steps=200)
    print(f"  FIFO migrations:   {oracle['fifo_migrations']}")
    print(f"  EMA migrations:    {oracle['ema_migrations']}  ({oracle['ema_vs_fifo_reduction']}x reduction)")
    print(f"  Oracle migrations: {oracle['oracle_migrations']}  ({oracle['oracle_vs_fifo_reduction']}x reduction)")
    print(f"  EMA optimality:    {oracle['ema_optimality']}% of oracle")

    # Summary
    print(f"\n{'='*65}")
    print(f"  FAIR BASELINE SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<20s} {'SDPA':>7s} {'Eager':>7s} {'F-FIFO':>7s} {'F-Orch':>7s} {'Policy':>7s} {'Evict':>6s}")
    print(f"  {'':20s} {'ceil':>7s} {'ceil':>7s} {'':>7s} {'':>7s} {'spdup':>7s} {'reduc':>6s}")
    for r in all_results:
        m = r["model"][:18]
        print(f"  {m:<20s} {r['gpu_only_sdpa']:>7.0f} {r['gpu_only_eager']:>7.0f} "
              f"{r['fast_fifo']:>7.0f} {r['fast_orchkv']:>7.0f} "
              f"{r['policy_speedup']:>6.3f}x {r['eviction_reduction']:>5.1f}x")

    save_json({"results": all_results, "oracle": oracle}, "exp_fair_baseline")
    print(f"\nSaved to {RESULTS_DIR}/exp_fair_baseline.json")


if __name__ == "__main__":
    main()

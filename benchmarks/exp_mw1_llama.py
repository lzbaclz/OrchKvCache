#!/usr/bin/env python3
"""
MW1 Experiment on LLaMA-2-7B: validate optimization on MHA architecture.
Also profiles Fast version overhead breakdown.
"""
from __future__ import annotations
import gc, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bench_utils import save_json, RESULTS_DIR


def load_model(name, attn_impl="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation=attn_impl)
    mdl.eval()
    return mdl, tok


def profile_fast_orchkv(model, input_ids, max_new, warmup, budget_mb, sample_N):
    from orchkv.fast_kvcache_manager import FastKVCacheManager
    cfg = model.config
    mgr = FastKVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget_mb * (1 << 20),
        max_seq_len=input_ids.shape[1] + max_new + 256,
    )
    times = {"forward": [], "build_past_kv": [], "append_token": [],
             "report_attn": [], "step_schedule": [], "total": []}
    cur, past = input_ids.clone(), None

    for s in range(warmup + max_new):
        torch.cuda.synchronize(); tt0 = time.perf_counter()
        want_attn = sample_N > 0 and s % sample_N == 0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=want_attn)
        torch.cuda.synchronize(); t_fwd = time.perf_counter() - t0

        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        if s == 0: mgr.ingest_step(out.past_key_values)
        else: mgr.append_token(out.past_key_values)
        torch.cuda.synchronize(); t_app = time.perf_counter() - t0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        if want_attn and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr.report_attention(li, a)
        torch.cuda.synchronize(); t_rpt = time.perf_counter() - t0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        mgr.step_done(); mgr.schedule()
        torch.cuda.synchronize(); t_sch = time.perf_counter() - t0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        past = mgr.build_past_kv()
        torch.cuda.synchronize(); t_bld = time.perf_counter() - t0

        cur = nt
        torch.cuda.synchronize(); t_tot = time.perf_counter() - tt0

        if s >= warmup:
            for k, v in [("forward", t_fwd), ("build_past_kv", t_bld),
                         ("append_token", t_app), ("report_attn", t_rpt),
                         ("step_schedule", t_sch), ("total", t_tot)]:
                times[k].append(v * 1000)

    mgr.destroy()
    return times


def run_gpu_only(model, input_ids, max_new, warmup):
    cur, past = input_ids.clone(), None
    for _ in range(warmup):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    cur, past = input_ids.clone(), None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return (input_ids.shape[1] + max_new) / (time.perf_counter() - t0)


def run_original(model, input_ids, max_new, warmup, budget_mb, sample_N):
    from orchkv.kvcache_manager import KVCacheManager
    cfg = model.config
    mgr = KVCacheManager(
        n_layers=cfg.num_hidden_layers, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget_mb * (1 << 20))

    cur, past = input_ids.clone(), None
    for s in range(warmup):
        wa = sample_N > 0 and s % sample_N == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True, output_attentions=wa)
        if s == 0: mgr.ingest_step(out.past_key_values)
        else: mgr.append_token(out.past_key_values)
        if wa and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions): mgr.report_attention(li, a)
        mgr.step_done(); mgr.schedule(); past = mgr.build_past_kv()
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    mgr2 = KVCacheManager(
        n_layers=cfg.num_hidden_layers, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget_mb * (1 << 20))
    cur, past = input_ids.clone(), None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(max_new):
        wa = sample_N > 0 and s % sample_N == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True, output_attentions=wa)
        if s == 0: mgr2.ingest_step(out.past_key_values)
        else: mgr2.append_token(out.past_key_values)
        if wa and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions): mgr2.report_attention(li, a)
        mgr2.step_done(); mgr2.schedule(); past = mgr2.build_past_kv()
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    thr = (input_ids.shape[1] + max_new) / (time.perf_counter() - t0)
    mgr.destroy(); mgr2.destroy()
    return thr


def main():
    models = [
        ("Qwen/Qwen2.5-7B", 1024, 50),
        ("meta-llama/Llama-2-7b-hf", 1024, 50),
    ]
    max_new, warmup = 64, 3
    all_results = []

    for model_name, seq_len, budget_mb in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_name, "eager")
        text = "The transformer " * (seq_len // 2)
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=seq_len)["input_ids"].to("cuda:0")
        pl = ids.shape[1]
        print(f"  prompt={pl}, gen={max_new}, budget={budget_mb}MB")

        # GPU-Only
        gc.collect(); torch.cuda.empty_cache()
        thr_go = run_gpu_only(model, ids, max_new, warmup)
        print(f"  GPU-Only (eager):        {thr_go:.1f} tok/s")

        # Original OrchKv
        gc.collect(); torch.cuda.empty_cache()
        thr_orig = run_original(model, ids, max_new, warmup, budget_mb, 10)
        print(f"  Original OrchKv (N=10):  {thr_orig:.1f} tok/s")

        # Fast OrchKv + profile
        gc.collect(); torch.cuda.empty_cache()
        times = profile_fast_orchkv(model, ids, max_new, warmup, budget_mb, 10)
        total_ms = sum(times["total"])
        thr_fast = (pl + max_new) / (total_ms / 1000)
        print(f"  Fast OrchKv (N=10):      {thr_fast:.1f} tok/s")
        print(f"  Speedup: {thr_fast/thr_orig:.2f}x")

        print(f"\n  Fast OrchKv Breakdown:")
        for comp in ["forward", "build_past_kv", "append_token",
                     "report_attn", "step_schedule"]:
            vals = times.get(comp, [])
            if vals:
                avg = sum(vals) / len(vals)
                pct = sum(vals) / total_ms * 100
                print(f"    {comp:<20s} {avg:>8.2f} ms  ({pct:>5.1f}%)")

        all_results.append({
            "model": model_name, "prompt_len": pl,
            "gpu_only": round(thr_go, 1),
            "original_orchkv": round(thr_orig, 1),
            "fast_orchkv": round(thr_fast, 1),
            "speedup": round(thr_fast / thr_orig, 2),
            "overhead_original": round((1 - thr_orig / thr_go) * 100, 1),
            "overhead_fast": round((1 - thr_fast / thr_go) * 100, 1),
            "breakdown": {k: round(sum(v)/len(v), 2) for k, v in times.items() if v},
        })

        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<25s} {'GPU-Only':>9s} {'Original':>9s} {'Fast':>9s} {'Speedup':>8s} {'OH Orig':>8s} {'OH Fast':>8s}")
    for r in all_results:
        mn = r["model"].split("/")[-1][:20]
        print(f"  {mn:<25s} {r['gpu_only']:>8.1f} {r['original_orchkv']:>9.1f} "
              f"{r['fast_orchkv']:>8.1f} {r['speedup']:>7.2f}x "
              f"{r['overhead_original']:>6.1f}% {r['overhead_fast']:>6.1f}%")

    save_json(all_results, "exp_mw1_llama")
    print(f"\nSaved to {RESULTS_DIR}/exp_mw1_llama.json")


if __name__ == "__main__":
    main()

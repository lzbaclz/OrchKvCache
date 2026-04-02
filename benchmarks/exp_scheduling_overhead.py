#!/usr/bin/env python3
"""
P1a: Scheduling Loop Overhead Decomposition

Precisely measures the time breakdown of each component in the
FastOrchKv scheduling loop to demonstrate that:
  1. C-side tm_schedule_once() takes <40μs
  2. Python-level attention reporting + eviction iteration adds 10-30ms
  3. The scheduling ALGORITHM is efficient; the IMPLEMENTATION overhead
     is the bottleneck (removable by porting to C/CUDA)

Also runs a "C-scheduling-only" variant that calls tm_* without
Python eviction overhead to show the minimum possible overhead.
"""
from __future__ import annotations
import gc, os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None


def measure_c_scheduling_overhead(n_blocks=64, n_steps=200):
    """Measure pure C-side scheduling overhead (no Python eviction)."""
    if _C is None:
        return {}

    params = dict(
        tracker_cap=n_blocks * 2, max_blocks=n_blocks + 64,
        alpha=0.7, beta=0.2, gamma=0.1,
        prefetch_budget=8, schedule_interval_us=500,
        gpu_hwm=0.80, gpu_lwm=0.60,
        dram_hwm=0.80, dram_lwm=0.60,
        threshold_to_gpu=0.5, threshold_to_dram=0.15,
    )
    tm = _C.tm_create(**params)
    for bid in range(n_blocks):
        _C.tm_register_block_id(tm, bid, 0, 0)

    report_times = []
    step_done_times = []
    schedule_times = []

    import random
    random.seed(42)

    for step in range(n_steps):
        # Simulate attention reporting (per-block)
        t0 = time.perf_counter()
        for bid in range(n_blocks):
            score = random.random() * (1.0 / (bid + 1))
            _C.tm_report_attn(tm, bid, score)
        t1 = time.perf_counter()
        report_times.append((t1 - t0) * 1e6)

        # step_done
        t2 = time.perf_counter()
        _C.tm_step_done(tm)
        t3 = time.perf_counter()
        step_done_times.append((t3 - t2) * 1e6)

        # schedule_once
        t4 = time.perf_counter()
        _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.5)
        _C.tm_schedule_once(tm)
        t5 = time.perf_counter()
        schedule_times.append((t5 - t4) * 1e6)

    _C.tm_destroy(tm)

    return {
        "n_blocks": n_blocks,
        "n_steps": n_steps,
        "report_attn_us": {
            "mean": round(np.mean(report_times), 1),
            "p50": round(np.median(report_times), 1),
            "p99": round(np.percentile(report_times, 99), 1),
        },
        "step_done_us": {
            "mean": round(np.mean(step_done_times), 1),
            "p50": round(np.median(step_done_times), 1),
            "p99": round(np.percentile(step_done_times, 99), 1),
        },
        "schedule_once_us": {
            "mean": round(np.mean(schedule_times), 1),
            "p50": round(np.median(schedule_times), 1),
            "p99": round(np.percentile(schedule_times, 99), 1),
        },
        "total_c_per_step_us": {
            "mean": round(np.mean(report_times) + np.mean(step_done_times) + np.mean(schedule_times), 1),
        },
    }


def measure_python_overhead_e2e(model_name, seq_len, budget_mb, max_new=64):
    """Measure per-step timing breakdown in actual E2E inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from orchkv.fast_kvcache_manager import FastKVCacheManager

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    mdl.eval()

    text = "The transformer " * (seq_len // 2)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=seq_len)["input_ids"].to("cuda:0")

    cfg = mdl.config
    mgr = FastKVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget_mb * (1 << 20),
        max_seq_len=ids.shape[1] + max_new + 256,
    )

    t_forward = []
    t_report = []
    t_step_done = []
    t_schedule = []
    t_build = []

    cur, past = ids.clone(), None
    sample_interval = 10

    for s in range(max_new):
        wa = sample_interval > 0 and s % sample_interval == 0

        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        with torch.no_grad():
            out = mdl(cur, past_key_values=past, use_cache=True,
                      output_attentions=wa)
        torch.cuda.synchronize()
        tf1 = time.perf_counter()
        t_forward.append((tf1 - tf0) * 1000)

        if s == 0:
            mgr.ingest_step(out.past_key_values)
        else:
            mgr.append_token(out.past_key_values)

        tr0 = time.perf_counter()
        if wa and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr.report_attention(li, a)
        tr1 = time.perf_counter()
        t_report.append((tr1 - tr0) * 1000)

        ts0 = time.perf_counter()
        mgr.step_done()
        ts1 = time.perf_counter()
        t_step_done.append((ts1 - ts0) * 1000)

        tsc0 = time.perf_counter()
        mgr.schedule()
        tsc1 = time.perf_counter()
        t_schedule.append((tsc1 - tsc0) * 1000)

        tb0 = time.perf_counter()
        past = mgr.build_past_kv()
        tb1 = time.perf_counter()
        t_build.append((tb1 - tb0) * 1000)

        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    mgr.destroy()
    del mdl; gc.collect(); torch.cuda.empty_cache()

    return {
        "model": model_name.split("/")[-1],
        "seq_len": seq_len,
        "budget_mb": budget_mb,
        "max_new": max_new,
        "forward_ms":    {"mean": round(np.mean(t_forward), 2), "p50": round(np.median(t_forward), 2)},
        "report_ms":     {"mean": round(np.mean(t_report), 2), "p50": round(np.median(t_report), 2)},
        "step_done_ms":  {"mean": round(np.mean(t_step_done), 3), "p50": round(np.median(t_step_done), 3)},
        "schedule_ms":   {"mean": round(np.mean(t_schedule), 2), "p50": round(np.median(t_schedule), 2)},
        "build_kv_ms":   {"mean": round(np.mean(t_build), 2), "p50": round(np.median(t_build), 2)},
        "total_ms":      {"mean": round(np.mean(t_forward) + np.mean(t_report) + np.mean(t_step_done) + np.mean(t_schedule) + np.mean(t_build), 2)},
        "python_sched_overhead_ms": {"mean": round(np.mean(t_report) + np.mean(t_step_done) + np.mean(t_schedule), 2)},
    }


def main():
    results = {}

    print("=" * 65)
    print("  Part 1: Pure C-side scheduling overhead")
    print("=" * 65)
    for nb in [64, 128, 256, 512]:
        r = measure_c_scheduling_overhead(n_blocks=nb, n_steps=200)
        results[f"c_overhead_{nb}blocks"] = r
        total = r["total_c_per_step_us"]["mean"]
        sched = r["schedule_once_us"]["mean"]
        report = r["report_attn_us"]["mean"]
        print(f"  n_blocks={nb:>4d}:  total={total:>7.1f}μs  "
              f"(report={report:.1f}μs  schedule={sched:.1f}μs)")

    print(f"\n{'=' * 65}")
    print("  Part 2: E2E Python overhead breakdown")
    print("=" * 65)
    for model_name, seq_len, budget_mb in [
        ("Qwen/Qwen2.5-7B", 1024, 50),
        ("meta-llama/Llama-2-7b-hf", 1024, 50),
    ]:
        short = model_name.split("/")[-1]
        print(f"\n  {short} (seq={seq_len}, budget={budget_mb}MB):")
        r = measure_python_overhead_e2e(model_name, seq_len, budget_mb)
        results[f"e2e_{short}"] = r
        print(f"    forward:        {r['forward_ms']['mean']:>7.2f} ms/step")
        print(f"    report_attn:    {r['report_ms']['mean']:>7.2f} ms/step")
        print(f"    step_done:      {r['step_done_ms']['mean']:>7.3f} ms/step")
        print(f"    schedule:       {r['schedule_ms']['mean']:>7.2f} ms/step")
        print(f"    build_past_kv:  {r['build_kv_ms']['mean']:>7.2f} ms/step")
        print(f"    ---")
        total = r['total_ms']['mean']
        sched_oh = r['python_sched_overhead_ms']['mean']
        print(f"    total:          {total:>7.2f} ms/step")
        print(f"    Python sched:   {sched_oh:>7.2f} ms/step ({sched_oh/total*100:.1f}%)")

    save_json(results, "exp_scheduling_overhead")
    print(f"\nSaved to {RESULTS_DIR}/exp_scheduling_overhead.json")


if __name__ == "__main__":
    main()

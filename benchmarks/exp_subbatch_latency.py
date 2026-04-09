#!/usr/bin/env python3
"""
Sub-batch rotation with per-request latency and fairness metrics.

Measures P50/P95/P99 per-request completion latency and Jain's fairness
index across sub-batch sizes K=1,2,4,N.
"""
from __future__ import annotations

import gc
import json
import math
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
    print(f" done")
    return mdl, tok, mc


def jains_fairness(latencies):
    n = len(latencies)
    if n == 0:
        return 1.0
    s = sum(latencies)
    s2 = sum(x * x for x in latencies)
    return (s * s) / (n * s2) if s2 > 0 else 1.0


def run_subbatch_latency(model, tokenizer, mc, n_req, seq_len, max_new,
                         total_budget_mb, sub_k):
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

    # Track per-request per-step latency
    # For each request, measure time from "my turn starts" to "my token produced"
    per_req_step_latencies = [[] for _ in range(n_req)]
    n_subbatches = (n_req + sub_k - 1) // sub_k

    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    for step in range(max_new):
        for sb in range(n_subbatches):
            start_ri = sb * sub_k
            end_ri = min(start_ri + sub_k, n_req)

            for ri in range(start_ri, end_ri):
                torch.cuda.synchronize()
                t_req_start = time.perf_counter()

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
                t_req_end = time.perf_counter()
                per_req_step_latencies[ri].append(
                    (t_req_end - t_req_start) * 1000)  # ms

    torch.cuda.synchronize()
    t_total_end = time.perf_counter()
    total_elapsed = t_total_end - t_total_start

    total_tokens = sum(ids.shape[1] for ids in all_ids) + max_new * n_req
    throughput = total_tokens / total_elapsed

    # Per-request completion time (sum of all step latencies for that request)
    req_completion_times = []
    for ri in range(n_req):
        req_completion_times.append(sum(per_req_step_latencies[ri]))

    # Per-step latency across all requests (flatten)
    all_step_latencies = []
    for ri in range(n_req):
        all_step_latencies.extend(per_req_step_latencies[ri])
    all_step_latencies.sort()

    n = len(all_step_latencies)
    total_evictions = 0
    for mgr in managers:
        s = mgr.get_stats()
        total_evictions += s["migrations"]["gpu_to_dram"]
        mgr.destroy()

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sub_k": sub_k,
        "throughput": round(throughput, 1),
        "evictions": total_evictions,
        "step_latency_p50_ms": round(all_step_latencies[n // 2], 2),
        "step_latency_p95_ms": round(all_step_latencies[int(n * 0.95)], 2),
        "step_latency_p99_ms": round(all_step_latencies[int(n * 0.99)], 2),
        "req_completion_mean_ms": round(
            sum(req_completion_times) / len(req_completion_times), 1),
        "req_completion_max_ms": round(max(req_completion_times), 1),
        "req_completion_min_ms": round(min(req_completion_times), 1),
        "jains_fairness": round(jains_fairness(req_completion_times), 4),
    }


def main():
    model_name = "Qwen/Qwen2.5-7B"
    seq_len = 1024
    max_new = 64
    total_budget_mb = 50
    n_req = 8

    print(f"{'='*70}")
    print(f"  Sub-Batch Latency & Fairness Experiment")
    print(f"  {model_name}, {n_req} req, seq={seq_len}, budget={total_budget_mb}MB")
    print(f"{'='*70}")

    model, tokenizer, mc = load_model(model_name)

    # Warmup
    print("  [warmup]...", flush=True)
    _ = run_subbatch_latency(model, tokenizer, mc, 2, seq_len, max_new,
                             total_budget_mb, 1)
    gc.collect(); torch.cuda.empty_cache()

    sub_ks = [1, 2, 4, n_req]
    results = []

    for k in sub_ks:
        label = "round-robin" if k == 1 else f"K={k}" if k < n_req else f"full-batch"
        print(f"\n  --- {label} ---")

        r = run_subbatch_latency(model, tokenizer, mc, n_req, seq_len,
                                 max_new, total_budget_mb, k)
        r["label"] = label
        results.append(r)

        print(f"    tok/s={r['throughput']}, evict={r['evictions']}")
        print(f"    step latency: P50={r['step_latency_p50_ms']:.1f}ms "
              f"P95={r['step_latency_p95_ms']:.1f}ms "
              f"P99={r['step_latency_p99_ms']:.1f}ms")
        print(f"    req completion: mean={r['req_completion_mean_ms']:.0f}ms "
              f"max={r['req_completion_max_ms']:.0f}ms "
              f"fairness={r['jains_fairness']:.4f}")

    baseline_thr = results[-1]["throughput"]
    print(f"\n{'='*75}")
    print(f"  SUMMARY")
    print(f"{'='*75}")
    print(f"  {'K':>3s} {'Label':<12s} {'tok/s':>7s} {'Speed':>6s} "
          f"{'P50ms':>7s} {'P99ms':>7s} {'CompMs':>7s} {'Fair':>6s}")
    print(f"  {'-'*60}")
    for r in results:
        sp = r["throughput"] / baseline_thr if baseline_thr > 0 else 0
        print(f"  {r['sub_k']:>3d} {r['label']:<12s} "
              f"{r['throughput']:>7.1f} {sp:>5.2f}× "
              f"{r['step_latency_p50_ms']:>7.1f} "
              f"{r['step_latency_p99_ms']:>7.1f} "
              f"{r['req_completion_mean_ms']:>7.0f} "
              f"{r['jains_fairness']:>6.4f}")

    out = RESULTS_DIR / "exp_subbatch_latency.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 4 Extended Experiments:
  1. Stronger baseline: OrchKvCache with no-attention (α=0, recency+freq only)
  2. Per-step latency recording for P95/P99

Runs on Qwen2.5-7B, budget=50MB, seq=2048, nreq=4, gen=64.
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B"
SHORT_NAME = "Qwen2.5-7B"
BUDGET_MB = 50
SEQ_LEN = 2048
N_REQ = 4
MAX_NEW = 64
ATTN_EVERY = 10


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  {SHORT_NAME}: {n_layers}L, {n_kv_heads}KV, d={head_dim}")
    return model, tokenizer, {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
    }


def generate_prompt(tokenizer, seq_len):
    text = "Artificial intelligence is transforming every aspect of modern life. " * (seq_len // 10 + 1)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"]
    return ids.to("cuda:0")


def run_decode_with_latency(model, input_ids, max_new, manager=None,
                            attn_every=10, skip_attn_report=False):
    """Decode with per-step latency recording."""
    generated = []
    step_latencies_ms = []
    cur_ids = input_ids.clone()
    past_kv = None
    t0 = time.perf_counter()
    t_first = None

    for step in range(max_new):
        t_step_start = time.perf_counter()

        want_attn = (manager is not None
                     and isinstance(manager, KVCacheManager)
                     and not skip_attn_report
                     and step % attn_every == 0)

        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv, use_cache=True,
                        output_attentions=want_attn)

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        if t_first is None:
            t_first = time.perf_counter()

        if manager is not None:
            new_past = out.past_key_values
            if step == 0:
                manager.ingest_step(new_past)
            else:
                manager.append_token(new_past)
            if not skip_attn_report and hasattr(out, 'attentions') and out.attentions is not None:
                for li, attn in enumerate(out.attentions):
                    manager.report_attention(li, attn)
            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = out.past_key_values
        cur_ids = next_tok

        t_step_end = time.perf_counter()
        if step > 0:
            step_latencies_ms.append((t_step_end - t_step_start) * 1000)

    elapsed = time.perf_counter() - t0
    prompt_len = input_ids.shape[1]
    total_tok = prompt_len + len(generated)
    ttft = (t_first - t0) * 1000 if t_first else 0

    lat = np.array(step_latencies_ms) if step_latencies_ms else np.array([0.0])

    return {
        "generated": generated,
        "prompt_len": prompt_len,
        "gen_len": len(generated),
        "elapsed_s": round(elapsed, 3),
        "throughput": round(total_tok / elapsed, 1),
        "ttft_ms": round(ttft, 2),
        "tpot_mean_ms": round(float(lat.mean()), 2),
        "tpot_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "tpot_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "tpot_p99_ms": round(float(np.percentile(lat, 99)), 2),
        "tpot_max_ms": round(float(lat.max()), 2),
        "step_latencies_ms": [round(x, 2) for x in step_latencies_ms],
    }


def run_mode(model, tokenizer, model_cfg, mode, n_req, budget_mb, seq_len, max_new):
    per_req_budget = (budget_mb * (1 << 20)) // max(n_req, 1)
    req_results = []
    total_evict = 0

    for ri in range(n_req):
        input_ids = generate_prompt(tokenizer, seq_len)
        mgr = None
        skip_attn = False

        if mode == "orchkv":
            mgr = KVCacheManager(
                n_layers=model_cfg["n_layers"],
                n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"],
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=per_req_budget,
            )
        elif mode == "orchkv-noattn":
            mgr = KVCacheManager(
                n_layers=model_cfg["n_layers"],
                n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"],
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=per_req_budget,
            )
            skip_attn = True
        elif mode == "naive":
            mgr = NaiveOffloadManager(
                n_layers=model_cfg["n_layers"],
                n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"],
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=per_req_budget,
            )

        try:
            res = run_decode_with_latency(
                model, input_ids, max_new, manager=mgr,
                attn_every=ATTN_EVERY, skip_attn_report=skip_attn,
            )
            if mgr:
                stats = mgr.get_stats()
                mig = stats.get("migrations", {})
                evict = mig.get("gpu_to_dram", 0) + mig.get("dram_to_ssd", 0)
                total_evict += evict
                res["evictions"] = evict
            req_results.append(res)
            print(f"    req {ri}: {res['throughput']} tok/s, "
                  f"TPOT p50={res['tpot_p50_ms']:.1f} p95={res['tpot_p95_ms']:.1f} "
                  f"p99={res['tpot_p99_ms']:.1f}ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            req_results.append({"status": "OOM"})
            print(f"    req {ri}: OOM", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        finally:
            if mgr:
                mgr.destroy()
            gc.collect(); torch.cuda.empty_cache()

    ok = [r for r in req_results if "throughput" in r]
    oom = len(req_results) - len(ok)

    all_latencies = []
    for r in ok:
        all_latencies.extend(r.get("step_latencies_ms", []))
    lat = np.array(all_latencies) if all_latencies else np.array([0.0])

    return {
        "model": SHORT_NAME,
        "mode": mode,
        "n_requests": n_req,
        "seq_len": seq_len,
        "gpu_budget_mb": budget_mb,
        "completed": len(ok),
        "oom": oom,
        "avg_throughput": round(sum(r["throughput"] for r in ok) / max(len(ok), 1), 1),
        "avg_ttft_ms": round(sum(r["ttft_ms"] for r in ok) / max(len(ok), 1), 2),
        "tpot_mean_ms": round(float(lat.mean()), 2),
        "tpot_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "tpot_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "tpot_p99_ms": round(float(np.percentile(lat, 99)), 2),
        "tpot_max_ms": round(float(lat.max()), 2),
        "total_evictions": total_evict,
        "per_request": [{k: v for k, v in r.items() if k != "step_latencies_ms"}
                        for r in req_results],
    }


def main():
    model, tokenizer, model_cfg = load_model()

    modes = ["baseline", "naive", "orchkv", "orchkv-noattn"]
    results = []

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}  (budget={BUDGET_MB}MB, seq={SEQ_LEN}, nreq={N_REQ})")
        print(f"{'='*60}")

        row = run_mode(model, tokenizer, model_cfg, mode,
                       N_REQ, BUDGET_MB, SEQ_LEN, MAX_NEW)
        results.append(row)

        print(f"\n  Summary: {row['avg_throughput']} tok/s, "
              f"evictions={row['total_evictions']}, "
              f"TPOT mean={row['tpot_mean_ms']:.1f} "
              f"p95={row['tpot_p95_ms']:.1f} p99={row['tpot_p99_ms']:.1f}ms")

    out_path = RESULTS_DIR / "exp_phase4_extended.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*70}")
    print(f"{'Mode':<20} {'Thr(tok/s)':>10} {'Evictions':>10} "
          f"{'TPOT-p50':>10} {'TPOT-p95':>10} {'TPOT-p99':>10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['mode']:<20} {r['avg_throughput']:>10.1f} {r['total_evictions']:>10} "
              f"{r['tpot_p50_ms']:>10.1f} {r['tpot_p95_ms']:>10.1f} {r['tpot_p99_ms']:>10.1f}")


if __name__ == "__main__":
    main()

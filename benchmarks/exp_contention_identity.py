#!/usr/bin/env python3
"""
Multi-request contention identity metric.

1. Run N requests sequentially, collect per-step per-block attention traces
2. Offline: simulate shared GPU cache with limited capacity
3. At each step, pool all active blocks, rank by EMA vs actual attention
4. Compute precision@K, recall@K under contention
5. Compare Full EMA (α=0.7) vs No-attn (α=0.0, recency+freq)
"""
from __future__ import annotations
import gc, os, sys, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    print("ERROR: orchkv_core not found")
    sys.exit(1)

BLOCK_SIZE = 16


def collect_traces(model_name, n_requests=8, seq_len=1024, max_new=32):
    """Run N requests, collect per-step attention traces."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    short = model_name.split("/")[-1]
    print(f"\n  Collecting traces: {short}, {n_requests} requests, seq={seq_len}, gen={max_new}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    mdl.eval()

    prompts = [
        "The history of artificial intelligence began in ancient times. ",
        "Quantum computing is a rapidly emerging technology that harnesses ",
        "Climate change represents one of the most pressing challenges facing ",
        "The development of the internet has revolutionized how people ",
        "Modern medicine has made remarkable advances in treating diseases ",
        "Space exploration has always captivated human imagination since ",
        "The global economy is increasingly interconnected through trade and ",
        "Renewable energy sources like solar and wind power are becoming ",
        "Neural networks have transformed the field of machine learning ",
        "The philosophy of consciousness remains one of the deepest puzzles ",
        "Blockchain technology provides a decentralized approach to recording ",
        "The evolution of programming languages reflects changing computing needs ",
        "Biodiversity loss threatens ecosystem stability across the planet ",
        "Advances in robotics are transforming manufacturing and logistics ",
        "The psychology of decision making reveals systematic biases in human ",
        "Nanotechnology enables manipulation of matter at the molecular scale ",
    ]

    all_traces = []

    for ri in range(n_requests):
        prompt_text = prompts[ri % len(prompts)] * (seq_len // 10 + 1)
        ids = tok(prompt_text, return_tensors="pt", truncation=True,
                  max_length=seq_len)["input_ids"].to("cuda:0")
        actual_len = ids.shape[1]
        n_blocks = (actual_len + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE

        req_trace = {"request_id": ri, "n_blocks_init": n_blocks, "steps": []}
        cur, past = ids.clone(), None

        for s in range(max_new):
            with torch.no_grad():
                out = mdl(cur, past_key_values=past, use_cache=True,
                          output_attentions=True)
            past = out.past_key_values

            cur_len = actual_len + s + 1
            n_blk = (cur_len + BLOCK_SIZE - 1) // BLOCK_SIZE

            if out.attentions is not None:
                block_attn = np.zeros(n_blk)
                for li, attn_w in enumerate(out.attentions):
                    avg = attn_w.float().mean(dim=(0, 2)).squeeze(0).cpu().numpy()
                    for bid in range(n_blk):
                        start = bid * BLOCK_SIZE
                        end = min(start + BLOCK_SIZE, len(avg))
                        if start < len(avg):
                            block_attn[bid] += float(avg[start:end].sum())

                req_trace["steps"].append({
                    "step": s, "n_blocks": n_blk,
                    "block_attn": block_attn.tolist(),
                })

            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        all_traces.append(req_trace)
        print(f"    req {ri}: {len(req_trace['steps'])} steps, {n_blocks} blocks")
        gc.collect(); torch.cuda.empty_cache()

    del mdl; gc.collect(); torch.cuda.empty_cache()
    return all_traces, short


def simulate_contention(traces, gpu_capacity_frac=0.10):
    """Simulate shared GPU cache, compare EMA vs No-attn vs Random."""

    configs = {
        "Full_EMA":   {"alpha": 0.7, "beta": 0.2, "gamma": 0.1},
        "No_attn":    {"alpha": 0.0, "beta": 0.6, "gamma": 0.4},
        "Recency":    {"alpha": 0.0, "beta": 1.0, "gamma": 0.0},
    }

    n_requests = len(traces)
    n_steps = min(len(t["steps"]) for t in traces)

    total_blocks = sum(t["steps"][0]["n_blocks"] for t in traces)
    gpu_cap = max(2, int(total_blocks * gpu_capacity_frac))
    print(f"\n  Contention sim: {n_requests} req, {total_blocks} total blocks, "
          f"GPU cap={gpu_cap} ({gpu_capacity_frac:.0%})")

    results = {}

    for cfg_name, weights in configs.items():
        tm = _C.tm_create(
            tracker_cap=4096, max_blocks=total_blocks + 64,
            alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"],
            prefetch_budget=8, schedule_interval_us=500,
            threshold_to_gpu=0.5, threshold_to_dram=0.15,
        )

        block_offset = {}
        bid_global = 0
        for ri, t in enumerate(traces):
            block_offset[ri] = bid_global
            for b in range(t["steps"][0]["n_blocks"]):
                flags = 1 if b == 0 else 0
                _C.tm_register_block_id(tm, bid_global, 0, flags)
                bid_global += 1

        step_metrics = []

        for s in range(n_steps):
            all_attn = np.zeros(bid_global)

            for ri, t in enumerate(traces):
                step_data = t["steps"][s]
                off = block_offset[ri]
                for b, a in enumerate(step_data["block_attn"]):
                    gid = off + b
                    if gid < bid_global:
                        all_attn[gid] = a
                        if weights["alpha"] > 0:
                            _C.tm_report_attn(tm, gid, a)
                        else:
                            _C.tm_report_attn(tm, gid, 0.0)

            _C.tm_step_done(tm)
            _C.tm_schedule_once(tm)

            all_ema = np.zeros(bid_global)
            for gid in range(bid_global):
                all_ema[gid] = _C.tm_get_block_score(tm, gid)

            nonsink = np.array([gid for gid in range(bid_global)
                                if gid not in [block_offset[ri] for ri in range(n_requests)]])

            if len(nonsink) > gpu_cap:
                ns_attn = all_attn[nonsink]
                ns_ema = all_ema[nonsink]

                K = gpu_cap
                gt_topK = set(np.argsort(-ns_attn)[:K])
                pred_topK = set(np.argsort(-ns_ema)[:K])

                tp = len(gt_topK & pred_topK)
                precision = tp / K
                recall = tp / K

                step_metrics.append({
                    "step": s, "precision": precision, "recall": recall,
                    "K": K, "n_nonsink": len(nonsink),
                })

        _C.tm_destroy(tm)

        avg_p = np.mean([m["precision"] for m in step_metrics]) if step_metrics else 0
        avg_r = np.mean([m["recall"] for m in step_metrics]) if step_metrics else 0
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

        results[cfg_name] = {
            "avg_precision": round(float(avg_p), 4),
            "avg_recall": round(float(avg_r), 4),
            "avg_f1": round(float(f1), 4),
            "n_steps": len(step_metrics),
            "gpu_cap": gpu_cap,
        }
        print(f"    {cfg_name:12s}: prec={avg_p:.3f}  recall={avg_r:.3f}  F1={f1:.3f}")

    return results


def main():
    traces, model_short = collect_traces(
        "Qwen/Qwen2.5-7B", n_requests=8, seq_len=1024, max_new=32)

    all_results = {}
    for cap in [0.05, 0.10, 0.20, 0.30]:
        print(f"\n{'='*50}")
        print(f"  GPU capacity = {cap:.0%} of total blocks")
        r = simulate_contention(traces, gpu_capacity_frac=cap)
        all_results[f"cap_{int(cap*100)}pct"] = r

    out_path = RESULTS_DIR / "exp_contention_identity.json"
    save_json({"model": model_short, "results": all_results}, out_path)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*70}")
    print(f"{'GPU Cap':>8} {'Config':>12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*70}")
    for cap_key, cap_data in all_results.items():
        for cfg, m in cap_data.items():
            print(f"{cap_key:>8} {cfg:>12} {m['avg_precision']:>10.3f} "
                  f"{m['avg_recall']:>10.3f} {m['avg_f1']:>10.3f}")


if __name__ == "__main__":
    main()

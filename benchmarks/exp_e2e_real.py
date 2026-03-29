#!/usr/bin/env python3
"""
E2E Experiment: Capacity Extension and Throughput under GPU Memory Pressure.

Compares three KV-cache management strategies:
  1. baseline  — all KV on GPU, OOM when full
  2. naive     — FIFO offload oldest blocks to DRAM
  3. orchkv    — attention-driven hot-cold offload via orchkv_core

Sweeps (gpu_budget, seq_len, num_requests) and measures:
  - max servable requests, throughput (tokens/s), latency, GPU memory

Usage:
    python benchmarks/exp_e2e_real.py
    python benchmarks/exp_e2e_real.py --quick          # fast smoke run
    python benchmarks/exp_e2e_real.py --seq-lens 4096  # single seq_len
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

MODEL_PATH = "Qwen/Qwen2.5-7B"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def generate_prompt_ids(tokenizer, seq_len: int) -> torch.Tensor:
    text = "Artificial intelligence is transforming every aspect of modern life. " * (seq_len // 10 + 1)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"]
    return ids.to("cuda:0")


def run_single_request(
    model, input_ids: torch.Tensor, max_new: int,
    manager=None, collect_attn_every: int = 5,
) -> dict:
    """Run one request through manual decode, optionally with a KV manager."""
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    prompt_len = input_ids.shape[1]
    t_start = time.perf_counter()
    t_first_token = None

    for step in range(max_new):
        want_attn = (manager is not None) and (step % collect_attn_every == 0)
        with torch.no_grad():
            outputs = model(
                cur_ids, past_key_values=past_kv,
                use_cache=True, output_attentions=want_attn,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if t_first_token is None:
            t_first_token = time.perf_counter()

        if manager is not None:
            new_past = outputs.past_key_values
            if step == 0:
                manager.ingest_step(new_past)
            else:
                manager.append_token(new_past)

            if outputs.attentions is not None:
                for li, attn in enumerate(outputs.attentions):
                    manager.report_attention(li, attn)

            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = outputs.past_key_values

        cur_ids = next_token

    t_end = time.perf_counter()
    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else 0
    tpot = (t_end - t_first_token) / max(len(generated) - 1, 1) if t_first_token else 0

    return {
        "prompt_len": prompt_len,
        "generated_len": len(generated),
        "total_time_s": round(total_time, 4),
        "ttft_ms": round(ttft * 1000, 2),
        "tpot_ms": round(tpot * 1000, 2),
        "throughput_tok_s": round((prompt_len + len(generated)) / total_time, 1),
        "tokens": generated,
    }


def run_multi_request(
    model, tokenizer, n_requests: int, seq_len: int,
    max_new: int, mode: str, gpu_budget_bytes: int,
) -> dict:
    """
    Sequentially process n_requests, reusing the model.
    Each request has its own KV manager.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    per_request_budget = gpu_budget_bytes // max(n_requests, 1)

    results_per_req = []
    total_oom = 0
    managers = []

    for ri in range(n_requests):
        input_ids = generate_prompt_ids(tokenizer, seq_len)

        mgr = None
        if mode == "orchkv":
            mgr = KVCacheManager(
                n_layers=n_layers, n_kv_heads=n_kv_heads,
                head_dim=head_dim, block_size=16,
                dtype=torch.float16,
                gpu_budget_bytes=per_request_budget,
            )
        elif mode == "naive":
            mgr = NaiveOffloadManager(
                n_layers=n_layers, n_kv_heads=n_kv_heads,
                head_dim=head_dim, block_size=16,
                dtype=torch.float16,
                gpu_budget_bytes=per_request_budget,
            )

        try:
            res = run_single_request(model, input_ids, max_new, manager=mgr)
            res["request_id"] = ri
            res["status"] = "OK"

            if mgr is not None:
                stats = mgr.get_stats()
                res["blocks_gpu"] = stats.get("blocks_gpu", 0)
                res["blocks_dram"] = stats.get("blocks_dram", 0)
                res["blocks_ssd"] = stats.get("blocks_ssd", 0)
                res["gpu_kv_mb"] = stats.get("gpu_kv_mb", 0)
                mig = stats.get("migrations", {})
                res["evictions"] = mig.get("gpu_to_dram", 0) + mig.get("dram_to_ssd", 0)
                res["promotions"] = mig.get("dram_to_gpu", 0) + mig.get("ssd_to_dram", 0)
                if "tm" in stats:
                    res["n_hot"] = stats["tm"].get("n_hot", 0)
                    res["n_cold"] = stats["tm"].get("n_cold", 0)

            results_per_req.append(res)

        except torch.cuda.OutOfMemoryError:
            total_oom += 1
            results_per_req.append({
                "request_id": ri, "status": "OOM",
            })
            gc.collect()
            torch.cuda.empty_cache()

        finally:
            if mgr is not None:
                mgr.destroy()
            gc.collect()
            torch.cuda.empty_cache()

    ok_results = [r for r in results_per_req if r["status"] == "OK"]
    avg_throughput = sum(r["throughput_tok_s"] for r in ok_results) / max(len(ok_results), 1)
    avg_tpot = sum(r["tpot_ms"] for r in ok_results) / max(len(ok_results), 1)
    total_evictions = sum(r.get("evictions", 0) for r in ok_results)

    return {
        "mode": mode,
        "n_requests": n_requests,
        "seq_len": seq_len,
        "gpu_budget_mb": gpu_budget_bytes // (1 << 20),
        "completed": len(ok_results),
        "oom": total_oom,
        "avg_throughput_tok_s": round(avg_throughput, 1),
        "avg_tpot_ms": round(avg_tpot, 2),
        "total_evictions": total_evictions,
        "per_request": results_per_req,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick smoke run")
    parser.add_argument("--seq-lens", type=str, default=None)
    parser.add_argument("--max-new", type=int, default=64)
    args = parser.parse_args()

    if args.quick:
        gpu_budgets_mb = [100]
        seq_lens = [512]
        n_requests_list = [2, 4]
        modes = ["baseline", "orchkv"]
    else:
        gpu_budgets_mb = [50, 100, 200, 500]
        seq_lens = [1024, 2048, 4096]
        n_requests_list = [1, 4, 8, 16]
        modes = ["baseline", "naive", "orchkv"]

    if args.seq_lens:
        seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 80)
    print("E2E Experiment: Capacity Extension under GPU Memory Pressure")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU budgets (MB): {gpu_budgets_mb}")
    print(f"Seq lengths: {seq_lens}")
    print(f"Request counts: {n_requests_list}")
    print(f"Modes: {modes}")
    print(f"Max new tokens: {args.max_new}")
    print()

    model, tokenizer = load_model()
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    all_results = []
    total_points = len(gpu_budgets_mb) * len(seq_lens) * len(n_requests_list) * len(modes)
    done = 0

    for budget_mb in gpu_budgets_mb:
        budget_bytes = budget_mb * (1 << 20)
        for seq_len in seq_lens:
            for n_req in n_requests_list:
                for mode in modes:
                    done += 1
                    tag = f"[{done}/{total_points}] {mode} budget={budget_mb}MB seq={seq_len} nreq={n_req}"
                    print(f"\n--- {tag} ---")

                    res = run_multi_request(
                        model, tokenizer, n_req, seq_len,
                        args.max_new, mode, budget_bytes,
                    )
                    all_results.append(res)

                    print(f"  completed={res['completed']}/{n_req} "
                          f"oom={res['oom']} "
                          f"throughput={res['avg_throughput_tok_s']} tok/s "
                          f"tpot={res['avg_tpot_ms']}ms "
                          f"evictions={res['total_evictions']}")

    out_path = RESULTS_DIR / "exp_e2e_real.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Mode':<10} {'Budget':>8} {'SeqLen':>8} {'NReq':>5} {'OK':>4} {'OOM':>4} {'Tok/s':>8} {'TPOT':>8} {'Evict':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['mode']:<10} {r['gpu_budget_mb']:>6}MB {r['seq_len']:>8} "
              f"{r['n_requests']:>5} {r['completed']:>4} {r['oom']:>4} "
              f"{r['avg_throughput_tok_s']:>8.1f} {r['avg_tpot_ms']:>7.2f}ms "
              f"{r['total_evictions']:>6}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

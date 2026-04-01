#!/usr/bin/env python3
"""
Multi-Model E2E Experiment: OrchKvCache vs Naive vs Baseline on
Qwen2.5-7B (GQA-4, 56KB/tok) and Mistral-7B (GQA-8, 128KB/tok).

Produces paper-ready data covering:
  1. Throughput & eviction efficiency under memory pressure
  2. Quality verification (lossless)
  3. Component ablation

Usage:
    python benchmarks/exp_multimodel.py
    python benchmarks/exp_multimodel.py --quick   # fast smoke run
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

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Qwen2.5-7B": {
        "path": "Qwen/Qwen2.5-7B",
        "attn_impl": "eager",
    },
    "Mistral-7B": {
        "path": "/raid/models/Mistral-7B-v0.1",
        "attn_impl": "eager",
    },
}


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    info = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(info["path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        info["path"], dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation=info["attn_impl"],
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    print(f"  Loaded {model_name}: {n_layers}L, {n_kv_heads}KV, d={head_dim}, "
          f"KV/tok={kv_per_token/1024:.0f}KB, model={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer, {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "kv_per_token": kv_per_token,
    }


def generate_prompt(tokenizer, seq_len):
    text = "Artificial intelligence is transforming every aspect of modern life and technology. " * (seq_len // 10 + 1)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"]
    return ids.to("cuda:0")


def run_decode(model, input_ids, max_new, manager=None, attn_every=10):
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    t0 = time.perf_counter()
    t_first = None

    for step in range(max_new):
        want_attn = manager is not None and isinstance(manager, KVCacheManager) and step % attn_every == 0
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv, use_cache=True, output_attentions=want_attn)

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
            if hasattr(out, 'attentions') and out.attentions is not None:
                for li, attn in enumerate(out.attentions):
                    manager.report_attention(li, attn)
            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = out.past_key_values
        cur_ids = next_tok

    elapsed = time.perf_counter() - t0
    prompt_len = input_ids.shape[1]
    total_tok = prompt_len + len(generated)
    ttft = (t_first - t0) * 1000 if t_first else 0
    tpot = ((time.perf_counter() - t_first) / max(len(generated) - 1, 1)) * 1000 if t_first else 0

    return {
        "generated": generated,
        "prompt_len": prompt_len,
        "gen_len": len(generated),
        "elapsed_s": round(elapsed, 3),
        "throughput": round(total_tok / elapsed, 1),
        "ttft_ms": round(ttft, 2),
        "tpot_ms": round(tpot, 2),
    }


# ======================================================================
# Experiment 1: E2E Throughput + Eviction
# ======================================================================
def exp_e2e(model, tokenizer, model_cfg, model_name, gpu_budgets, seq_lens, n_reqs_list, max_new):
    print(f"\n{'='*70}")
    print(f"EXP E2E: {model_name}")
    print(f"{'='*70}")
    results = []
    modes = ["baseline", "naive", "orchkv"]

    for budget_mb in gpu_budgets:
        for seq_len in seq_lens:
            for n_req in n_reqs_list:
                for mode in modes:
                    tag = f"{mode} budget={budget_mb}MB seq={seq_len} nreq={n_req}"
                    print(f"\n  [{tag}]", end=" ", flush=True)

                    per_req_budget = (budget_mb * (1 << 20)) // max(n_req, 1)
                    req_results = []
                    total_evict = 0

                    for ri in range(n_req):
                        input_ids = generate_prompt(tokenizer, seq_len)
                        mgr = None
                        if mode == "orchkv":
                            mgr = KVCacheManager(
                                n_layers=model_cfg["n_layers"],
                                n_kv_heads=model_cfg["n_kv_heads"],
                                head_dim=model_cfg["head_dim"],
                                block_size=16, dtype=torch.float16,
                                gpu_budget_bytes=per_req_budget,
                            )
                        elif mode == "naive":
                            mgr = NaiveOffloadManager(
                                n_layers=model_cfg["n_layers"],
                                n_kv_heads=model_cfg["n_kv_heads"],
                                head_dim=model_cfg["head_dim"],
                                block_size=16, dtype=torch.float16,
                                gpu_budget_bytes=per_req_budget,
                            )

                        try:
                            res = run_decode(model, input_ids, max_new, manager=mgr)
                            if mgr:
                                stats = mgr.get_stats()
                                mig = stats.get("migrations", {})
                                evict = mig.get("gpu_to_dram", 0) + mig.get("dram_to_ssd", 0)
                                total_evict += evict
                            req_results.append(res)
                        except torch.cuda.OutOfMemoryError:
                            req_results.append({"status": "OOM"})
                            gc.collect(); torch.cuda.empty_cache()
                        finally:
                            if mgr:
                                mgr.destroy()
                            gc.collect(); torch.cuda.empty_cache()

                    ok = [r for r in req_results if "throughput" in r]
                    oom = len(req_results) - len(ok)
                    avg_tps = sum(r["throughput"] for r in ok) / max(len(ok), 1)
                    avg_tpot = sum(r["tpot_ms"] for r in ok) / max(len(ok), 1)

                    row = {
                        "model": model_name,
                        "mode": mode,
                        "gpu_budget_mb": budget_mb,
                        "seq_len": seq_len,
                        "n_requests": n_req,
                        "completed": len(ok),
                        "oom": oom,
                        "avg_throughput": round(avg_tps, 1),
                        "avg_tpot_ms": round(avg_tpot, 2),
                        "total_evictions": total_evict,
                    }
                    results.append(row)
                    print(f"OK={len(ok)} OOM={oom} tps={avg_tps:.0f} evict={total_evict}")

    return results


# ======================================================================
# Experiment 2: Quality
# ======================================================================
def exp_quality(model, tokenizer, model_cfg, model_name, max_new=128):
    print(f"\n{'='*70}")
    print(f"EXP QUALITY: {model_name}")
    print(f"{'='*70}")

    prompts = {
        "short": "Artificial intelligence is transforming every aspect of modern life. " * 15,
        "medium": "Machine learning focuses on building systems that learn from data. Deep learning uses neural networks. " * 20,
        "long": "Large language models have transformed NLP. The transformer uses self-attention. KV cache is critical. " * 35,
    }
    budget_mb = 20
    results = []

    for label, text in prompts.items():
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to("cuda:0")
        prompt_len = input_ids.shape[1]
        print(f"\n  {label} ({prompt_len} tokens):", end=" ", flush=True)

        # Baseline
        base_res = run_decode(model, input_ids, max_new, manager=None)
        gc.collect(); torch.cuda.empty_cache()

        # OrchKv
        mgr = KVCacheManager(
            n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
            head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=budget_mb * (1 << 20),
        )
        orch_res = run_decode(model, input_ids, max_new, manager=mgr)
        stats = mgr.get_stats()
        mgr.destroy()
        gc.collect(); torch.cuda.empty_cache()

        n_match = sum(1 for a, b in zip(base_res["generated"], orch_res["generated"]) if a == b)
        n_total = min(len(base_res["generated"]), len(orch_res["generated"]))
        match_rate = n_match / n_total * 100 if n_total > 0 else 0
        evictions = stats.get("migrations", {}).get("gpu_to_dram", 0)

        results.append({
            "model": model_name, "prompt": label, "prompt_len": prompt_len,
            "generated": n_total, "match_rate": round(match_rate, 4),
            "evictions": evictions,
        })
        status = "PASS" if match_rate == 100.0 else "FAIL"
        print(f"match={match_rate:.2f}% [{status}] evict={evictions}")

    return results


# ======================================================================
# Experiment 3: Ablation
# ======================================================================
def exp_ablation(model, tokenizer, model_cfg, model_name, seq_len=2048, max_new=128, budget_mb=50):
    print(f"\n{'='*70}")
    print(f"EXP ABLATION: {model_name}")
    print(f"{'='*70}")

    input_ids = generate_prompt(tokenizer, seq_len)
    prompt_len = input_ids.shape[1]
    budget = budget_mb * (1 << 20)
    results = []

    configs = [
        ("gpu-only", None),
        ("naive-fifo", "naive"),
        ("orchkv", "orchkv"),
    ]

    for name, mgr_type in configs:
        print(f"\n  {name}:", end=" ", flush=True)
        gc.collect(); torch.cuda.empty_cache()

        mgr = None
        if mgr_type == "orchkv":
            mgr = KVCacheManager(
                n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
                gpu_budget_bytes=budget,
            )
        elif mgr_type == "naive":
            mgr = NaiveOffloadManager(
                n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
                gpu_budget_bytes=budget,
            )

        try:
            res = run_decode(model, input_ids, max_new, manager=mgr)
            evictions = 0
            if mgr:
                s = mgr.get_stats()
                evictions = s.get("migrations", {}).get("gpu_to_dram", 0)
            row = {
                "model": model_name, "config": name,
                "throughput": res["throughput"], "evictions": evictions,
                "status": "OK",
            }
            print(f"{res['throughput']:.0f} tok/s evict={evictions}")
        except torch.cuda.OutOfMemoryError:
            row = {"model": model_name, "config": name, "status": "OOM"}
            print("OOM")
            gc.collect(); torch.cuda.empty_cache()
        finally:
            if mgr:
                mgr.destroy()

        results.append(row)

    return results


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        gpu_budgets = [100]
        seq_lens = [512]
        n_reqs = [2]
        max_new = 32
        model_list = ["Qwen2.5-7B"]
    else:
        gpu_budgets = [50, 100, 200]
        seq_lens = [2048, 4096]
        n_reqs = [1, 4, 8, 16]
        max_new = 64
        model_list = ["Qwen2.5-7B", "Mistral-7B"]

    all_e2e = []
    all_quality = []
    all_ablation = []

    for model_name in model_list:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")

        model, tokenizer, model_cfg = load_model(model_name)

        e2e = exp_e2e(model, tokenizer, model_cfg, model_name,
                      gpu_budgets, seq_lens, n_reqs, max_new)
        all_e2e.extend(e2e)

        qual = exp_quality(model, tokenizer, model_cfg, model_name, max_new=128)
        all_quality.extend(qual)

        abl = exp_ablation(model, tokenizer, model_cfg, model_name,
                           seq_len=2048, max_new=128, budget_mb=50)
        all_ablation.extend(abl)

        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # Save results
    for name, data in [("multimodel_e2e", all_e2e),
                        ("multimodel_quality", all_quality),
                        ("multimodel_ablation", all_ablation)]:
        path = RESULTS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nSaved {path}")

    # Print summary tables
    print(f"\n{'='*80}")
    print("E2E SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<14} {'Mode':<10} {'Budget':>7} {'Seq':>6} {'NReq':>5} {'OK':>4} {'Tok/s':>7} {'Evict':>10}")
    print("-" * 75)
    for r in all_e2e:
        print(f"{r['model']:<14} {r['mode']:<10} {r['gpu_budget_mb']:>5}MB {r['seq_len']:>6} "
              f"{r['n_requests']:>5} {r['completed']:>4} {r['avg_throughput']:>7.0f} {r['total_evictions']:>10}")

    print(f"\n{'='*80}")
    print("QUALITY SUMMARY")
    print(f"{'='*80}")
    for r in all_quality:
        s = "PASS" if r["match_rate"] == 100.0 else "FAIL"
        print(f"  {r['model']:<14} {r['prompt']:<8} len={r['prompt_len']:>5} "
              f"match={r['match_rate']:>7.2f}% [{s}] evict={r['evictions']}")

    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    for r in all_ablation:
        if r["status"] == "OK":
            print(f"  {r['model']:<14} {r['config']:<12} {r['throughput']:>7.0f} tok/s evict={r['evictions']}")
        else:
            print(f"  {r['model']:<14} {r['config']:<12} OOM")


if __name__ == "__main__":
    main()

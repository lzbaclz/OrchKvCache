#!/usr/bin/env python3
"""
LLaMA-2 E2E Experiment: OrchKvCache on MHA models with large KV-Cache.

LLaMA-2-7B:  MHA-32, 512 KB/token — 9x larger KV than Qwen2.5-7B
LLaMA-2-13B: MHA-40, 800 KB/token — 14x larger KV than Qwen2.5-7B

These models create much stronger memory pressure, demonstrating
OrchKvCache's value on realistic workloads.

Usage:
    nohup python -u benchmarks/exp_llama.py > benchmarks/results/llama_run.log 2>&1 &
"""
from __future__ import annotations

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
    "LLaMA-2-7B": {
        "path": "/raid/models/Llama-2-7b-hf",
        "attn_impl": "eager",
    },
    "LLaMA-2-13B": {
        "path": "/raid/models/Llama-2-13b-hf",
        "attn_impl": "eager",
    },
}


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    info = MODELS[model_name]
    print(f"  Loading {model_name} from {info['path']}...")
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
    print(f"  OK: {n_layers}L, {n_kv_heads}KV, d={head_dim}, "
          f"KV/tok={kv_per_token/1024:.0f}KB, model={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer, {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "kv_per_token": kv_per_token,
    }


def generate_prompt(tokenizer, seq_len):
    text = "The history of artificial intelligence began in the 1950s when researchers first proposed machines could think. " * (seq_len // 12 + 1)
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
    tpot = ((time.perf_counter() - t_first) / max(len(generated) - 1, 1)) * 1000 if t_first else 0

    return {
        "generated": generated,
        "prompt_len": prompt_len,
        "gen_len": len(generated),
        "elapsed_s": round(elapsed, 3),
        "throughput": round(total_tok / elapsed, 1),
        "tpot_ms": round(tpot, 2),
    }


def exp_e2e(model, tokenizer, model_cfg, model_name, gpu_budgets, seq_lens, n_reqs_list, max_new):
    print(f"\n{'='*70}\nEXP E2E: {model_name}\n{'='*70}")
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
                                n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
                                head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
                                gpu_budget_bytes=per_req_budget)
                        elif mode == "naive":
                            mgr = NaiveOffloadManager(
                                n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
                                head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
                                gpu_budget_bytes=per_req_budget)
                        try:
                            res = run_decode(model, input_ids, max_new, manager=mgr)
                            if mgr:
                                stats = mgr.get_stats()
                                mig = stats.get("migrations", {})
                                total_evict += mig.get("gpu_to_dram", 0) + mig.get("dram_to_ssd", 0)
                            req_results.append(res)
                        except torch.cuda.OutOfMemoryError:
                            req_results.append({"status": "OOM"})
                            gc.collect(); torch.cuda.empty_cache()
                        finally:
                            if mgr: mgr.destroy()
                            gc.collect(); torch.cuda.empty_cache()

                    ok = [r for r in req_results if "throughput" in r]
                    oom = len(req_results) - len(ok)
                    avg_tps = sum(r["throughput"] for r in ok) / max(len(ok), 1)
                    avg_tpot = sum(r["tpot_ms"] for r in ok) / max(len(ok), 1)
                    row = {
                        "model": model_name, "mode": mode,
                        "gpu_budget_mb": budget_mb, "seq_len": seq_len,
                        "n_requests": n_req, "completed": len(ok), "oom": oom,
                        "avg_throughput": round(avg_tps, 1),
                        "avg_tpot_ms": round(avg_tpot, 2),
                        "total_evictions": total_evict,
                    }
                    results.append(row)
                    print(f"OK={len(ok)} OOM={oom} tps={avg_tps:.0f} evict={total_evict}")
    return results


def exp_quality(model, tokenizer, model_cfg, model_name, max_new=64):
    print(f"\n{'='*70}\nEXP QUALITY: {model_name}\n{'='*70}")
    prompts = {
        "short": "The history of artificial intelligence began in the 1950s. " * 15,
        "medium": "Machine learning focuses on building systems that learn from data. " * 25,
        "long": "Large language models have transformed NLP through self-attention. " * 40,
    }
    budget_mb = 50
    results = []
    for label, text in prompts.items():
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to("cuda:0")
        prompt_len = input_ids.shape[1]
        print(f"\n  {label} ({prompt_len} tokens):", end=" ", flush=True)

        base_res = run_decode(model, input_ids, max_new, manager=None)
        gc.collect(); torch.cuda.empty_cache()

        mgr = KVCacheManager(
            n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
            head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=budget_mb * (1 << 20))
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
            "generated": n_total, "match_rate": round(match_rate, 4), "evictions": evictions,
        })
        s = "PASS" if match_rate == 100.0 else "FAIL"
        print(f"match={match_rate:.2f}% [{s}] evict={evictions}")
    return results


def exp_ablation(model, tokenizer, model_cfg, model_name, seq_len=1024, max_new=64, budget_mb=50):
    print(f"\n{'='*70}\nEXP ABLATION: {model_name}\n{'='*70}")
    input_ids = generate_prompt(tokenizer, seq_len)
    budget = budget_mb * (1 << 20)
    results = []
    for name, mgr_type in [("gpu-only", None), ("naive-fifo", "naive"), ("orchkv", "orchkv")]:
        print(f"\n  {name}:", end=" ", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        mgr = None
        if mgr_type == "orchkv":
            mgr = KVCacheManager(n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
                                 head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
                                 gpu_budget_bytes=budget)
        elif mgr_type == "naive":
            mgr = NaiveOffloadManager(n_layers=model_cfg["n_layers"], n_kv_heads=model_cfg["n_kv_heads"],
                                      head_dim=model_cfg["head_dim"], block_size=16, dtype=torch.float16,
                                      gpu_budget_bytes=budget)
        try:
            res = run_decode(model, input_ids, max_new, manager=mgr)
            evictions = mgr.get_stats().get("migrations", {}).get("gpu_to_dram", 0) if mgr else 0
            row = {"model": model_name, "config": name, "throughput": res["throughput"],
                   "evictions": evictions, "status": "OK"}
            print(f"{res['throughput']:.0f} tok/s evict={evictions}")
        except torch.cuda.OutOfMemoryError:
            row = {"model": model_name, "config": name, "status": "OOM"}
            print("OOM"); gc.collect(); torch.cuda.empty_cache()
        finally:
            if mgr: mgr.destroy()
        results.append(row)
    return results


def main():
    print("=" * 70)
    print("LLaMA-2 Experiment Suite (MHA models, large KV-Cache)")
    print("=" * 70)

    # Use SAME parameters as Qwen+Mistral experiment for fair comparison
    gpu_budgets = [50, 100, 200]
    seq_lens = [2048, 4096]
    n_reqs = [1, 4, 8, 16]
    max_new = 64

    all_e2e = []
    all_quality = []
    all_ablation = []

    # ---- LLaMA-2-7B ----
    print(f"\n{'#'*70}\n# MODEL: LLaMA-2-7B (MHA-32, 512 KB/token)\n{'#'*70}")
    model, tokenizer, cfg = load_model("LLaMA-2-7B")

    e2e = exp_e2e(model, tokenizer, cfg, "LLaMA-2-7B", gpu_budgets, seq_lens, n_reqs, max_new)
    all_e2e.extend(e2e)

    qual = exp_quality(model, tokenizer, cfg, "LLaMA-2-7B", max_new=128)
    all_quality.extend(qual)

    abl = exp_ablation(model, tokenizer, cfg, "LLaMA-2-7B", seq_len=2048, max_new=128, budget_mb=50)
    all_ablation.extend(abl)

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    # ---- LLaMA-2-13B ----
    print(f"\n{'#'*70}\n# MODEL: LLaMA-2-13B (MHA-40, 800 KB/token)\n{'#'*70}")
    model, tokenizer, cfg = load_model("LLaMA-2-13B")

    e2e = exp_e2e(model, tokenizer, cfg, "LLaMA-2-13B", gpu_budgets, seq_lens, n_reqs, max_new)
    all_e2e.extend(e2e)

    qual = exp_quality(model, tokenizer, cfg, "LLaMA-2-13B", max_new=128)
    all_quality.extend(qual)

    abl = exp_ablation(model, tokenizer, cfg, "LLaMA-2-13B", seq_len=2048, max_new=128, budget_mb=50)
    all_ablation.extend(abl)

    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

    # Save results
    for name, data in [("llama_e2e", all_e2e), ("llama_quality", all_quality), ("llama_ablation", all_ablation)]:
        path = RESULTS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nSaved {path}")

    # Summary
    print(f"\n{'='*80}\nLLaMA E2E SUMMARY\n{'='*80}")
    print(f"{'Model':<14} {'Mode':<10} {'Budget':>7} {'Seq':>6} {'NReq':>5} {'OK':>4} {'Tok/s':>7} {'Evict':>10}")
    print("-" * 75)
    for r in all_e2e:
        print(f"{r['model']:<14} {r['mode']:<10} {r['gpu_budget_mb']:>5}MB {r['seq_len']:>6} "
              f"{r['n_requests']:>5} {r['completed']:>4} {r['avg_throughput']:>7.0f} {r['total_evictions']:>10}")

    print(f"\n{'='*80}\nQUALITY\n{'='*80}")
    for r in all_quality:
        s = "PASS" if r["match_rate"] == 100.0 else "FAIL"
        print(f"  {r['model']:<14} {r['prompt']:<8} match={r['match_rate']:.2f}% [{s}] evict={r['evictions']}")

    print(f"\n{'='*80}\nABLATION\n{'='*80}")
    for r in all_ablation:
        if r["status"] == "OK":
            print(f"  {r['model']:<14} {r['config']:<12} {r['throughput']:.0f} tok/s evict={r['evictions']}")
        else:
            print(f"  {r['model']:<14} {r['config']:<12} OOM")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Model E2E Experiment: Add LLaMA-2-7B and LLaMA-2-13B to the existing
Qwen2.5-7B / Mistral-7B benchmark suite.

Produces E2E throughput + eviction data under the same parameter grid:
  gpu_budgets = [50, 100, 200]  (MB)
  seq_lens    = [2048, 4096]
  n_reqs      = [1, 4, 8, 16]
  max_new     = 64

Results are **appended** to the existing 4-model JSON for downstream plotting.

Usage:
    python paper/plot_figures_code_data/update_figures/exp_multimodel_llama.py
    python paper/plot_figures_code_data/update_figures/exp_multimodel_llama.py --quick
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

HERE = Path(__file__).resolve().parent

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
    text = ("Artificial intelligence is transforming every aspect of modern "
            "life and technology. ") * (seq_len // 10 + 1)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=seq_len)["input_ids"]
    return ids.to("cuda:0")


def run_decode(model, input_ids, max_new, manager=None, attn_every=10):
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    t0 = time.perf_counter()
    t_first = None

    for step in range(max_new):
        want_attn = (manager is not None
                     and isinstance(manager, KVCacheManager)
                     and step % attn_every == 0)
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv,
                        use_cache=True, output_attentions=want_attn)

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
    tpot = (((time.perf_counter() - t_first) /
             max(len(generated) - 1, 1)) * 1000 if t_first else 0)

    return {
        "generated": generated,
        "prompt_len": prompt_len,
        "gen_len": len(generated),
        "elapsed_s": round(elapsed, 3),
        "throughput": round(total_tok / elapsed, 1),
        "ttft_ms": round(ttft, 2),
        "tpot_ms": round(tpot, 2),
    }


def exp_e2e(model, tokenizer, model_cfg, model_name,
            gpu_budgets, seq_lens, n_reqs_list, max_new):
    print(f"\n{'='*70}")
    print(f"EXP E2E: {model_name}")
    print(f"{'='*70}")
    results = []
    modes = ["baseline", "naive", "orchkv"]

    for budget_mb in gpu_budgets:
        for seq_len in seq_lens:
            for n_req in n_reqs_list:
                for mode in modes:
                    tag = (f"{mode} budget={budget_mb}MB "
                           f"seq={seq_len} nreq={n_req}")
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
                            res = run_decode(model, input_ids, max_new,
                                             manager=mgr)
                            if mgr:
                                stats = mgr.get_stats()
                                mig = stats.get("migrations", {})
                                evict = (mig.get("gpu_to_dram", 0)
                                         + mig.get("dram_to_ssd", 0))
                                total_evict += evict
                            req_results.append(res)
                        except torch.cuda.OutOfMemoryError:
                            req_results.append({"status": "OOM"})
                            gc.collect()
                            torch.cuda.empty_cache()
                        finally:
                            if mgr:
                                mgr.destroy()
                            gc.collect()
                            torch.cuda.empty_cache()

                    ok = [r for r in req_results if "throughput" in r]
                    oom = len(req_results) - len(ok)
                    avg_tps = (sum(r["throughput"] for r in ok)
                               / max(len(ok), 1))
                    avg_tpot = (sum(r["tpot_ms"] for r in ok)
                                / max(len(ok), 1))

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
                    print(f"OK={len(ok)} OOM={oom} "
                          f"tps={avg_tps:.0f} evict={total_evict}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        gpu_budgets = [50]
        seq_lens = [2048]
        n_reqs = [4]
        max_new = 32
        model_list = ["LLaMA-2-7B"]
    else:
        gpu_budgets = [50, 100, 200]
        seq_lens = [2048, 4096]
        n_reqs = [1, 4, 8, 16]
        max_new = 64
        model_list = ["LLaMA-2-7B", "LLaMA-2-13B"]

    all_e2e = []
    for model_name in model_list:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")

        model, tokenizer, model_cfg = load_model(model_name)
        e2e = exp_e2e(model, tokenizer, model_cfg, model_name,
                      gpu_budgets, seq_lens, n_reqs, max_new)
        all_e2e.extend(e2e)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    out_path = HERE / "exp_llama_e2e_results.json"
    with open(out_path, "w") as f:
        json.dump(all_e2e, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    existing_path = HERE / "multimodel_e2e_4models.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        existing_models = {r["model"] for r in existing}
        for r in all_e2e:
            if r["model"] not in existing_models:
                existing.append(r)
        with open(existing_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"Merged into {existing_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ablation experiment for LLaMA-2-7B and LLaMA-2-13B.

Same config as the Qwen/Mistral ablation:
  seq_len=2048, max_new=128, budget_mb=50
  configs: gpu-only, naive-fifo, orchkv

Usage:
    python paper/plot_figures_code_data/update_figures/exp_ablation_llama.py
"""
from __future__ import annotations

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
          f"KV/tok={kv_per_token/1024:.0f}KB")
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

    return {
        "throughput": round(total_tok / elapsed, 1),
        "gen_len": len(generated),
    }


def exp_ablation(model, tokenizer, model_cfg, model_name,
                 seq_len=2048, max_new=128, budget_mb=50):
    print(f"\n{'='*70}")
    print(f"EXP ABLATION: {model_name}")
    print(f"{'='*70}")

    input_ids = generate_prompt(tokenizer, seq_len)
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
                n_layers=model_cfg["n_layers"],
                n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"],
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=budget,
            )
        elif mgr_type == "naive":
            mgr = NaiveOffloadManager(
                n_layers=model_cfg["n_layers"],
                n_kv_heads=model_cfg["n_kv_heads"],
                head_dim=model_cfg["head_dim"],
                block_size=16, dtype=torch.float16,
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
            row = {"model": model_name, "config": name, "status": "OOM",
                   "throughput": 0, "evictions": 0}
            print("OOM")
            gc.collect(); torch.cuda.empty_cache()
        finally:
            if mgr:
                mgr.destroy()

        results.append(row)

    return results


def main():
    all_results = []
    for model_name in ["LLaMA-2-7B", "LLaMA-2-13B"]:
        model, tokenizer, model_cfg = load_model(model_name)
        res = exp_ablation(model, tokenizer, model_cfg, model_name)
        all_results.extend(res)
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    out_path = HERE / "exp_ablation_llama_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n--- Summary ---")
    for r in all_results:
        print(f"  {r['model']:<14} {r['config']:<12} "
              f"tps={r['throughput']:>7} evict={r['evictions']}")


if __name__ == "__main__":
    main()

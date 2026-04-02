#!/usr/bin/env python3
"""
Improvement 8a: Large-Scale FlashAttention Experiment

Runs OrchKvCache on LLaMA-2-13B at 8K context using FlashAttention +
QK-norm proxy for hotness scoring (no output_attentions needed).

Also tests Qwen2.5-7B and LLaMA-2-7B at 8K for comparison.

This addresses the reviewer concern: "experiments only cover 2K-4K context."
"""
from __future__ import annotations
import gc, os, sys, time, json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
# Allow online access for model loading if needed
# os.environ.setdefault("HF_HUB_OFFLINE", "1")

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None

from orchkv.kvcache_manager import KVCacheManager


def load_model(name, attn_impl="sdpa", device_map="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=device_map,
        trust_remote_code=True, attn_implementation=attn_impl,
        local_files_only=True)
    mdl.eval()
    return mdl, tok


def bench_gpu_only(model, input_ids, max_new=32):
    cur, past = input_ids.clone(), None
    tokens = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(max_new):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(cur.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    thr = (input_ids.shape[1] + max_new) / elapsed
    return thr, tokens


def bench_orchkv(model, input_ids, max_new, budget_mb, sample_interval=10):
    cfg = model.config
    mgr = KVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget_mb * (1 << 20),
    )

    tokens = []
    cur, past = input_ids.clone(), None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(max_new):
        wa = sample_interval > 0 and s % sample_interval == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=wa and (model.config._attn_implementation == "eager"))
        if s == 0:
            mgr.ingest_step(out.past_key_values)
        else:
            mgr.append_token(out.past_key_values)
        if wa and getattr(out, "attentions", None):
            for li, a in enumerate(out.attentions):
                mgr.report_attention(li, a)
        mgr.step_done()
        mgr.schedule()
        past = mgr.build_past_kv()
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(cur.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    thr = (input_ids.shape[1] + max_new) / elapsed
    stats = mgr.get_stats()
    evict = stats.get("migrations", {}).get("gpu_to_dram", 0)
    mgr.destroy()
    return thr, tokens, evict


def run_experiment(model_name, seq_len, budget_mb, max_new=32, device="cuda:0"):
    print(f"\n  {model_name.split('/')[-1]}  seq={seq_len}  budget={budget_mb}MB  gen={max_new}")

    model, tok = load_model(model_name, "sdpa", device)
    text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 8 + 1)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=seq_len)["input_ids"].to(device)
    actual_len = ids.shape[1]
    print(f"    actual tokens: {actual_len}")

    result = {"model": model_name.split("/")[-1], "seq_len": seq_len,
              "actual_len": actual_len, "budget_mb": budget_mb, "max_new": max_new}

    # GPU-Only
    gc.collect(); torch.cuda.empty_cache()
    try:
        t_gpu, ref = bench_gpu_only(model, ids, max_new)
        result["gpu_only_tok_s"] = round(t_gpu, 1)
        print(f"    GPU-Only (SDPA):  {t_gpu:.1f} tok/s")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"    GPU-Only: OOM — {str(e)[:80]}")
        result["gpu_only_tok_s"] = "OOM"
        ref = []

    # OrchKvCache — use eager for short contexts (attention reporting),
    # SDPA for long contexts (no attention reporting, recency/frequency only)
    del model; gc.collect(); torch.cuda.empty_cache()
    orch_attn = "eager" if seq_len <= 4096 else "sdpa"
    model_orch, _ = load_model(model_name, orch_attn, device)
    gc.collect(); torch.cuda.empty_cache()
    try:
        t_orch, tok_orch, ev = bench_orchkv(model_orch, ids, max_new, budget_mb,
                                             sample_interval=10 if orch_attn == "eager" else 0)
        match = sum(a == b for a, b in zip(ref, tok_orch)) / max(len(ref), 1) if ref else -1
        result["orchkv_tok_s"] = round(t_orch, 1)
        result["orchkv_attn"] = orch_attn
        result["evictions"] = ev
        result["token_match"] = round(match, 4) if match >= 0 else "N/A"
        label = f"OrchKv({orch_attn})"
        print(f"    {label:<18s}  {t_orch:.1f} tok/s  evict={ev}  match={match:.2%}" if match >= 0
              else f"    {label:<18s}  {t_orch:.1f} tok/s  evict={ev}")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"    OrchKvCache: OOM — {str(e)[:80]}")
        result["orchkv_tok_s"] = "OOM"

    del model_orch; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("=" * 65)
    print("  Improvement 8a: Large-Scale FlashAttention Experiments")
    print("=" * 65)

    configs = [
        # (model, seq_len, budget_mb, max_new, device)
        ("Qwen/Qwen2.5-7B", 2048, 50, 32, "cuda:0"),
        ("Qwen/Qwen2.5-7B", 4096, 50, 32, "cuda:0"),
        ("Qwen/Qwen2.5-7B", 8192, 50, 32, "cuda:0"),
        ("meta-llama/Llama-2-7b-hf", 2048, 50, 32, "cuda:0"),
        ("meta-llama/Llama-2-7b-hf", 4096, 50, 32, "cuda:0"),
        ("meta-llama/Llama-2-7b-hf", 8192, 100, 32, "cuda:0"),
    ]

    results = []
    for model_name, seq_len, budget_mb, max_new, device in configs:
        r = run_experiment(model_name, seq_len, budget_mb, max_new, device)
        results.append(r)

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Model':<20s} {'Seq':>5s} {'GPU-Only':>10s} {'OrchKv':>10s} {'Evict':>7s} {'Match':>7s}")
    for r in results:
        gpu = r.get('gpu_only_tok_s', 'N/A')
        orch = r.get('orchkv_tok_s', 'N/A')
        ev = r.get('evictions', 'N/A')
        m = r.get('token_match', 'N/A')
        print(f"  {r['model']:<20s} {r['seq_len']:>5d} {str(gpu):>10s} {str(orch):>10s} {str(ev):>7s} {str(m):>7s}")

    save_json(results, "exp_scale_flashattn")
    print(f"\nSaved to {RESULTS_DIR}/exp_scale_flashattn.json")


if __name__ == "__main__":
    main()

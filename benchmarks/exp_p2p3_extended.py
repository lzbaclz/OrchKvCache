#!/usr/bin/env python3
"""
P2: 8K+ Context Experiments — test OrchKvCache at longer sequences
P3: Hyperparameter E2E Validation — compare trace-sim results with actual inference

Uses FastFIFO and KVCacheManager (original, not fast) to measure:
  - P2: throughput and eviction at seq_len=4096, 8192 on Qwen2.5-7B
  - P3: E2E throughput with different EMA lambda values
"""
from __future__ import annotations
import gc, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None

from orchkv.kvcache_manager import KVCacheManager


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    mdl.eval()
    return mdl, tok


def bench_gpu_only(model, input_ids, max_new=64):
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
    thr = (input_ids.shape[1] + max_new) / (time.perf_counter() - t0)
    return thr, tokens


def bench_managed(model, input_ids, max_new, budget_mb,
                  sample_interval=10, ema_lambda=None):
    cfg = model.config
    kwargs = dict(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16, dtype=torch.float16,
        gpu_budget_bytes=budget_mb * (1 << 20),
    )

    mgr = KVCacheManager(**kwargs)
    if ema_lambda is not None and _C and hasattr(mgr, '_tm_handle') and mgr._tm_handle:
        _C.tm_set_policy(mgr._tm_handle, ema_lambda, 0.2, 0.1)

    tokens = []
    cur, past = input_ids.clone(), None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(max_new):
        wa = sample_interval > 0 and s % sample_interval == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=wa)
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


def run_p2_8k_context():
    """P2: 8K+ context experiments on Qwen2.5-7B."""
    print(f"\n{'='*65}")
    print(f"  P2: Extended Context Length Experiments (Qwen2.5-7B)")
    print(f"{'='*65}")

    model_name = "Qwen/Qwen2.5-7B"
    model, tok = load_model(model_name)
    max_new = 32
    budget_mb = 50
    results = []

    for seq_len in [2048, 4096, 8192]:
        print(f"\n  seq_len={seq_len}, budget={budget_mb}MB, gen={max_new}")

        text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 8)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=seq_len)["input_ids"].to("cuda:0")
        actual_len = ids.shape[1]
        print(f"    actual prompt tokens: {actual_len}")

        gc.collect(); torch.cuda.empty_cache()
        try:
            t_gpu, ref = bench_gpu_only(model, ids, max_new)
            print(f"    GPU-Only: {t_gpu:.1f} tok/s")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"    GPU-Only: OOM ({e})")
            t_gpu, ref = 0, []

        gc.collect(); torch.cuda.empty_cache()
        try:
            t_orch, tok_orch, ev_orch = bench_managed(model, ids, max_new, budget_mb)
            match = sum(a == b for a, b in zip(ref, tok_orch)) / max(len(ref), 1) if ref else -1
            print(f"    OrchKv:   {t_orch:.1f} tok/s  evict={ev_orch}  match={match:.2%}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"    OrchKv:   OOM ({e})")
            t_orch, ev_orch, match = 0, 0, -1

        results.append({
            "seq_len": seq_len, "actual_len": actual_len,
            "budget_mb": budget_mb, "max_new": max_new,
            "gpu_only_tok_s": round(t_gpu, 1),
            "orchkv_tok_s": round(t_orch, 1),
            "evictions": ev_orch,
            "token_match": round(match, 4) if match >= 0 else "N/A",
        })

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


def run_p3_hyperparam_e2e():
    """P3: Hyperparameter E2E validation on Qwen2.5-7B."""
    print(f"\n{'='*65}")
    print(f"  P3: Hyperparameter E2E Validation (Qwen2.5-7B)")
    print(f"{'='*65}")

    model_name = "Qwen/Qwen2.5-7B"
    model, tok = load_model(model_name)
    seq_len = 1024
    max_new = 64
    budget_mb = 50
    results = []

    text = "The transformer " * (seq_len // 2)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=seq_len)["input_ids"].to("cuda:0")

    gc.collect(); torch.cuda.empty_cache()
    t_gpu, ref = bench_gpu_only(model, ids, max_new)
    print(f"  GPU-Only baseline: {t_gpu:.1f} tok/s")

    for lam in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
        gc.collect(); torch.cuda.empty_cache()
        try:
            t, toks, ev = bench_managed(model, ids, max_new, budget_mb, ema_lambda=lam)
            match = sum(a == b for a, b in zip(ref, toks)) / len(ref) if ref else -1
            speedup = t / t_gpu if t_gpu > 0 else 0
            print(f"  lambda={lam:.2f}:  {t:>7.1f} tok/s  "
                  f"evict={ev:>5d}  match={match:.2%}  vs_gpu={speedup:.3f}x")
            results.append({
                "ema_lambda": lam, "tok_s": round(t, 1),
                "evictions": ev, "token_match": round(match, 4),
                "speedup_vs_gpu": round(speedup, 3),
            })
        except Exception as e:
            print(f"  lambda={lam:.2f}:  ERROR: {e}")
            results.append({"ema_lambda": lam, "error": str(e)})

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    all_results = {}

    p2 = run_p2_8k_context()
    all_results["p2_extended_context"] = p2

    p3 = run_p3_hyperparam_e2e()
    all_results["p3_hyperparam_e2e"] = p3

    save_json(all_results, "exp_p2p3_extended")
    print(f"\nAll results saved to {RESULTS_DIR}/exp_p2p3_extended.json")


if __name__ == "__main__":
    main()

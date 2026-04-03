#!/usr/bin/env python3
"""
Improvement 8b: Selective Restore / Attention Weight Coverage

Measures how well OrchKvCache's EMA-based scoring predicts which KV
blocks receive the most attention weight. This demonstrates the VALUE
of intelligent scheduling for kernel-level selective block placement.

Experiment:
  1. Run inference with full attention weights collected each step
  2. Simultaneously collect OrchKvCache EMA scores per block
  3. Rank blocks by EMA score and compute cumulative attention coverage
  4. Report: "top-K% blocks by EMA score capture X% of attention weight"

If top-50% blocks capture >95% of attention, this means a kernel-level
selective restore could skip restoring 50% of blocks with <5% quality impact.
"""
from __future__ import annotations
import gc, os, sys, time, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None


def run_coverage_experiment(model_name, seq_len=1024, max_new=64, budget_mb=50):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  {model_name.split('/')[-1]}  seq={seq_len}  gen={max_new}  budget={budget_mb}MB")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    mdl.eval()
    cfg = mdl.config
    block_size = 16

    text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 8 + 1)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=seq_len)["input_ids"].to("cuda:0")
    actual_len = ids.shape[1]

    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)

    if _C is None:
        print("    orchkv_core not available, skipping")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return None

    params = dict(
        tracker_cap=2048, max_blocks=1024,
        alpha=0.7, beta=0.2, gamma=0.1,
        prefetch_budget=8, schedule_interval_us=500,
        gpu_hwm=0.80, gpu_lwm=0.60,
        dram_hwm=0.80, dram_lwm=0.60,
        threshold_to_gpu=0.5, threshold_to_dram=0.15,
    )
    tm = _C.tm_create(**params)

    n_blocks_total = (actual_len + max_new + block_size - 1) // block_size
    for bid in range(n_blocks_total):
        flags = 1 if bid == 0 else 0
        _C.tm_register_block_id(tm, bid, 0, flags)

    coverage_per_step = []
    cur, past = ids.clone(), None

    for s in range(max_new):
        with torch.no_grad():
            out = mdl(cur, past_key_values=past, use_cache=True,
                      output_attentions=True)
        past = out.past_key_values

        cur_len = actual_len + s + 1
        n_blocks = (cur_len + block_size - 1) // block_size

        if out.attentions is not None and len(out.attentions) > 0:
            all_block_attn = np.zeros(n_blocks)
            all_block_ema = np.zeros(n_blocks)
            n_layers_counted = 0

            for li, attn_w in enumerate(out.attentions):
                avg = attn_w.float().mean(dim=(0, 2)).squeeze(0).cpu().numpy()
                for bid in range(n_blocks):
                    start = bid * block_size
                    end = min(start + block_size, len(avg))
                    if start >= len(avg):
                        break
                    block_attn = float(avg[start:end].sum())
                    all_block_attn[bid] += block_attn
                    _C.tm_report_attn(tm, bid, block_attn)
                n_layers_counted += 1

            _C.tm_step_done(tm)
            _C.tm_schedule_once(tm)

            for bid in range(n_blocks):
                all_block_ema[bid] = _C.tm_get_block_score(tm, bid)

            sink_blocks = 1
            nonsink_mask = np.ones(n_blocks, dtype=bool)
            nonsink_mask[:sink_blocks] = False
            nonsink_idx = np.where(nonsink_mask)[0]

            if len(nonsink_idx) > 0:
                nonsink_attn = all_block_attn[nonsink_idx]
                nonsink_ema = all_block_ema[nonsink_idx]
                total_nonsink = nonsink_attn.sum()

                if total_nonsink > 0:
                    ema_order = np.argsort(-nonsink_ema)
                    cum_ema = np.cumsum(nonsink_attn[ema_order]) / total_nonsink

                    rand_order = np.random.permutation(len(nonsink_idx))
                    cum_rand = np.cumsum(nonsink_attn[rand_order]) / total_nonsink

                    step_coverage = {"step": s, "n_blocks": int(len(nonsink_idx))}
                    for pct in [1, 3, 5, 10, 20, 30, 50, 70, 90]:
                        k = max(1, int(len(nonsink_idx) * pct / 100))
                        idx = min(k - 1, len(cum_ema) - 1)
                        step_coverage[f"ema_top{pct}pct"] = round(float(cum_ema[idx]) * 100, 2)
                        step_coverage[f"rand_top{pct}pct"] = round(float(cum_rand[idx]) * 100, 2)
                    coverage_per_step.append(step_coverage)
        else:
            _C.tm_step_done(tm)

        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _C.tm_destroy(tm)
    del mdl; gc.collect(); torch.cuda.empty_cache()

    if not coverage_per_step:
        return None

    avg_ema = {}
    avg_rand = {}
    for pct in [1, 3, 5, 10, 20, 30, 50, 70, 90]:
        ema_key = f"ema_top{pct}pct"
        rand_key = f"rand_top{pct}pct"
        ema_vals = [c[ema_key] for c in coverage_per_step if ema_key in c]
        rand_vals = [c[rand_key] for c in coverage_per_step if rand_key in c]
        avg_ema[f"top{pct}pct"] = round(np.mean(ema_vals), 2) if ema_vals else 0
        avg_rand[f"top{pct}pct"] = round(np.mean(rand_vals), 2) if rand_vals else 0

    return {
        "model": model_name.split("/")[-1],
        "seq_len": seq_len,
        "max_new": max_new,
        "n_steps_measured": len(coverage_per_step),
        "avg_ema_coverage": avg_ema,
        "avg_random_coverage": avg_rand,
        "per_step_samples": coverage_per_step[:5],
    }


def main():
    print("=" * 65)
    print("  Improvement 8b: Attention Weight Coverage by EMA Score")
    print("=" * 65)

    configs = [
        ("Qwen/Qwen2.5-7B", 1024, 64, 50),
        ("meta-llama/Llama-2-7b-hf", 1024, 64, 50),
    ]

    results = []
    for model_name, seq_len, max_new, budget_mb in configs:
        r = run_coverage_experiment(model_name, seq_len, max_new, budget_mb)
        if r:
            results.append(r)
            ema = r["avg_ema_coverage"]
            rnd = r["avg_random_coverage"]
            print(f"\n    Coverage (excluding sink, non-sink blocks only):")
            print(f"    {'Top-K%':>8s}  {'EMA':>8s}  {'Random':>8s}  {'Gain':>6s}")
            for pct in [1, 3, 5, 10, 20, 30, 50, 70, 90]:
                key = f"top{pct}pct"
                gain = ema[key] - rnd[key]
                print(f"    Top-{pct:>2d}%  {ema[key]:>7.1f}%  {rnd[key]:>7.1f}%  {gain:>+5.1f}%")

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY: EMA vs Random Coverage (non-sink blocks)")
    print(f"{'=' * 65}")
    print(f"  {'Model':<18s} {'Metric':>6s} {'t1%':>5s} {'t3%':>5s} {'t5%':>5s} {'t10%':>5s} {'t30%':>5s} {'t50%':>5s} {'t90%':>5s}")
    for r in results:
        e = r["avg_ema_coverage"]
        rd = r["avg_random_coverage"]
        print(f"  {r['model']:<18s}  {'EMA':>5s} {e['top1pct']:>4.0f}% {e['top3pct']:>4.0f}% {e['top5pct']:>4.0f}% "
              f"{e['top10pct']:>4.0f}% {e['top30pct']:>4.0f}% {e['top50pct']:>4.0f}% {e['top90pct']:>4.0f}%")
        print(f"  {'':18s}  {'Rand':>5s} {rd['top1pct']:>4.0f}% {rd['top3pct']:>4.0f}% {rd['top5pct']:>4.0f}% "
              f"{rd['top10pct']:>4.0f}% {rd['top30pct']:>4.0f}% {rd['top50pct']:>4.0f}% {rd['top90pct']:>4.0f}%")

    save_json(results, "exp_selective_restore")
    print(f"\nSaved to {RESULTS_DIR}/exp_selective_restore.json")


if __name__ == "__main__":
    main()

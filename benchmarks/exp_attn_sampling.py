#!/usr/bin/env python3
"""
E10: Attention Sampling Sensitivity

Measures the trade-off between attention collection frequency and:
  (A) classification accuracy   — orchkv_core trace simulation, no GPU needed
  (B) end-to-end throughput     — HuggingFace model decode on real hardware

Directly addresses W3 (Eager Attention 7.2x overhead):
  By collecting attention only every N steps and using SDPA (Flash-equivalent)
  on non-sampling steps, the amortised overhead drops to ~1/N of the full-eager
  cost.  Part A quantifies how much classification quality degrades; Part B
  quantifies the resulting throughput gain.

Usage:
    python benchmarks/exp_attn_sampling.py                        # both parts
    python benchmarks/exp_attn_sampling.py --part a               # trace only
    python benchmarks/exp_attn_sampling.py --part b               # E2E only
    python benchmarks/exp_attn_sampling.py --part b --model Qwen/Qwen2.5-7B
    python benchmarks/exp_attn_sampling.py --part a --n-blocks 512 --n-steps 300
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, save_csv, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    _C = None

SAMPLE_INTERVALS = [1, 2, 5, 10, 20, 50]


# ═══════════════════════════════════════════════════════════════════
#  Attention Trace Generation
# ═══════════════════════════════════════════════════════════════════

def generate_attention_trace(n_blocks: int, n_steps: int,
                             hot_fraction: float = 0.10,
                             warm_fraction: float = 0.15,
                             shift_every: int = 5,
                             shift_fraction: float = 0.15,
                             seed: int = 42) -> list[list[tuple[int, float]]]:
    """
    Synthetic attention trace matching observed characteristics:
      - Top ~10% blocks receive ~90% of total weight  (Gini ~0.9)
      - Slowly shifting hot set                       (Jaccard ~0.7)
      - Warm tier with moderate, noisy weights
    """
    rng = random.Random(seed)

    n_hot = max(1, int(n_blocks * hot_fraction))
    n_warm_pool = max(1, int(n_blocks * warm_fraction * 2))
    hot_set = set(range(n_hot))
    warm_pool = set(range(n_hot, n_hot + n_warm_pool))

    trace: list[list[tuple[int, float]]] = []

    for step in range(n_steps):
        if step > 0 and step % shift_every == 0:
            n_shift = max(1, int(n_hot * shift_fraction))
            removable = list(hot_set)
            to_remove = set(rng.sample(removable, min(n_shift, len(removable))))
            cold_candidates = [b for b in range(n_blocks) if b not in hot_set]
            to_add = set(rng.sample(cold_candidates,
                                    min(n_shift, len(cold_candidates))))
            hot_set = (hot_set - to_remove) | to_add
            warm_pool |= to_remove

        step_data: list[tuple[int, float]] = []
        for bid in range(n_blocks):
            if bid in hot_set:
                w = max(0.0, rng.gauss(0.80, 0.10))
            elif bid in warm_pool:
                w = max(0.0, rng.gauss(0.15, 0.05))
            else:
                w = max(0.0, rng.gauss(0.02, 0.01))
            step_data.append((bid, w))

        trace.append(step_data)

    return trace


# ═══════════════════════════════════════════════════════════════════
#  Part A: orchkv_core Trace-Based Classification Accuracy
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_TM_PARAMS = dict(
    alpha=0.5, beta=0.3, gamma=0.2,
    prefetch_budget=16,
    schedule_interval_us=500,
    gpu_hwm=0.80, gpu_lwm=0.60,
    dram_hwm=0.80, dram_lwm=0.60,
    threshold_to_gpu=0.4,
    threshold_to_dram=0.15,
)


def _simulate_sampling(trace: list[list[tuple[int, float]]],
                       n_blocks: int,
                       sample_interval: int,
                       tm_params: dict,
                       report_threshold: float = 0.05) -> tuple[list[dict], dict]:
    """
    Replay *trace* through orchkv_core's tiered_manager, reporting attention
    only every *sample_interval* steps.  On non-sampling steps, EMA decays
    naturally (ema *= lambda) without fresh data.

    Only blocks with weight >= report_threshold are reported on sampling
    steps.  This matches real behaviour: blocks with negligible attention
    contribute nothing meaningful to the classifier, and their recency /
    frequency should decay naturally to reflect disuse.

    Returns (step_records, final_stats).
    """
    params = dict(
        tracker_cap=n_blocks * 2,
        max_blocks=n_blocks + 64,
        **tm_params,
    )
    tm = _C.tm_create(**params)

    for bid in range(n_blocks):
        _C.tm_register_block_id(tm, bid, int(_C.GPU_HBM), 0)

    step_records: list[dict] = []

    for step, step_data in enumerate(trace):
        if step % sample_interval == 0:
            for bid, w in step_data:
                if w >= report_threshold:
                    _C.tm_report_attn(tm, bid, w)

        _C.tm_step_done(tm)

        if step % 5 == 0:
            _C.tm_set_usage(tm, gpu_ratio=0.85, dram_ratio=0.55)
            _C.tm_schedule_once(tm)

        s = _C.tm_get_stats(tm)
        step_records.append({
            "step": step,
            "n_hot": s["n_hot"],
            "n_warm": s["n_warm"],
            "n_cold": s["n_cold"],
        })

    final = _C.tm_get_stats(tm)
    _C.tm_destroy(tm)
    return step_records, final


def _ground_truth_counts(trace: list[list[tuple[int, float]]],
                         n_blocks: int,
                         hot_thresh: float = 0.50,
                         warm_thresh: float = 0.08):
    """
    Derive expected (n_hot, n_warm, n_cold) per step from the known trace.
    A block with weight >= hot_thresh is "true hot", >= warm_thresh is "true warm".
    """
    gt: list[dict] = []
    for step_data in trace:
        weights = {bid: w for bid, w in step_data}
        n_hot = sum(1 for b in range(n_blocks) if weights.get(b, 0) >= hot_thresh)
        n_warm = sum(1 for b in range(n_blocks)
                     if warm_thresh <= weights.get(b, 0) < hot_thresh)
        n_cold = n_blocks - n_hot - n_warm
        gt.append({"n_hot": n_hot, "n_warm": n_warm, "n_cold": n_cold})
    return gt


def _distribution_accuracy(records: list[dict], gt: list[dict],
                            n_blocks: int) -> float:
    """
    Aggregate distribution accuracy vs ground truth in [0, 1].

    At each step: 1 - |Δ hot| + |Δ warm| + |Δ cold| / (2 · n_blocks).
    """
    total = 0.0
    n = min(len(records), len(gt))
    for rec, ref in zip(records[:n], gt[:n]):
        diff = (abs(rec["n_hot"] - ref["n_hot"])
                + abs(rec["n_warm"] - ref["n_warm"])
                + abs(rec["n_cold"] - ref["n_cold"]))
        total += max(0.0, 1.0 - diff / (2 * n_blocks))
    return total / max(n, 1)


def _baseline_agreement(ref: list[dict], test: list[dict],
                         n_blocks: int) -> float:
    """Agreement between test and N=1 baseline distributions."""
    total = 0.0
    n = min(len(ref), len(test))
    for r, t in zip(ref[:n], test[:n]):
        diff = (abs(r["n_hot"] - t["n_hot"])
                + abs(r["n_warm"] - t["n_warm"])
                + abs(r["n_cold"] - t["n_cold"]))
        total += max(0.0, 1.0 - diff / (2 * n_blocks))
    return total / max(n, 1)


def run_part_a(n_blocks: int = 256, n_steps: int = 200,
               n_runs: int = 3, seed: int = 42) -> list[dict]:
    """
    Part A: classification quality at varying sample intervals.

    Two accuracy metrics:
      - gt_accuracy:       agreement with ground-truth labels from the trace
      - baseline_agreement: agreement with the N=1 classifier output
    """
    print(f"\n{'=' * 70}")
    print(f"  Part A: Classification Quality  (orchkv_core trace simulation)")
    print(f"  n_blocks={n_blocks}  n_steps={n_steps}  n_runs={n_runs}")
    print(f"{'=' * 70}")

    if _C is None:
        print("  [SKIP] orchkv_core not available")
        return []

    raw_rows: list[dict] = []

    for run_id in range(n_runs):
        run_seed = seed + run_id * 1000
        trace = generate_attention_trace(n_blocks, n_steps, seed=run_seed)
        gt = _ground_truth_counts(trace, n_blocks)

        ref_records, ref_stats = _simulate_sampling(
            trace, n_blocks, sample_interval=1, tm_params=_DEFAULT_TM_PARAMS)

        for N in SAMPLE_INTERVALS:
            test_records, test_stats = _simulate_sampling(
                trace, n_blocks, sample_interval=N, tm_params=_DEFAULT_TM_PARAMS)

            gt_acc = _distribution_accuracy(test_records, gt, n_blocks)
            bl_agr = _baseline_agreement(ref_records, test_records, n_blocks)

            ref_mig = ref_stats["gpu_demotes"] + ref_stats.get("dram_demotes", 0)
            test_mig = test_stats["gpu_demotes"] + test_stats.get("dram_demotes", 0)

            raw_rows.append({
                "run": run_id,
                "sample_interval": N,
                "gt_accuracy": round(gt_acc, 4),
                "baseline_agreement": round(bl_agr, 4),
                "n_hot_gt": gt[-1]["n_hot"],
                "n_hot_test": test_stats["n_hot"],
                "n_warm_gt": gt[-1]["n_warm"],
                "n_warm_test": test_stats["n_warm"],
                "n_cold_gt": gt[-1]["n_cold"],
                "n_cold_test": test_stats["n_cold"],
                "gpu_demotes": test_stats["gpu_demotes"],
                "migration_ratio": round(test_mig / max(ref_mig, 1), 4),
            })

    summary: list[dict] = []
    for N in SAMPLE_INTERVALS:
        rows = [r for r in raw_rows if r["sample_interval"] == N]
        if not rows:
            continue
        avg_gt = sum(r["gt_accuracy"] for r in rows) / len(rows)
        avg_bl = sum(r["baseline_agreement"] for r in rows) / len(rows)

        entry = {
            "sample_interval": N,
            "gt_accuracy": round(avg_gt, 4),
            "baseline_agreement": round(avg_bl, 4),
            "sampling_overhead_pct": round(100.0 / N, 1),
        }
        summary.append(entry)
        print(f"  N={N:3d}  gt_accuracy={avg_gt:.3f}  "
              f"baseline_agreement={avg_bl:.3f}  "
              f"overhead={100/N:.0f}%")

    save_json({"raw": raw_rows, "summary": summary,
               "config": {"n_blocks": n_blocks, "n_steps": n_steps,
                           "n_runs": n_runs, "intervals": SAMPLE_INTERVALS}},
              "exp_e10_sampling_sim")
    if summary:
        save_csv(summary, "exp_e10_sampling_sim")

    return summary


# ═══════════════════════════════════════════════════════════════════
#  Part B: End-to-End Throughput with Real Model
# ═══════════════════════════════════════════════════════════════════

def _load_model(model_name: str, attn_impl: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    mdl.eval()
    return mdl, tok


def _run_decode(model, input_ids: torch.Tensor, max_new: int,
                manager, sample_interval: int):
    """
    Decode loop with variable attention collection.

    sample_interval > 0:
        Every N steps → output_attentions=True (triggers eager fallback in SDPA
        models, returns softmax weights).  Other steps use the default fast path.
    sample_interval == 0:
        Never collect attention.  No manager interaction.
    """
    generated: list[int] = []
    cur_ids = input_ids.clone()
    past_kv = None
    step_times: list[float] = []

    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()

    for step in range(max_new):
        st0 = time.perf_counter()

        want_attn = sample_interval > 0 and (step % sample_interval == 0)

        with torch.no_grad():
            outputs = model(
                cur_ids, past_key_values=past_kv,
                use_cache=True, output_attentions=want_attn,
            )

        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())

        if manager is not None:
            new_past = outputs.past_key_values
            if step == 0:
                manager.ingest_step(new_past)
            else:
                manager.append_token(new_past)

            if want_attn and getattr(outputs, "attentions", None) is not None:
                for li, attn in enumerate(outputs.attentions):
                    manager.report_attention(li, attn)

            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = outputs.past_key_values

        cur_ids = next_tok
        step_times.append(time.perf_counter() - st0)

    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_t0
    return generated, wall_elapsed, step_times


def run_part_b(model_name: str = "Qwen/Qwen2.5-7B",
               seq_len: int = 2048, max_new: int = 128,
               gpu_budget_mb: int = 50,
               n_warmup: int = 1, n_runs: int = 2) -> list[dict]:
    """Part B: end-to-end throughput at different sampling intervals."""
    print(f"\n{'=' * 70}")
    print(f"  Part B: E2E Throughput")
    print(f"  model={model_name}  seq={seq_len}  gen={max_new}  budget={gpu_budget_mb}MB")
    print(f"{'=' * 70}")

    from orchkv.kvcache_manager import KVCacheManager

    attn_impl = "sdpa"
    try:
        model, tokenizer = _load_model(model_name, attn_impl)
        test_ids = tokenizer("test", return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            model(test_ids, output_attentions=True)
        print(f"  attn_implementation = {attn_impl}")
    except Exception as exc:
        print(f"  SDPA failed ({exc}), falling back to eager")
        attn_impl = "eager"
        model, tokenizer = _load_model(model_name, attn_impl)
        print(f"  attn_implementation = {attn_impl}")

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    budget_bytes = gpu_budget_mb * (1 << 20)

    text = ("The transformer architecture uses self-attention to process "
            "sequences. ") * (seq_len // 8)
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=seq_len)["input_ids"].to("cuda:0")
    prompt_len = input_ids.shape[1]
    print(f"  prompt_len={prompt_len}  max_new={max_new}")

    sweep = [0] + SAMPLE_INTERVALS          # 0 = no-sample / GPU-only reference
    results: list[dict] = []
    ref_tokens: list[int] | None = None     # N=1 output for token match rate

    for N in sweep:
        label = f"N={N}" if N > 0 else "no-sample"
        print(f"\n  --- {label} ---")
        gc.collect(); torch.cuda.empty_cache()

        run_thrs: list[float] = []
        run_lats: list[float] = []
        last_stats: dict = {}
        last_tokens: list[int] = []

        for rid in range(n_warmup + n_runs):
            gc.collect(); torch.cuda.empty_cache()

            mgr = KVCacheManager(
                n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim,
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=budget_bytes,
            ) if N > 0 else None

            try:
                tokens, elapsed, step_times = _run_decode(
                    model, input_ids, max_new, mgr, N)
                total_tok = prompt_len + len(tokens)
                thr = total_tok / elapsed

                tag = "warmup" if rid < n_warmup else "RUN"
                print(f"    [{tag}] {thr:.1f} tok/s  {elapsed:.2f}s")

                if rid >= n_warmup:
                    run_thrs.append(thr)
                    run_lats.extend(step_times)
                    last_tokens = tokens
                    if mgr:
                        last_stats = mgr.get_stats()

            except torch.cuda.OutOfMemoryError:
                print(f"    [OOM]")
                gc.collect(); torch.cuda.empty_cache()
                break
            finally:
                if mgr:
                    mgr.destroy()

        if not run_thrs:
            results.append({"sample_interval": N, "label": label, "status": "oom"})
            continue

        if N == 1:
            ref_tokens = last_tokens

        avg_thr = sum(run_thrs) / len(run_thrs)
        avg_step = (sum(run_lats) / len(run_lats)) * 1000 if run_lats else 0

        evict = last_stats.get("migrations", {}).get("gpu_to_dram", 0)
        promo = last_stats.get("migrations", {}).get("dram_to_gpu", 0)

        token_match = 1.0
        if ref_tokens and last_tokens and N != 1:
            n_cmp = min(len(ref_tokens), len(last_tokens))
            matches = sum(1 for a, b in zip(ref_tokens[:n_cmp], last_tokens[:n_cmp])
                          if a == b)
            token_match = matches / max(n_cmp, 1)

        row = {
            "sample_interval": N,
            "label": label,
            "attn_impl": attn_impl,
            "avg_throughput_tok_s": round(avg_thr, 1),
            "avg_step_ms": round(avg_step, 3),
            "evictions": evict,
            "promotions": promo,
            "total_migrations": evict + promo,
            "blocks_gpu": last_stats.get("blocks_gpu", 0),
            "blocks_dram": last_stats.get("blocks_dram", 0),
            "gpu_kv_mb": last_stats.get("gpu_kv_mb", 0),
            "token_match_rate": round(token_match, 4),
            "sampling_overhead_pct": round(100.0 / N, 1) if N > 0 else 0.0,
            "status": "ok",
        }
        results.append(row)
        tmr_str = f"  match={token_match:.2%}" if N > 1 else ""
        print(f"    => AVG {avg_thr:.1f} tok/s  evict={evict}  promo={promo}{tmr_str}")

    n1 = next((r for r in results
               if r["sample_interval"] == 1 and r.get("status") == "ok"), None)
    if n1:
        base = n1["avg_throughput_tok_s"]
        for r in results:
            if r.get("status") == "ok" and base > 0:
                r["speedup_vs_n1"] = round(r["avg_throughput_tok_s"] / base, 3)

    save_json({"results": results, "config": {
        "model": model_name, "attn_impl": attn_impl,
        "seq_len": seq_len, "prompt_len": prompt_len,
        "max_new": max_new, "gpu_budget_mb": gpu_budget_mb,
        "intervals": sweep, "n_warmup": n_warmup, "n_runs": n_runs,
    }}, "exp_e10_sampling_e2e")

    ok_res = [r for r in results if r.get("status") == "ok"]
    if ok_res:
        save_csv(ok_res, "exp_e10_sampling_e2e")

    print(f"\n{'=' * 70}")
    print(f"  E10 Part B  —  SUMMARY")
    print(f"{'=' * 70}")
    hdr = (f"{'Interval':>10} {'Throughput':>12} {'Speedup':>8} "
           f"{'Evict':>7} {'Promo':>7} {'Overhead%':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("status") != "ok":
            print(f"  N={r['sample_interval']:<4}  OOM")
            continue
        su = r.get("speedup_vs_n1", "—")
        print(f"  N={r['sample_interval']:<4}  "
              f"{r['avg_throughput_tok_s']:>10.1f}  "
              f"{su:>8}  "
              f"{r.get('evictions', 0):>7}  "
              f"{r.get('promotions', 0):>7}  "
              f"{r['sampling_overhead_pct']:>8.1f}%")

    del model
    gc.collect(); torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="E10: Attention Sampling Sensitivity")
    ap.add_argument("--part", default="all", choices=["a", "b", "all"],
                    help="a = trace sim only, b = E2E only, all = both")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--gpu-budget-mb", type=int, default=50)
    ap.add_argument("--n-blocks", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("  E10: Attention Sampling Sensitivity")
    print("=" * 70)

    if args.part in ("a", "all"):
        run_part_a(args.n_blocks, args.n_steps, args.n_runs, args.seed)

    if args.part in ("b", "all"):
        run_part_b(args.model, args.seq_len, args.max_new,
                   args.gpu_budget_mb, n_runs=args.n_runs)


if __name__ == "__main__":
    main()

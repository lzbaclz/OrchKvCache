#!/usr/bin/env python3
"""
Main experiment runner for SIGMETRICS 2027 evaluation.

Runs head-to-head baseline comparisons and collects all metrics.
Supports both single-run mode and full-sweep mode.

Single run:
    python -m benchmarks.sigmetrics.run_baseline \\
        --model qwen2.5-7b --workload sharegpt --baseline orchkv \\
        --budget 0.25 --num_prompts 32

Full sweep:
    python -m benchmarks.sigmetrics.run_baseline --sweep \\
        --models qwen2.5-7b llama-3.1-8b \\
        --workloads sharegpt longbench \\
        --baselines gpu_only fifo_offload orchkv
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from benchmarks.sigmetrics.config import (
    MODELS, BASELINES, BUDGET_FRACTIONS, METRIC_KEYS,
    SLO_TARGETS_MS, ExperimentPoint, budget_bytes, budget_mb,
    iter_experiment_grid,
)
from benchmarks.sigmetrics.workload_loader import load_workload

try:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from bench_utils import save_json, CPUTimer, gpu_mem_mb, reset_gpu_peak
except ImportError:
    def save_json(data, name):
        p = RESULTS_DIR / f"{name}.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[run] saved → {p}")
        return p

    class CPUTimer:
        def __init__(self):
            self._start = 0.0
            self._elapsed = []
        def start(self):
            self._start = time.perf_counter_ns()
        def stop(self):
            e = (time.perf_counter_ns() - self._start) / 1e3
            self._elapsed.append(e)
            return e
        @property
        def all_us(self):
            return self._elapsed
        def stats(self):
            if not self._elapsed:
                return {}
            s = sorted(self._elapsed)
            return {"count": len(s), "avg_us": statistics.mean(s),
                    "p50_us": s[len(s)//2], "p99_us": s[int(len(s)*0.99)]}

    def gpu_mem_mb(device=0):
        torch.cuda.synchronize(device)
        return {"allocated_mb": torch.cuda.memory_allocated(device) / (1<<20),
                "reserved_mb": torch.cuda.memory_reserved(device) / (1<<20),
                "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1<<20)}

    def reset_gpu_peak(device=0):
        torch.cuda.reset_peak_memory_stats(device)


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =====================================================================
#  Model loading
# =====================================================================

_MODEL_CACHE: dict[str, tuple] = {}


def load_model(model_key: str, device: str = "cuda:0"):
    """Load and cache a model + tokenizer."""
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_name = MODELS[model_key]["hf_name"]
    print(f"[run] Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    _MODEL_CACHE[model_key] = (model, tokenizer)
    return model, tokenizer


def unload_model(model_key: str):
    if model_key in _MODEL_CACHE:
        del _MODEL_CACHE[model_key]
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
#  Manager factory
# =====================================================================

def create_manager(
    baseline_name: str,
    model_config: dict,
    budget_bytes_val: int,
    max_seq_len: int,
    ssd_dir: str = "/tmp/orchkv_spill",
):
    """Instantiate the KV cache manager for a given baseline config."""
    bcfg = BASELINES[baseline_name]

    if bcfg.manager_cls == "none":
        return None

    base_kwargs = dict(
        n_layers=model_config["n_layers"],
        n_kv_heads=model_config["n_kv_heads"],
        head_dim=model_config["head_dim"],
        block_size=16,
        dtype=torch.float16,
        gpu_budget_bytes=budget_bytes_val,
        max_seq_len=max_seq_len,
    )

    if bcfg.manager_cls == "FastKVCacheManager":
        from orchkv.fast_kvcache_manager import FastKVCacheManager
        return FastKVCacheManager(**base_kwargs)
    elif bcfg.manager_cls == "FastFIFOManager":
        from orchkv.fast_fifo_manager import FastFIFOManager
        return FastFIFOManager(**base_kwargs)
    elif bcfg.manager_cls == "KVCacheManager":
        from orchkv.kvcache_manager import KVCacheManager
        base_kwargs.pop("max_seq_len", None)
        base_kwargs["ssd_dir"] = ssd_dir
        return KVCacheManager(**base_kwargs)
    elif bcfg.manager_cls == "NaiveOffloadManager":
        from orchkv.kvcache_manager import NaiveOffloadManager
        base_kwargs.pop("max_seq_len", None)
        base_kwargs["ssd_dir"] = ssd_dir
        return NaiveOffloadManager(**base_kwargs)
    else:
        raise ValueError(f"Unknown manager class: {bcfg.manager_cls}")


# =====================================================================
#  Single-prompt decode loop
# =====================================================================

def run_single_prompt(
    model,
    tokenizer,
    prompt_text: str,
    manager,
    baseline_config,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run one prompt through the decode loop, collecting all metrics."""
    ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                    max_length=8192)["input_ids"].to(device)
    prompt_len = ids.shape[1]
    bcfg = baseline_config

    reset_gpu_peak()
    itl_times: list[float] = []
    tokens_out: list[int] = []
    ref_tokens: list[int] = []

    gpu_only = (manager is None)
    sample_interval = bcfg.sample_interval
    report_attn = bcfg.report_attn

    cur, past = ids, None

    # Prefill
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(cur, past_key_values=past, use_cache=True,
                    output_attentions=(report_attn and sample_interval > 0))
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t_prefill_start) * 1000.0

    past = out.past_key_values
    cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens_out.append(cur.item())

    if manager is not None:
        manager.ingest_step(out.past_key_values)
        if report_attn and getattr(out, "attentions", None):
            for li, attn in enumerate(out.attentions):
                if bcfg.use_qk_proxy and hasattr(manager, "report_qk_norm"):
                    pass  # QK proxy uses query vectors, handled separately
                else:
                    manager.report_attention(li, attn)
        manager.step_done()
        manager.schedule()
        past = manager.build_past_kv()

    # Decode
    for step in range(1, max_new_tokens):
        want_attn = (report_attn and sample_interval > 0
                     and step % sample_interval == 0)

        torch.cuda.synchronize()
        t_step_start = time.perf_counter()
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=(want_attn and not gpu_only))
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t_step_start) * 1000.0
        itl_times.append(step_ms)

        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_out.append(nt.item())

        if manager is not None:
            manager.append_token(out.past_key_values)
            if want_attn and getattr(out, "attentions", None):
                for li, attn in enumerate(out.attentions):
                    manager.report_attention(li, attn)
            manager.step_done()
            manager.schedule()
            past = manager.build_past_kv()
        else:
            past = out.past_key_values

        cur = nt

    # Collect metrics
    total_tokens = prompt_len + len(tokens_out)
    total_time_s = ttft / 1000.0 + sum(itl_times) / 1000.0
    throughput = total_tokens / max(total_time_s, 1e-9)

    itl_sorted = sorted(itl_times) if itl_times else [0.0]
    n = len(itl_sorted)

    mgr_stats = {}
    if manager is not None:
        try:
            mgr_stats = manager.get_stats()
        except Exception:
            pass

    migrations = mgr_stats.get("migrations", {})

    mem = gpu_mem_mb()

    tpot = statistics.mean(itl_times) if itl_times else 0.0
    itl_p99 = itl_sorted[min(int(n * 0.99), n - 1)]
    goodput = 1.0
    if tpot > SLO_TARGETS_MS["tpot"]:
        goodput *= SLO_TARGETS_MS["tpot"] / tpot
    if ttft > SLO_TARGETS_MS["ttft"]:
        goodput *= SLO_TARGETS_MS["ttft"] / ttft

    promo_stats = {}
    if hasattr(manager, "get_promotion_latency_stats"):
        try:
            promo_stats = manager.get_promotion_latency_stats()
        except Exception:
            pass

    return {
        "prompt_len": prompt_len,
        "generated_tokens": len(tokens_out),
        "ttft_ms": round(ttft, 3),
        "tpot_ms": round(tpot, 3),
        "itl_p50_ms": round(itl_sorted[n // 2], 3),
        "itl_p95_ms": round(itl_sorted[min(int(n * 0.95), n - 1)], 3),
        "itl_p99_ms": round(itl_p99, 3),
        "throughput_tok_s": round(throughput, 1),
        "goodput_under_slo": round(goodput, 4),
        "gpu_mem_used_mb": round(mem["max_allocated_mb"], 1),
        "dram_mem_used_mb": round(mgr_stats.get("dram_kv_mb", 0.0), 1),
        "ssd_traffic_mb": 0.0,
        "evictions": migrations.get("gpu_to_dram", 0),
        "promotions": migrations.get("dram_to_gpu", 0),
        "promotion_stall_count": promo_stats.get("stall_count", 0),
        "promotion_stall_p99_us": promo_stats.get("p99_us", 0.0),
        "tokens_out": tokens_out,
    }


# =====================================================================
#  Experiment runner
# =====================================================================

def run_experiment(point: ExperimentPoint, device: str = "cuda:0") -> dict[str, Any]:
    """Run a full experiment for one ExperimentPoint."""
    print(f"\n{'='*65}")
    print(f"  {point.model} × {point.workload} × {point.baseline} "
          f"× budget={point.budget_fraction:.0%}")
    print(f"{'='*65}")

    model, tokenizer = load_model(point.model, device)
    mcfg = point.model_config
    bcfg = point.baseline_config

    prompts = load_workload(point.workload, num_prompts=point.num_prompts,
                            seed=point.seed)
    if not prompts:
        return {"error": f"No prompts loaded for workload '{point.workload}'",
                "config": point.to_dict()}

    bud = point.budget_bytes
    max_seq = point.seq_len + point.max_new_tokens + 256

    # Run GPU-only reference first for bit-exact comparison
    ref_tokens_map: dict[int, list[int]] = {}
    if point.baseline != "gpu_only":
        print(f"[run] Collecting GPU-only reference for bit-exact comparison...")
        for pi, p in enumerate(prompts[:min(8, len(prompts))]):
            try:
                ref = run_single_prompt(model, tokenizer, p["prompt"],
                                         manager=None,
                                         baseline_config=BASELINES["gpu_only"],
                                         max_new_tokens=point.max_new_tokens,
                                         device=device)
                ref_tokens_map[pi] = ref["tokens_out"]
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                gc.collect(); torch.cuda.empty_cache()
                break

    per_prompt_results: list[dict] = []
    n_completed = 0
    n_oom = 0

    for pi, prompt_data in enumerate(prompts):
        manager = None
        try:
            manager = create_manager(point.baseline, mcfg, bud, max_seq)
            result = run_single_prompt(
                model, tokenizer, prompt_data["prompt"],
                manager=manager,
                baseline_config=bcfg,
                max_new_tokens=point.max_new_tokens,
                device=device,
            )

            if pi in ref_tokens_map:
                ref = ref_tokens_map[pi]
                gen = result["tokens_out"]
                match_len = min(len(ref), len(gen))
                matches = sum(a == b for a, b in zip(ref[:match_len], gen[:match_len]))
                result["bit_exact_match"] = round(matches / max(match_len, 1), 4)
            else:
                result["bit_exact_match"] = -1.0

            del result["tokens_out"]
            result["prompt_idx"] = pi
            result["status"] = "OK"
            per_prompt_results.append(result)
            n_completed += 1

            if (pi + 1) % 8 == 0 or pi == len(prompts) - 1:
                _print_progress(pi + 1, len(prompts), result)

        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            per_prompt_results.append({"prompt_idx": pi, "status": "OOM"})
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            per_prompt_results.append({"prompt_idx": pi, "status": f"ERROR: {exc}"})
        finally:
            if manager is not None:
                try:
                    manager.destroy()
                except Exception:
                    pass
            gc.collect(); torch.cuda.empty_cache()

    summary = _summarize_results(per_prompt_results, point)
    summary["n_completed"] = n_completed
    summary["n_oom"] = n_oom
    summary["n_total"] = len(prompts)

    return summary


def _print_progress(done: int, total: int, last_result: dict):
    thr = last_result.get("throughput_tok_s", 0)
    tpot = last_result.get("tpot_ms", 0)
    evict = last_result.get("evictions", 0)
    print(f"  [{done:>3d}/{total}] thr={thr:.0f} tok/s  tpot={tpot:.1f}ms  evict={evict}")


def _summarize_results(
    per_prompt: list[dict], point: ExperimentPoint,
) -> dict[str, Any]:
    """Aggregate per-prompt results into a summary."""
    ok = [r for r in per_prompt if r.get("status") == "OK"]
    if not ok:
        return {"config": point.to_dict(), "per_prompt": per_prompt, "error": "all failed"}

    def _agg(key):
        vals = [r[key] for r in ok if key in r and isinstance(r[key], (int, float))]
        if not vals:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}
        vals.sort()
        n = len(vals)
        return {
            "mean": round(statistics.mean(vals), 3),
            "p50": round(vals[n // 2], 3),
            "p95": round(vals[min(int(n * 0.95), n - 1)], 3),
            "p99": round(vals[min(int(n * 0.99), n - 1)], 3),
        }

    summary = {
        "config": point.to_dict(),
        "ttft": _agg("ttft_ms"),
        "tpot": _agg("tpot_ms"),
        "itl_p99": _agg("itl_p99_ms"),
        "throughput": _agg("throughput_tok_s"),
        "goodput_under_slo": _agg("goodput_under_slo"),
        "gpu_mem": _agg("gpu_mem_used_mb"),
        "evictions": _agg("evictions"),
        "promotions": _agg("promotions"),
        "bit_exact_match": _agg("bit_exact_match"),
        "per_prompt": per_prompt,
    }
    return summary


# =====================================================================
#  Sweep mode
# =====================================================================

def run_sweep(
    models: list[str],
    workloads: list[str],
    baselines: list[str],
    budgets: list[float],
    num_prompts: int = 32,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    output_tag: str = "",
) -> list[dict]:
    """Run the full experiment grid and save results."""
    from benchmarks.sigmetrics.config import grid_size

    total = grid_size(models, workloads, baselines, budgets)
    print(f"\n[sweep] Total experiment points: {total}")
    print(f"[sweep] Models:    {models}")
    print(f"[sweep] Workloads: {workloads}")
    print(f"[sweep] Baselines: {baselines}")
    print(f"[sweep] Budgets:   {budgets}")

    all_results = []
    done = 0
    t0 = time.time()

    for point in iter_experiment_grid(models, workloads, baselines, budgets):
        point.num_prompts = num_prompts
        point.max_new_tokens = max_new_tokens
        done += 1
        print(f"\n[sweep] === Point {done}/{total} ===")

        result = run_experiment(point, device)
        all_results.append(result)

        tag = output_tag or "sigmetrics_sweep"
        save_json(all_results, f"{tag}_partial")

    elapsed_min = (time.time() - t0) / 60
    print(f"\n[sweep] Completed {done} points in {elapsed_min:.1f} minutes")

    tag = output_tag or "sigmetrics_sweep"
    save_json(all_results, tag)

    _print_sweep_summary(all_results)
    return all_results


def _print_sweep_summary(results: list[dict]):
    """Print a compact summary table of sweep results."""
    print(f"\n{'='*80}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*80}")
    header = (f"  {'Model':<15s} {'Workload':<12s} {'Baseline':<18s} "
              f"{'Budget':>7s} {'Thr':>8s} {'TPOT':>7s} {'Evict':>7s} {'Match':>6s}")
    print(header)
    print(f"  {'-'*76}")

    for r in results:
        cfg = r.get("config", {})
        thr = r.get("throughput", {}).get("mean", 0)
        tpot = r.get("tpot", {}).get("mean", 0)
        evict = r.get("evictions", {}).get("mean", 0)
        match = r.get("bit_exact_match", {}).get("mean", -1)
        bud = f"{cfg.get('budget_fraction', 0):.0%}"
        match_str = f"{match:.2f}" if match >= 0 else "N/A"
        print(f"  {cfg.get('model', '?'):<15s} {cfg.get('workload', '?'):<12s} "
              f"{cfg.get('baseline', '?'):<18s} {bud:>7s} "
              f"{thr:>7.0f}t {tpot:>6.1f}ms {evict:>7.0f} {match_str:>6s}")


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SIGMETRICS 2027 baseline comparison runner")

    # Single-run args
    parser.add_argument("--model", type=str, default="qwen2.5-7b",
                        choices=list(MODELS))
    parser.add_argument("--workload", type=str, default="sharegpt")
    parser.add_argument("--baseline", type=str, default="orchkv",
                        choices=list(BASELINES))
    parser.add_argument("--budget", type=float, default=0.25,
                        help="Budget fraction (0.05-0.75)")
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run full experiment grid")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models for sweep (default: all)")
    parser.add_argument("--workloads", nargs="+", default=None,
                        help="Workloads for sweep (default: all)")
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="Baselines for sweep (default: all)")
    parser.add_argument("--budgets", nargs="+", type=float, default=None,
                        help="Budget fractions for sweep (default: all)")
    parser.add_argument("--tag", type=str, default="",
                        help="Output file tag")

    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick sweep: 1 model, 2 workloads, 3 baselines, 2 budgets")

    args = parser.parse_args()

    if args.quick:
        args.sweep = True
        args.models = args.models or ["qwen2.5-7b"]
        args.workloads = args.workloads or ["sharegpt", "ruler"]
        args.baselines = args.baselines or ["gpu_only", "fifo_offload", "orchkv"]
        args.budgets = args.budgets or [0.25, 0.50]
        args.num_prompts = min(args.num_prompts, 8)
        args.max_new_tokens = min(args.max_new_tokens, 64)
        args.tag = args.tag or "sigmetrics_quick"

    if args.sweep:
        run_sweep(
            models=args.models or list(MODELS),
            workloads=args.workloads or list(WORKLOADS),
            baselines=args.baselines or list(BASELINES),
            budgets=args.budgets or BUDGET_FRACTIONS,
            num_prompts=args.num_prompts,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            output_tag=args.tag,
        )
    else:
        point = ExperimentPoint(
            model=args.model,
            workload=args.workload,
            baseline=args.baseline,
            budget_fraction=args.budget,
            num_prompts=args.num_prompts,
            max_new_tokens=args.max_new_tokens,
            tag=args.tag,
        )
        result = run_experiment(point, args.device)
        name = point.result_name()
        save_json(result, f"sigmetrics_{name}")

        cfg = result.get("config", {})
        thr = result.get("throughput", {}).get("mean", 0)
        tpot = result.get("tpot", {}).get("mean", 0)
        evict = result.get("evictions", {}).get("mean", 0)
        print(f"\n[run] Done: thr={thr:.0f} tok/s  tpot={tpot:.1f}ms  evict={evict:.0f}")


if __name__ == "__main__":
    main()

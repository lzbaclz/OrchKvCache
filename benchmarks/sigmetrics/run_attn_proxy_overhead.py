#!/usr/bin/env python3
"""
Attention proxy overhead measurement and workload characterization.

Measures the cost of different importance-signal acquisition strategies:
  1. Full attention extraction (output_attentions=True vs False)
  2. N-step periodic sampling (N=1,5,10,20)
  3. QK proxy (Q·K_max per block without full softmax)
  4. Workload characterization (Gini, Jaccard, concentration, reuse distance)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

from orchkv.reuse_distance import (
    WorkloadCharacterizer,
    SignalAcquisition,
    SignalMode,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BLOCK_SIZE = 64


def load_model(model_path: str, device: str = "cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[load] Model loaded. Layers={model.config.num_hidden_layers}, "
          f"KV heads={model.config.num_key_value_heads}, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")
    return model, tokenizer


def make_prompt_ids(tokenizer, prompt_len: int, device: str = "cuda:0"):
    """Create a prompt of approximately `prompt_len` tokens."""
    text = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 8 + 1)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=prompt_len)["input_ids"].to(device)
    return ids


def timed_decode_loop(
    model, input_ids, gen_len: int, output_attentions: bool = False,
    collect_attentions: bool = False,
):
    """Run prefill + decode, return per-step times and optionally attention weights.

    Returns:
        step_times_ms: list of per-decode-step times in ms
        all_attentions: list of attention tuples (one per collected step), or empty
        prefill_time_ms: float
    """
    device = input_ids.device
    step_times_ms = []
    all_attentions = []

    with torch.no_grad():
        # Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True, output_attentions=output_attentions)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        past = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(dim=-1)

        if collect_attentions and output_attentions and out.attentions is not None:
            all_attentions.append(out.attentions)

        # Decode
        for step in range(gen_len):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=output_attentions)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            step_times_ms.append(elapsed_ms)

            past = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)

            if collect_attentions and output_attentions and out.attentions is not None:
                all_attentions.append(out.attentions)

    return step_times_ms, all_attentions, prefill_ms


def timed_decode_periodic(
    model, input_ids, gen_len: int, sample_interval: int,
):
    """Decode collecting attention only every `sample_interval` steps.

    Returns per-step times and collected attention steps.
    """
    device = input_ids.device
    step_times_ms = []
    collected_attentions = []
    collected_steps = []

    with torch.no_grad():
        # Prefill (always collect attention for initial state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True, output_attentions=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        past = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(dim=-1)
        if out.attentions is not None:
            collected_attentions.append(out.attentions)
            collected_steps.append(-1)

        # Decode
        for step in range(gen_len):
            want_attn = (step % sample_interval == 0)

            torch.cuda.synchronize()
            t_start = time.perf_counter()
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=want_attn)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            step_times_ms.append(elapsed_ms)

            past = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)

            if want_attn and out.attentions is not None:
                collected_attentions.append(out.attentions)
                collected_steps.append(step)

    return step_times_ms, collected_attentions, collected_steps, prefill_ms


def extract_block_importance_from_attn(attentions, block_size: int = BLOCK_SIZE):
    """Given a tuple of layer attentions, compute per-block importance.

    attentions: tuple of [B, H, seq_q, seq_kv] tensors (one per layer)
    Returns: [num_blocks] tensor averaged across layers and heads.
    """
    importances = []
    for layer_attn in attentions:
        # layer_attn: [B, H, Q, KV] - take last query position
        attn = layer_attn[:, :, -1:, :]  # [B, H, 1, KV]
        B, H, _, S = attn.shape
        n_blocks = math.ceil(S / block_size)
        padded = n_blocks * block_size
        if S < padded:
            attn = F.pad(attn, (0, padded - S), value=0.0)
        attn_blocks = attn.view(B, H, 1, n_blocks, block_size)
        block_imp = attn_blocks.max(dim=-1).values.squeeze(2)  # [B, H, n_blocks]
        block_imp = block_imp.mean(dim=(0, 1))  # [n_blocks]
        importances.append(block_imp)
    # Average across layers
    stacked = torch.stack(importances, dim=0)  # [L, n_blocks]
    return stacked.mean(dim=0)  # [n_blocks]


def classify_blocks(importance: torch.Tensor, hot_frac=0.2, warm_frac=0.3):
    """Classify blocks into hot/warm/cold based on importance ranking."""
    n = importance.shape[0]
    n_hot = max(1, int(hot_frac * n))
    n_warm = max(1, int(warm_frac * n))
    ranked = importance.argsort(descending=True)
    labels = torch.full((n,), 2, dtype=torch.long)  # cold=2
    labels[ranked[:n_hot]] = 0  # hot=0
    labels[ranked[n_hot:n_hot + n_warm]] = 1  # warm=1
    return labels


# =====================================================================
# Experiment 1: Full Attention Extraction Overhead
# =====================================================================

def experiment_attention_overhead(model, tokenizer, prompt_len, gen_len, device):
    print("\n" + "=" * 70)
    print("  Experiment 1: Full Attention Extraction Overhead")
    print("=" * 70)

    input_ids = make_prompt_ids(tokenizer, prompt_len, device)
    actual_prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {actual_prompt_len}, Decode steps: {gen_len}")

    # Warmup
    print("  [warmup] Running 2 warmup iterations...")
    for _ in range(2):
        timed_decode_loop(model, input_ids, gen_len=4, output_attentions=False)
        torch.cuda.empty_cache()

    # Without attention
    print("  [run] Decoding WITHOUT output_attentions...")
    times_no_attn, _, pf_no = timed_decode_loop(
        model, input_ids, gen_len, output_attentions=False)
    torch.cuda.empty_cache()
    gc.collect()

    # With attention
    print("  [run] Decoding WITH output_attentions=True...")
    times_with_attn, _, pf_with = timed_decode_loop(
        model, input_ids, gen_len, output_attentions=True)
    torch.cuda.empty_cache()
    gc.collect()

    tpot_no = sum(times_no_attn) / len(times_no_attn)
    tpot_with = sum(times_with_attn) / len(times_with_attn)
    overhead_ms = tpot_with - tpot_no
    overhead_pct = 100.0 * overhead_ms / tpot_no if tpot_no > 0 else 0.0

    result = {
        "prompt_len": actual_prompt_len,
        "gen_len": gen_len,
        "tpot_no_attn_ms": round(tpot_no, 3),
        "tpot_with_attn_ms": round(tpot_with, 3),
        "overhead_ms": round(overhead_ms, 3),
        "overhead_pct": round(overhead_pct, 2),
        "prefill_no_attn_ms": round(pf_no, 2),
        "prefill_with_attn_ms": round(pf_with, 2),
        "per_step_no_attn": [round(t, 3) for t in times_no_attn],
        "per_step_with_attn": [round(t, 3) for t in times_with_attn],
    }

    print(f"\n  Results:")
    print(f"    TPOT (no attn):   {tpot_no:.3f} ms")
    print(f"    TPOT (with attn): {tpot_with:.3f} ms")
    print(f"    Overhead:         {overhead_ms:.3f} ms ({overhead_pct:.1f}%)")
    print(f"    Prefill (no):     {pf_no:.2f} ms")
    print(f"    Prefill (with):   {pf_with:.2f} ms")

    return result


# =====================================================================
# Experiment 2: N-step Sampling Overhead
# =====================================================================

def experiment_nstep_sampling(model, tokenizer, prompt_len, gen_len, device):
    print("\n" + "=" * 70)
    print("  Experiment 2: N-step Sampling Overhead")
    print("=" * 70)

    input_ids = make_prompt_ids(tokenizer, prompt_len, device)
    actual_prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {actual_prompt_len}, Decode steps: {gen_len}")

    # First run the oracle (full attention every step) to get ground-truth labels
    print("  [oracle] Collecting full attention every step...")
    oracle_times, oracle_attns, oracle_steps, _ = timed_decode_periodic(
        model, input_ids, gen_len, sample_interval=1)
    torch.cuda.empty_cache()
    gc.collect()

    # Compute oracle block labels at each step
    oracle_labels_per_step = []
    for attn_tuple in oracle_attns[1:]:  # skip prefill
        imp = extract_block_importance_from_attn(attn_tuple)
        labels = classify_blocks(imp)
        oracle_labels_per_step.append(labels)

    intervals = [1, 5, 10, 20]
    results = {}

    for N in intervals:
        print(f"  [N={N}] Running periodic sampling...")
        times, collected_attns, collected_steps, _ = timed_decode_periodic(
            model, input_ids, gen_len, sample_interval=N)
        torch.cuda.empty_cache()
        gc.collect()

        tpot = sum(times) / len(times)

        # Compute classification accuracy vs oracle
        # For non-collection steps, we reuse the last collected attention
        last_labels = None
        matches = 0
        total = 0
        for step_idx in range(gen_len):
            # Find the last collected attention at or before this step
            attn_idx = None
            for ci, cs in enumerate(collected_steps):
                if cs == -1:
                    continue  # prefill
                if cs <= step_idx:
                    attn_idx = ci
            if attn_idx is not None and attn_idx < len(collected_attns):
                imp = extract_block_importance_from_attn(collected_attns[attn_idx])
                last_labels = classify_blocks(imp)

            if last_labels is not None and step_idx < len(oracle_labels_per_step):
                oracle_lab = oracle_labels_per_step[step_idx]
                n_blocks = min(last_labels.shape[0], oracle_lab.shape[0])
                matches += (last_labels[:n_blocks] == oracle_lab[:n_blocks]).sum().item()
                total += n_blocks

        accuracy = matches / total if total > 0 else 0.0

        results[f"N={N}"] = {
            "sample_interval": N,
            "tpot_ms": round(tpot, 3),
            "classification_accuracy": round(accuracy, 4),
            "n_collections": len([s for s in collected_steps if s >= 0]),
        }
        print(f"    TPOT: {tpot:.3f} ms, Classification accuracy: {accuracy:.4f}")

    # Build summary table
    print(f"\n  {'N':>4s} | {'TPOT (ms)':>10s} | {'Accuracy':>10s} | {'Collections':>12s}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for N in intervals:
        r = results[f"N={N}"]
        print(f"  {N:>4d} | {r['tpot_ms']:>10.3f} | {r['classification_accuracy']:>10.4f} | {r['n_collections']:>12d}")

    return results


# =====================================================================
# Experiment 3: QK Proxy Overhead
# =====================================================================

def experiment_qk_proxy(model, tokenizer, prompt_len, gen_len, device):
    print("\n" + "=" * 70)
    print("  Experiment 3: QK Proxy Overhead")
    print("=" * 70)

    input_ids = make_prompt_ids(tokenizer, prompt_len, device)
    actual_prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {actual_prompt_len}, Decode steps: {gen_len}")

    # Baseline: no signal (just decode)
    print("  [baseline] Decoding with no signal collection...")
    times_baseline, _, pf_base = timed_decode_loop(
        model, input_ids, gen_len, output_attentions=False)
    tpot_baseline = sum(times_baseline) / len(times_baseline)
    torch.cuda.empty_cache()
    gc.collect()

    # QK proxy: decode without attention, but compute Q·K_max after each step
    print("  [qk_proxy] Decoding with QK proxy computation...")
    step_times_proxy = []
    signal_acq = SignalAcquisition(mode=SignalMode.QK_PROXY, proxy_top_k=4)

    with torch.no_grad():
        # Prefill
        out = model(input_ids, use_cache=True, output_attentions=False)
        past = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(dim=-1)

        # Extract key cache shape info from past_key_values
        # past is a tuple of (key, value) per layer
        n_layers = len(past)
        sample_k = past[0][0]  # [B, H, S, D]
        B, H, S, D = sample_k.shape

        for step in range(gen_len):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=False)

            # QK proxy: compute importance using last query from model internals
            # We approximate by using the current token embedding projected through
            # the first layer's Q projection as a proxy query
            # Instead, use the key cache directly with a synthetic query direction
            past_new = out.past_key_values
            key_cache = past_new[0][0]  # [B, H, S_new, D]
            # Use the last key as a rough query proxy (cheap)
            proxy_query = key_cache[:, :, -1:, :]  # [B, H, 1, D]
            _ = signal_acq.compute_block_importance(
                proxy_query, key_cache, block_size=BLOCK_SIZE
            )

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            step_times_proxy.append(elapsed_ms)

            past = past_new
            cur = out.logits[:, -1:, :].argmax(dim=-1)

    tpot_proxy = sum(step_times_proxy) / len(step_times_proxy)
    proxy_overhead_ms = tpot_proxy - tpot_baseline
    proxy_overhead_pct = 100.0 * proxy_overhead_ms / tpot_baseline if tpot_baseline > 0 else 0

    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "tpot_baseline_ms": round(tpot_baseline, 3),
        "tpot_qk_proxy_ms": round(tpot_proxy, 3),
        "qk_proxy_overhead_ms": round(proxy_overhead_ms, 3),
        "qk_proxy_overhead_pct": round(proxy_overhead_pct, 2),
        "n_blocks": math.ceil(S / BLOCK_SIZE),
        "per_step_baseline": [round(t, 3) for t in times_baseline],
        "per_step_qk_proxy": [round(t, 3) for t in step_times_proxy],
    }

    print(f"\n  Results:")
    print(f"    TPOT (baseline):  {tpot_baseline:.3f} ms")
    print(f"    TPOT (QK proxy):  {tpot_proxy:.3f} ms")
    print(f"    Overhead:         {proxy_overhead_ms:.3f} ms ({proxy_overhead_pct:.1f}%)")
    print(f"    Blocks:           {math.ceil(S / BLOCK_SIZE)}")

    return result


# =====================================================================
# Experiment 4: Workload Characterization
# =====================================================================

def experiment_workload_characterization(model, tokenizer, prompt_len, gen_len, device):
    print("\n" + "=" * 70)
    print("  Experiment 4: Workload Characterization")
    print("=" * 70)

    input_ids = make_prompt_ids(tokenizer, prompt_len, device)
    actual_prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {actual_prompt_len}, Decode steps: {gen_len}")

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_blocks_est = math.ceil(actual_prompt_len / BLOCK_SIZE)

    characterizer = WorkloadCharacterizer(
        num_blocks=n_blocks_est, num_layers=n_layers, num_heads=n_kv_heads
    )

    print("  [run] Collecting full attention for characterization...")
    all_block_importances = []
    per_layer_gini = [[] for _ in range(n_layers)]
    concentration_scores = []  # top-10% coverage

    with torch.no_grad():
        # Prefill
        out = model(input_ids, use_cache=True, output_attentions=True)
        past = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(dim=-1)

        if out.attentions is not None:
            imp = extract_block_importance_from_attn(out.attentions)
            characterizer.observe_step(imp)
            all_block_importances.append(imp)

        # Decode
        for step in range(gen_len):
            out = model(cur, past_key_values=past, use_cache=True,
                        output_attentions=True)
            past = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)

            if out.attentions is not None:
                imp = extract_block_importance_from_attn(out.attentions)
                characterizer.observe_step(imp)
                all_block_importances.append(imp)

                # Per-layer Gini
                for li, layer_attn in enumerate(out.attentions):
                    attn = layer_attn[:, :, -1:, :]  # [B, H, 1, S]
                    B, H2, _, S = attn.shape
                    nb = math.ceil(S / BLOCK_SIZE)
                    padded = nb * BLOCK_SIZE
                    if S < padded:
                        attn = F.pad(attn, (0, padded - S), value=0.0)
                    attn_blocks = attn.view(B, H2, 1, nb, BLOCK_SIZE)
                    layer_block_imp = attn_blocks.max(dim=-1).values.squeeze(2).mean(dim=(0, 1))
                    gini = WorkloadCharacterizer.gini_coefficient(layer_block_imp)
                    per_layer_gini[li].append(gini)

                # Attention concentration: top-10% blocks cover what % of total
                sorted_imp, _ = imp.sort(descending=True)
                n_top = max(1, int(0.1 * imp.shape[0]))
                top_sum = sorted_imp[:n_top].sum().item()
                total_sum = sorted_imp.sum().item()
                concentration = top_sum / total_sum if total_sum > 0 else 0.0
                concentration_scores.append(concentration)

            if (step + 1) % 16 == 0:
                print(f"    Step {step + 1}/{gen_len} done")

    torch.cuda.empty_cache()
    gc.collect()

    # Compute per-block reuse distances
    n_blocks = all_block_importances[0].shape[0] if all_block_importances else n_blocks_est
    criticality_threshold = 0.02
    last_critical = [-1] * n_blocks
    reuse_distances = []

    for step_idx, imp in enumerate(all_block_importances):
        for bid in range(min(n_blocks, imp.shape[0])):
            if imp[bid].item() > criticality_threshold:
                if last_critical[bid] >= 0:
                    rd = step_idx - last_critical[bid]
                    reuse_distances.append(rd)
                last_critical[bid] = step_idx

    characterizer.observe_reuse_distances(reuse_distances)

    # Hot-set Jaccard stability
    summary = characterizer.summary()

    # Per-layer Gini stats
    layer_gini_stats = []
    for li in range(n_layers):
        if per_layer_gini[li]:
            g = torch.tensor(per_layer_gini[li])
            layer_gini_stats.append({
                "layer": li,
                "gini_mean": round(g.mean().item(), 4),
                "gini_std": round(g.std().item(), 4),
            })

    # Concentration stats
    conc_t = torch.tensor(concentration_scores) if concentration_scores else torch.zeros(1)

    result = {
        "prompt_len": actual_prompt_len,
        "gen_len": gen_len,
        "n_blocks": n_blocks,
        "block_size": BLOCK_SIZE,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "gini_coefficient": {
            "mean": round(summary["attention_concentration"]["gini_mean"], 4),
            "std": round(summary["attention_concentration"]["gini_std"], 4),
            "median": round(summary["attention_concentration"]["gini_p50"], 4),
        },
        "hot_set_jaccard_stability": {
            "mean": round(summary["hot_set_stability"]["jaccard_mean"], 4),
            "std": round(summary["hot_set_stability"]["jaccard_std"], 4),
        },
        "attention_concentration_top10pct": {
            "mean": round(conc_t.mean().item(), 4),
            "std": round(conc_t.std().item(), 4),
            "min": round(conc_t.min().item(), 4),
            "max": round(conc_t.max().item(), 4),
        },
        "reuse_distance": summary["reuse_distance"],
        "reuse_distance_samples": len(reuse_distances),
        "per_layer_gini": layer_gini_stats,
        "gini_trace": summary.get("gini_trace", characterizer._gini_history),
        "jaccard_trace": characterizer._jaccard_history,
    }

    print(f"\n  Results:")
    print(f"    Gini (attention concentration): {result['gini_coefficient']['mean']:.4f} "
          f"± {result['gini_coefficient']['std']:.4f}")
    print(f"    Hot-set Jaccard stability:      {result['hot_set_jaccard_stability']['mean']:.4f} "
          f"± {result['hot_set_jaccard_stability']['std']:.4f}")
    print(f"    Top-10% concentration:          {result['attention_concentration_top10pct']['mean']:.4f}")
    print(f"    Reuse distance (mean):          {result['reuse_distance']['mean']:.2f} steps")
    print(f"    Reuse distance (p90):           {result['reuse_distance']['p90']:.2f} steps")
    print(f"    Reuse distance samples:         {result['reuse_distance_samples']}")

    # Per-layer Gini table (show first/last few)
    if layer_gini_stats:
        print(f"\n    Per-layer Gini (first 5 + last 5):")
        show = layer_gini_stats[:5] + layer_gini_stats[-5:]
        for s in show:
            print(f"      Layer {s['layer']:>2d}: {s['gini_mean']:.4f} ± {s['gini_std']:.4f}")

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attention proxy overhead measurement & workload characterization")
    parser.add_argument("--model", type=str, default="/public/model_zoo/Qwen2.5-7B")
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 70)
    print("  Attention Proxy Overhead & Workload Characterization")
    print(f"  Model: {args.model}")
    print(f"  Prompt len: {args.prompt_len}, Gen len: {args.gen_len}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    model, tokenizer = load_model(args.model, args.device)

    # Run experiments
    results = {}

    results["attention_overhead"] = experiment_attention_overhead(
        model, tokenizer, args.prompt_len, args.gen_len, args.device)

    results["nstep_sampling"] = experiment_nstep_sampling(
        model, tokenizer, args.prompt_len, args.gen_len, args.device)

    results["qk_proxy"] = experiment_qk_proxy(
        model, tokenizer, args.prompt_len, args.gen_len, args.device)

    workload_result = experiment_workload_characterization(
        model, tokenizer, args.prompt_len, args.gen_len, args.device)

    # Save results
    results["metadata"] = {
        "model": args.model,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "device": args.device,
        "block_size": BLOCK_SIZE,
    }

    overhead_path = RESULTS_DIR / "attn_proxy_overhead.json"
    with open(overhead_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[save] Overhead results → {overhead_path}")

    char_path = RESULTS_DIR / "workload_characterization.json"
    with open(char_path, "w") as f:
        json.dump(workload_result, f, indent=2, default=str)
    print(f"[save] Characterization → {char_path}")

    # Final summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    r1 = results["attention_overhead"]
    print(f"\n  1. Full Attention Extraction:")
    print(f"     TPOT baseline:    {r1['tpot_no_attn_ms']:.3f} ms")
    print(f"     TPOT w/ attn:     {r1['tpot_with_attn_ms']:.3f} ms")
    print(f"     Overhead:         +{r1['overhead_ms']:.3f} ms ({r1['overhead_pct']:.1f}%)")

    print(f"\n  2. N-step Sampling:")
    print(f"     {'N':>4s} | {'TPOT (ms)':>10s} | {'Accuracy':>10s}")
    print(f"     {'-'*4}-+-{'-'*10}-+-{'-'*10}")
    for key in sorted(results["nstep_sampling"].keys()):
        r = results["nstep_sampling"][key]
        print(f"     {r['sample_interval']:>4d} | {r['tpot_ms']:>10.3f} | {r['classification_accuracy']:>10.4f}")

    r3 = results["qk_proxy"]
    print(f"\n  3. QK Proxy:")
    print(f"     TPOT baseline:    {r3['tpot_baseline_ms']:.3f} ms")
    print(f"     TPOT w/ proxy:    {r3['tpot_qk_proxy_ms']:.3f} ms")
    print(f"     Overhead:         +{r3['qk_proxy_overhead_ms']:.3f} ms ({r3['qk_proxy_overhead_pct']:.1f}%)")

    print(f"\n  4. Workload Characterization:")
    print(f"     Gini coefficient: {workload_result['gini_coefficient']['mean']:.4f}")
    print(f"     Jaccard stability:{workload_result['hot_set_jaccard_stability']['mean']:.4f}")
    print(f"     Top-10% coverage: {workload_result['attention_concentration_top10pct']['mean']:.4f}")
    print(f"     Reuse dist mean:  {workload_result['reuse_distance']['mean']:.2f} steps")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()

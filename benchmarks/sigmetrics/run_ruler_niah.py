#!/usr/bin/env python3
"""
RULER-style Needle-In-A-Haystack (NIAH) benchmark for SIGMETRICS 2027.

Generates synthetic NIAH prompts at controlled lengths, embeds a secret
needle at a random position, then checks whether the model correctly
retrieves the needle under:
  a) GPU-only baseline (full KV on GPU)
  b) OrchKvCache at 50% GPU budget

Reports: accuracy, throughput (tok/s), TPOT, eviction/promotion stats.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n orchkv \
        PYTHONPATH=build/bindings:python \
        python benchmarks/sigmetrics/run_ruler_niah.py
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orchkv.kvcache_manager import KVCacheManager

MODEL_PATH = "/public/model_zoo/Qwen2.5-7B"
RESULTS_DIR = ROOT / "benchmarks" / "sigmetrics" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "ruler_niah.json"

PROMPT_LENGTHS = [1024, 2048, 4096]
NUM_PROMPTS_PER_LENGTH = 20
MAX_NEW_TOKENS = 32
BUDGET_FRACTION = 0.50
ATTN_SAMPLE_INTERVAL = 5
SEED = 42

NEEDLE_TEXT = "The secret number is 42."
QUESTION = "\nQuestion: What is the secret number mentioned in the passage above?\nAnswer: The secret number is"

FILLER_SENTENCE = (
    "The history of distributed systems spans decades of innovation in "
    "fault tolerance, consensus algorithms, and replicated state machines. "
    "Modern cloud-native architectures leverage micro-service orchestration "
    "and container-based deployments across geographically distributed data "
    "centers. The CAP theorem fundamentally constrains the design space of "
    "distributed databases, forcing engineers to choose between consistency "
    "and availability during network partitions. "
)


def generate_niah_prompt(tokenizer, target_len: int, rng: random.Random) -> dict:
    """Generate a single NIAH prompt with the needle at a random position."""
    needle_tokens = len(tokenizer.encode(NEEDLE_TEXT, add_special_tokens=False))
    question_tokens = len(tokenizer.encode(QUESTION, add_special_tokens=False))
    filler_budget = target_len - needle_tokens - question_tokens - 10

    filler_tokens = tokenizer.encode(FILLER_SENTENCE, add_special_tokens=False)
    filler_per_repeat = len(filler_tokens)
    n_repeats = max(1, filler_budget // filler_per_repeat)

    filler_blocks = [FILLER_SENTENCE] * n_repeats
    insert_pos = rng.randint(1, max(1, len(filler_blocks) - 1))
    filler_blocks.insert(insert_pos, f"\n{NEEDLE_TEXT}\n")

    full_text = "".join(filler_blocks) + QUESTION
    input_ids = tokenizer.encode(full_text, add_special_tokens=True)

    if len(input_ids) > target_len + 64:
        input_ids = input_ids[:target_len]
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    needle_frac = insert_pos / max(len(filler_blocks), 1)

    return {
        "prompt": full_text,
        "target_len": target_len,
        "actual_len": len(input_ids),
        "needle_position_frac": round(needle_frac, 3),
        "expected_answer": "42",
    }


def check_accuracy(generated_text: str) -> bool:
    """Check if the generated text contains the correct needle answer."""
    return "42" in generated_text


def run_baseline_decode(model, tokenizer, input_ids: torch.Tensor) -> dict:
    """GPU-only baseline: full KV on GPU, no offloading."""
    t0 = time.time()
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv, use_cache=True,
                        output_attentions=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        past_kv = out.past_key_values
        cur_ids = next_token
        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.time() - t0
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    n_gen = len(generated)

    return {
        "generated_text": gen_text,
        "n_tokens": n_gen,
        "elapsed_s": round(elapsed, 4),
        "throughput_tok_s": round(n_gen / max(elapsed, 1e-9), 2),
        "tpot_ms": round((elapsed * 1000) / max(n_gen, 1), 2),
        "accurate": check_accuracy(gen_text),
    }


def run_orchkv_decode(model, tokenizer, input_ids: torch.Tensor,
                      gpu_budget_bytes: int) -> dict:
    """OrchKvCache decode with attention-aware offloading."""
    cfg = model.config
    mgr = KVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16,
        dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_bytes,
        sink_tokens=4,
    )

    t0 = time.time()
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(MAX_NEW_TOKENS):
        want_attn = (step % ATTN_SAMPLE_INTERVAL == 0)
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv, use_cache=True,
                        output_attentions=want_attn)

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        new_past = out.past_key_values
        if step == 0:
            mgr.ingest_step(new_past)
        else:
            mgr.append_token(new_past)

        if out.attentions is not None:
            for li, attn in enumerate(out.attentions):
                mgr.report_attention(li, attn)

        mgr.step_done()
        sched = mgr.schedule()
        past_kv = mgr.build_past_kv()
        cur_ids = next_token

        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.time() - t0
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    n_gen = len(generated)
    stats = mgr.get_stats()

    result = {
        "generated_text": gen_text,
        "n_tokens": n_gen,
        "elapsed_s": round(elapsed, 4),
        "throughput_tok_s": round(n_gen / max(elapsed, 1e-9), 2),
        "tpot_ms": round((elapsed * 1000) / max(n_gen, 1), 2),
        "accurate": check_accuracy(gen_text),
        "evictions": stats["migrations"]["gpu_to_dram"],
        "promotions": stats["migrations"]["dram_to_gpu"],
        "blocks_gpu": stats["blocks_gpu"],
        "blocks_dram": stats["blocks_dram"],
        "gpu_kv_mb": round(stats["gpu_kv_mb"], 2),
    }

    mgr.destroy()
    return result


def compute_budget_bytes(model, prompt_len: int, fraction: float) -> int:
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    seq_len = prompt_len + MAX_NEW_TOKENS
    full_kv_bytes = 2 * n_layers * n_kv_heads * seq_len * head_dim * 2
    return int(full_kv_bytes * fraction)


def main():
    print("=" * 70)
    print("  RULER-style Needle-In-A-Haystack (NIAH) Benchmark")
    print("=" * 70)
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Lengths:     {PROMPT_LENGTHS}")
    print(f"  Prompts/len: {NUM_PROMPTS_PER_LENGTH}")
    print(f"  Budget:      {BUDGET_FRACTION*100:.0f}%")
    print(f"  Gen tokens:  {MAX_NEW_TOKENS}")
    print("=" * 70)

    rng = random.Random(SEED)

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded: {model.config.num_hidden_layers}L, "
          f"{model.config.num_key_value_heads}KV, "
          f"d={model.config.hidden_size // model.config.num_attention_heads}")

    all_results = {}

    for target_len in PROMPT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"  Length = {target_len} tokens")
        print(f"{'='*60}")

        prompts = [
            generate_niah_prompt(tokenizer, target_len, rng)
            for _ in range(NUM_PROMPTS_PER_LENGTH)
        ]

        baseline_results = []
        orchkv_results = []

        consistency_matches = 0

        for pi, p in enumerate(prompts):
            input_ids = tokenizer(
                p["prompt"], return_tensors="pt", truncation=True,
                max_length=target_len + 64,
            )["input_ids"].cuda()
            actual_len = input_ids.shape[1]

            # --- baseline ---
            try:
                br = run_baseline_decode(model, tokenizer, input_ids)
                baseline_results.append(br)
                bl_mark = "Y" if br["accurate"] else "N"
            except Exception as e:
                traceback.print_exc()
                baseline_results.append({
                    "accurate": False, "error": str(e),
                    "generated_text": "",
                    "n_tokens": 0, "throughput_tok_s": 0, "tpot_ms": 0,
                })
                bl_mark = "E"

            torch.cuda.empty_cache()
            gc.collect()

            # --- orchkv ---
            budget = compute_budget_bytes(model, actual_len, BUDGET_FRACTION)
            try:
                okr = run_orchkv_decode(model, tokenizer, input_ids, budget)
                orchkv_results.append(okr)
                ok_mark = "Y" if okr["accurate"] else "N"
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                orchkv_results.append({
                    "accurate": False, "error": str(e)[:200],
                    "generated_text": "",
                    "n_tokens": 0, "throughput_tok_s": 0, "tpot_ms": 0,
                    "evictions": 0, "promotions": 0,
                })
                ok_mark = "E"
            except Exception as e:
                traceback.print_exc()
                orchkv_results.append({
                    "accurate": False, "error": str(e)[:200],
                    "generated_text": "",
                    "n_tokens": 0, "throughput_tok_s": 0, "tpot_ms": 0,
                    "evictions": 0, "promotions": 0,
                })
                ok_mark = "E"

            torch.cuda.empty_cache()
            gc.collect()

            bl_text = baseline_results[-1].get("generated_text", "")
            ok_text = orchkv_results[-1].get("generated_text", "")
            consistent = (bl_text == ok_text) and bl_mark != "E" and ok_mark != "E"
            if consistent:
                consistency_matches += 1

            print(f"  [{pi+1:2d}/{NUM_PROMPTS_PER_LENGTH}] "
                  f"len={actual_len:5d}  "
                  f"baseline={bl_mark}  orchkv={ok_mark}  "
                  f"consist={'Y' if consistent else 'N'}  "
                  f"needle@{p['needle_position_frac']:.2f}")

        def summarize(results: list[dict]) -> dict:
            valid = [r for r in results if "error" not in r]
            n_valid = len(valid)
            if n_valid == 0:
                return {"accuracy": 0.0, "n_valid": 0}
            accuracies = [r["accurate"] for r in valid]
            throughputs = [r["throughput_tok_s"] for r in valid]
            tpots = [r["tpot_ms"] for r in valid]
            return {
                "accuracy": sum(accuracies) / len(accuracies),
                "n_valid": n_valid,
                "n_errors": len(results) - n_valid,
                "mean_throughput_tok_s": round(sum(throughputs) / len(throughputs), 2),
                "mean_tpot_ms": round(sum(tpots) / len(tpots), 2),
            }

        def summarize_orchkv(results: list[dict]) -> dict:
            base = summarize(results)
            valid = [r for r in results if "error" not in r]
            if valid:
                evictions = [r.get("evictions", 0) for r in valid]
                promotions = [r.get("promotions", 0) for r in valid]
                base["mean_evictions"] = round(sum(evictions) / len(evictions), 1)
                base["mean_promotions"] = round(sum(promotions) / len(promotions), 1)
                base["total_evictions"] = sum(evictions)
                base["total_promotions"] = sum(promotions)
            return base

        bl_summary = summarize(baseline_results)
        ok_summary = summarize_orchkv(orchkv_results)

        n_valid_pairs = sum(
            1 for b, o in zip(baseline_results, orchkv_results)
            if "error" not in b and "error" not in o
        )
        consistency_rate = consistency_matches / max(n_valid_pairs, 1)

        all_results[str(target_len)] = {
            "target_len": target_len,
            "num_prompts": NUM_PROMPTS_PER_LENGTH,
            "baseline": bl_summary,
            "orchkv_50pct": ok_summary,
            "consistency": {
                "matches": consistency_matches,
                "valid_pairs": n_valid_pairs,
                "rate": round(consistency_rate, 4),
            },
            "per_prompt": {
                "baseline": baseline_results,
                "orchkv_50pct": orchkv_results,
            },
        }

        print(f"\n  Summary for len={target_len}:")
        print(f"    Baseline   accuracy={bl_summary['accuracy']*100:.1f}%  "
              f"throughput={bl_summary.get('mean_throughput_tok_s', 0):.1f} tok/s")
        print(f"    OrchKv50%  accuracy={ok_summary['accuracy']*100:.1f}%  "
              f"throughput={ok_summary.get('mean_throughput_tok_s', 0):.1f} tok/s  "
              f"evictions={ok_summary.get('mean_evictions', 0):.0f}")
        print(f"    Consistency: {consistency_matches}/{n_valid_pairs} "
              f"({consistency_rate*100:.1f}%)")

    final = {
        "experiment": "ruler_niah",
        "model": MODEL_PATH,
        "budget_fraction": BUDGET_FRACTION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_lengths": PROMPT_LENGTHS,
        "num_prompts_per_length": NUM_PROMPTS_PER_LENGTH,
        "seed": SEED,
        "needle": NEEDLE_TEXT,
        "results": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("  RULER NIAH SUMMARY")
    print("=" * 70)
    for length_str, data in all_results.items():
        bl = data["baseline"]
        ok = data["orchkv_50pct"]
        con = data["consistency"]
        print(f"  len={length_str:>5s}:  "
              f"baseline_acc={bl['accuracy']*100:5.1f}%  "
              f"orchkv_acc={ok['accuracy']*100:5.1f}%  "
              f"consist={con['rate']*100:5.1f}%  "
              f"orchkv_evict={ok.get('mean_evictions', 0):.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

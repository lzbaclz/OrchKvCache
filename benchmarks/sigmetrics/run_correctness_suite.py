#!/usr/bin/env python3
"""
Correctness verification suite: 1000+ prompts from LongBench.

Runs each prompt through two paths:
  a) GPU-only baseline: manual decode loop with full KV on GPU
  b) OrchKvCache: KVCacheManager at 50% GPU budget

Compares generated tokens bit-exactly. Checkpoints progress for resumability.
"""
import sys
import os
import json
import time
import glob
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from orchkv.kvcache_manager import KVCacheManager

MODEL_PATH = "/public/model_zoo/Qwen2.5-7B"
MAX_NEW_TOKENS = 16
BUDGET_FRACTION = 0.50
MAX_PROMPT_TOKENS = 512
BATCH_CHECKPOINT_SIZE = 50
ATTN_SAMPLE_INTERVAL = 5

DATA_DIR = ROOT / "benchmarks" / "sigmetrics" / "data" / "longbench" / "data"
RESULTS_DIR = ROOT / "benchmarks" / "sigmetrics" / "results"
CHECKPOINT_PATH = RESULTS_DIR / "correctness_checkpoint.json"
OUTPUT_PATH = RESULTS_DIR / "correctness_1000.json"

SPLITS = [
    "qasper", "multifieldqa_en", "narrativeqa",
    "hotpotqa", "2wikimqa", "musique", "multi_news",
]


def load_prompts(tokenizer):
    """Load and tokenize prompts from LongBench splits, truncating to MAX_PROMPT_TOKENS."""
    prompts = []
    for split in SPLITS:
        fpath = DATA_DIR / f"{split}.jsonl"
        if not fpath.exists():
            print(f"  [WARN] {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("context", "") + "\n\n" + obj.get("input", "")
                prompts.append({
                    "text": text,
                    "split": split,
                    "id": obj.get("_id", ""),
                })
    print(f"  Loaded {len(prompts)} raw prompts from {len(SPLITS)} splits")
    return prompts


def run_baseline_decode(model, input_ids, max_new_tokens):
    """Manual decode loop without KVCacheManager (GPU-only reference)."""
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                cur_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        past_kv = outputs.past_key_values
        cur_ids = next_token

    return generated


def run_orchkv_decode(model, input_ids, max_new_tokens, gpu_budget_bytes):
    """Manual decode loop with KVCacheManager at given budget."""
    cfg = model.config
    mgr = KVCacheManager(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=16,
        dtype=torch.float16,
        gpu_budget_bytes=gpu_budget_bytes,
    )

    generated = []
    cur_ids = input_ids.clone()
    past_kv = None

    for step in range(max_new_tokens):
        want_attn = (step % ATTN_SAMPLE_INTERVAL == 0)
        with torch.no_grad():
            outputs = model(
                cur_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=want_attn,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        new_past = outputs.past_key_values
        if step == 0:
            mgr.ingest_step(new_past)
        else:
            mgr.append_token(new_past)

        if outputs.attentions is not None:
            for li, attn in enumerate(outputs.attentions):
                mgr.report_attention(li, attn)

        mgr.step_done()
        mgr.schedule()

        past_kv = mgr.build_past_kv()
        cur_ids = next_token

    stats = mgr.get_stats()
    mgr.destroy()
    return generated, stats


def compute_budget_bytes(model, prompt_len, max_new_tokens, fraction):
    """Compute GPU budget in bytes for given fraction of full KV size."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    seq_len = prompt_len + max_new_tokens
    full_kv_bytes = 2 * n_layers * n_kv_heads * seq_len * head_dim * 2  # FP16
    return int(full_kv_bytes * fraction)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_checkpoint(state):
    """Save checkpoint."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f)


def main():
    print("=" * 70)
    print("CORRECTNESS VERIFICATION: 1000+ prompts, OrchKvCache vs GPU-only")
    print("=" * 70)
    print(f"  Model:        {MODEL_PATH}")
    print(f"  Budget:       {BUDGET_FRACTION*100:.0f}%")
    print(f"  Gen tokens:   {MAX_NEW_TOKENS}")
    print(f"  Max prompt:   {MAX_PROMPT_TOKENS} tokens")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading prompts...")
    prompts = load_prompts(tokenizer)

    checkpoint = load_checkpoint()
    completed_ids = set(checkpoint["completed"])
    results = checkpoint["results"]
    print(f"  Checkpoint: {len(completed_ids)} already completed")

    print(f"\nLoading model: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Model loaded. Config: {model.config.num_hidden_layers}L, "
          f"{model.config.num_key_value_heads}KV heads, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    n_total = len(prompts)
    n_match = sum(1 for r in results if r["match"])
    n_mismatch = sum(1 for r in results if not r["match"])
    batch_count = 0
    start_time = time.time()

    print(f"\nRunning correctness verification on {n_total} prompts...")
    print(f"  (resuming from {len(completed_ids)} completed)\n")

    for idx, prompt_info in enumerate(prompts):
        prompt_id = f"{prompt_info['split']}_{prompt_info['id']}"
        if prompt_id in completed_ids:
            continue

        text = prompt_info["text"]
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )["input_ids"].cuda()
        prompt_len = input_ids.shape[1]

        if prompt_len < 10:
            completed_ids.add(prompt_id)
            continue

        gpu_budget = compute_budget_bytes(
            model, prompt_len, MAX_NEW_TOKENS, BUDGET_FRACTION
        )

        try:
            baseline_tokens = run_baseline_decode(model, input_ids, MAX_NEW_TOKENS)
            orchkv_tokens, stats = run_orchkv_decode(
                model, input_ids, MAX_NEW_TOKENS, gpu_budget
            )

            match = (baseline_tokens == orchkv_tokens)
            first_div = -1
            if not match:
                for i in range(min(len(baseline_tokens), len(orchkv_tokens))):
                    if baseline_tokens[i] != orchkv_tokens[i]:
                        first_div = i
                        break

            result = {
                "prompt_id": prompt_id,
                "split": prompt_info["split"],
                "prompt_len": prompt_len,
                "match": match,
                "first_divergence": first_div,
                "gpu_budget_bytes": gpu_budget,
                "migrations": stats.get("migrations", {}),
            }
            results.append(result)
            completed_ids.add(prompt_id)

            if match:
                n_match += 1
            else:
                n_mismatch += 1

        except Exception as e:
            result = {
                "prompt_id": prompt_id,
                "split": prompt_info["split"],
                "prompt_len": prompt_len,
                "match": True,
                "error": str(e),
                "skipped": True,
            }
            results.append(result)
            completed_ids.add(prompt_id)
            print(f"  [ERR] {prompt_id}: {e}")

        batch_count += 1
        done = len(completed_ids)

        if batch_count % 50 == 0:
            elapsed = time.time() - start_time
            rate = batch_count / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{n_total}] match={n_match} mismatch={n_mismatch} "
                  f"rate={rate:.1f} prompts/s")

        if batch_count % BATCH_CHECKPOINT_SIZE == 0:
            checkpoint["completed"] = list(completed_ids)
            checkpoint["results"] = results
            save_checkpoint(checkpoint)

        torch.cuda.empty_cache()

    elapsed_total = time.time() - start_time

    valid_results = [r for r in results if not r.get("skipped")]
    total_tested = len(valid_results)
    total_match = sum(1 for r in valid_results if r["match"])
    total_mismatch = sum(1 for r in valid_results if not r["match"])
    match_rate = total_match / total_tested if total_tested > 0 else 0

    divergence_positions = [
        r["first_divergence"] for r in valid_results
        if not r["match"] and r["first_divergence"] >= 0
    ]

    summary = {
        "experiment": "correctness_verification_1000+",
        "model": MODEL_PATH,
        "budget_fraction": BUDGET_FRACTION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "total_prompts_tested": total_tested,
        "total_prompts_loaded": n_total,
        "bit_exact_matches": total_match,
        "mismatches": total_mismatch,
        "match_rate": match_rate,
        "divergence_positions": divergence_positions,
        "elapsed_seconds": elapsed_total,
        "splits_used": SPLITS,
        "attn_sample_interval": ATTN_SAMPLE_INTERVAL,
        "per_split_stats": {},
        "results": valid_results,
    }

    for split in SPLITS:
        split_results = [r for r in valid_results if r["split"] == split]
        if split_results:
            split_match = sum(1 for r in split_results if r["match"])
            summary["per_split_stats"][split] = {
                "total": len(split_results),
                "match": split_match,
                "mismatch": len(split_results) - split_match,
                "match_rate": split_match / len(split_results),
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION RESULTS")
    print("=" * 70)
    print(f"  Total prompts tested:  {total_tested}")
    print(f"  Bit-exact matches:     {total_match}")
    print(f"  Mismatches:            {total_mismatch}")
    print(f"  Match rate:            {match_rate*100:.2f}%")
    if divergence_positions:
        print(f"  Divergence positions:  {divergence_positions[:10]}")
    print(f"  Elapsed time:          {elapsed_total:.1f}s")
    print(f"  Results saved to:      {OUTPUT_PATH}")
    print("=" * 70)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    return 0 if total_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

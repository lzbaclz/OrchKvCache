#!/usr/bin/env python3
"""
Smoke test: verify orchkv_core's tiered_manager is actually invoked
during real model inference via the KVCacheManager.

Expected output:
  - Model loads and generates tokens
  - orchkv_core tiered_manager receives attention reports
  - Block classification (hot/warm/cold) produces non-zero counts
  - Output tokens match baseline (no offload) exactly
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.kvcache_manager import KVCacheManager

MODEL_PATH = "Qwen/Qwen2.5-7B"
PROMPT = "The key insight behind transformer-based language models is"
MAX_NEW_TOKENS = 20
GPU_BUDGET_MB = 200  # Small budget to force evictions


def run_baseline(model, tokenizer, input_ids):
    """Standard generation, no KV management."""
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, use_cache=True,
        )
    return out[0, input_ids.shape[1]:]


def run_with_orchkv(model, tokenizer, input_ids, gpu_budget_bytes):
    """Manual decode loop with KVCacheManager."""
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

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(
                cur_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=(step % 5 == 0),
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
        sched = mgr.schedule()

        past_kv = mgr.build_past_kv()
        cur_ids = next_token

    stats = mgr.get_stats()
    mgr.destroy()
    return torch.tensor(generated), stats


def main():
    print("=" * 70)
    print("SMOKE TEST: OrchKvCache E2E with real model inference")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    input_ids = inputs["input_ids"]
    print(f"Prompt: {PROMPT!r} ({input_ids.shape[1]} tokens)")

    print("\n--- Running baseline (standard generation) ---")
    baseline_tokens = run_baseline(model, tokenizer, input_ids)
    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
    print(f"Baseline output: {baseline_text!r}")

    print("\n--- Running with OrchKvCache (gpu_budget={:.0f}MB) ---".format(
        GPU_BUDGET_MB))
    gpu_budget = GPU_BUDGET_MB * (1 << 20)
    orchkv_tokens, stats = run_with_orchkv(model, tokenizer, input_ids, gpu_budget)
    orchkv_text = tokenizer.decode(orchkv_tokens, skip_special_tokens=True)
    print(f"OrchKv output:   {orchkv_text!r}")

    print("\n--- KVCacheManager Stats ---")
    for k, v in stats.items():
        if k == "tm":
            print(f"  tiered_manager:")
            for tk, tv in v.items():
                print(f"    {tk}: {tv}")
        elif k == "migrations":
            print(f"  migrations:")
            for mk, mv in v.items():
                print(f"    {mk}: {mv}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Verification ---")
    match = torch.equal(baseline_tokens.cpu(), orchkv_tokens.cpu())
    print(f"Token-level match: {'PASS' if match else 'FAIL'}")
    if not match:
        for i in range(min(len(baseline_tokens), len(orchkv_tokens))):
            if baseline_tokens[i] != orchkv_tokens[i]:
                print(f"  First mismatch at position {i}: "
                      f"baseline={baseline_tokens[i].item()} "
                      f"orchkv={orchkv_tokens[i].item()}")
                break

    tm = stats.get("tm", {})
    has_scheduling = tm.get("schedule_cycles", 0) > 0
    has_classification = (tm.get("n_hot", 0) + tm.get("n_warm", 0) + tm.get("n_cold", 0)) > 0
    print(f"orchkv_core scheduling active: {'PASS' if has_scheduling else 'FAIL'}")
    print(f"orchkv_core classification active: {'PASS' if has_classification else 'FAIL'}")

    has_evictions = stats["migrations"]["gpu_to_dram"] > 0
    print(f"GPU->DRAM evictions occurred: {'PASS' if has_evictions else 'SKIPPED (budget not exceeded)'}")

    all_pass = match and has_scheduling and has_classification
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    del model
    torch.cuda.empty_cache()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

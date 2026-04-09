#!/usr/bin/env python3
"""
P3: Hotness Identity Metrics — precision@K, recall@K, top-K hit rate.

For each decode step, compare:
  - Ground truth: top-K blocks by actual attention weight
  - Predicted: top-K blocks by EMA score

Reports precision@K, recall@K, F1@K averaged across steps.
"""
from __future__ import annotations
import gc, os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, RESULTS_DIR

try:
    import orchkv_core as _C
except ImportError:
    print("ERROR: orchkv_core not found")
    sys.exit(1)


def run_identity_experiment(model_name, seq_len=1024, max_new=32, budget_mb=50):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    short = model_name.split("/")[-1]
    print(f"\n  {short}  seq={seq_len}  gen={max_new}")

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

    tm = _C.tm_create(
        tracker_cap=2048, max_blocks=1024,
        alpha=0.7, beta=0.2, gamma=0.1,
        prefetch_budget=8, schedule_interval_us=500,
        threshold_to_gpu=0.5, threshold_to_dram=0.15,
    )

    n_blocks_total = (actual_len + max_new + block_size - 1) // block_size
    for bid in range(n_blocks_total):
        flags = 1 if bid == 0 else 0
        _C.tm_register_block_id(tm, bid, 0, flags)

    step_metrics = []
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

            _C.tm_step_done(tm)
            _C.tm_schedule_once(tm)

            for bid in range(n_blocks):
                all_block_ema[bid] = _C.tm_get_block_score(tm, bid)

            sink_blocks = 1
            nonsink = np.arange(sink_blocks, n_blocks)

            if len(nonsink) > 1:
                ns_attn = all_block_attn[nonsink]
                ns_ema = all_block_ema[nonsink]

                step_result = {"step": s, "n_nonsink_blocks": len(nonsink)}

                for K_pct in [5, 10, 25]:
                    K = max(1, int(len(nonsink) * K_pct / 100))

                    gt_topK = set(np.argsort(-ns_attn)[:K])
                    pred_topK = set(np.argsort(-ns_ema)[:K])

                    tp = len(gt_topK & pred_topK)
                    precision = tp / K if K > 0 else 0
                    recall = tp / K if K > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    step_result[f"precision@{K_pct}pct"] = round(precision, 4)
                    step_result[f"recall@{K_pct}pct"] = round(recall, 4)
                    step_result[f"f1@{K_pct}pct"] = round(f1, 4)
                    step_result[f"K_{K_pct}pct"] = K

                step_metrics.append(step_result)
        else:
            _C.tm_step_done(tm)

        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _C.tm_destroy(tm)
    del mdl; gc.collect(); torch.cuda.empty_cache()

    if not step_metrics:
        return None

    summary = {"model": short, "seq_len": seq_len, "n_steps": len(step_metrics)}
    for K_pct in [5, 10, 25]:
        for metric in ["precision", "recall", "f1"]:
            key = f"{metric}@{K_pct}pct"
            vals = [m[key] for m in step_metrics if key in m]
            summary[f"avg_{key}"] = round(float(np.mean(vals)), 4) if vals else 0
    summary["per_step"] = step_metrics

    print(f"    Results for {short}:")
    for K_pct in [5, 10, 25]:
        p = summary[f"avg_precision@{K_pct}pct"]
        r = summary[f"avg_recall@{K_pct}pct"]
        f = summary[f"avg_f1@{K_pct}pct"]
        print(f"      @{K_pct}%: precision={p:.3f}  recall={r:.3f}  F1={f:.3f}")

    return summary


def main():
    models = [
        "Qwen/Qwen2.5-7B",
        "meta-llama/Llama-2-7b-hf",
    ]

    results = []
    for m in models:
        r = run_identity_experiment(m, seq_len=1024, max_new=32)
        if r:
            results.append(r)

    out_path = RESULTS_DIR / "exp_hotness_identity.json"
    out_data = [
        {k: v for k, v in r.items() if k != "per_step"} for r in results
    ]
    save_json(out_data, out_path)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'K%':>4} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"{'-'*60}")
    for r in results:
        for K_pct in [5, 10, 25]:
            print(f"{r['model']:<20} {K_pct:>3}% "
                  f"{r[f'avg_precision@{K_pct}pct']:>8.3f} "
                  f"{r[f'avg_recall@{K_pct}pct']:>8.3f} "
                  f"{r[f'avg_f1@{K_pct}pct']:>8.3f}")


if __name__ == "__main__":
    main()

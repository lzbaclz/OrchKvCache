#!/usr/bin/env python3
"""
W7: Validate attention distribution findings on Qwen2.5-7B.

Reproduces the Motivation analysis (§2.3) on an Evaluation-scale model
to confirm that attention skewness generalizes from 1.5B to 7B.

Metrics:
  - Top-10% token attention concentration
  - Gini coefficient per layer
  - Block-level concentration (block_size=16)
  - Jaccard similarity between consecutive decode steps
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def gini_coefficient(weights: np.ndarray) -> float:
    """Gini coefficient of a 1D array."""
    w = np.sort(np.abs(weights.ravel()))
    n = len(w)
    if n == 0 or np.nansum(w) == 0:
        return 0.0
    w = np.nan_to_num(w, nan=0.0)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * w) / (n * np.sum(w))) - (n + 1) / n)


def top_k_concentration(weights: np.ndarray, k_frac: float = 0.10) -> float:
    """Fraction of total weight captured by top k% of entries."""
    w = np.nan_to_num(np.abs(weights.ravel()), nan=0.0)
    total = w.sum()
    if total == 0 or len(w) == 0:
        return 0.0
    k = max(1, int(len(w) * k_frac))
    topk = np.sort(w)[-k:]
    return float(topk.sum() / total)


def block_concentration(weights: np.ndarray, block_size: int = 16,
                        k_frac: float = 0.10) -> float:
    """Top-k% block-level concentration."""
    w = np.nan_to_num(np.abs(weights.ravel()), nan=0.0)
    n_blocks = len(w) // block_size
    if n_blocks == 0:
        return 0.0
    block_weights = np.array([
        w[i * block_size:(i + 1) * block_size].sum()
        for i in range(n_blocks)
    ])
    total = block_weights.sum()
    if total == 0:
        return 0.0
    k = max(1, int(n_blocks * k_frac))
    topk = np.sort(block_weights)[-k:]
    return float(topk.sum() / total)


def jaccard_topk(w1: np.ndarray, w2: np.ndarray, k_frac: float = 0.10) -> float:
    """Jaccard similarity of top-k% token sets between two steps."""
    k = max(1, int(len(w1) * k_frac))
    s1 = set(np.argsort(w1.ravel())[-k:])
    s2 = set(np.argsort(w2.ravel())[-k:])
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


@torch.no_grad()
def profile_model(model_name: str, prompt_lens: list[int], n_decode: int = 10):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads

    print(f"  layers={n_layers}, kv_heads={n_kv_heads}")

    text_base = ("The transformer architecture uses self-attention "
                 "to process sequences of tokens efficiently. ") * 200

    all_results = []

    for plen in prompt_lens:
        print(f"\n  --- prompt_len={plen} ---")
        input_ids = tokenizer(text_base, return_tensors="pt",
                              truncation=True, max_length=plen
                              )["input_ids"].to("cuda:0")
        actual_len = input_ids.shape[1]

        layer_ginis = []
        layer_top10 = []
        layer_block_conc = []
        step_jaccards = []
        prev_per_layer = None

        past_kv = None
        cur_ids = input_ids

        for step in range(n_decode + 1):
            out = model(cur_ids, past_key_values=past_kv,
                        use_cache=True, output_attentions=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_ids = next_tok

            if step == 0:
                continue

            per_layer_weights = []
            step_ginis = []
            step_top10 = []
            step_block = []

            for li, attn in enumerate(out.attentions):
                w = attn[0].float().cpu().numpy()  # [n_heads, q_len, kv_len]
                w_flat = w.reshape(-1, w.shape[-1])  # [n_heads*q_len, kv_len]
                w_mean = w_flat.mean(axis=0)  # [kv_len]
                per_layer_weights.append(w_mean)

                g = gini_coefficient(w_mean)
                t10 = top_k_concentration(w_mean, 0.10)
                bc = block_concentration(w_mean, block_size=16, k_frac=0.10)
                step_ginis.append(g)
                step_top10.append(t10)
                step_block.append(bc)

            layer_ginis.append(step_ginis)
            layer_top10.append(step_top10)
            layer_block_conc.append(step_block)

            if prev_per_layer is not None:
                jacs = []
                for li in range(n_layers):
                    j = jaccard_topk(prev_per_layer[li],
                                     per_layer_weights[li], 0.10)
                    jacs.append(j)
                step_jaccards.append(jacs)

            prev_per_layer = per_layer_weights

        arr_gini = np.array(layer_ginis)
        arr_top10 = np.array(layer_top10)
        arr_block = np.array(layer_block_conc)

        mean_gini = arr_gini.mean(axis=0)
        mean_top10 = arr_top10.mean(axis=0)
        mean_block = arr_block.mean(axis=0)

        mean_jaccard = np.zeros(n_layers)
        if step_jaccards:
            arr_jac = np.array(step_jaccards)
            mean_jaccard = arr_jac.mean(axis=0)

        result = {
            "prompt_len": actual_len,
            "n_decode_steps": n_decode,
            "gini_range": [round(float(mean_gini.min()), 3),
                           round(float(mean_gini.max()), 3)],
            "gini_mean": round(float(mean_gini.mean()), 3),
            "top10_range": [round(float(mean_top10.min()), 3),
                            round(float(mean_top10.max()), 3)],
            "top10_mean": round(float(mean_top10.mean()), 3),
            "block_top10_range": [round(float(mean_block.min()), 3),
                                  round(float(mean_block.max()), 3)],
            "block_top10_mean": round(float(mean_block.mean()), 3),
            "jaccard_range": [round(float(mean_jaccard.min()), 3),
                              round(float(mean_jaccard.max()), 3)],
            "jaccard_mean": round(float(mean_jaccard.mean()), 3),
        }

        print(f"    Gini: {result['gini_range']}  mean={result['gini_mean']}")
        print(f"    Top10%: {result['top10_range']}  mean={result['top10_mean']}")
        print(f"    Block top10%: {result['block_top10_range']}  mean={result['block_top10_mean']}")
        print(f"    Jaccard: {result['jaccard_range']}  mean={result['jaccard_mean']}")

        all_results.append(result)

        del out
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return all_results


def main():
    model_name = "Qwen/Qwen2.5-7B"
    prompt_lens = [256, 512, 1024]

    print("=" * 60)
    print("  W7: Attention Distribution Validation on Qwen2.5-7B")
    print("=" * 60)

    results = profile_model(model_name, prompt_lens, n_decode=10)

    summary = {"model": model_name, "prompts": results}

    out_path = os.path.join(RESULTS_DIR, "w7_attn_profile.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  prompt={r['prompt_len']:>5}  "
              f"Gini={r['gini_range']}  "
              f"Top10={r['top10_range']}  "
              f"Jaccard={r['jaccard_range']}")


if __name__ == "__main__":
    main()

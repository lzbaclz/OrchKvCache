#!/usr/bin/env python3
"""
Exp-M2: 注意力分数分布分析 — 冷热分化验证

目标:
  1. 收集真实推理时每层每头的注意力分数
  2. 验证注意力分数是否呈幂律分布 (少量 token 贡献绝大部分注意力)
  3. 分析 block 粒度聚合后是否仍有明显冷热分化
  4. 分析不同层、不同 decode step 的差异

论文用途: Figure 3 — CDF 图 + 热力图, 论证冷热分级的理论基础
"""
import torch
import json
import os
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/raid/models/Qwen/Qwen2___5-1___5B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = "cuda:0"

BLOCK_SIZES = [8, 16, 32, 64]
INPUT_TEXTS = {
    "short_512": "The history of artificial intelligence began in the 1950s when researchers first proposed that machines could simulate human intelligence. " * 30,
    "medium_1024": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning, a further subset, uses neural networks with many layers. The transformer architecture revolutionized natural language processing by introducing the self-attention mechanism, which allows the model to weigh the importance of different parts of the input sequence. " * 20,
    "long_2048": "Large language models have transformed the field of natural language processing. These models, trained on vast amounts of text data, can generate human-like text, answer questions, translate languages, and perform various other language tasks. The key innovation behind modern LLMs is the transformer architecture, which uses self-attention mechanisms to process input sequences in parallel. The KV-cache is a critical optimization that stores the key and value projections from previous tokens, avoiding redundant computation during autoregressive generation. However, as sequence lengths grow, the KV-cache becomes a major memory bottleneck. " * 15,
}

def analyze_single_input(model, tokenizer, text, label):
    """Analyze attention distribution for a single input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    print(f"\n  Input '{label}': {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    attentions = outputs.attentions  # tuple of (1, n_heads, seq_len, seq_len)
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    result = {
        "label": label,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "layer_stats": [],
        "block_analysis": {},
        "overall_stats": {},
    }

    all_top10 = []
    all_top20 = []
    all_gini = []

    print(f"  {'Layer':>6s} {'Top5%':>8s} {'Top10%':>8s} {'Top20%':>8s} {'Top50%':>8s} {'Gini':>8s} {'MaxAttn':>8s}")
    print(f"  {'-'*55}")

    for layer_idx, attn in enumerate(attentions):
        # attn: (1, n_heads, seq_len, seq_len)
        # Use the LAST query token's attention over all previous tokens
        last_query_attn = attn[0, :, -1, :].float()  # (n_heads, seq_len)

        # Replace NaN with 0 (causal mask area)
        last_query_attn = torch.nan_to_num(last_query_attn, nan=0.0)

        # Average across heads
        avg_attn = last_query_attn.mean(dim=0)  # (seq_len,)

        # Sort descending
        sorted_attn, sorted_idx = avg_attn.sort(descending=True)
        total = sorted_attn.sum()
        if total < 1e-10:
            continue

        cumsum = sorted_attn.cumsum(dim=0) / total

        # Top-K% coverage
        top5 = cumsum[max(int(seq_len * 0.05) - 1, 0)].item()
        top10 = cumsum[max(int(seq_len * 0.10) - 1, 0)].item()
        top20 = cumsum[max(int(seq_len * 0.20) - 1, 0)].item()
        top50 = cumsum[max(int(seq_len * 0.50) - 1, 0)].item()

        # Gini coefficient (measure of inequality, 0=equal, 1=maximally unequal)
        n = len(avg_attn)
        sorted_vals = sorted_attn.cpu().numpy()
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n

        max_attn = sorted_attn[0].item()

        all_top10.append(top10)
        all_top20.append(top20)
        all_gini.append(gini)

        if layer_idx % 4 == 0 or layer_idx == n_layers - 1:
            print(f"  {layer_idx:>6d} {top5:>7.1%} {top10:>7.1%} {top20:>7.1%} {top50:>7.1%} {gini:>8.3f} {max_attn:>8.4f}")

        # Attention sink analysis: check first few tokens
        first_token_attn = avg_attn[0].item() / total.item()
        first_5_attn = avg_attn[:5].sum().item() / total.item()

        layer_stat = {
            "layer": layer_idx,
            "top5_coverage": round(top5, 4),
            "top10_coverage": round(top10, 4),
            "top20_coverage": round(top20, 4),
            "top50_coverage": round(top50, 4),
            "gini": round(float(gini), 4),
            "max_attention": round(max_attn, 6),
            "first_token_attn_pct": round(first_token_attn * 100, 2),
            "first_5_tokens_attn_pct": round(first_5_attn * 100, 2),
        }
        result["layer_stats"].append(layer_stat)

    # Block-level analysis
    print(f"\n  Block-level aggregation:")
    print(f"  {'BlockSize':>10s} {'nBlocks':>8s} {'Top10%blk':>10s} {'Top20%blk':>10s} {'Gini':>8s}")
    print(f"  {'-'*50}")

    for bs in BLOCK_SIZES:
        if seq_len < bs * 4:
            continue

        block_coverages = []
        block_ginis = []

        for layer_idx, attn in enumerate(attentions):
            raw = torch.nan_to_num(attn[0, :, -1, :].float(), nan=0.0)
            avg_attn = raw.mean(dim=0)  # (seq_len,)
            n_blocks = seq_len // bs
            if n_blocks < 2:
                continue

            block_attn = avg_attn[:n_blocks * bs].reshape(n_blocks, bs).sum(dim=1)
            block_attn = block_attn / block_attn.sum()

            sorted_block, _ = block_attn.sort(descending=True)
            block_cumsum = sorted_block.cumsum(dim=0)
            b_top10 = block_cumsum[max(int(n_blocks * 0.10) - 1, 0)].item()
            b_top20 = block_cumsum[max(int(n_blocks * 0.20) - 1, 0)].item()

            vals = sorted_block.cpu().numpy()
            idx = np.arange(1, len(vals) + 1)
            b_gini = (2 * np.sum(idx * vals) / (len(vals) * np.sum(vals))) - (len(vals) + 1) / len(vals)

            block_coverages.append((b_top10, b_top20))
            block_ginis.append(b_gini)

        if block_coverages:
            avg_b10 = np.mean([c[0] for c in block_coverages])
            avg_b20 = np.mean([c[1] for c in block_coverages])
            avg_bgini = np.mean(block_ginis)
            n_blk = seq_len // bs
            print(f"  {bs:>10d} {n_blk:>8d} {avg_b10:>9.1%} {avg_b20:>9.1%} {avg_bgini:>8.3f}")

            result["block_analysis"][str(bs)] = {
                "n_blocks": n_blk,
                "avg_top10_coverage": round(avg_b10, 4),
                "avg_top20_coverage": round(avg_b20, 4),
                "avg_gini": round(float(avg_bgini), 4),
            }

    # Cross-decode-step stability (generate a few tokens and check consistency)
    print(f"\n  Cross-decode-step Top-K stability:")
    top_k_sets = []
    k = max(int(seq_len * 0.1), 1)

    past_kv = None
    current_ids = inputs["input_ids"]

    for step in range(5):
        with torch.no_grad():
            out = model(current_ids, past_key_values=past_kv, output_attentions=True, use_cache=True)

        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_ids = next_token

        mid_layer = len(out.attentions) // 2
        step_attn = out.attentions[mid_layer]  # (1, n_heads, cur_seq, full_seq)
        avg_step_attn = torch.nan_to_num(step_attn[0, :, -1, :].float(), nan=0.0).mean(dim=0)
        _, top_indices = avg_step_attn.topk(k)
        top_k_sets.append(set(top_indices.cpu().numpy().tolist()))

    # Jaccard similarity between consecutive steps
    jaccards = []
    for i in range(len(top_k_sets) - 1):
        inter = len(top_k_sets[i] & top_k_sets[i+1])
        union = len(top_k_sets[i] | top_k_sets[i+1])
        jaccards.append(inter / union if union > 0 else 0)

    avg_jaccard = np.mean(jaccards) if jaccards else 0
    print(f"  Mid-layer Top-10% Jaccard similarity across 5 decode steps: {avg_jaccard:.3f}")
    result["decode_step_stability"] = {
        "k": k,
        "n_steps": 5,
        "jaccard_similarities": [round(j, 4) for j in jaccards],
        "avg_jaccard": round(float(avg_jaccard), 4),
    }

    # Overall stats
    result["overall_stats"] = {
        "avg_top10_coverage": round(float(np.mean(all_top10)), 4),
        "avg_top20_coverage": round(float(np.mean(all_top20)), 4),
        "avg_gini": round(float(np.mean(all_gini)), 4),
        "std_top10": round(float(np.std(all_top10)), 4),
        "std_top20": round(float(np.std(all_top20)), 4),
    }

    print(f"\n  Overall: Top-10% covers {np.mean(all_top10):.1%} (±{np.std(all_top10):.1%}), "
          f"Top-20% covers {np.mean(all_top20):.1%} (±{np.std(all_top20):.1%}), "
          f"Gini={np.mean(all_gini):.3f}")

    del outputs, attentions, past_kv
    gc.collect()
    torch.cuda.empty_cache()

    return result

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("M2: Attention Score Distribution Analysis")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    all_results = []
    for label, text in INPUT_TEXTS.items():
        result = analyze_single_input(model, tokenizer, text, label)
        all_results.append(result)

    out_path = os.path.join(RESULTS_DIR, "m2_attention_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

if __name__ == "__main__":
    main()

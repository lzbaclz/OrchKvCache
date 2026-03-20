#!/usr/bin/env python3
"""
Exp-M1b: 真实 KV-Cache 显存测量

用 HuggingFace Transformers 加载模型, 实际 prefill 不同长度序列,
测量 GPU 显存中模型权重 vs KV-Cache 的真实占用, 验证理论计算。

测量方法:
  1. 加载模型后记录 baseline 显存 (= 模型权重 + 框架开销)
  2. 用 use_cache=True 跑 forward pass, 测量 KV-Cache 实际占用
  3. 不同 seq_len 下重复测量
"""
import torch
import json
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/raid/models/Qwen/Qwen2___5-1___5B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SEQ_LENS = [256, 512, 1024, 2048, 4096]
DEVICE = "cuda:0"
DEVICE_IDX = 0

def get_gpu_mem_mb():
    return torch.cuda.memory_allocated(DEVICE_IDX) / 1024**2

def get_gpu_mem_reserved_mb():
    return torch.cuda.memory_reserved(DEVICE_IDX) / 1024**2

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(DEVICE_IDX)

    print("=" * 80)
    print("M1b: Real KV-Cache Memory Measurement")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print()

    mem_before_model = get_gpu_mem_mb()
    print(f"GPU memory before loading: {mem_before_model:.1f} MB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    mem_after_model = get_gpu_mem_mb()
    model_weight_mb = mem_after_model - mem_before_model
    print(f"GPU memory after loading:  {mem_after_model:.1f} MB")
    print(f"Model weight on GPU:       {model_weight_mb:.1f} MB ({model_weight_mb/1024:.2f} GB)")

    config = model.config
    print(f"\nModel config:")
    print(f"  n_layers:    {config.num_hidden_layers}")
    print(f"  n_heads:     {config.num_attention_heads}")
    print(f"  n_kv_heads:  {config.num_key_value_heads}")
    print(f"  d_head:      {config.hidden_size // config.num_attention_heads}")
    print(f"  hidden_size: {config.hidden_size}")

    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    d_head = config.hidden_size // config.num_attention_heads

    results = []

    print(f"\n{'SeqLen':>8s} {'KV-Theory(MB)':>14s} {'KV-Actual(MB)':>14s} {'Ratio':>8s} {'Weight(MB)':>11s} {'KV/Weight':>10s}")
    print("-" * 70)

    for seq_len in SEQ_LENS:
        gc.collect()
        torch.cuda.empty_cache()

        # Theoretical KV-cache size
        kv_theory_bytes = 2 * n_layers * n_kv_heads * d_head * seq_len * 2  # FP16
        kv_theory_mb = kv_theory_bytes / 1024**2

        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=DEVICE)

        torch.cuda.reset_peak_memory_stats(DEVICE)
        mem_before = get_gpu_mem_mb()

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values

        mem_after = get_gpu_mem_mb()
        kv_actual_mb = mem_after - mem_before

        # Measure KV cache tensor sizes directly
        kv_tensor_bytes = 0
        for layer_kv in past_kv:
            for t in layer_kv:
                if t is not None:
                    kv_tensor_bytes += t.nelement() * t.element_size()
        kv_tensor_mb = kv_tensor_bytes / 1024**2

        ratio = kv_actual_mb / kv_theory_mb if kv_theory_mb > 0 else 0
        kv_weight_ratio = kv_actual_mb / model_weight_mb * 100

        print(f"{seq_len:>8d} {kv_theory_mb:>14.2f} {kv_actual_mb:>14.2f} {ratio:>8.2f}x {model_weight_mb:>10.1f} {kv_weight_ratio:>9.1f}%")

        results.append({
            "seq_len": seq_len,
            "kv_theory_MB": round(kv_theory_mb, 2),
            "kv_actual_MB": round(kv_actual_mb, 2),
            "kv_tensor_MB": round(kv_tensor_mb, 2),
            "actual_vs_theory_ratio": round(ratio, 3),
            "model_weight_MB": round(model_weight_mb, 1),
            "kv_weight_ratio_pct": round(kv_weight_ratio, 2),
        })

        del outputs, past_kv
        gc.collect()
        torch.cuda.empty_cache()

    # Batch size scaling test
    print(f"\n{'='*80}")
    print("Batch size scaling (seq_len=2048)")
    print(f"{'='*80}")
    print(f"{'Batch':>6s} {'KV-Cache(MB)':>13s} {'Total(MB)':>11s} {'KV/Total':>9s}")
    print("-" * 45)

    batch_results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        gc.collect()
        torch.cuda.empty_cache()

        input_ids = torch.randint(0, config.vocab_size, (bs, 2048), device=DEVICE)

        mem_before = get_gpu_mem_mb()
        try:
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
            mem_after = get_gpu_mem_mb()
            kv_mb = mem_after - mem_before
            total_mb = mem_after
            kv_pct = kv_mb / total_mb * 100

            print(f"{bs:>6d} {kv_mb:>13.1f} {total_mb:>11.1f} {kv_pct:>8.1f}%")
            batch_results.append({
                "batch_size": bs, "seq_len": 2048,
                "kv_cache_MB": round(kv_mb, 1),
                "total_GPU_MB": round(total_mb, 1),
                "kv_total_pct": round(kv_pct, 1),
            })

            del outputs
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6d}  OOM!")
            batch_results.append({"batch_size": bs, "seq_len": 2048, "error": "OOM"})
            gc.collect()
            torch.cuda.empty_cache()
            break

    out_path = os.path.join(RESULTS_DIR, "m1_real_memory.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_PATH,
            "model_weight_MB": round(model_weight_mb, 1),
            "model_config": {
                "n_layers": n_layers,
                "n_kv_heads": n_kv_heads,
                "d_head": d_head,
            },
            "seq_len_scaling": results,
            "batch_size_scaling": batch_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exp-M1: KV-Cache 显存瓶颈量化

目标:
  1. 理论计算不同模型/序列长度的 KV-Cache 大小
  2. 与模型权重大小对比, 展示 KV-Cache 的主导地位
  3. 计算不同 batch_size 下 KV-Cache 的显存占用 vs GPU 显存容量
  4. 找到 "OOM 临界点": 给定 GPU 显存, 最大可服务的 batch_size

论文用途: Figure 2 — "KV-Cache 随序列长度增长, 显存占用超越模型权重"
"""
import json
import os
import csv

# ============================================================
# 1. 模型配置
# ============================================================
MODELS = {
    "LLaMA-2-7B":   {"n_layers": 32, "n_kv_heads": 32, "d_head": 128, "params_B": 7,   "weight_GB": 14.0},
    "LLaMA-2-13B":  {"n_layers": 40, "n_kv_heads": 40, "d_head": 128, "params_B": 13,  "weight_GB": 26.0},
    "LLaMA-2-70B":  {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128, "params_B": 70,  "weight_GB": 140.0},
    "LLaMA-3-8B":   {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128, "params_B": 8,   "weight_GB": 16.0},
    "LLaMA-3-70B":  {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128, "params_B": 70,  "weight_GB": 140.0},
    "Mistral-7B":   {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128, "params_B": 7.3, "weight_GB": 14.6},
    "Qwen2-7B":     {"n_layers": 28, "n_kv_heads": 4,  "d_head": 128, "params_B": 7.6, "weight_GB": 15.2},
    "Qwen2-72B":    {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128, "params_B": 72,  "weight_GB": 144.0},
}

GPU_MEMORY_GB = {
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100-80GB": 80,
    "L40S-48GB": 48,
    "RTX4090-24GB": 24,
}

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# ============================================================
# 2. 核心计算函数
# ============================================================
def kvcache_size_bytes(n_layers, n_kv_heads, d_head, seq_len, batch_size=1, dtype_bytes=2):
    """KV-Cache 大小 (bytes). dtype_bytes=2 for FP16."""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * dtype_bytes

def kvcache_size_gb(n_layers, n_kv_heads, d_head, seq_len, batch_size=1, dtype_bytes=2):
    return kvcache_size_bytes(n_layers, n_kv_heads, d_head, seq_len, batch_size, dtype_bytes) / (1024**3)

def per_token_kv_mb(n_layers, n_kv_heads, d_head, dtype_bytes=2):
    """每个 token 的 KV-Cache 大小 (MB)."""
    return 2 * n_layers * n_kv_heads * d_head * dtype_bytes / (1024**2)

def max_batch_size(gpu_gb, weight_gb, n_layers, n_kv_heads, d_head, seq_len, overhead_gb=2, dtype_bytes=2):
    """给定 GPU 显存, 最大可服务 batch_size."""
    available = (gpu_gb - weight_gb - overhead_gb) * (1024**3)
    per_seq = kvcache_size_bytes(n_layers, n_kv_heads, d_head, seq_len, 1, dtype_bytes)
    if per_seq <= 0 or available <= 0:
        return 0
    return int(available // per_seq)

# ============================================================
# 3. 运行分析
# ============================================================
def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # --- Part A: 理论 KV-Cache 大小 vs 序列长度 ---
    print("=" * 90)
    print("Part A: KV-Cache 理论大小 (单请求, FP16)")
    print("=" * 90)
    print(f"{'Model':<18s} {'SeqLen':>8s} {'KV-Cache':>10s} {'Weight':>10s} {'KV/Weight':>10s} {'per-tok':>10s}")
    print("-" * 70)

    part_a = []
    for model_name, cfg in MODELS.items():
        pt_mb = per_token_kv_mb(cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"])
        for seq_len in SEQ_LENS:
            kv_gb = kvcache_size_gb(cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"], seq_len)
            ratio = kv_gb / cfg["weight_GB"] * 100
            row = {
                "model": model_name,
                "seq_len": seq_len,
                "kv_cache_GB": round(kv_gb, 4),
                "weight_GB": cfg["weight_GB"],
                "kv_weight_ratio_pct": round(ratio, 2),
                "per_token_MB": round(pt_mb, 4),
            }
            part_a.append(row)
            if seq_len in [2048, 8192, 32768, 131072]:
                print(f"{model_name:<18s} {seq_len:>8d} {kv_gb:>9.2f}G {cfg['weight_GB']:>9.1f}G {ratio:>9.1f}% {pt_mb:>9.4f}M")

    # --- Part B: 不同 batch_size 下 KV-Cache 总占用 ---
    print("\n" + "=" * 90)
    print("Part B: KV-Cache 总占用 (多 batch, seq=4096, FP16)")
    print("=" * 90)
    print(f"{'Model':<18s} {'Batch':>6s} {'KV-Cache':>10s} {'Weight':>10s} {'Total':>10s} {'A100-80G?':>10s}")
    print("-" * 70)

    part_b = []
    for model_name, cfg in MODELS.items():
        for bs in BATCH_SIZES:
            kv_gb = kvcache_size_gb(cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"], 4096, bs)
            total = kv_gb + cfg["weight_GB"]
            fits = "OK" if total < 80 else "OOM"
            row = {
                "model": model_name,
                "batch_size": bs,
                "seq_len": 4096,
                "kv_cache_GB": round(kv_gb, 2),
                "weight_GB": cfg["weight_GB"],
                "total_GB": round(total, 2),
                "fits_A100_80G": total < 80,
            }
            part_b.append(row)
            if bs in [1, 8, 32, 128]:
                print(f"{model_name:<18s} {bs:>6d} {kv_gb:>9.2f}G {cfg['weight_GB']:>9.1f}G {total:>9.1f}G {fits:>10s}")

    # --- Part C: OOM 临界点 ---
    print("\n" + "=" * 90)
    print("Part C: A100-80GB 最大 Batch Size (FP16, 2GB overhead)")
    print("=" * 90)
    print(f"{'Model':<18s}", end="")
    for sl in [2048, 4096, 8192, 16384, 32768, 131072]:
        print(f" {'seq='+str(sl):>12s}", end="")
    print()
    print("-" * 90)

    part_c = []
    for model_name, cfg in MODELS.items():
        print(f"{model_name:<18s}", end="")
        for sl in [2048, 4096, 8192, 16384, 32768, 131072]:
            mbs = max_batch_size(80, cfg["weight_GB"], cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"], sl)
            print(f" {mbs:>12d}", end="")
            part_c.append({
                "model": model_name, "gpu": "A100-80GB", "seq_len": sl,
                "max_batch_size": mbs,
            })
        print()

    # --- Part D: KV-Cache 超越模型权重的 "交叉点" ---
    print("\n" + "=" * 90)
    print("Part D: KV-Cache 超越模型权重的序列长度 (batch=1, FP16)")
    print("=" * 90)

    part_d = []
    for model_name, cfg in MODELS.items():
        cross_seq = None
        for sl in range(256, 262144, 256):
            kv_gb = kvcache_size_gb(cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"], sl)
            if kv_gb >= cfg["weight_GB"]:
                cross_seq = sl
                break
        print(f"  {model_name:<18s}: KV-Cache >= Weight at seq_len = {cross_seq if cross_seq else '>262144'}")
        part_d.append({"model": model_name, "crossover_seq_len": cross_seq})

    # --- Save all results ---
    with open(os.path.join(results_dir, "m1_kvcache_theory.json"), "w") as f:
        json.dump({
            "part_a_size_vs_seqlen": part_a,
            "part_b_batch_scaling": part_b,
            "part_c_max_batch": part_c,
            "part_d_crossover": part_d,
        }, f, indent=2)

    with open(os.path.join(results_dir, "m1_kvcache_size.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=part_a[0].keys())
        writer.writeheader()
        writer.writerows(part_a)

    print(f"\nResults saved to {results_dir}/m1_kvcache_theory.json")
    print(f"CSV saved to {results_dir}/m1_kvcache_size.csv")

if __name__ == "__main__":
    main()

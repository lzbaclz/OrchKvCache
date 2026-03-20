#!/usr/bin/env python3
"""
Exp-M3: KV-Cache Offloading IO 效率分析 (精简版)

模拟不同 eviction/reload 策略, 测量端到端带宽与 SSD 利用率。

论文用途: Figure 4 — 不同 IO 策略 vs SSD 峰值的带宽利用率对比
"""
import torch
import time
import json
import os
import gc
import shutil
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TEST_DIR = "/raid/orchkv_bench_m3"
DEVICE = "cuda:0"

SSD_PEAK_WRITE_GBPS = 5.3
SSD_PEAK_READ_GBPS = 17.8

WARMUP = 3
ITERS = 10

CONFIGS = [
    {"label": "vLLM_blk16_1layer",   "block_tok": 16,  "n_kv_heads": 8, "d_head": 128, "n_layers": 1},
    {"label": "vLLM_blk16_8layer",   "block_tok": 16,  "n_kv_heads": 8, "d_head": 128, "n_layers": 8},
    {"label": "vLLM_blk16_32layer",  "block_tok": 16,  "n_kv_heads": 8, "d_head": 128, "n_layers": 32},
    {"label": "blk64_1layer",        "block_tok": 64,  "n_kv_heads": 8, "d_head": 128, "n_layers": 1},
    {"label": "blk64_8layer",        "block_tok": 64,  "n_kv_heads": 8, "d_head": 128, "n_layers": 8},
    {"label": "blk64_32layer",       "block_tok": 64,  "n_kv_heads": 8, "d_head": 128, "n_layers": 32},
    {"label": "MHA32_blk16_32layer", "block_tok": 16,  "n_kv_heads": 32,"d_head": 128, "n_layers": 32},
    {"label": "MHA32_blk64_32layer", "block_tok": 64,  "n_kv_heads": 32,"d_head": 128, "n_layers": 32},
]

def block_bytes(bt, nkv, dh):
    return 2 * bt * nkv * dh * 2  # K+V, FP16

def naive_evict(blocks_gpu, cpu_bufs, test_dir):
    """Per-layer: D2H sync copy + file write."""
    for i, (g, c) in enumerate(zip(blocks_gpu, cpu_bufs)):
        c.copy_(g, non_blocking=False)
        torch.cuda.synchronize()
        fpath = os.path.join(test_dir, f"n_{i}.bin")
        fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, c.numpy().tobytes())
        os.fsync(fd)
        os.close(fd)

def batched_evict(blocks_gpu, big_cpu_buf, test_dir, per_block_elems):
    """All layers: D2H copy into contiguous buffer, single file write."""
    offset = 0
    for g in blocks_gpu:
        big_cpu_buf[offset:offset+per_block_elems].copy_(g.flatten(), non_blocking=False)
        offset += per_block_elems
    torch.cuda.synchronize()
    fpath = os.path.join(test_dir, "batched.bin")
    fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, big_cpu_buf.numpy().tobytes())
    os.fsync(fd)
    os.close(fd)

def naive_reload(n_layers, cpu_buf, gpu_buf, test_dir):
    """Per-layer: file read + H2D sync copy."""
    for i in range(n_layers):
        fpath = os.path.join(test_dir, f"r_{i}.bin")
        with open(fpath, "rb") as f:
            raw = f.read()
        cpu_buf.copy_(torch.frombuffer(bytearray(raw), dtype=torch.float16))
        gpu_buf.copy_(cpu_buf, non_blocking=False)
        torch.cuda.synchronize()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.cuda.init()

    print("=" * 90)
    print("M3: KV-Cache Offloading IO Efficiency Analysis")
    print("=" * 90)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SSD peak write: {SSD_PEAK_WRITE_GBPS} GB/s, read: {SSD_PEAK_READ_GBPS} GB/s (fio)")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print()

    results = []

    for cfg in CONFIGS:
        bb = block_bytes(cfg["block_tok"], cfg["n_kv_heads"], cfg["d_head"])
        nl = cfg["n_layers"]
        total = bb * nl
        total_mb = total / 1024**2

        print(f"\n--- {cfg['label']}: block={bb/1024:.0f}KB × {nl} layers = {total_mb:.1f} MB ---")

        # Prepare tensors
        blocks_gpu = [torch.randn(bb // 2, dtype=torch.float16, device=DEVICE) for _ in range(nl)]
        cpu_bufs = [torch.empty_like(b, device="cpu", pin_memory=True) for b in blocks_gpu]
        per_elems = bb // 2
        big_cpu = torch.empty(per_elems * nl, dtype=torch.float16, pin_memory=True)

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR)

        # --- Naive eviction ---
        for _ in range(WARMUP):
            naive_evict(blocks_gpu, cpu_bufs, TEST_DIR)
        lats = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            naive_evict(blocks_gpu, cpu_bufs, TEST_DIR)
            lats.append((time.perf_counter() - t0) * 1e6)
        naive_lat = np.mean(lats)
        naive_bw = total / (naive_lat * 1e-6) / 1e9

        # --- Batched eviction ---
        for _ in range(WARMUP):
            batched_evict(blocks_gpu, big_cpu, TEST_DIR, per_elems)
        lats = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batched_evict(blocks_gpu, big_cpu, TEST_DIR, per_elems)
            lats.append((time.perf_counter() - t0) * 1e6)
        batch_lat = np.mean(lats)
        batch_bw = total / (batch_lat * 1e-6) / 1e9

        # --- Prepare reload files ---
        for i in range(nl):
            fpath = os.path.join(TEST_DIR, f"r_{i}.bin")
            with open(fpath, "wb") as f:
                f.write(np.random.bytes(bb))

        cpu_r = torch.empty(bb // 2, dtype=torch.float16, pin_memory=True)
        gpu_r = torch.empty(bb // 2, dtype=torch.float16, device=DEVICE)

        for _ in range(WARMUP):
            naive_reload(nl, cpu_r, gpu_r, TEST_DIR)
        lats = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            naive_reload(nl, cpu_r, gpu_r, TEST_DIR)
            lats.append((time.perf_counter() - t0) * 1e6)
        reload_lat = np.mean(lats)
        reload_bw = total / (reload_lat * 1e-6) / 1e9

        print(f"  {'Strategy':<22s} {'Lat(us)':>12s} {'BW(GB/s)':>10s} {'SSD Util':>9s}")
        print(f"  {'-'*55}")
        print(f"  {'Naive evict':<22s} {naive_lat:>12.0f} {naive_bw:>10.3f} {naive_bw/SSD_PEAK_WRITE_GBPS*100:>8.1f}%")
        print(f"  {'Batched evict':<22s} {batch_lat:>12.0f} {batch_bw:>10.3f} {batch_bw/SSD_PEAK_WRITE_GBPS*100:>8.1f}%")
        print(f"  {'Naive reload':<22s} {reload_lat:>12.0f} {reload_bw:>10.3f} {reload_bw/SSD_PEAK_READ_GBPS*100:>8.1f}%")

        results.append({
            "config": cfg["label"],
            "block_bytes": bb,
            "n_layers": nl,
            "total_MB": round(total_mb, 2),
            "naive_evict":   {"lat_us": round(naive_lat),  "bw_gbps": round(naive_bw, 4), "util_pct": round(naive_bw/SSD_PEAK_WRITE_GBPS*100, 1)},
            "batched_evict": {"lat_us": round(batch_lat),  "bw_gbps": round(batch_bw, 4), "util_pct": round(batch_bw/SSD_PEAK_WRITE_GBPS*100, 1)},
            "naive_reload":  {"lat_us": round(reload_lat), "bw_gbps": round(reload_bw, 4), "util_pct": round(reload_bw/SSD_PEAK_READ_GBPS*100, 1)},
        })

        del blocks_gpu, cpu_bufs, big_cpu
        gc.collect()
        torch.cuda.empty_cache()

    shutil.rmtree(TEST_DIR, ignore_errors=True)

    out_path = os.path.join(RESULTS_DIR, "m3_io_efficiency.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

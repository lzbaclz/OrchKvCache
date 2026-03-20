#!/usr/bin/env python3
"""
Exp-M4: 存储层级延迟对比 + DRAM 缓冲层价值量化

汇总 Exp0 的数据 + 补充 DRAM-DRAM 拷贝基线, 生成完整的层级对比。
同时量化 DRAM 缓冲层作为"温存储"的价值。

论文用途: Figure 5 — 存储层级延迟柱状图 + 缓冲层分析
"""
import torch
import time
import json
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = "cuda:0"

SIZES = [
    ("64KB",   64 * 1024),
    ("256KB",  256 * 1024),
    ("1MB",    1 * 1024**2),
    ("4MB",    4 * 1024**2),
    ("16MB",   16 * 1024**2),
    ("64MB",   64 * 1024**2),
]

WARMUP = 50
ITERS = 200

def bench_gpu_internal(size_bytes):
    n = size_bytes // 2
    a = torch.randn(n, dtype=torch.float16, device=DEVICE)
    b = torch.empty_like(a)
    for _ in range(WARMUP):
        b.copy_(a); torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        b.copy_(a); torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / ITERS * 1e6
    bw = size_bytes / (lat * 1e-6) / 1e9
    return lat, bw

def bench_d2h_pinned(size_bytes):
    n = size_bytes // 2
    g = torch.randn(n, dtype=torch.float16, device=DEVICE)
    c = torch.empty(n, dtype=torch.float16, pin_memory=True)
    for _ in range(WARMUP):
        c.copy_(g, non_blocking=False); torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c.copy_(g, non_blocking=False); torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / ITERS * 1e6
    bw = size_bytes / (lat * 1e-6) / 1e9
    return lat, bw

def bench_h2d_pinned(size_bytes):
    n = size_bytes // 2
    c = torch.randn(n, dtype=torch.float16, pin_memory=True)
    g = torch.empty(n, dtype=torch.float16, device=DEVICE)
    for _ in range(WARMUP):
        g.copy_(c, non_blocking=False); torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        g.copy_(c, non_blocking=False); torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / ITERS * 1e6
    bw = size_bytes / (lat * 1e-6) / 1e9
    return lat, bw

def bench_dram_copy(size_bytes):
    src = torch.randn(size_bytes // 2, dtype=torch.float16, pin_memory=True)
    dst = torch.empty_like(src)
    for _ in range(WARMUP):
        dst.copy_(src)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        dst.copy_(src)
    lat = (time.perf_counter() - t0) / ITERS * 1e6
    bw = size_bytes / (lat * 1e-6) / 1e9
    return lat, bw

def bench_ssd_write(size_bytes, test_dir="/raid/orchkv_bench_m4"):
    os.makedirs(test_dir, exist_ok=True)
    data = os.urandom(size_bytes)
    fpath = os.path.join(test_dir, "test.bin")
    for _ in range(3):
        fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
    t0 = time.perf_counter()
    iters = min(ITERS, 50)
    for _ in range(iters):
        fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
    lat = (time.perf_counter() - t0) / iters * 1e6
    bw = size_bytes / (lat * 1e-6) / 1e9
    os.remove(fpath)
    return lat, bw

def bench_ssd_read(size_bytes, test_dir="/raid/orchkv_bench_m4"):
    os.makedirs(test_dir, exist_ok=True)
    fpath = os.path.join(test_dir, "test_read.bin")
    data = os.urandom(size_bytes)
    with open(fpath, "wb") as f:
        f.write(data)
    for _ in range(3):
        with open(fpath, "rb") as f:
            f.read()
    t0 = time.perf_counter()
    iters = min(ITERS, 50)
    for _ in range(iters):
        with open(fpath, "rb") as f:
            f.read()
    lat = (time.perf_counter() - t0) / iters * 1e6
    bw = size_bytes / (lat * 1e-6) / 1e9
    os.remove(fpath)
    return lat, bw

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.cuda.init()

    print("=" * 100)
    print("M4: Storage Tier Latency & Bandwidth Comparison")
    print("=" * 100)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = []

    print(f"{'Size':>8s} | {'GPU D2D':>14s} | {'D2H pin':>14s} | {'H2D pin':>14s} | {'DRAM copy':>14s} | {'SSD write':>14s} | {'SSD read':>14s}")
    print(f"{'':>8s} | {'lat/bw':>14s} | {'lat/bw':>14s} | {'lat/bw':>14s} | {'lat/bw':>14s} | {'lat/bw':>14s} | {'lat/bw':>14s}")
    print("-" * 110)

    for label, size in SIZES:
        d2d_lat, d2d_bw = bench_gpu_internal(size)
        d2h_lat, d2h_bw = bench_d2h_pinned(size)
        h2d_lat, h2d_bw = bench_h2d_pinned(size)
        dram_lat, dram_bw = bench_dram_copy(size)
        ssd_w_lat, ssd_w_bw = bench_ssd_write(size)
        ssd_r_lat, ssd_r_bw = bench_ssd_read(size)

        print(f"{label:>8s} | {d2d_lat:7.0f}/{d2d_bw:5.1f} | {d2h_lat:7.0f}/{d2h_bw:5.1f} | "
              f"{h2d_lat:7.0f}/{h2d_bw:5.1f} | {dram_lat:7.0f}/{dram_bw:5.1f} | "
              f"{ssd_w_lat:7.0f}/{ssd_w_bw:5.1f} | {ssd_r_lat:7.0f}/{ssd_r_bw:5.1f}")

        results.append({
            "size_label": label,
            "size_bytes": size,
            "gpu_d2d":    {"lat_us": round(d2d_lat, 1), "bw_gbps": round(d2d_bw, 2)},
            "d2h_pinned": {"lat_us": round(d2h_lat, 1), "bw_gbps": round(d2h_bw, 2)},
            "h2d_pinned": {"lat_us": round(h2d_lat, 1), "bw_gbps": round(h2d_bw, 2)},
            "dram_copy":  {"lat_us": round(dram_lat, 1), "bw_gbps": round(dram_bw, 2)},
            "ssd_write":  {"lat_us": round(ssd_w_lat, 1), "bw_gbps": round(ssd_w_bw, 2)},
            "ssd_read":   {"lat_us": round(ssd_r_lat, 1), "bw_gbps": round(ssd_r_bw, 2)},
        })

    # Compute ratios relative to GPU D2D
    print(f"\n{'='*80}")
    print("Latency ratios (relative to GPU D2D)")
    print(f"{'='*80}")
    print(f"{'Size':>8s} {'D2H/D2D':>10s} {'H2D/D2D':>10s} {'DRAM/D2D':>10s} {'SSDw/D2D':>10s} {'SSDr/D2D':>10s}")
    print("-" * 60)
    for r in results:
        d2d = r["gpu_d2d"]["lat_us"]
        if d2d < 1:
            d2d = 1
        print(f"{r['size_label']:>8s} "
              f"{r['d2h_pinned']['lat_us']/d2d:>10.1f}x "
              f"{r['h2d_pinned']['lat_us']/d2d:>10.1f}x "
              f"{r['dram_copy']['lat_us']/d2d:>10.1f}x "
              f"{r['ssd_write']['lat_us']/d2d:>10.1f}x "
              f"{r['ssd_read']['lat_us']/d2d:>10.1f}x")

    # DRAM buffering value: compare direct SSD access vs DRAM-cached
    print(f"\n{'='*80}")
    print("DRAM Buffer Value: SSD latency vs DRAM latency")
    print(f"{'='*80}")
    print(f"{'Size':>8s} {'SSD read(us)':>14s} {'DRAM read(us)':>14s} {'Speedup':>10s} {'SSD write(us)':>14s} {'DRAM write(us)':>14s} {'Speedup':>10s}")
    print("-" * 80)
    for r in results:
        ssd_r = r["ssd_read"]["lat_us"]
        dram_r = r["dram_copy"]["lat_us"]
        ssd_w = r["ssd_write"]["lat_us"]
        print(f"{r['size_label']:>8s} {ssd_r:>14.1f} {dram_r:>14.1f} {ssd_r/max(dram_r,0.1):>9.1f}x "
              f"{ssd_w:>14.1f} {dram_r:>14.1f} {ssd_w/max(dram_r,0.1):>9.1f}x")

    import shutil
    shutil.rmtree("/raid/orchkv_bench_m4", ignore_errors=True)

    out_path = os.path.join(RESULTS_DIR, "m4_tier_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

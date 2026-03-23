#!/usr/bin/env python3
"""
D4 / E8: Storage bandwidth benchmark.

Measures transfer bandwidth (GB/s) for each tier:
  - GPU HBM ↔ Host DRAM (cudaMemcpy)
  - Host DRAM ↔ tmpfs/NVM (memcpy / pwrite/pread)
  - DRAM ↔ SSD (pwrite/pread via OrchFS or POSIX)

Uses real CUDA operations for GPU↔DRAM.
Uses orchkv_core for full-stack DRAM↔Storage measurement.

Usage:
    python benchmarks/benchmark_storage_bw.py
    python benchmarks/benchmark_storage_bw.py --sizes 1,4,16,64
"""
from __future__ import annotations

import argparse
import time
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, save_csv, cuda_timer


def measure_gpu_dram(sizes_mb: list[float], n_iter: int = 10) -> list[dict]:
    """Measure GPU↔DRAM transfer bandwidth at various sizes."""
    results = []
    for size_mb in sizes_mb:
        n_bytes = int(size_mb * (1 << 20))
        n_elem = n_bytes // 2  # fp16

        gpu_buf = torch.randn(n_elem, dtype=torch.float16, device="cuda")
        cpu_buf = torch.empty(n_elem, dtype=torch.float16,
                              device="cpu", pin_memory=True)

        # GPU→DRAM (D2H)
        d2h_times = []
        for _ in range(n_iter):
            with cuda_timer() as t:
                cpu_buf.copy_(gpu_buf, non_blocking=True)
            d2h_times.append(t["elapsed_ms"])

        # DRAM→GPU (H2D)
        h2d_times = []
        for _ in range(n_iter):
            with cuda_timer() as t:
                gpu_buf.copy_(cpu_buf, non_blocking=True)
            h2d_times.append(t["elapsed_ms"])

        avg_d2h = sum(d2h_times[2:]) / max(len(d2h_times) - 2, 1)
        avg_h2d = sum(h2d_times[2:]) / max(len(h2d_times) - 2, 1)

        r = {
            "size_mb": size_mb,
            "d2h_avg_ms": round(avg_d2h, 3),
            "h2d_avg_ms": round(avg_h2d, 3),
            "d2h_gbps": round(size_mb / 1024 / (avg_d2h / 1000), 2) if avg_d2h > 0 else 0,
            "h2d_gbps": round(size_mb / 1024 / (avg_h2d / 1000), 2) if avg_h2d > 0 else 0,
        }
        results.append(r)
        print(f"  GPU↔DRAM {size_mb:6.1f}MB  "
              f"D2H={r['d2h_gbps']:6.2f}GB/s  "
              f"H2D={r['h2d_gbps']:6.2f}GB/s")

        del gpu_buf, cpu_buf

    return results


def measure_dram_storage(sizes_mb: list[float], n_iter: int = 10,
                          base_dir: str = "/dev/shm/orchkv_bench") -> list[dict]:
    """Measure DRAM↔tmpfs write/read bandwidth."""
    os.makedirs(base_dir, exist_ok=True)
    results = []

    for size_mb in sizes_mb:
        n_bytes = int(size_mb * (1 << 20))
        data = os.urandom(n_bytes)
        fpath = os.path.join(base_dir, f"bench_{size_mb}mb.bin")

        # Write
        write_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            with open(fpath, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            write_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Read
        read_times = []
        for _ in range(n_iter):
            os.popen(f"echo 3 > /proc/sys/vm/drop_caches 2>/dev/null").read()
            t0 = time.perf_counter_ns()
            with open(fpath, "rb") as f:
                _ = f.read()
            read_times.append((time.perf_counter_ns() - t0) / 1e6)

        os.unlink(fpath)

        avg_w = sum(write_times[2:]) / max(len(write_times) - 2, 1)
        avg_r = sum(read_times[2:]) / max(len(read_times) - 2, 1)

        r = {
            "size_mb": size_mb,
            "backend": "tmpfs",
            "write_avg_ms": round(avg_w, 3),
            "read_avg_ms": round(avg_r, 3),
            "write_gbps": round(size_mb / 1024 / (avg_w / 1000), 2) if avg_w > 0 else 0,
            "read_gbps": round(size_mb / 1024 / (avg_r / 1000), 2) if avg_r > 0 else 0,
        }
        results.append(r)
        print(f"  DRAM↔tmpfs {size_mb:6.1f}MB  "
              f"W={r['write_gbps']:6.2f}GB/s  "
              f"R={r['read_gbps']:6.2f}GB/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="E8 Storage Bandwidth")
    parser.add_argument("--sizes", default="0.5,1,2,4,8,16,32,64")
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    sizes = list(map(float, args.sizes.split(",")))

    print(f"\n{'='*60}")
    print(f"  E8: Storage Bandwidth Benchmark")
    print(f"{'='*60}\n")

    gpu_dram = measure_gpu_dram(sizes, args.n_iter)
    dram_stor = measure_dram_storage(sizes, args.n_iter)

    all_results = {
        "gpu_dram": gpu_dram,
        "dram_storage": dram_stor,
    }
    save_json(all_results, "benchmark_e8_storage_bw")

    if gpu_dram:
        save_csv(gpu_dram, "benchmark_e8_gpu_dram")
    if dram_stor:
        save_csv(dram_stor, "benchmark_e8_dram_storage")


if __name__ == "__main__":
    main()

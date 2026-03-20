#!/usr/bin/env python3
"""
Benchmark: GPU HBM <-> Host DRAM transfer latency and bandwidth.
Simulates KV-cache block offloading/loading scenarios.

Measures:
  1. GPU->CPU (D2H): cudaMemcpy device-to-host
  2. CPU->GPU (H2D): cudaMemcpy host-to-device
  3. GPU->GPU (D2D): intra-GPU copy baseline
  4. Pinned vs Non-pinned host memory comparison
  5. Different tensor sizes matching typical KV-cache block sizes
"""
import torch
import torch.cuda
import time
import json
import os
import sys

WARMUP = 50
ITERATIONS = 200

KV_BLOCK_CONFIGS = [
    ("1-token_1head",    2 * 1 * 1 * 128),          # 256 B
    ("64tok_GQA8",       2 * 64 * 8 * 128),          # 128 KB  (typical vLLM block)
    ("64tok_MHA32",      2 * 64 * 32 * 128),         # 512 KB
    ("256tok_MHA32",     2 * 256 * 32 * 128),        # 2 MB
    ("1024tok_MHA32",    2 * 1024 * 32 * 128),       # 8 MB
    ("4096tok_MHA32",    2 * 4096 * 32 * 128),       # 32 MB
    ("16K_tok_MHA32",    2 * 16384 * 32 * 128),      # 128 MB
    ("32MB_flat",        32 * 1024 * 1024 // 2),     # 32 MB
    ("128MB_flat",       128 * 1024 * 1024 // 2),    # 128 MB
    ("512MB_flat",       512 * 1024 * 1024 // 2),    # 512 MB
]

def fmt_size(n_bytes):
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024**2:
        return f"{n_bytes/1024:.1f} KB"
    elif n_bytes < 1024**3:
        return f"{n_bytes/1024**2:.1f} MB"
    else:
        return f"{n_bytes/1024**3:.2f} GB"

def bench_transfer(src, dst, size_elements, warmup=WARMUP, iters=ITERATIONS):
    """Time a .copy_ between two tensors, return (latency_us, bw_GBps)."""
    s = torch.randn(size_elements, dtype=torch.float16, device=src)
    d = torch.empty_like(s) if src == dst else torch.empty(size_elements, dtype=torch.float16, device=dst)

    for _ in range(warmup):
        d.copy_(s, non_blocking=False)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        d.copy_(s, non_blocking=False)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    lat_us = elapsed / iters * 1e6
    nbytes = size_elements * 2  # fp16
    bw_gbps = nbytes / (elapsed / iters) / 1e9
    return lat_us, bw_gbps

def bench_pinned_transfer(direction, size_elements, warmup=WARMUP, iters=ITERATIONS):
    """Bench with pinned host memory."""
    if direction == "H2D":
        host_t = torch.randn(size_elements, dtype=torch.float16, pin_memory=True)
        dev_t = torch.empty(size_elements, dtype=torch.float16, device="cuda:0")
        src, dst = host_t, dev_t
    else:
        dev_t = torch.randn(size_elements, dtype=torch.float16, device="cuda:0")
        host_t = torch.empty(size_elements, dtype=torch.float16, pin_memory=True)
        src, dst = dev_t, host_t

    for _ in range(warmup):
        dst.copy_(src, non_blocking=False)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=False)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    lat_us = elapsed / iters * 1e6
    nbytes = size_elements * 2
    bw_gbps = nbytes / (elapsed / iters) / 1e9
    return lat_us, bw_gbps

def bench_async_transfer(direction, size_elements, warmup=WARMUP, iters=ITERATIONS):
    """Bench async (non_blocking=True) with pinned memory and CUDA events."""
    stream = torch.cuda.Stream()
    if direction == "H2D":
        host_t = torch.randn(size_elements, dtype=torch.float16, pin_memory=True)
        dev_t = torch.empty(size_elements, dtype=torch.float16, device="cuda:0")
        src, dst = host_t, dev_t
    else:
        dev_t = torch.randn(size_elements, dtype=torch.float16, device="cuda:0")
        host_t = torch.empty(size_elements, dtype=torch.float16, pin_memory=True)
        src, dst = dev_t, host_t

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()

    latencies = []
    for _ in range(iters):
        start_event.record(stream)
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        end_event.record(stream)
        stream.synchronize()
        latencies.append(start_event.elapsed_time(end_event) * 1000)  # ms -> us

    avg_lat = sum(latencies) / len(latencies)
    nbytes = size_elements * 2
    bw_gbps = nbytes / (avg_lat * 1e-6) / 1e9
    return avg_lat, bw_gbps

def main():
    print("=" * 80)
    print("GPU HBM <-> Host DRAM Transfer Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERATIONS}")
    print()

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "benchmarks": []
    }

    directions = [
        ("D2H (GPU->CPU) pageable",   "cuda:0", "cpu",    False, False),
        ("H2D (CPU->GPU) pageable",   "cpu",    "cuda:0", False, False),
        ("D2H (GPU->CPU) pinned",     None,     None,     True,  False),
        ("H2D (CPU->GPU) pinned",     None,     None,     True,  False),
        ("D2H (GPU->CPU) async+pin",  None,     None,     False, True),
        ("H2D (CPU->GPU) async+pin",  None,     None,     False, True),
        ("D2D (GPU->GPU) baseline",   "cuda:0", "cuda:0", False, False),
    ]

    for label, src, dst, is_pinned, is_async in directions:
        print(f"\n--- {label} ---")
        print(f"{'Size':>20s} {'Latency(us)':>14s} {'BW(GB/s)':>12s}")
        print("-" * 50)
        for name, n_elem in KV_BLOCK_CONFIGS:
            nbytes = n_elem * 2
            try:
                if is_async:
                    d = "D2H" if "D2H" in label else "H2D"
                    lat, bw = bench_async_transfer(d, n_elem)
                elif is_pinned:
                    d = "D2H" if "D2H" in label else "H2D"
                    lat, bw = bench_pinned_transfer(d, n_elem)
                else:
                    lat, bw = bench_transfer(src, dst, n_elem)
                print(f"{name+' ('+fmt_size(nbytes)+')':>20s} {lat:>14.1f} {bw:>12.2f}")
                results["benchmarks"].append({
                    "direction": label,
                    "config": name,
                    "size_bytes": nbytes,
                    "latency_us": round(lat, 2),
                    "bandwidth_gbps": round(bw, 3),
                })
            except Exception as e:
                print(f"{name:>20s}  ERROR: {e}")
                results["benchmarks"].append({
                    "direction": label,
                    "config": name,
                    "size_bytes": nbytes,
                    "error": str(e),
                })

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bench_gpu_dram.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

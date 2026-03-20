#!/usr/bin/env python3
"""
End-to-end KV-cache offload path benchmark.

Measures the full pipeline latency of typical KV-cache migration operations:
  1. GPU HBM -> Host DRAM -> SSD (eviction / cold-path)
  2. SSD -> Host DRAM -> GPU HBM (loading / hot-path)
  3. GPU HBM -> Host DRAM (warm offload)
  4. Host DRAM -> GPU HBM (warm reload)

For each path, measures:
  - Total latency (us)
  - Effective bandwidth (GB/s)
  - Breakdown per hop

Simulates realistic KV-cache block sizes for:
  - Llama-2-7B:  n_heads=32, d_head=128, GQA=32 -> per-layer KV block
  - Llama-2-70B: n_heads=64, d_head=128, GQA=8  -> per-layer KV block
  - Mixtral-8x7B: n_heads=32, d_head=128, GQA=8
"""
import torch
import time
import json
import os
import tempfile

WARMUP = 20
ITERATIONS = 100

MODEL_CONFIGS = {
    "Llama-7B_bs1_seq256": {
        "desc": "Llama-2-7B, batch=1, 256 tokens, 1 layer KV",
        "size_elements": 2 * 256 * 32 * 128,   # K+V, seq=256, heads=32, d=128
    },
    "Llama-7B_bs1_seq2048": {
        "desc": "Llama-2-7B, batch=1, 2048 tokens, 1 layer KV",
        "size_elements": 2 * 2048 * 32 * 128,
    },
    "Llama-7B_bs1_seq4096": {
        "desc": "Llama-2-7B, batch=1, 4096 tokens, 1 layer KV",
        "size_elements": 2 * 4096 * 32 * 128,
    },
    "Llama-70B_bs1_seq256": {
        "desc": "Llama-2-70B, batch=1, 256 tokens, 1 layer KV (GQA=8)",
        "size_elements": 2 * 256 * 8 * 128,
    },
    "Llama-70B_bs1_seq2048": {
        "desc": "Llama-2-70B, batch=1, 2048 tokens, 1 layer KV (GQA=8)",
        "size_elements": 2 * 2048 * 8 * 128,
    },
    "vLLM_block_16tok": {
        "desc": "vLLM default block (16 tokens), MHA-32, 1 layer",
        "size_elements": 2 * 16 * 32 * 128,
    },
    "vLLM_block_64tok": {
        "desc": "vLLM block (64 tokens), MHA-32, 1 layer",
        "size_elements": 2 * 64 * 32 * 128,
    },
}

SSD_TEST_DIR = "/raid/orchkv_bench"

def fmt_size(n_bytes):
    if n_bytes < 1024: return f"{n_bytes}B"
    if n_bytes < 1024**2: return f"{n_bytes/1024:.1f}KB"
    if n_bytes < 1024**3: return f"{n_bytes/1024**2:.1f}MB"
    return f"{n_bytes/1024**3:.2f}GB"

def bench_gpu_to_dram_to_ssd(size_elements, ssd_dir, warmup=WARMUP, iters=ITERATIONS):
    """Eviction path: GPU -> pinned DRAM -> SSD file."""
    gpu_tensor = torch.randn(size_elements, dtype=torch.float16, device="cuda:0")
    cpu_tensor = torch.empty(size_elements, dtype=torch.float16, pin_memory=True)
    fpath = os.path.join(ssd_dir, "evict_test.bin")
    nbytes = size_elements * 2

    for _ in range(warmup):
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        torch.cuda.synchronize()
        with open(fpath, "wb") as f:
            f.write(cpu_tensor.numpy().tobytes())

    lats_d2h = []
    lats_disk = []
    lats_total = []

    for _ in range(iters):
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        torch.cuda.synchronize()
        t_d2h = time.perf_counter()

        fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, cpu_tensor.numpy().tobytes())
        os.fsync(fd)
        os.close(fd)
        t_end = time.perf_counter()

        lats_d2h.append((t_d2h - t_start) * 1e6)
        lats_disk.append((t_end - t_d2h) * 1e6)
        lats_total.append((t_end - t_start) * 1e6)

    os.remove(fpath)
    return {
        "d2h_us": round(sum(lats_d2h) / len(lats_d2h), 1),
        "disk_write_us": round(sum(lats_disk) / len(lats_disk), 1),
        "total_us": round(sum(lats_total) / len(lats_total), 1),
        "total_bw_gbps": round(nbytes / (sum(lats_total) / len(lats_total) * 1e-6) / 1e9, 3),
    }

def bench_ssd_to_dram_to_gpu(size_elements, ssd_dir, warmup=WARMUP, iters=ITERATIONS):
    """Loading path: SSD file -> pinned DRAM -> GPU."""
    gpu_tensor = torch.empty(size_elements, dtype=torch.float16, device="cuda:0")
    cpu_tensor = torch.empty(size_elements, dtype=torch.float16, pin_memory=True)
    fpath = os.path.join(ssd_dir, "load_test.bin")
    nbytes = size_elements * 2

    # Write test file
    data = torch.randn(size_elements, dtype=torch.float16)
    with open(fpath, "wb") as f:
        f.write(data.numpy().tobytes())

    for _ in range(warmup):
        with open(fpath, "rb") as f:
            raw = f.read()
        cpu_tensor.copy_(torch.frombuffer(bytearray(raw), dtype=torch.float16))
        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
        torch.cuda.synchronize()

    lats_disk = []
    lats_h2d = []
    lats_total = []

    for _ in range(iters):
        # drop page cache if possible
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        except PermissionError:
            pass

        t_start = time.perf_counter()
        with open(fpath, "rb") as f:
            raw = f.read()
        cpu_tensor.copy_(torch.frombuffer(bytearray(raw), dtype=torch.float16))
        t_disk = time.perf_counter()

        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        lats_disk.append((t_disk - t_start) * 1e6)
        lats_h2d.append((t_end - t_disk) * 1e6)
        lats_total.append((t_end - t_start) * 1e6)

    os.remove(fpath)
    return {
        "disk_read_us": round(sum(lats_disk) / len(lats_disk), 1),
        "h2d_us": round(sum(lats_h2d) / len(lats_h2d), 1),
        "total_us": round(sum(lats_total) / len(lats_total), 1),
        "total_bw_gbps": round(nbytes / (sum(lats_total) / len(lats_total) * 1e-6) / 1e9, 3),
    }

def bench_gpu_dram_roundtrip(size_elements, warmup=WARMUP, iters=ITERATIONS):
    """Warm tier: GPU <-> pinned DRAM."""
    gpu_tensor = torch.randn(size_elements, dtype=torch.float16, device="cuda:0")
    cpu_tensor = torch.empty(size_elements, dtype=torch.float16, pin_memory=True)
    gpu_reload = torch.empty(size_elements, dtype=torch.float16, device="cuda:0")
    nbytes = size_elements * 2

    for _ in range(warmup):
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        torch.cuda.synchronize()
        gpu_reload.copy_(cpu_tensor, non_blocking=False)
        torch.cuda.synchronize()

    lats_offload = []
    lats_reload = []

    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_reload.copy_(cpu_tensor, non_blocking=False)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        lats_offload.append((t1 - t0) * 1e6)
        lats_reload.append((t2 - t1) * 1e6)

    return {
        "offload_d2h_us": round(sum(lats_offload) / len(lats_offload), 1),
        "reload_h2d_us": round(sum(lats_reload) / len(lats_reload), 1),
        "offload_bw_gbps": round(nbytes / (sum(lats_offload) / len(lats_offload) * 1e-6) / 1e9, 3),
        "reload_bw_gbps": round(nbytes / (sum(lats_reload) / len(lats_reload) * 1e-6) / 1e9, 3),
    }

def main():
    print("=" * 80)
    print("End-to-End KV-Cache Offload Path Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SSD test dir: {SSD_TEST_DIR}")
    print()

    os.makedirs(SSD_TEST_DIR, exist_ok=True)

    results = {"e2e_benchmarks": []}

    for config_name, cfg in MODEL_CONFIGS.items():
        n_elem = cfg["size_elements"]
        nbytes = n_elem * 2
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  {cfg['desc']}")
        print(f"  Size: {fmt_size(nbytes)}")
        print(f"{'='*60}")

        entry = {"config": config_name, "desc": cfg["desc"], "size_bytes": nbytes}

        # GPU <-> DRAM (warm tier)
        print("\n  [Warm] GPU <-> DRAM (pinned):")
        warm = bench_gpu_dram_roundtrip(n_elem)
        print(f"    Offload (D2H): {warm['offload_d2h_us']:.1f} us  ({warm['offload_bw_gbps']:.3f} GB/s)")
        print(f"    Reload  (H2D): {warm['reload_h2d_us']:.1f} us  ({warm['reload_bw_gbps']:.3f} GB/s)")
        entry["warm_gpu_dram"] = warm

        # GPU -> DRAM -> SSD (eviction)
        print("\n  [Cold] GPU -> DRAM -> SSD (eviction):")
        evict = bench_gpu_to_dram_to_ssd(n_elem, SSD_TEST_DIR)
        print(f"    D2H:        {evict['d2h_us']:.1f} us")
        print(f"    Disk write: {evict['disk_write_us']:.1f} us")
        print(f"    Total:      {evict['total_us']:.1f} us  ({evict['total_bw_gbps']:.3f} GB/s)")
        entry["cold_evict"] = evict

        # SSD -> DRAM -> GPU (loading)
        print("\n  [Cold] SSD -> DRAM -> GPU (loading):")
        load = bench_ssd_to_dram_to_gpu(n_elem, SSD_TEST_DIR)
        print(f"    Disk read: {load['disk_read_us']:.1f} us")
        print(f"    H2D:       {load['h2d_us']:.1f} us")
        print(f"    Total:     {load['total_us']:.1f} us  ({load['total_bw_gbps']:.3f} GB/s)")
        entry["cold_load"] = load

        results["e2e_benchmarks"].append(entry)

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bench_e2e_offload.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

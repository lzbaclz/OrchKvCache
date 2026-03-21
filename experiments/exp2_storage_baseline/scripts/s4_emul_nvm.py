#!/usr/bin/env python3
"""
S4: DRAM-emulated NVM Latency/Bandwidth Benchmark
Emulates NVM behavior using DRAM with optional delay injection.

Since we don't have real NVM (Intel Optane PM), we measure
DRAM performance at NVM-relevant granularities and estimate
what NVM-tier performance would look like with delay injection.

Reference NVM latencies (Intel Optane PM):
  - Sequential read:  ~6-8 GB/s
  - Sequential write: ~2-3 GB/s
  - Random 4KB read:  ~300ns
  - Random 4KB write: ~100ns (before wear-leveling)
"""

import os
import time
import json
import mmap
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

REGION_SIZE = 1 * 1024 * 1024 * 1024  # 1GB mmap region
ACCESS_SIZES = [256, 4096, 32768, 65536, 262144, 1048576]
ITERATIONS = 100000
LARGE_ITERATIONS = 10000

NVM_LITERATURE = {
    'optane_seq_read_gbps': 6.6,
    'optane_seq_write_gbps': 2.3,
    'optane_rand_read_4k_ns': 305,
    'optane_rand_write_4k_ns': 94,
    'optane_rand_read_4k_gbps': 1.7,
    'optane_rand_write_4k_gbps': 0.5,
    'optane_idle_lat_ns': 169,
    'source': 'Intel Optane DC PMM Datasheet + Izraelevitz et al. ASPLOS 2019',
}


def bench_dram_sequential_rw(region_size, block_size, mode='read'):
    """Measure DRAM sequential read/write bandwidth."""
    buf = mmap.mmap(-1, region_size)
    data = os.urandom(block_size)
    n_blocks = region_size // block_size

    if mode == 'write':
        start = time.perf_counter()
        for i in range(n_blocks):
            buf.seek(i * block_size)
            buf.write(data)
        elapsed = time.perf_counter() - start
    else:
        buf.write(os.urandom(region_size))
        buf.seek(0)
        start = time.perf_counter()
        for i in range(n_blocks):
            buf.seek(i * block_size)
            buf.read(block_size)
        elapsed = time.perf_counter() - start

    buf.close()
    bw = region_size / elapsed / 1e9
    return bw, elapsed


def bench_dram_random_latency(region_size, access_size, n_accesses):
    """Measure DRAM random access latency at given granularity."""
    buf = mmap.mmap(-1, region_size)
    buf.write(os.urandom(region_size))

    max_offset = region_size - access_size
    offsets = np.random.randint(0, max_offset // access_size, size=n_accesses) * access_size

    latencies = []
    for off in offsets[:min(n_accesses, 50000)]:
        start = time.perf_counter_ns()
        buf.seek(int(off))
        buf.read(access_size)
        lat = time.perf_counter_ns() - start
        latencies.append(lat)

    buf.close()
    latencies = np.array(latencies)
    return {
        'avg_ns': round(float(np.mean(latencies)), 1),
        'p50_ns': round(float(np.median(latencies)), 1),
        'p99_ns': round(float(np.percentile(latencies, 99)), 1),
        'min_ns': round(float(np.min(latencies)), 1),
        'max_ns': round(float(np.max(latencies)), 1),
    }


def estimate_with_nvm_delay(dram_result, access_size):
    """Estimate NVM-tier performance by adding typical NVM penalty."""
    nvm_extra_ns = 200 if access_size <= 4096 else 150
    return {
        'estimated_avg_ns': round(dram_result['avg_ns'] + nvm_extra_ns, 1),
        'estimated_p50_ns': round(dram_result['p50_ns'] + nvm_extra_ns, 1),
        'estimated_p99_ns': round(dram_result['p99_ns'] + nvm_extra_ns, 1),
        'nvm_extra_ns': nvm_extra_ns,
        'note': 'DRAM latency + estimated NVM penalty (Optane PM typical)',
    }


def main():
    results = {
        'nvm_literature': NVM_LITERATURE,
        'dram_sequential': [],
        'dram_random_latency': [],
        'nvm_estimated': [],
        'tier_comparison': [],
    }

    print("=" * 60)
    print("S4: DRAM-emulated NVM Benchmark")
    print("=" * 60)

    print("\n--- Sequential Read/Write Bandwidth ---")
    for bs in [4096, 32768, 65536, 262144, 1048576]:
        bs_label = f"{bs//1024}KB" if bs < 1048576 else f"{bs//1048576}MB"
        for mode in ['read', 'write']:
            bw, elapsed = bench_dram_sequential_rw(REGION_SIZE, bs, mode)
            results['dram_sequential'].append({
                'block_size': bs,
                'block_size_label': bs_label,
                'mode': mode,
                'bw_gbps': round(bw, 3),
            })
            print(f"  {mode:>5s} {bs_label:>6s}: {bw:.3f} GB/s")

    print("\n--- Random Access Latency ---")
    for access_size in ACCESS_SIZES:
        sz_label = f"{access_size}B" if access_size < 1024 else (f"{access_size//1024}KB" if access_size < 1048576 else f"{access_size//1048576}MB")
        n = ITERATIONS if access_size <= 4096 else LARGE_ITERATIONS

        lat = bench_dram_random_latency(REGION_SIZE, access_size, n)
        results['dram_random_latency'].append({
            'access_size': access_size,
            'access_size_label': sz_label,
            **lat,
        })

        nvm_est = estimate_with_nvm_delay(lat, access_size)
        results['nvm_estimated'].append({
            'access_size': access_size,
            'access_size_label': sz_label,
            **nvm_est,
        })

        print(f"  {sz_label:>6s}: DRAM avg={lat['avg_ns']:.0f}ns p50={lat['p50_ns']:.0f}ns p99={lat['p99_ns']:.0f}ns | "
              f"NVM est={nvm_est['estimated_avg_ns']:.0f}ns")

    print("\n--- Tier Comparison (4KB random read, for KV-Cache reload) ---")
    dram_4k = next((r for r in results['dram_random_latency'] if r['access_size'] == 4096), None)
    nvm_4k = next((r for r in results['nvm_estimated'] if r['access_size'] == 4096), None)

    ssd_read_lat_us = 158.08
    ssd_read_lat_ns = ssd_read_lat_us * 1000

    if dram_4k and nvm_4k:
        tier_data = {
            'access_size': 4096,
            'dram_ns': dram_4k['avg_ns'],
            'nvm_estimated_ns': nvm_4k['estimated_avg_ns'],
            'ssd_ns': ssd_read_lat_ns,
            'nvm_vs_ssd_speedup': round(ssd_read_lat_ns / nvm_4k['estimated_avg_ns'], 1),
            'dram_vs_ssd_speedup': round(ssd_read_lat_ns / dram_4k['avg_ns'], 1),
            'dram_vs_nvm_speedup': round(nvm_4k['estimated_avg_ns'] / dram_4k['avg_ns'], 1),
        }
        results['tier_comparison'].append(tier_data)

        print(f"  DRAM:          {dram_4k['avg_ns']:.0f} ns")
        print(f"  NVM (est):     {nvm_4k['estimated_avg_ns']:.0f} ns  ({tier_data['dram_vs_nvm_speedup']}x slower than DRAM)")
        print(f"  SSD:           {ssd_read_lat_ns:.0f} ns  ({tier_data['nvm_vs_ssd_speedup']}x slower than NVM)")
        print(f"  SSD vs DRAM:   {tier_data['dram_vs_ssd_speedup']}x")

    print("\n--- Tier Comparison (32KB, for KV block reload) ---")
    dram_32k = next((r for r in results['dram_random_latency'] if r['access_size'] == 32768), None)
    nvm_32k = next((r for r in results['nvm_estimated'] if r['access_size'] == 32768), None)

    ssd_32k_lat_us = 250.5
    ssd_32k_lat_ns = ssd_32k_lat_us * 1000

    if dram_32k and nvm_32k:
        tier_data_32k = {
            'access_size': 32768,
            'dram_ns': dram_32k['avg_ns'],
            'nvm_estimated_ns': nvm_32k['estimated_avg_ns'],
            'ssd_ns': ssd_32k_lat_ns,
            'nvm_vs_ssd_speedup': round(ssd_32k_lat_ns / nvm_32k['estimated_avg_ns'], 1),
            'dram_vs_ssd_speedup': round(ssd_32k_lat_ns / dram_32k['avg_ns'], 1),
        }
        results['tier_comparison'].append(tier_data_32k)

        print(f"  DRAM:          {dram_32k['avg_ns']:.0f} ns")
        print(f"  NVM (est):     {nvm_32k['estimated_avg_ns']:.0f} ns")
        print(f"  SSD:           {ssd_32k_lat_ns:.0f} ns  ({tier_data_32k['nvm_vs_ssd_speedup']}x slower than NVM)")

    out_path = os.path.join(RESULTS_DIR, 's4_emul_nvm.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

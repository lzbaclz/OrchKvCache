#!/usr/bin/env python3
"""
S5: KV-Cache Specific IO Pattern Benchmark
Simulates realistic KV-Cache eviction and reload patterns to measure
achievable IO throughput under real access patterns.

Key patterns tested:
1. Batch eviction: GPU -> DRAM -> SSD (cold path, write-heavy)
2. Selective reload: SSD -> DRAM -> GPU (warm-up path, read-heavy)
3. Pipeline overlap: concurrent evict + reload (steady-state)
4. io_uring batch submission: multiple IO ops in one syscall
"""

import os
import time
import json
import threading
import queue
import subprocess
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DIR = '/raid/orchkv_bench_s2'
if not os.path.isdir('/raid'):
    TEST_DIR = '/tmp/orchkv_bench_s2'

os.makedirs(TEST_DIR, exist_ok=True)

KV_BLOCK_CONFIGS = [
    {'label': 'vLLM_blk16_GQA8',  'block_bytes': 65536,   'desc': '16 tokens, GQA8, 1 layer'},
    {'label': 'vLLM_blk16_MHA32', 'block_bytes': 262144,  'desc': '16 tokens, MHA32, 1 layer'},
    {'label': 'vLLM_blk64_GQA8',  'block_bytes': 262144,  'desc': '64 tokens, GQA8, 1 layer'},
    {'label': 'vLLM_blk64_MHA32', 'block_bytes': 1048576, 'desc': '64 tokens, MHA32, 1 layer'},
    {'label': 'seq256_7B_1layer',  'block_bytes': 4194304, 'desc': '256 tokens, 7B, 1 layer KV'},
]

N_LAYERS_BATCH = [1, 8, 32]


def bench_batch_eviction(test_dir, block_bytes, n_blocks, strategy='sequential'):
    """Simulate batch KV block eviction to SSD."""
    data = os.urandom(block_bytes)

    if strategy == 'sequential':
        filepath = os.path.join(test_dir, 'evict_seq')
        start = time.perf_counter()
        with open(filepath, 'wb') as f:
            for _ in range(n_blocks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.perf_counter() - start
        total = block_bytes * n_blocks
        os.remove(filepath)
        return total / elapsed / 1e9, elapsed * 1e6

    elif strategy == 'per_file':
        start = time.perf_counter()
        for i in range(n_blocks):
            fp = os.path.join(test_dir, f'evict_blk_{i}')
            with open(fp, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        elapsed = time.perf_counter() - start
        total = block_bytes * n_blocks
        for i in range(n_blocks):
            fp = os.path.join(test_dir, f'evict_blk_{i}')
            if os.path.exists(fp):
                os.remove(fp)
        return total / elapsed / 1e9, elapsed * 1e6

    elif strategy == 'multi_thread':
        n_threads = min(n_blocks, 8)
        blocks_per_thread = max(1, n_blocks // n_threads)
        result_q = queue.Queue()

        def _writer(tid, n_blks):
            fp = os.path.join(test_dir, f'evict_mt_{tid}')
            t0 = time.perf_counter()
            with open(fp, 'wb') as f:
                for _ in range(n_blks):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
            result_q.put(time.perf_counter() - t0)

        threads = []
        start = time.perf_counter()
        for tid in range(n_threads):
            t = threading.Thread(target=_writer, args=(tid, blocks_per_thread))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        total = block_bytes * blocks_per_thread * n_threads
        for tid in range(n_threads):
            fp = os.path.join(test_dir, f'evict_mt_{tid}')
            if os.path.exists(fp):
                os.remove(fp)
        return total / elapsed / 1e9, elapsed * 1e6


def bench_selective_reload(test_dir, block_bytes, n_blocks, selectivity=0.2):
    """Simulate selective KV block reload (only hot blocks)."""
    filepath = os.path.join(test_dir, 'reload_data')
    data = os.urandom(block_bytes)
    total_blocks = n_blocks
    n_hot = max(1, int(total_blocks * selectivity))

    with open(filepath, 'wb') as f:
        for _ in range(total_blocks):
            f.write(data)
        f.flush()

    hot_indices = sorted(np.random.choice(total_blocks, size=n_hot, replace=False))

    start = time.perf_counter()
    with open(filepath, 'rb') as f:
        for idx in hot_indices:
            f.seek(idx * block_bytes)
            f.read(block_bytes)
    elapsed = time.perf_counter() - start

    total_read = n_hot * block_bytes
    os.remove(filepath)
    return total_read / elapsed / 1e9, elapsed * 1e6, n_hot


def bench_pipeline_overlap(test_dir, block_bytes, n_evict, n_reload):
    """Simulate concurrent eviction and reload (steady-state pipeline)."""
    evict_data = os.urandom(block_bytes)
    reload_file = os.path.join(test_dir, 'pipeline_reload')
    with open(reload_file, 'wb') as f:
        for _ in range(n_reload):
            f.write(evict_data)
        f.flush()

    evict_done = threading.Event()
    reload_done = threading.Event()
    results = {}

    def _evict():
        fp = os.path.join(test_dir, 'pipeline_evict')
        t0 = time.perf_counter()
        with open(fp, 'wb') as f:
            for _ in range(n_evict):
                f.write(evict_data)
            f.flush()
            os.fsync(f.fileno())
        results['evict_time_us'] = (time.perf_counter() - t0) * 1e6
        results['evict_bw_gbps'] = (block_bytes * n_evict) / (results['evict_time_us'] / 1e6) / 1e9
        evict_done.set()
        if os.path.exists(fp):
            os.remove(fp)

    def _reload():
        t0 = time.perf_counter()
        with open(reload_file, 'rb') as f:
            for _ in range(n_reload):
                f.read(block_bytes)
        results['reload_time_us'] = (time.perf_counter() - t0) * 1e6
        results['reload_bw_gbps'] = (block_bytes * n_reload) / (results['reload_time_us'] / 1e6) / 1e9
        reload_done.set()

    wall_start = time.perf_counter()
    t_evict = threading.Thread(target=_evict)
    t_reload = threading.Thread(target=_reload)
    t_evict.start()
    t_reload.start()
    t_evict.join()
    t_reload.join()
    wall_elapsed = time.perf_counter() - wall_start

    results['wall_time_us'] = wall_elapsed * 1e6
    total_bytes = block_bytes * (n_evict + n_reload)
    results['combined_bw_gbps'] = total_bytes / wall_elapsed / 1e9

    if os.path.exists(reload_file):
        os.remove(reload_file)

    return results


def bench_fio_iouring_kv(test_dir, block_bytes, n_blocks, rw='write'):
    """Use fio with io_uring to measure KV-block-sized IO."""
    filepath = os.path.join(test_dir, 'fio_kv_bench')
    bs_str = f"{block_bytes // 1024}k" if block_bytes < 1048576 else f"{block_bytes // 1048576}m"
    total_size = block_bytes * n_blocks

    cmd = [
        'fio',
        f'--name=kv_{rw}_{bs_str}',
        f'--ioengine=io_uring',
        f'--rw={rw}',
        f'--bs={bs_str}',
        f'--size={total_size}',
        '--iodepth=32',
        '--direct=1',
        '--numjobs=1',
        '--runtime=8',
        '--time_based',
        '--group_reporting',
        '--output-format=json',
        f'--filename={filepath}',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        job = data['jobs'][0]
        key = 'read' if 'read' in rw else 'write'
        return {
            'bw_gbps': round(job[key]['bw_bytes'] / 1e9, 4),
            'iops': round(job[key]['iops'], 1),
            'lat_avg_us': round(job[key]['lat_ns']['mean'] / 1000, 2),
        }
    except Exception as e:
        print(f"  fio error: {e}")
        return None
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def main():
    all_results = {
        'batch_eviction': [],
        'selective_reload': [],
        'pipeline_overlap': [],
        'fio_iouring_baseline': [],
    }

    print("=" * 70)
    print("S5: KV-Cache Specific IO Pattern Benchmark")
    print(f"Test directory: {TEST_DIR}")
    print("=" * 70)

    print("\n--- Part A: Batch Eviction Strategies ---")
    for cfg in KV_BLOCK_CONFIGS:
        for n_layers in N_LAYERS_BATCH:
            n_blocks = n_layers
            total_mb = cfg['block_bytes'] * n_blocks / 1e6

            if total_mb < 0.05:
                continue

            print(f"\n  {cfg['label']} x {n_layers} layers ({total_mb:.1f} MB):")
            for strategy in ['sequential', 'per_file', 'multi_thread']:
                bw, lat_us = bench_batch_eviction(TEST_DIR, cfg['block_bytes'], n_blocks, strategy)
                result = {
                    'config': cfg['label'],
                    'block_bytes': cfg['block_bytes'],
                    'n_layers': n_layers,
                    'total_MB': round(total_mb, 2),
                    'strategy': strategy,
                    'bw_gbps': round(bw, 4),
                    'latency_us': round(lat_us, 1),
                }
                all_results['batch_eviction'].append(result)
                print(f"    {strategy:>15s}: {bw:.3f} GB/s, {lat_us:.0f} us")

    print("\n--- Part B: Selective Reload (simulate hot block fetch) ---")
    for cfg in KV_BLOCK_CONFIGS[:3]:
        n_total = 64
        for selectivity in [0.1, 0.2, 0.5, 1.0]:
            bw, lat_us, n_hot = bench_selective_reload(TEST_DIR, cfg['block_bytes'], n_total, selectivity)
            result = {
                'config': cfg['label'],
                'block_bytes': cfg['block_bytes'],
                'n_total_blocks': n_total,
                'selectivity': selectivity,
                'n_hot_blocks': n_hot,
                'bw_gbps': round(bw, 4),
                'latency_us': round(lat_us, 1),
            }
            all_results['selective_reload'].append(result)
            print(f"  {cfg['label']:>20s}, top {selectivity*100:.0f}% ({n_hot} blocks): {bw:.3f} GB/s, {lat_us:.0f} us")

    print("\n--- Part C: Pipeline Overlap (concurrent evict + reload) ---")
    for cfg in KV_BLOCK_CONFIGS[:3]:
        pipe_result = bench_pipeline_overlap(TEST_DIR, cfg['block_bytes'], n_evict=32, n_reload=16)
        all_results['pipeline_overlap'].append({
            'config': cfg['label'],
            'block_bytes': cfg['block_bytes'],
            'n_evict': 32,
            'n_reload': 16,
            **{k: round(v, 3) for k, v in pipe_result.items()},
        })
        print(f"  {cfg['label']:>20s}: evict={pipe_result['evict_bw_gbps']:.3f} + reload={pipe_result['reload_bw_gbps']:.3f} "
              f"= combined {pipe_result['combined_bw_gbps']:.3f} GB/s (wall {pipe_result['wall_time_us']:.0f} us)")

    print("\n--- Part D: fio io_uring Baseline (KV block sizes) ---")
    for cfg in KV_BLOCK_CONFIGS:
        for rw in ['write', 'randread']:
            fio_result = bench_fio_iouring_kv(TEST_DIR, cfg['block_bytes'], 1024, rw)
            if fio_result:
                all_results['fio_iouring_baseline'].append({
                    'config': cfg['label'],
                    'block_bytes': cfg['block_bytes'],
                    'rw': rw,
                    **fio_result,
                })
                print(f"  {cfg['label']:>20s} {rw:>10s}: {fio_result['bw_gbps']:.3f} GB/s, "
                      f"{fio_result['iops']:.0f} IOPS, {fio_result['lat_avg_us']:.1f} us")

    out_path = os.path.join(RESULTS_DIR, 's5_kv_io_patterns.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("DESIGN INSIGHTS for OrchKvCache")
    print("=" * 70)
    print("""
  1. Sequential batch eviction >> per-file eviction (avoid fsync per block)
  2. Multi-threaded eviction helps for large batches
  3. Selective reload: reading only hot blocks saves significant IO
  4. Pipeline overlap: concurrent evict + reload utilizes more SSD bandwidth
  5. io_uring with high queue depth achieves near-peak SSD performance
    """)


if __name__ == '__main__':
    main()

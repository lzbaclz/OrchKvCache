#!/usr/bin/env python3
"""
S3: Multi-threaded Concurrent IO Scaling Benchmark
Tests how IO bandwidth scales with thread count.

OrchFS uses separate thread pools for NVM and SSD IO
(e.g., 4 NVM threads + 16 SSD threads). This benchmark
quantifies the IO scaling to guide thread pool sizing.
"""

import os
import time
import json
import threading
import queue

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DIRS = {
    'raid0_samsung': '/raid/orchkv_bench_s2',
    'root_nvme': '/tmp/orchkv_bench_s2',
}

THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
BLOCK_SIZES = [32768, 262144, 1048576]  # 32KB, 256KB, 1MB
TOTAL_PER_THREAD = 128 * 1024 * 1024  # 128MB per thread
RW_MODES = ['write', 'read']


def writer_thread(filepath, block_size, total_bytes, result_q):
    data = os.urandom(block_size)
    n_blocks = total_bytes // block_size

    start = time.perf_counter()
    with open(filepath, 'wb') as f:
        for _ in range(n_blocks):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())
    elapsed = time.perf_counter() - start
    result_q.put(('write', total_bytes, elapsed))


def reader_thread(filepath, block_size, total_bytes, result_q):
    n_blocks = total_bytes // block_size
    os.system(f'dd if=/dev/urandom of={filepath} bs={block_size} count={n_blocks} 2>/dev/null')

    start = time.perf_counter()
    with open(filepath, 'rb') as f:
        for _ in range(n_blocks):
            f.read(block_size)
    elapsed = time.perf_counter() - start
    result_q.put(('read', total_bytes, elapsed))


def bench_concurrent(test_dir, n_threads, block_size, rw):
    result_q = queue.Queue()
    threads = []

    for i in range(n_threads):
        filepath = os.path.join(test_dir, f'mt_bench_{rw}_{i}')
        if rw == 'write':
            t = threading.Thread(target=writer_thread, args=(filepath, block_size, TOTAL_PER_THREAD, result_q))
        else:
            t = threading.Thread(target=reader_thread, args=(filepath, block_size, TOTAL_PER_THREAD, result_q))
        threads.append((t, filepath))

    wall_start = time.perf_counter()
    for t, _ in threads:
        t.start()
    for t, _ in threads:
        t.join()
    wall_elapsed = time.perf_counter() - wall_start

    total_bytes = 0
    results_list = []
    while not result_q.empty():
        rw_type, nbytes, elapsed = result_q.get()
        total_bytes += nbytes
        results_list.append(elapsed)

    for _, filepath in threads:
        if os.path.exists(filepath):
            os.remove(filepath)

    aggregate_bw = total_bytes / wall_elapsed / 1e9
    avg_per_thread_bw = (TOTAL_PER_THREAD / (sum(results_list) / len(results_list))) / 1e9 if results_list else 0

    return aggregate_bw, wall_elapsed, avg_per_thread_bw


def main():
    all_results = []

    for disk_label, test_dir in TEST_DIRS.items():
        if not os.path.isdir(os.path.dirname(test_dir)):
            print(f"Skipping {disk_label}")
            continue

        os.makedirs(test_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Disk: {disk_label}")
        print(f"{'='*60}")

        for rw in RW_MODES:
            for bs in BLOCK_SIZES:
                bs_label = f"{bs//1024}KB" if bs < 1048576 else f"{bs//1048576}MB"
                print(f"\n  {rw} {bs_label}:")

                base_bw = None
                for n_threads in THREAD_COUNTS:
                    agg_bw, wall_time, per_thread_bw = bench_concurrent(test_dir, n_threads, bs, rw)
                    if base_bw is None:
                        base_bw = agg_bw

                    scaling = agg_bw / base_bw if base_bw > 0 else 0
                    result = {
                        'disk': disk_label,
                        'rw': rw,
                        'block_size': bs,
                        'block_size_label': bs_label,
                        'n_threads': n_threads,
                        'aggregate_bw_gbps': round(agg_bw, 4),
                        'per_thread_bw_gbps': round(per_thread_bw, 4),
                        'wall_time_s': round(wall_time, 3),
                        'scaling_vs_1t': round(scaling, 2),
                    }
                    all_results.append(result)
                    print(f"    {n_threads:>2d} threads: {agg_bw:.3f} GB/s agg, {per_thread_bw:.3f} GB/s/thread, {scaling:.2f}x scaling")

    out_path = os.path.join(RESULTS_DIR, 's3_multithread_io.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("DESIGN INSIGHT: Optimal thread count for OrchFS IO pools")
    print("=" * 70)
    for disk_label in TEST_DIRS:
        disk_results = [r for r in all_results if r['disk'] == disk_label]
        if not disk_results:
            continue
        print(f"\n  {disk_label}:")
        for rw in RW_MODES:
            for bs in BLOCK_SIZES:
                bs_label = f"{bs//1024}KB" if bs < 1048576 else f"{bs//1048576}MB"
                matches = [r for r in disk_results if r['rw'] == rw and r['block_size'] == bs]
                if matches:
                    best = max(matches, key=lambda x: x['aggregate_bw_gbps'])
                    print(f"    {rw} {bs_label}: peak at {best['n_threads']} threads = {best['aggregate_bw_gbps']:.3f} GB/s")


if __name__ == '__main__':
    main()

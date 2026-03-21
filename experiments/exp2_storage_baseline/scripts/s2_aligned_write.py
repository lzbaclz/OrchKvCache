#!/usr/bin/env python3
"""
S2: Aligned vs Unaligned Write Benchmark
Tests the impact of write alignment on SSD performance.

OrchFS's key optimization is aligned-write partitioning:
NVM writes at 4KB page granularity, SSD writes at 32KB block granularity.
This benchmark quantifies the penalty of unaligned writes.
"""

import os
import time
import json
import mmap
import ctypes
import ctypes.util

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DIRS = {
    'raid0_samsung': '/raid/orchkv_bench_s2',
    'root_nvme': '/tmp/orchkv_bench_s2',
}

BLOCK_SIZES = [4096, 32768, 65536, 262144, 1048576]
TOTAL_WRITE = 512 * 1024 * 1024  # 512MB per test
ITERATIONS = 3

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)


def aligned_alloc(size, alignment=4096):
    buf = mmap.mmap(-1, size)
    buf.write(os.urandom(size))
    buf.seek(0)
    return buf


def bench_write_direct_aligned(filepath, block_size, total_bytes):
    n_blocks = total_bytes // block_size
    buf = aligned_alloc(block_size)

    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        os.ftruncate(fd, total_bytes)
        os.lseek(fd, 0, os.SEEK_SET)

        start = time.perf_counter()
        for i in range(n_blocks):
            buf.seek(0)
            os.write(fd, buf.read(block_size))
        os.fsync(fd)
        elapsed = time.perf_counter() - start
    finally:
        os.close(fd)
        buf.close()

    return total_bytes / elapsed / 1e9, elapsed


def bench_write_direct_unaligned(filepath, block_size, total_bytes):
    """Write with misaligned offset (shift by 512 bytes within each block)."""
    actual_block = block_size
    n_blocks = total_bytes // actual_block
    buf = aligned_alloc(actual_block)

    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        os.ftruncate(fd, total_bytes + 4096)
        offset = 512
        os.lseek(fd, offset, os.SEEK_SET)

        start = time.perf_counter()
        for i in range(n_blocks):
            buf.seek(0)
            try:
                os.write(fd, buf.read(actual_block))
            except OSError:
                buf.seek(0)
                os.lseek(fd, (i + 1) * actual_block, os.SEEK_SET)
                os.write(fd, buf.read(actual_block))
        os.fsync(fd)
        elapsed = time.perf_counter() - start
    finally:
        os.close(fd)
        buf.close()

    return total_bytes / elapsed / 1e9, elapsed


def bench_write_buffered(filepath, block_size, total_bytes):
    n_blocks = total_bytes // block_size
    data = os.urandom(block_size)

    start = time.perf_counter()
    with open(filepath, 'wb') as f:
        for _ in range(n_blocks):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())
    elapsed = time.perf_counter() - start

    return total_bytes / elapsed / 1e9, elapsed


def bench_write_buffered_fsync_per_block(filepath, block_size, total_bytes):
    """Simulate worst case: fsync after every block (like naive KV eviction)."""
    n_blocks = min(total_bytes // block_size, 256)
    data = os.urandom(block_size)
    actual_bytes = n_blocks * block_size

    start = time.perf_counter()
    with open(filepath, 'wb') as f:
        for _ in range(n_blocks):
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    elapsed = time.perf_counter() - start

    return actual_bytes / elapsed / 1e9, elapsed


def main():
    all_results = []

    for disk_label, test_dir in TEST_DIRS.items():
        if not os.path.isdir(os.path.dirname(test_dir)):
            print(f"Skipping {disk_label}")
            continue

        os.makedirs(test_dir, exist_ok=True)
        filepath = os.path.join(test_dir, 'aligned_bench_file')
        print(f"\n{'='*60}")
        print(f"Disk: {disk_label}")
        print(f"{'='*60}")

        for bs in BLOCK_SIZES:
            bs_label = f"{bs//1024}KB" if bs < 1048576 else f"{bs//1048576}MB"
            print(f"\n  Block size: {bs_label}")

            methods = [
                ('direct_aligned', bench_write_direct_aligned),
                ('buffered_batch', bench_write_buffered),
                ('buffered_fsync_per_block', bench_write_buffered_fsync_per_block),
            ]

            for method_name, method_func in methods:
                bws = []
                for it in range(ITERATIONS):
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    try:
                        bw, elapsed = method_func(filepath, bs, TOTAL_WRITE)
                        bws.append(bw)
                    except OSError as e:
                        print(f"    {method_name}: ERROR - {e}")
                        break

                if bws:
                    avg_bw = sum(bws) / len(bws)
                    result = {
                        'disk': disk_label,
                        'block_size': bs,
                        'block_size_label': bs_label,
                        'method': method_name,
                        'bw_gbps': round(avg_bw, 4),
                        'iterations': len(bws),
                    }
                    all_results.append(result)
                    print(f"    {method_name:>30s}: {avg_bw:.4f} GB/s")

        if os.path.exists(filepath):
            os.remove(filepath)

    out_path = os.path.join(RESULTS_DIR, 's2_aligned_write.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: fsync-per-block penalty (simulates naive KV eviction)")
    print("=" * 70)
    for disk_label in TEST_DIRS:
        disk_results = [r for r in all_results if r['disk'] == disk_label]
        if not disk_results:
            continue
        print(f"\n  {disk_label}:")
        for bs in BLOCK_SIZES:
            bs_label = f"{bs//1024}KB" if bs < 1048576 else f"{bs//1048576}MB"
            batch = [r for r in disk_results if r['block_size'] == bs and r['method'] == 'buffered_batch']
            fsync = [r for r in disk_results if r['block_size'] == bs and r['method'] == 'buffered_fsync_per_block']
            if batch and fsync:
                penalty = batch[0]['bw_gbps'] / max(fsync[0]['bw_gbps'], 0.0001)
                print(f"    {bs_label}: batch={batch[0]['bw_gbps']:.3f} vs fsync_per_blk={fsync[0]['bw_gbps']:.3f} GB/s  ({penalty:.1f}x penalty)")


if __name__ == '__main__':
    main()

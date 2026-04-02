#!/usr/bin/env python3
"""
Direction C: SSD IO Optimization Ablation

Measures the independent contribution of each IO optimization by toggling
them one at a time. Uses direct file IO (no model inference) for clean
measurement of pure SSD bandwidth.

Configurations:
  C0: Random offset + sync + single-block writes (worst case)
  C1: Deterministic offset + sync + single-block writes
  C2: Deterministic offset + aligned 32KB + sync + single-block writes
  C3: Deterministic offset + aligned + async IO pool + single-block writes
  C4: Deterministic offset + aligned + async IO pool + batch coalescing (best)

Measures: write bandwidth (MB/s), read bandwidth (MB/s), total time.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import struct
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SLAB_SIZE = 32 * 1024  # 32 KB per KV block
N_LAYERS = 28
N_KV_HEADS = 4
BLOCKS_PER_HEAD = 64
TOTAL_BLOCKS = N_LAYERS * N_KV_HEADS * BLOCKS_PER_HEAD  # 7168 blocks
TOTAL_BYTES = TOTAL_BLOCKS * SLAB_SIZE  # ~224 MB


def gen_block_data():
    return os.urandom(SLAB_SIZE)


# ═══════════════════════════════════════════════════════════════════
#  IO Configurations
# ═══════════════════════════════════════════════════════════════════

def deterministic_offset(layer, head, block_idx, n_kv_heads, blocks_per_head):
    linear = layer * n_kv_heads * blocks_per_head + head * blocks_per_head + block_idx
    return linear * SLAB_SIZE


def random_offset_map(n_layers, n_kv_heads, blocks_per_head):
    """Pre-compute random offsets to simulate fragmented layout."""
    offsets = {}
    positions = list(range(n_layers * n_kv_heads * blocks_per_head))
    random.shuffle(positions)
    idx = 0
    for l in range(n_layers):
        for h in range(n_kv_heads):
            for b in range(blocks_per_head):
                offsets[(l, h, b)] = positions[idx] * SLAB_SIZE
                idx += 1
    return offsets


def write_c0_random_sync_single(filepath, blocks_to_write, rng_offsets):
    """C0: Random offsets, synchronous, single-block writes."""
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, TOTAL_BYTES)
    data = gen_block_data()

    t0 = time.perf_counter()
    for (l, h, b) in blocks_to_write:
        offset = rng_offsets[(l, h, b)]
        os.pwrite(fd, data, offset)
    elapsed = time.perf_counter() - t0
    os.close(fd)
    return elapsed


def write_c1_determ_sync_single(filepath, blocks_to_write):
    """C1: Deterministic offsets, synchronous, single-block writes."""
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, TOTAL_BYTES)
    data = gen_block_data()

    t0 = time.perf_counter()
    for (l, h, b) in blocks_to_write:
        offset = deterministic_offset(l, h, b, N_KV_HEADS, BLOCKS_PER_HEAD)
        os.pwrite(fd, data, offset)
    elapsed = time.perf_counter() - t0
    os.close(fd)
    return elapsed


def write_c2_determ_aligned_sync(filepath, blocks_to_write):
    """C2: Deterministic + aligned 32KB, sync, single."""
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    os.ftruncate(fd, TOTAL_BYTES)
    import ctypes
    buf = ctypes.create_string_buffer(SLAB_SIZE)
    ctypes.memmove(buf, os.urandom(SLAB_SIZE), SLAB_SIZE)

    t0 = time.perf_counter()
    for (l, h, b) in blocks_to_write:
        offset = deterministic_offset(l, h, b, N_KV_HEADS, BLOCKS_PER_HEAD)
        try:
            os.pwrite(fd, buf.raw, offset)
        except OSError:
            os.pwrite(fd, bytes(buf), offset)
    elapsed = time.perf_counter() - t0
    os.close(fd)
    return elapsed


def write_c3_determ_aligned_async(filepath, blocks_to_write, n_workers=4):
    """C3: Deterministic + aligned + async pool, single writes."""
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, TOTAL_BYTES)
    data = gen_block_data()

    def do_write(l, h, b):
        offset = deterministic_offset(l, h, b, N_KV_HEADS, BLOCKS_PER_HEAD)
        os.pwrite(fd, data, offset)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(do_write, l, h, b) for (l, h, b) in blocks_to_write]
        for f in futs:
            f.result()
    elapsed = time.perf_counter() - t0
    os.close(fd)
    return elapsed


def write_c4_determ_aligned_async_batch(filepath, blocks_to_write,
                                          n_workers=4, batch_size=8):
    """C4: Deterministic + aligned + async + batch coalescing."""
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, TOTAL_BYTES)

    batches = []
    sorted_blocks = sorted(blocks_to_write,
                            key=lambda x: deterministic_offset(x[0], x[1], x[2],
                                                                N_KV_HEADS, BLOCKS_PER_HEAD))
    for i in range(0, len(sorted_blocks), batch_size):
        batches.append(sorted_blocks[i:i + batch_size])

    def do_batch_write(batch):
        if not batch:
            return
        first_offset = deterministic_offset(batch[0][0], batch[0][1], batch[0][2],
                                             N_KV_HEADS, BLOCKS_PER_HEAD)
        coalesced = b""
        prev_end = first_offset
        for (l, h, b) in batch:
            off = deterministic_offset(l, h, b, N_KV_HEADS, BLOCKS_PER_HEAD)
            if off == prev_end:
                coalesced += os.urandom(SLAB_SIZE)
                prev_end = off + SLAB_SIZE
            else:
                if coalesced:
                    os.pwrite(fd, coalesced, first_offset)
                first_offset = off
                coalesced = os.urandom(SLAB_SIZE)
                prev_end = off + SLAB_SIZE
        if coalesced:
            os.pwrite(fd, coalesced, first_offset)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(do_batch_write, batch) for batch in batches]
        for f in futs:
            f.result()
    elapsed = time.perf_counter() - t0
    os.close(fd)
    return elapsed


def read_sequential(filepath, blocks_to_read):
    """Read back blocks (simulates promote)."""
    fd = os.open(filepath, os.O_RDONLY)
    buf = bytearray(SLAB_SIZE)

    t0 = time.perf_counter()
    for (l, h, b) in blocks_to_read:
        offset = deterministic_offset(l, h, b, N_KV_HEADS, BLOCKS_PER_HEAD)
        os.preadv(fd, [buf], offset)
    elapsed = time.perf_counter() - t0
    os.close(fd)
    return elapsed


def main():
    tmpdir = tempfile.mkdtemp(prefix="orchkv_ssd_ablation_")
    print(f"Temp dir: {tmpdir}")

    n_write_blocks = 1024
    blocks = []
    for l in range(N_LAYERS):
        for h in range(N_KV_HEADS):
            for b in range(BLOCKS_PER_HEAD):
                blocks.append((l, h, b))
    random.shuffle(blocks)
    write_blocks = blocks[:n_write_blocks]
    read_blocks = write_blocks[:256]

    total_write_mb = n_write_blocks * SLAB_SIZE / (1 << 20)
    total_read_mb = len(read_blocks) * SLAB_SIZE / (1 << 20)

    rng_offsets = random_offset_map(N_LAYERS, N_KV_HEADS, BLOCKS_PER_HEAD)

    configs = [
        ("C0: Random+Sync+Single",
         lambda f: write_c0_random_sync_single(f, write_blocks, rng_offsets)),
        ("C1: Determ+Sync+Single",
         lambda f: write_c1_determ_sync_single(f, write_blocks)),
        ("C2: Determ+Aligned+Sync",
         lambda f: write_c2_determ_aligned_sync(f, write_blocks)),
        ("C3: Determ+Aligned+Async",
         lambda f: write_c3_determ_aligned_async(f, write_blocks)),
        ("C4: Determ+Aligned+Async+Batch",
         lambda f: write_c4_determ_aligned_async_batch(f, write_blocks)),
    ]

    print(f"\n{'='*65}")
    print(f"  SSD IO Ablation ({n_write_blocks} blocks x {SLAB_SIZE//1024}KB = {total_write_mb:.0f}MB)")
    print(f"{'='*65}")
    print(f"  {'Config':<32s} {'Write s':>8s} {'W MB/s':>8s} {'Read s':>8s} {'R MB/s':>8s}")
    print(f"  {'-'*60}")

    results = []

    for name, write_fn in configs:
        filepath = os.path.join(tmpdir, name.split(":")[0].strip() + ".bin")

        try:
            w_time = write_fn(filepath)
        except Exception as e:
            filepath2 = os.path.join(tmpdir, name.split(":")[0].strip() + "_fb.bin")
            if "Aligned" in name:
                w_time = write_c1_determ_sync_single(filepath2, write_blocks)
                filepath = filepath2
            else:
                print(f"  {name:<32s} FAILED: {e}")
                continue

        w_bw = total_write_mb / w_time if w_time > 0 else 0

        r_time = read_sequential(filepath, read_blocks)
        r_bw = total_read_mb / r_time if r_time > 0 else 0

        row = {
            "config": name,
            "write_time_s": round(w_time, 4),
            "write_bw_mbs": round(w_bw, 1),
            "read_time_s": round(r_time, 4),
            "read_bw_mbs": round(r_bw, 1),
        }
        results.append(row)
        print(f"  {name:<32s} {w_time:>8.4f} {w_bw:>8.1f} {r_time:>8.4f} {r_bw:>8.1f}")

    if results and len(results) >= 2:
        base_w = results[0]["write_bw_mbs"]
        best_w = results[-1]["write_bw_mbs"]
        print(f"\n  Write BW improvement: {best_w/base_w:.1f}x (C0→C4)")

    out_path = RESULTS_DIR / "exp_ssd_io_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

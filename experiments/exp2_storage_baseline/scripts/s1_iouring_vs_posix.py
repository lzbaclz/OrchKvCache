#!/usr/bin/env python3
"""
S1: io_uring vs POSIX IO Engine Comparison
Uses fio to compare io_uring / libaio / psync across block sizes relevant to KV-Cache.

OrchFS uses io_uring as its core IO engine. This benchmark quantifies
the bandwidth advantage of io_uring over traditional POSIX IO,
which is what vLLM/FlexGen use for KV-Cache offloading.
"""

import subprocess
import json
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DIRS = {
    'raid0_samsung': '/raid/orchkv_bench_s2',
    'root_nvme': '/tmp/orchkv_bench_s2',
}

ENGINES = ['io_uring', 'libaio', 'psync']
BLOCK_SIZES = ['4k', '32k', '64k', '256k', '1m']
IO_DEPTHS = [1, 8, 32]
RW_MODES = ['write', 'read', 'randwrite', 'randread']
FILE_SIZE = '2G'
RUNTIME_SEC = 10
NUM_JOBS = 1


def run_fio(engine, bs, iodepth, rw, filepath, runtime=RUNTIME_SEC, size=FILE_SIZE, numjobs=NUM_JOBS):
    direct = 1
    if engine == 'psync':
        iodepth = 1

    cmd = [
        'fio',
        f'--name=bench_{engine}_{rw}_{bs}',
        f'--ioengine={engine}',
        f'--rw={rw}',
        f'--bs={bs}',
        f'--size={size}',
        f'--iodepth={iodepth}',
        f'--direct={direct}',
        f'--numjobs={numjobs}',
        f'--runtime={runtime}',
        '--time_based',
        '--group_reporting',
        '--output-format=json',
        f'--filename={filepath}',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=runtime + 30)
        if result.returncode != 0:
            print(f"  fio error: {result.stderr[:200]}")
            return None
        data = json.loads(result.stdout)
        job = data['jobs'][0]

        if 'read' in rw:
            bw_bytes = job['read']['bw_bytes']
            iops = job['read']['iops']
            lat_ns = job['read']['lat_ns']['mean']
            clat_p99_us = job['read']['clat_ns']['percentile'].get('99.000000', 0) / 1000
        else:
            bw_bytes = job['write']['bw_bytes']
            iops = job['write']['iops']
            lat_ns = job['write']['lat_ns']['mean']
            clat_p99_us = job['write']['clat_ns']['percentile'].get('99.000000', 0) / 1000

        return {
            'bw_gbps': round(bw_bytes / 1e9, 4),
            'iops': round(iops, 1),
            'lat_avg_us': round(lat_ns / 1000, 2),
            'clat_p99_us': round(clat_p99_us, 2),
        }
    except subprocess.TimeoutExpired:
        print(f"  fio timeout for {engine}/{rw}/{bs}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  fio parse error: {e}")
        return None


def main():
    all_results = []

    for disk_label, test_dir in TEST_DIRS.items():
        if not os.path.isdir(os.path.dirname(test_dir)):
            print(f"Skipping {disk_label}: parent dir does not exist")
            continue

        os.makedirs(test_dir, exist_ok=True)
        filepath = os.path.join(test_dir, 'fio_bench_file')
        print(f"\n{'='*60}")
        print(f"Disk: {disk_label} ({test_dir})")
        print(f"{'='*60}")

        for rw in RW_MODES:
            for bs in BLOCK_SIZES:
                for iodepth in IO_DEPTHS:
                    for engine in ENGINES:
                        if engine == 'psync' and iodepth > 1:
                            continue

                        label = f"{engine}/{rw}/{bs}/qd{iodepth}"
                        print(f"  {label} ...", end=' ', flush=True)

                        result = run_fio(engine, bs, iodepth, rw, filepath)
                        if result:
                            print(f"{result['bw_gbps']:.3f} GB/s, {result['lat_avg_us']:.1f} us")
                            all_results.append({
                                'disk': disk_label,
                                'engine': engine,
                                'rw': rw,
                                'bs': bs,
                                'iodepth': iodepth,
                                **result,
                            })
                        else:
                            print("FAILED")

        if os.path.exists(filepath):
            os.remove(filepath)

    out_path = os.path.join(RESULTS_DIR, 's1_iouring_vs_posix.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY: io_uring vs libaio vs psync (best iodepth per engine)")
    print("=" * 70)
    for disk_label in TEST_DIRS:
        disk_results = [r for r in all_results if r['disk'] == disk_label]
        if not disk_results:
            continue
        print(f"\n--- {disk_label} ---")
        for rw in RW_MODES:
            for bs in BLOCK_SIZES:
                row = {}
                for engine in ENGINES:
                    matches = [r for r in disk_results if r['engine'] == engine and r['rw'] == rw and r['bs'] == bs]
                    if matches:
                        best = max(matches, key=lambda x: x['bw_gbps'])
                        row[engine] = best['bw_gbps']
                if row:
                    parts = [f"{e}: {row.get(e, 0):.3f}" for e in ENGINES if e in row]
                    psync_bw = row.get('psync', 0.001)
                    iouring_bw = row.get('io_uring', 0)
                    speedup = iouring_bw / psync_bw if psync_bw > 0 else 0
                    print(f"  {rw:>10s} {bs:>6s}: {', '.join(parts)}  (io_uring {speedup:.1f}x vs psync)")


if __name__ == '__main__':
    main()

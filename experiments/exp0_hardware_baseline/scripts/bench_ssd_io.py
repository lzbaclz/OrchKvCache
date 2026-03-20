#!/usr/bin/env python3
"""
Benchmark: SSD I/O performance for KV-cache offloading scenarios.

Tests:
  1. Sequential write bandwidth (different block sizes)
  2. Sequential read bandwidth (different block sizes)
  3. Random write IOPS and bandwidth
  4. Random read IOPS and bandwidth
  5. Direct I/O (O_DIRECT) vs buffered I/O comparison
  6. fsync latency

Uses Python (os.open + O_DIRECT) for precise control.
Also wraps fio for standardized comparison.
"""
import os
import time
import json
import ctypes
import mmap
import subprocess
import tempfile
import shutil

TOTAL_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB total per test
BLOCK_SIZES = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]  # 4K to 16M
ITERATIONS_SMALL = 5000
TEST_DIR = None

def fmt_size(n):
    if n < 1024: return f"{n}B"
    if n < 1024**2: return f"{n//1024}K"
    if n < 1024**3: return f"{n//1024**2}M"
    return f"{n//1024**3}G"

def get_aligned_buffer(size):
    """Allocate page-aligned buffer for O_DIRECT."""
    buf = mmap.mmap(-1, size)
    buf.write(os.urandom(size))
    buf.seek(0)
    return buf

def bench_sequential_write(test_file, block_size, total, use_direct=False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if use_direct:
        flags |= os.O_DIRECT

    n_blocks = total // block_size
    if use_direct:
        buf = get_aligned_buffer(block_size)
        data = buf.read()
        buf.close()
        arr = bytearray(block_size)
        aligned_buf = (ctypes.c_char * block_size).from_buffer(arr)
        ctypes.memmove(aligned_buf, data, block_size)
    else:
        data = os.urandom(block_size)

    fd = os.open(test_file, flags, 0o644)
    try:
        t0 = time.perf_counter()
        for _ in range(n_blocks):
            if use_direct:
                os.write(fd, aligned_buf)
            else:
                os.write(fd, data)
        os.fsync(fd)
        elapsed = time.perf_counter() - t0
    finally:
        os.close(fd)

    bw = total / elapsed / 1e9  # GB/s
    lat = elapsed / n_blocks * 1e6  # us per block
    return bw, lat, elapsed

def bench_sequential_read(test_file, block_size, total, use_direct=False):
    flags = os.O_RDONLY
    if use_direct:
        flags |= os.O_DIRECT

    n_blocks = total // block_size
    fd = os.open(test_file, flags)
    try:
        # drop page cache
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        except PermissionError:
            pass

        t0 = time.perf_counter()
        for _ in range(n_blocks):
            d = os.read(fd, block_size)
            if len(d) == 0:
                break
        elapsed = time.perf_counter() - t0
    finally:
        os.close(fd)

    bw = total / elapsed / 1e9
    lat = elapsed / n_blocks * 1e6
    return bw, lat, elapsed

def bench_fsync_latency(test_file, n_iters=2000):
    """Measure fsync latency for small writes."""
    fd = os.open(test_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    data = os.urandom(4096)
    lats = []
    try:
        for _ in range(n_iters):
            os.write(fd, data)
            t0 = time.perf_counter()
            os.fsync(fd)
            lats.append((time.perf_counter() - t0) * 1e6)
    finally:
        os.close(fd)
    avg = sum(lats) / len(lats)
    p50 = sorted(lats)[len(lats)//2]
    p99 = sorted(lats)[int(len(lats)*0.99)]
    return avg, p50, p99

def run_fio_bench(test_dir, bs, rw, direct, size="1G", runtime=10, numjobs=1):
    """Run fio and parse results."""
    name = f"fio_{rw}_bs{fmt_size(bs)}_{'direct' if direct else 'buffered'}"
    cmd = [
        "fio",
        f"--name={name}",
        f"--directory={test_dir}",
        f"--bs={bs}",
        f"--rw={rw}",
        f"--size={size}",
        f"--runtime={runtime}",
        f"--direct={'1' if direct else '0'}",
        f"--numjobs={numjobs}",
        "--ioengine=libaio",
        "--iodepth=32",
        "--group_reporting",
        "--output-format=json",
        "--time_based",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(result.stdout)
        job = data["jobs"][0]
        if "read" in rw:
            bw_bytes = job["read"]["bw_bytes"]
            iops = job["read"]["iops"]
            lat_ns = job["read"]["lat_ns"]["mean"]
        else:
            bw_bytes = job["write"]["bw_bytes"]
            iops = job["write"]["iops"]
            lat_ns = job["write"]["lat_ns"]["mean"]
        return {
            "bw_gbps": round(bw_bytes / 1e9, 3),
            "iops": round(iops, 1),
            "lat_us": round(lat_ns / 1000, 2),
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    global TEST_DIR
    home = os.path.expanduser("~")

    mount_points = {
        "root_nvme (KIOXIA)": "/tmp/orchkv_bench",
        "raid0_samsung": "/raid/orchkv_bench",
        "sata_samsung": "/public/orchkv_bench",
    }

    results = {"ssd_benchmarks": []}

    for disk_label, test_dir in mount_points.items():
        if not os.path.exists(os.path.dirname(test_dir)):
            print(f"Skipping {disk_label}: mount point not found")
            continue

        os.makedirs(test_dir, exist_ok=True)
        TEST_DIR = test_dir

        print("=" * 80)
        print(f"SSD Benchmark: {disk_label} -> {test_dir}")
        print("=" * 80)

        disk_result = {"disk": disk_label, "path": test_dir, "tests": {}}

        # --- Python sequential write/read ---
        print("\n--- Sequential Write (Python, buffered) ---")
        print(f"{'BlockSize':>12s} {'BW(GB/s)':>10s} {'Lat(us)':>10s}")
        for bs in BLOCK_SIZES:
            total = min(TOTAL_SIZE, max(bs * 100, 256 * 1024 * 1024))
            tf = os.path.join(test_dir, "seq_write_test.bin")
            bw, lat, _ = bench_sequential_write(tf, bs, total, use_direct=False)
            print(f"{fmt_size(bs):>12s} {bw:>10.3f} {lat:>10.1f}")
            disk_result["tests"][f"seq_write_buffered_{fmt_size(bs)}"] = {
                "bw_gbps": round(bw, 3), "lat_us": round(lat, 1)
            }
            os.remove(tf)

        print("\n--- Sequential Write (Python, O_DIRECT) ---")
        print(f"{'BlockSize':>12s} {'BW(GB/s)':>10s} {'Lat(us)':>10s}")
        for bs in BLOCK_SIZES:
            total = min(TOTAL_SIZE, max(bs * 100, 256 * 1024 * 1024))
            tf = os.path.join(test_dir, "seq_write_direct_test.bin")
            try:
                bw, lat, _ = bench_sequential_write(tf, bs, total, use_direct=True)
                print(f"{fmt_size(bs):>12s} {bw:>10.3f} {lat:>10.1f}")
                disk_result["tests"][f"seq_write_direct_{fmt_size(bs)}"] = {
                    "bw_gbps": round(bw, 3), "lat_us": round(lat, 1)
                }
            except Exception as e:
                print(f"{fmt_size(bs):>12s}  ERROR: {e}")
            finally:
                if os.path.exists(tf):
                    os.remove(tf)

        # Write a file for read test
        read_file = os.path.join(test_dir, "read_test.bin")
        bench_sequential_write(read_file, 1048576, TOTAL_SIZE, use_direct=False)

        print("\n--- Sequential Read (Python, buffered) ---")
        print(f"{'BlockSize':>12s} {'BW(GB/s)':>10s} {'Lat(us)':>10s}")
        for bs in BLOCK_SIZES:
            bw, lat, _ = bench_sequential_read(read_file, bs, TOTAL_SIZE, use_direct=False)
            print(f"{fmt_size(bs):>12s} {bw:>10.3f} {lat:>10.1f}")
            disk_result["tests"][f"seq_read_buffered_{fmt_size(bs)}"] = {
                "bw_gbps": round(bw, 3), "lat_us": round(lat, 1)
            }

        # fsync latency
        print("\n--- fsync Latency (4KB write + fsync) ---")
        tf = os.path.join(test_dir, "fsync_test.bin")
        avg, p50, p99 = bench_fsync_latency(tf)
        print(f"  avg={avg:.1f}us  p50={p50:.1f}us  p99={p99:.1f}us")
        disk_result["tests"]["fsync_4k"] = {
            "avg_us": round(avg, 1), "p50_us": round(p50, 1), "p99_us": round(p99, 1)
        }
        os.remove(tf)

        # --- fio benchmarks ---
        fio_configs = [
            ("write",     "1M",  True,  "fio_seq_write_1M_direct"),
            ("read",      "1M",  True,  "fio_seq_read_1M_direct"),
            ("randwrite", "4k",  True,  "fio_rand_write_4k_direct"),
            ("randread",  "4k",  True,  "fio_rand_read_4k_direct"),
            ("randwrite", "64k", True,  "fio_rand_write_64k_direct"),
            ("randread",  "64k", True,  "fio_rand_read_64k_direct"),
            ("randwrite", "1M",  True,  "fio_rand_write_1M_direct"),
            ("randread",  "1M",  True,  "fio_rand_read_1M_direct"),
        ]

        print("\n--- fio Benchmarks ---")
        print(f"{'Test':>35s} {'BW(GB/s)':>10s} {'IOPS':>12s} {'Lat(us)':>10s}")
        for rw, bs_str, direct, label in fio_configs:
            bs_val = int(bs_str.replace("k","").replace("K","").replace("M","").replace("G",""))
            if "M" in bs_str: bs_val *= 1024*1024
            elif "k" in bs_str or "K" in bs_str: bs_val *= 1024
            r = run_fio_bench(test_dir, bs_val, rw, direct, size="1G", runtime=10)
            if "error" not in r:
                print(f"{label:>35s} {r['bw_gbps']:>10.3f} {r['iops']:>12.1f} {r['lat_us']:>10.1f}")
            else:
                print(f"{label:>35s}  ERROR: {r['error'][:50]}")
            disk_result["tests"][label] = r

        # cleanup
        for f in os.listdir(test_dir):
            fp = os.path.join(test_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

        results["ssd_benchmarks"].append(disk_result)

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bench_ssd_io.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

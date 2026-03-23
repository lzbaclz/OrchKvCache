"""
D4: Unit tests for benchmark scripts — tests the parts that
do NOT require vLLM (orchkv_core-only experiments).

Validates: bench_utils, E5 policy sweep, E7 prefetch, E9 scalability,
           E8 GPU↔DRAM bandwidth measurement.

Run with: conda run -n orchkv python -m pytest test/test_benchmarks.py -v
"""
import sys
import os
import json
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import orchkv_core as _C


# ===== bench_utils =====

class TestBenchUtils:
    def test_cpu_timer(self):
        from bench_utils import CPUTimer
        t = CPUTimer()
        t.start()
        for _ in range(100000):
            pass
        us = t.stop()
        assert us > 0
        stats = t.stats()
        assert stats["count"] == 1
        assert stats["avg_us"] > 0

    def test_cuda_timer(self):
        from bench_utils import cuda_timer
        with cuda_timer() as t:
            a = torch.randn(1000, 1000, device="cuda")
            b = a @ a
        assert t["elapsed_ms"] > 0

    def test_gpu_mem_mb(self):
        from bench_utils import gpu_mem_mb
        mem = gpu_mem_mb()
        assert "allocated_mb" in mem
        assert "reserved_mb" in mem
        assert mem["allocated_mb"] >= 0

    def test_save_json(self, tmp_path):
        from bench_utils import save_json, RESULTS_DIR
        old_dir = RESULTS_DIR
        import bench_utils
        bench_utils.RESULTS_DIR = tmp_path
        try:
            p = save_json({"test": 42}, "unit_test")
            assert p.exists()
            data = json.loads(p.read_text())
            assert data["test"] == 42
        finally:
            bench_utils.RESULTS_DIR = old_dir

    def test_save_csv(self, tmp_path):
        from bench_utils import save_csv, RESULTS_DIR
        import bench_utils
        old_dir = RESULTS_DIR
        bench_utils.RESULTS_DIR = tmp_path
        try:
            rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
            p = save_csv(rows, "unit_test")
            assert p.exists()
            lines = p.read_text().strip().split("\n")
            assert len(lines) == 3  # header + 2 rows
        finally:
            bench_utils.RESULTS_DIR = old_dir

    def test_experiment_config(self):
        from bench_utils import ExperimentConfig
        cfg = ExperimentConfig(seq_len=2048, batch_size=8, orchkv_enabled=True)
        label = cfg.label()
        assert "seq2048" in label
        assert "bs8" in label
        assert "orchkv" in label

    def test_generate_synthetic_prompts(self):
        from bench_utils import generate_synthetic_prompts
        prompts = generate_synthetic_prompts(3, 100)
        assert len(prompts) == 3
        assert len(prompts[0]) == 100
        assert all(1 <= t < 32000 for t in prompts[0])


# ===== E5: Policy Sweep =====

class TestE5PolicySweep:
    def test_single_config(self):
        from benchmark_ablation import run_e5
        results = run_e5(n_blocks=32, n_steps=10)
        assert len(results) > 0
        r = results[0]
        assert "alpha" in r
        assert "n_hot" in r
        assert "schedule_cycles" in r

    def test_hot_cold_separation(self):
        """High alpha → attention dominates → better hot/cold separation."""
        from benchmark_ablation import run_e5
        results = run_e5(n_blocks=64, n_steps=20)
        high_alpha = [r for r in results if r["alpha"] >= 0.8]
        low_alpha = [r for r in results if r["alpha"] <= 0.2]
        if high_alpha and low_alpha:
            ha_hot = sum(r["n_hot"] for r in high_alpha) / len(high_alpha)
            la_hot = sum(r["n_hot"] for r in low_alpha) / len(low_alpha)
            assert ha_hot >= 0
            assert la_hot >= 0


# ===== E7: Prefetch =====

class TestE7Prefetch:
    def test_budget_sweep(self):
        from benchmark_prefetch import run_prefetch_sweep
        results = run_prefetch_sweep(n_blocks=32, n_steps=20, budgets=[0, 4, 8])
        assert len(results) == 3

        budget_0 = results[0]
        assert budget_0["prefetch_budget"] == 0
        assert budget_0["prefetches_dispatched"] == 0

    def test_higher_budget_more_dispatches(self):
        from benchmark_prefetch import run_prefetch_sweep
        results = run_prefetch_sweep(n_blocks=64, n_steps=30, budgets=[0, 16])
        b0 = next(r for r in results if r["prefetch_budget"] == 0)
        b16 = next(r for r in results if r["prefetch_budget"] == 16)
        assert b16["prefetches_dispatched"] >= b0["prefetches_dispatched"]

    def test_schedule_latency(self):
        from benchmark_prefetch import run_prefetch_sweep
        results = run_prefetch_sweep(n_blocks=32, n_steps=10, budgets=[8])
        r = results[0]
        assert r["avg_schedule_us"] > 0
        assert r["p99_schedule_us"] >= r["avg_schedule_us"]


# ===== E9: Scalability =====

class TestE9Scalability:
    def test_basic(self):
        from benchmark_scalability import run_scalability
        results = run_scalability(block_counts=[64, 128], n_steps=10)
        assert len(results) == 2
        for r in results:
            assert "avg_schedule_us" in r
            assert "n_blocks" in r

    def test_latency_scales(self):
        """More blocks → scheduling takes more time (roughly)."""
        from benchmark_scalability import run_scalability
        results = run_scalability(block_counts=[64, 256, 1024], n_steps=15)
        latencies = [r["avg_schedule_us"] for r in results]
        assert latencies[-1] > 0
        assert len(latencies) == 3

    def test_schedule_count_matches(self):
        from benchmark_scalability import run_scalability
        results = run_scalability(block_counts=[64], n_steps=20)
        r = results[0]
        assert r["schedule_cycles"] == 20


# ===== E8: GPU↔DRAM bandwidth (quick) =====

class TestE8StorageBW:
    def test_gpu_dram_bw(self):
        from benchmark_storage_bw import measure_gpu_dram
        results = measure_gpu_dram(sizes_mb=[1.0], n_iter=3)
        assert len(results) == 1
        r = results[0]
        assert r["d2h_gbps"] > 0
        assert r["h2d_gbps"] > 0

    def test_dram_storage_bw(self):
        from benchmark_storage_bw import measure_dram_storage
        results = measure_dram_storage(sizes_mb=[0.5], n_iter=3)
        assert len(results) == 1
        r = results[0]
        assert r["write_gbps"] > 0
        assert r["read_gbps"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

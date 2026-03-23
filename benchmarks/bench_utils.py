"""
Shared utilities for OrchKvCache benchmark scripts.

Provides:
  - JSON/CSV output helpers
  - Timer context manager with ns precision
  - Percentile / statistics helpers
  - vLLM engine builder with/without OrchKvCache
  - GPU memory sampling
"""
from __future__ import annotations

import json
import csv
import time
import os
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Timing ──────────────────────────────────────────────────────────

@contextmanager
def cuda_timer():
    """Yield a dict that gets populated with elapsed_ms after the block."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    record = {"elapsed_ms": 0.0}
    start.record()
    yield record
    end.record()
    torch.cuda.synchronize()
    record["elapsed_ms"] = start.elapsed_time(end)


class CPUTimer:
    """High-resolution CPU wall-clock timer."""

    def __init__(self):
        self._start = 0.0
        self._elapsed: list[float] = []

    def start(self):
        self._start = time.perf_counter_ns()

    def stop(self) -> float:
        elapsed_us = (time.perf_counter_ns() - self._start) / 1e3
        self._elapsed.append(elapsed_us)
        return elapsed_us

    @property
    def all_us(self) -> list[float]:
        return self._elapsed

    def stats(self) -> dict[str, float]:
        if not self._elapsed:
            return {}
        s = sorted(self._elapsed)
        return {
            "count": len(s),
            "avg_us": statistics.mean(s),
            "p50_us": s[len(s) // 2],
            "p99_us": s[int(len(s) * 0.99)],
            "min_us": s[0],
            "max_us": s[-1],
        }


# ─── GPU Memory ──────────────────────────────────────────────────────

def gpu_mem_mb(device: int = 0) -> dict[str, float]:
    """Snapshot GPU memory usage in MB."""
    torch.cuda.synchronize(device)
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1 << 20),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1 << 20),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1 << 20),
    }


def reset_gpu_peak(device: int = 0):
    torch.cuda.reset_peak_memory_stats(device)


# ─── Output ──────────────────────────────────────────────────────────

def save_json(data: Any, name: str):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[bench] saved → {path}")
    return path


def save_csv(rows: list[dict], name: str):
    if not rows:
        return None
    path = RESULTS_DIR / f"{name}.csv"
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[bench] saved → {path}")
    return path


# ─── vLLM Engine Helpers ─────────────────────────────────────────────

def build_vllm_engine(
    model: str,
    orchkv_enabled: bool = False,
    gpu_pool_gb: float = 4.0,
    dram_pool_gb: float = 8.0,
    block_size: int = 16,
    max_model_len: int = 4096,
    extra_args: dict | None = None,
):
    """
    Build a vLLM LLM engine, optionally with OrchKvCache integration.

    Requires vLLM to be installed. Returns None if vLLM is unavailable.
    """
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except ImportError:
        print("[bench] vLLM not installed, returning None")
        return None

    engine_args = {
        "model": model,
        "max_model_len": max_model_len,
        "block_size": block_size,
        "enforce_eager": False,
        "gpu_memory_utilization": 0.9,
    }

    if orchkv_enabled:
        from orchkv.vllm_integration.engine_patch import register_orchkv_backend
        register_orchkv_backend()
        engine_args["kv_transfer_config"] = {
            "kv_connector": "OrchKvOffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "dram_pool_gb": dram_pool_gb,
            },
        }

    if extra_args:
        engine_args.update(extra_args)

    return LLM(**engine_args)


# ─── Experiment Parametrization ──────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Encapsulates one experiment point."""
    model: str = "meta-llama/Llama-2-7b-hf"
    seq_len: int = 1024
    batch_size: int = 1
    block_size: int = 16
    orchkv_enabled: bool = True
    dram_pool_gb: float = 8.0
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    prefetch_budget: int = 16
    tiers: str = "GPU+DRAM+NVM+SSD"
    dataset: str = "Synthetic-uniform"
    max_new_tokens: int = 128
    tag: str = ""

    def label(self) -> str:
        parts = [
            f"seq{self.seq_len}",
            f"bs{self.batch_size}",
            f"blk{self.block_size}",
        ]
        if self.orchkv_enabled:
            parts.append("orchkv")
        else:
            parts.append("baseline")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


@dataclass
class BenchmarkResult:
    """Stores results for one experiment point."""
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    gpu_mem: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def generate_synthetic_prompts(
    n: int, seq_len: int, vocab_size: int = 32000,
) -> list[list[int]]:
    """Generate n synthetic prompts of given token length."""
    import random
    return [
        [random.randint(1, vocab_size - 1) for _ in range(seq_len)]
        for _ in range(n)
    ]

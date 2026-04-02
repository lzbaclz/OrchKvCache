"""
FastFIFOManager — fair-baseline FIFO manager using the same pre-allocated
buffer architecture as FastKVCacheManager.

The ONLY difference from FastKVCacheManager:
  - Eviction policy: FIFO (oldest blocks first) instead of attention-based
  - No attention reporting (report_attention / report_qk_norm are no-ops)
  - No orchkv_core scheduling (no EMA, no classifier, no adaptive threshold)

This isolates the pure scheduling-policy benefit: any throughput difference
between FastFIFOManager and FastKVCacheManager comes solely from the
eviction decision quality, not from framework overhead differences.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Optional

import torch

try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False

TIER_GPU = 0
TIER_DRAM = 1


class FastFIFOManager:

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        gpu_budget_bytes: int = 0,
        max_seq_len: int = 4096,
        sink_tokens: int = 4,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_budget_bytes = gpu_budget_bytes
        self.max_seq_len = max_seq_len
        self.sink_tokens = sink_tokens

        self._block_bytes = 2 * n_kv_heads * block_size * head_dim * dtype.itemsize
        self._total_tokens = 0
        self._step = 0

        self._key_bufs = [
            torch.zeros(1, n_kv_heads, max_seq_len, head_dim,
                        dtype=dtype, device="cuda")
            for _ in range(n_layers)
        ]
        self._val_bufs = [
            torch.zeros(1, n_kv_heads, max_seq_len, head_dim,
                        dtype=dtype, device="cuda")
            for _ in range(n_layers)
        ]

        self._evicted_ranges: list[list[tuple[int, int]]] = [[] for _ in range(n_layers)]
        self._dram_store: dict[tuple[int, int, int], torch.Tensor] = {}
        self._fifo_queue: deque[int] = deque()

        self._stats = {
            "gpu_to_dram": 0, "dram_to_gpu": 0,
            "total_blocks": 0, "evictions_this_step": 0,
        }

    def destroy(self):
        pass

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def gpu_kv_bytes(self) -> int:
        n_blocks = (self._total_tokens + self.block_size - 1) // self.block_size
        evicted = sum(len(e) for e in self._evicted_ranges)
        return (n_blocks * self.n_layers - evicted) * self._block_bytes

    @staticmethod
    def _to_legacy(past_key_values):
        if HAS_DYNAMIC_CACHE and isinstance(past_key_values, DynamicCache):
            return past_key_values.to_legacy_cache()
        return past_key_values

    @staticmethod
    def _from_legacy(legacy_cache):
        if HAS_DYNAMIC_CACHE:
            return DynamicCache.from_legacy_cache(legacy_cache)
        return legacy_cache

    def ingest_step(self, past_key_values) -> None:
        legacy = self._to_legacy(past_key_values)
        for li, (k, v) in enumerate(legacy):
            seq_len = k.shape[2]
            if seq_len > self.max_seq_len:
                self._grow_buffers(seq_len)
            self._key_bufs[li][0, :, :seq_len, :] = k[0, :, :seq_len, :]
            self._val_bufs[li][0, :, :seq_len, :] = v[0, :, :seq_len, :]
        self._total_tokens = legacy[0][0].shape[2]
        n_blocks = (self._total_tokens + self.block_size - 1) // self.block_size
        for bid in range(n_blocks):
            self._fifo_queue.append(bid)

    def append_token(self, past_key_values) -> None:
        legacy = self._to_legacy(past_key_values)
        pos = self._total_tokens
        if pos >= self.max_seq_len:
            self._grow_buffers(pos + 256)
        for li, (k, v) in enumerate(legacy):
            self._key_bufs[li][0, :, pos, :] = k[0, :, -1, :]
            self._val_bufs[li][0, :, pos, :] = v[0, :, -1, :]
        self._total_tokens = pos + 1
        if pos % self.block_size == 0:
            self._fifo_queue.append(pos // self.block_size)

    def _grow_buffers(self, new_max: int):
        new_max = ((new_max + 255) // 256) * 256
        for li in range(self.n_layers):
            old_k, old_v = self._key_bufs[li], self._val_bufs[li]
            new_k = torch.zeros(1, self.n_kv_heads, new_max, self.head_dim,
                                dtype=self.dtype, device="cuda")
            new_v = torch.zeros_like(new_k)
            n = min(old_k.shape[2], new_max)
            new_k[0, :, :n, :] = old_k[0, :, :n, :]
            new_v[0, :, :n, :] = old_v[0, :, :n, :]
            self._key_bufs[li] = new_k
            self._val_bufs[li] = new_v
        self.max_seq_len = new_max

    def build_past_kv(self, device: str = "cuda"):
        n = self._total_tokens
        self._restore_evicted_blocks()
        result = []
        for li in range(self.n_layers):
            result.append((
                self._key_bufs[li][:, :, :n, :],
                self._val_bufs[li][:, :, :n, :],
            ))
        return self._from_legacy(tuple(result))

    def _restore_evicted_blocks(self):
        for li in range(self.n_layers):
            for (start, end) in self._evicted_ranges[li]:
                key = (li, start, end)
                if key in self._dram_store:
                    data = self._dram_store[key]
                    n = end - start
                    self._key_bufs[li][0, :, start:end, :] = data[0, :, :n, :].to("cuda", non_blocking=False)
                    self._val_bufs[li][0, :, start:end, :] = data[1, :, :n, :].to("cuda", non_blocking=False)
                    del self._dram_store[key]
                    self._stats["dram_to_gpu"] += 1
            self._evicted_ranges[li].clear()

    def report_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        pass

    def step_done(self):
        self._step += 1

    def schedule(self) -> dict:
        if self.gpu_budget_bytes <= 0:
            return {}
        self._stats["evictions_this_step"] = 0
        gpu_bytes = self.gpu_kv_bytes()
        if gpu_bytes > self.gpu_budget_bytes * 0.85:
            self._evict_fifo()
        return {"step": self._step}

    def _evict_fifo(self, max_evict: int = 8):
        evicted_bids = set()
        for ranges in self._evicted_ranges:
            for (s, e) in ranges:
                evicted_bids.add(s // self.block_size)

        evicted = 0
        while self._fifo_queue and evicted < max_evict:
            bid = self._fifo_queue.popleft()
            if bid in evicted_bids:
                continue
            if (bid * self.block_size) < self.sink_tokens:
                continue

            start = bid * self.block_size
            end = min(start + self.block_size, self._total_tokens)
            n = end - start

            for li in range(self.n_layers):
                data = torch.empty(2, self.n_kv_heads, self.block_size, self.head_dim,
                                   dtype=self.dtype, pin_memory=True)
                data[0, :, :n, :] = self._key_bufs[li][0, :, start:end, :].cpu()
                data[1, :, :n, :] = self._val_bufs[li][0, :, start:end, :].cpu()
                self._dram_store[(li, start, end)] = data
                self._evicted_ranges[li].append((start, end))
                self._stats["gpu_to_dram"] += 1

            evicted += 1
            self._stats["evictions_this_step"] += 1

            if self.gpu_kv_bytes() <= self.gpu_budget_bytes * 0.7:
                break

    def get_stats(self) -> dict:
        n_blocks = (self._total_tokens + self.block_size - 1) // self.block_size
        evicted = sum(len(e) for e in self._evicted_ranges)
        gpu_blocks = n_blocks * self.n_layers - evicted
        return {
            "step": self._step,
            "total_tokens": self._total_tokens,
            "blocks_gpu": gpu_blocks,
            "blocks_dram": evicted,
            "gpu_kv_mb": gpu_blocks * self._block_bytes / (1 << 20),
            "migrations": dict(self._stats),
        }

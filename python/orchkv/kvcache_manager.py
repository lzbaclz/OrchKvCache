"""
KVCacheManager — block-level KV-Cache management bridging HuggingFace
transformers' past_key_values and orchkv_core's tiered_manager.

Splits the KV cache into fixed-size blocks, tracks their tier placement
(GPU / DRAM / SSD), and uses orchkv_core's scheduling to decide eviction
and promotion.  Data movement uses PyTorch pin_memory + non_blocking copy
for GPU<->DRAM, and standard file I/O for DRAM<->SSD.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False

logger = logging.getLogger(__name__)

try:
    import orchkv_core as _C
except ImportError:
    _C = None
    logger.warning("orchkv_core not available; KVCacheManager will run in dummy mode")

TIER_GPU = 0
TIER_DRAM = 1
TIER_SSD = 2

_TIER_NAMES = {TIER_GPU: "GPU", TIER_DRAM: "DRAM", TIER_SSD: "SSD"}


@dataclass
class KVBlock:
    """Metadata and storage for one KV-cache block."""
    block_id: int
    layer: int
    token_start: int
    token_count: int = 0
    capacity: int = 16

    tier: int = TIER_GPU
    gpu_data: Optional[torch.Tensor] = None
    dram_data: Optional[torch.Tensor] = None
    ssd_path: Optional[str] = None

    is_sink: bool = False
    access_count: int = 0


class KVCacheManager:
    """
    Manages KV-cache blocks across GPU, DRAM, and SSD tiers.

    Args:
        n_layers: number of transformer layers
        n_kv_heads: number of KV attention heads
        head_dim: dimension per head
        block_size: tokens per block (default 16)
        dtype: tensor dtype (default torch.float16)
        gpu_budget_bytes: max GPU memory for KV blocks (0 = unlimited)
        ssd_dir: directory for SSD spill files (None = DRAM only)
        sink_tokens: number of initial tokens to pin on GPU permanently
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        gpu_budget_bytes: int = 0,
        ssd_dir: Optional[str] = None,
        sink_tokens: int = 4,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_budget_bytes = gpu_budget_bytes
        self.ssd_dir = ssd_dir
        self.sink_tokens = sink_tokens

        self._block_bytes = 2 * n_kv_heads * block_size * head_dim * dtype.itemsize
        self._next_block_id = 0
        self._total_tokens = 0

        # blocks[layer] = list of KVBlock in token order
        self._blocks: list[list[KVBlock]] = [[] for _ in range(n_layers)]

        self._tm_handle: Optional[int] = None
        self._step = 0

        self._stats = {
            "gpu_to_dram": 0,
            "dram_to_gpu": 0,
            "dram_to_ssd": 0,
            "ssd_to_dram": 0,
            "total_blocks": 0,
        }

        if ssd_dir:
            Path(ssd_dir).mkdir(parents=True, exist_ok=True)

        self._init_tiered_manager()

    def _init_tiered_manager(self):
        if _C is None:
            return
        self._tm_handle = _C.tm_create()
        logger.info("KVCacheManager: tiered_manager created (handle=%s)", self._tm_handle)

    def destroy(self):
        if self._tm_handle is not None and _C is not None:
            _C.tm_destroy(self._tm_handle)
            self._tm_handle = None

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def block_bytes(self) -> int:
        return self._block_bytes

    def gpu_kv_bytes(self) -> int:
        count = sum(
            1 for layer_blocks in self._blocks
            for b in layer_blocks if b.tier == TIER_GPU
        )
        return count * self._block_bytes

    def _alloc_block(self, layer: int, token_start: int) -> KVBlock:
        bid = self._next_block_id
        self._next_block_id += 1

        is_sink = token_start < self.sink_tokens
        blk = KVBlock(
            block_id=bid,
            layer=layer,
            token_start=token_start,
            capacity=self.block_size,
            is_sink=is_sink,
        )
        self._blocks[layer].append(blk)
        self._stats["total_blocks"] += 1

        if self._tm_handle is not None and _C is not None:
            flags = 1 if is_sink else 0  # KV_FLAG_ATTN_SINK = 1
            _C.tm_register_block_id(self._tm_handle, bid, flags, int(_C.GPU_HBM))

        return blk

    def _make_block_tensor(self, device: str = "cuda") -> torch.Tensor:
        """Allocate a [2, n_kv_heads, block_size, head_dim] tensor."""
        if device == "cpu":
            t = torch.zeros(
                2, self.n_kv_heads, self.block_size, self.head_dim,
                dtype=self.dtype, device="cpu", pin_memory=True,
            )
        else:
            t = torch.zeros(
                2, self.n_kv_heads, self.block_size, self.head_dim,
                dtype=self.dtype, device="cuda",
            )
        return t

    # ------------------------------------------------------------------
    # Ingest: absorb new KV data from a forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _to_legacy(past_key_values):
        """Convert DynamicCache to legacy tuple format if needed."""
        if HAS_DYNAMIC_CACHE and isinstance(past_key_values, DynamicCache):
            return past_key_values.to_legacy_cache()
        return past_key_values

    @staticmethod
    def _from_legacy(legacy_cache):
        """Convert legacy tuple back to DynamicCache."""
        if HAS_DYNAMIC_CACHE:
            return DynamicCache.from_legacy_cache(legacy_cache)
        return legacy_cache

    def ingest_step(self, past_key_values) -> None:
        """
        After a forward pass, absorb the KV cache produced by HF transformers.
        Accepts both DynamicCache and legacy tuple format.
        """
        legacy = self._to_legacy(past_key_values)
        for layer_idx, (k, v) in enumerate(legacy):
            seq_len = k.shape[2]
            self._ingest_layer(layer_idx, k, v, seq_len)
        self._total_tokens = legacy[0][0].shape[2]

    def _ingest_layer(
        self, layer: int, key: torch.Tensor, value: torch.Tensor, seq_len: int,
    ):
        existing_blocks = self._blocks[layer]
        existing_tokens = sum(b.token_count for b in existing_blocks)

        if seq_len <= existing_tokens:
            return

        new_start = existing_tokens
        for pos in range(new_start, seq_len, self.block_size):
            end = min(pos + self.block_size, seq_len)
            n_tok = end - pos

            blk = self._alloc_block(layer, pos)
            blk.token_count = n_tok

            data = self._make_block_tensor("cuda")
            data[0, :, :n_tok, :] = key[0, :, pos:end, :]
            data[1, :, :n_tok, :] = value[0, :, pos:end, :]
            blk.gpu_data = data
            blk.tier = TIER_GPU

    def append_token(self, past_key_values) -> None:
        """Optimized path: only absorb the last token from each layer."""
        past_key_values = self._to_legacy(past_key_values)
        for layer_idx, (k, v) in enumerate(past_key_values):
            seq_len = k.shape[2]
            blocks = self._blocks[layer_idx]

            if not blocks:
                self._ingest_layer(layer_idx, k, v, seq_len)
                continue

            last_blk = blocks[-1]
            total_managed = sum(b.token_count for b in blocks)

            if seq_len <= total_managed:
                continue

            if last_blk.token_count < last_blk.capacity:
                idx = last_blk.token_count
                if last_blk.tier != TIER_GPU:
                    self._promote_block(last_blk)
                last_blk.gpu_data[0, :, idx, :] = k[0, :, -1, :]
                last_blk.gpu_data[1, :, idx, :] = v[0, :, -1, :]
                last_blk.token_count += 1
            else:
                pos = total_managed
                blk = self._alloc_block(layer_idx, pos)
                blk.token_count = 1
                data = self._make_block_tensor("cuda")
                data[0, :, 0, :] = k[0, :, -1, :]
                data[1, :, 0, :] = v[0, :, -1, :]
                blk.gpu_data = data

        self._total_tokens = past_key_values[0][0].shape[2]

    # ------------------------------------------------------------------
    # Build: reconstruct past_key_values for HF forward pass
    # ------------------------------------------------------------------

    def build_past_kv(self, device: str = "cuda"):
        """
        Reconstruct HF-format past_key_values from managed blocks.
        Promotes any non-GPU blocks back to GPU before assembly.

        Returns: DynamicCache (or legacy tuple if DynamicCache unavailable)
        """
        result = []
        for layer_idx in range(self.n_layers):
            blocks = self._blocks[layer_idx]
            if not blocks:
                result.append((
                    torch.zeros(1, self.n_kv_heads, 0, self.head_dim,
                                dtype=self.dtype, device=device),
                    torch.zeros(1, self.n_kv_heads, 0, self.head_dim,
                                dtype=self.dtype, device=device),
                ))
                continue

            total_tokens = sum(b.token_count for b in blocks)
            key_out = torch.zeros(
                1, self.n_kv_heads, total_tokens, self.head_dim,
                dtype=self.dtype, device=device,
            )
            val_out = torch.zeros_like(key_out)

            pos = 0
            for blk in blocks:
                if blk.token_count == 0:
                    continue
                if blk.tier != TIER_GPU:
                    self._promote_block(blk)

                n = blk.token_count
                key_out[0, :, pos:pos + n, :] = blk.gpu_data[0, :, :n, :]
                val_out[0, :, pos:pos + n, :] = blk.gpu_data[1, :, :n, :]
                pos += n

            result.append((key_out, val_out))

        legacy = tuple(result)
        return self._from_legacy(legacy)

    # ------------------------------------------------------------------
    # Attention reporting
    # ------------------------------------------------------------------

    def report_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        """
        Report per-block attention scores to the tiered_manager.

        attn_weights: [batch, n_heads, q_len, kv_len] softmax output
        """
        if self._tm_handle is None or _C is None:
            return

        blocks = self._blocks[layer_idx]
        if not blocks:
            return

        with torch.no_grad():
            # Mean over batch and query dims -> [n_heads, kv_len]
            avg = attn_weights.float().mean(dim=(0, 2))

            for blk in blocks:
                start = blk.token_start
                end = start + blk.token_count
                if end > avg.shape[-1]:
                    end = avg.shape[-1]
                if start >= end:
                    continue
                score = float(avg[:, start:end].sum())
                _C.tm_report_attn(self._tm_handle, blk.block_id, score)
                blk.access_count += 1

    # ------------------------------------------------------------------
    # Scheduling: invoke orchkv_core's tiered_manager
    # ------------------------------------------------------------------

    def step_done(self):
        """Mark end of a decode step. Advances EMA in the tracker."""
        self._step += 1
        if self._tm_handle is not None and _C is not None:
            _C.tm_step_done(self._tm_handle)

    def schedule(self) -> dict:
        """
        Run one scheduling cycle.  Checks GPU memory pressure and
        evicts cold blocks / promotes predicted-hot blocks.

        Returns dict with scheduling stats for this cycle.
        """
        if self._tm_handle is None or _C is None:
            return {}

        gpu_bytes = self.gpu_kv_bytes()
        budget = self.gpu_budget_bytes
        if budget > 0:
            gpu_ratio = min(gpu_bytes / budget, 1.0)
        else:
            gpu_ratio = 0.0

        _C.tm_set_usage(self._tm_handle, gpu_ratio, 0.0)

        _C.tm_schedule_once(self._tm_handle)

        raw = _C.tm_get_stats(self._tm_handle)

        evicted = 0
        promoted = 0

        if budget > 0 and gpu_bytes > budget * 0.85:
            evicted = self._evict_cold_blocks()

        if self.ssd_dir:
            self._spill_dram_to_ssd()

        if budget > 0 and gpu_bytes < budget * 0.60:
            promoted = self._promote_warm_blocks()

        return {
            "step": self._step,
            "gpu_kv_bytes": gpu_bytes,
            "gpu_ratio": gpu_ratio,
            "evicted": evicted,
            "promoted": promoted,
            "tm_stats": raw,
        }

    def _evict_cold_blocks(self, max_evict: int = 8) -> int:
        """Evict the coldest GPU-resident blocks to DRAM (or SSD)."""
        if _C is None:
            return 0

        candidates = []
        for layer_blocks in self._blocks:
            for blk in layer_blocks:
                if blk.tier == TIER_GPU and not blk.is_sink and blk.gpu_data is not None:
                    score = 0.0
                    if self._tm_handle:
                        try:
                            s = _C.tm_get_stats(self._tm_handle)
                            score = s.get("n_cold", 0)
                        except Exception:
                            pass
                    candidates.append((blk, blk.access_count))

        candidates.sort(key=lambda x: x[1])

        evicted = 0
        for blk, _ in candidates[:max_evict]:
            gpu_bytes = self.gpu_kv_bytes()
            if self.gpu_budget_bytes > 0 and gpu_bytes <= self.gpu_budget_bytes * 0.7:
                break
            self._demote_block(blk)
            evicted += 1

        return evicted

    def _promote_warm_blocks(self, max_promote: int = 4) -> int:
        """Promote DRAM-resident blocks with high access counts back to GPU."""
        candidates = []
        for layer_blocks in self._blocks:
            for blk in layer_blocks:
                if blk.tier == TIER_DRAM and blk.dram_data is not None:
                    candidates.append((blk, blk.access_count))

        candidates.sort(key=lambda x: x[1], reverse=True)

        promoted = 0
        for blk, _ in candidates[:max_promote]:
            self._promote_block(blk)
            promoted += 1

        return promoted

    # ------------------------------------------------------------------
    # Data movement primitives
    # ------------------------------------------------------------------

    def _spill_dram_to_ssd(self, max_spill: int = 4):
        """Move oldest DRAM blocks to SSD when DRAM has many blocks."""
        dram_blocks = []
        for layer_blocks in self._blocks:
            for blk in layer_blocks:
                if blk.tier == TIER_DRAM and blk.dram_data is not None and not blk.is_sink:
                    dram_blocks.append(blk)

        if len(dram_blocks) < 4:
            return

        dram_blocks.sort(key=lambda b: b.access_count)
        for blk in dram_blocks[:max_spill]:
            self._save_to_ssd(blk)

    def _demote_block(self, blk: KVBlock):
        """GPU -> DRAM (and optionally DRAM -> SSD)."""
        if blk.tier == TIER_GPU and blk.gpu_data is not None:
            if blk.dram_data is None:
                blk.dram_data = torch.empty_like(blk.gpu_data, device="cpu", pin_memory=True)
            blk.dram_data.copy_(blk.gpu_data, non_blocking=False)
            blk.gpu_data = None
            blk.tier = TIER_DRAM
            self._stats["gpu_to_dram"] += 1

            if self._tm_handle and _C is not None:
                _C.tm_set_block_tier(self._tm_handle, blk.block_id, int(_C.HOST_DRAM))

            logger.debug("Demoted block %d (layer=%d, start=%d) to DRAM",
                         blk.block_id, blk.layer, blk.token_start)

    def _promote_block(self, blk: KVBlock):
        """DRAM -> GPU (or SSD -> DRAM -> GPU)."""
        if blk.tier == TIER_SSD:
            self._load_from_ssd(blk)

        if blk.tier == TIER_DRAM and blk.dram_data is not None:
            blk.gpu_data = blk.dram_data.to("cuda", non_blocking=False)
            blk.tier = TIER_GPU
            self._stats["dram_to_gpu"] += 1

            if self._tm_handle and _C is not None:
                _C.tm_set_block_tier(self._tm_handle, blk.block_id, int(_C.GPU_HBM))

            logger.debug("Promoted block %d (layer=%d, start=%d) to GPU",
                         blk.block_id, blk.layer, blk.token_start)

    def _save_to_ssd(self, blk: KVBlock):
        """DRAM -> SSD file."""
        if self.ssd_dir is None or blk.dram_data is None:
            return
        path = os.path.join(self.ssd_dir, f"blk_{blk.block_id}.bin")
        raw = blk.dram_data.numpy().tobytes()
        with open(path, "wb") as f:
            f.write(raw)
        blk.ssd_path = path
        blk.dram_data = None
        blk.tier = TIER_SSD
        self._stats["dram_to_ssd"] += 1

        if self._tm_handle and _C is not None:
            _C.tm_set_block_tier(self._tm_handle, blk.block_id, int(_C.SSD))

    def _load_from_ssd(self, blk: KVBlock):
        """SSD file -> DRAM."""
        if blk.ssd_path is None or not os.path.exists(blk.ssd_path):
            return
        with open(blk.ssd_path, "rb") as f:
            raw = f.read()
        shape = (2, self.n_kv_heads, self.block_size, self.head_dim)
        blk.dram_data = torch.frombuffer(
            bytearray(raw), dtype=self.dtype
        ).reshape(shape).clone()
        blk.dram_data = blk.dram_data.pin_memory()
        blk.tier = TIER_DRAM
        self._stats["ssd_to_dram"] += 1

    # ------------------------------------------------------------------
    # Stats and info
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        gpu_count = dram_count = ssd_count = 0
        for layer_blocks in self._blocks:
            for blk in layer_blocks:
                if blk.tier == TIER_GPU:
                    gpu_count += 1
                elif blk.tier == TIER_DRAM:
                    dram_count += 1
                elif blk.tier == TIER_SSD:
                    ssd_count += 1

        tm_stats = {}
        if self._tm_handle and _C is not None:
            tm_stats = _C.tm_get_stats(self._tm_handle)

        return {
            "step": self._step,
            "total_tokens": self._total_tokens,
            "blocks_gpu": gpu_count,
            "blocks_dram": dram_count,
            "blocks_ssd": ssd_count,
            "blocks_total": gpu_count + dram_count + ssd_count,
            "gpu_kv_mb": gpu_count * self._block_bytes / (1 << 20),
            "dram_kv_mb": dram_count * self._block_bytes / (1 << 20),
            "migrations": dict(self._stats),
            "tm": tm_stats,
        }

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"KVCacheManager(tokens={s['total_tokens']}, "
            f"blocks={s['blocks_total']} "
            f"[GPU:{s['blocks_gpu']} DRAM:{s['blocks_dram']} SSD:{s['blocks_ssd']}], "
            f"gpu_kv={s['gpu_kv_mb']:.1f}MB)"
        )


class NaiveOffloadManager:
    """
    FIFO-based offload baseline: when GPU budget is exceeded,
    evict the oldest blocks to DRAM. No attention awareness.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        gpu_budget_bytes: int = 0,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_budget_bytes = gpu_budget_bytes
        self._block_bytes = 2 * n_kv_heads * block_size * head_dim * dtype.itemsize

        self._blocks: list[list[KVBlock]] = [[] for _ in range(n_layers)]
        self._next_block_id = 0
        self._total_tokens = 0
        self._stats = {"gpu_to_dram": 0, "dram_to_gpu": 0, "total_blocks": 0}

    def gpu_kv_bytes(self) -> int:
        return sum(
            1 for lb in self._blocks for b in lb if b.tier == TIER_GPU
        ) * self._block_bytes

    def ingest_step(self, past_key_values):
        past_key_values = KVCacheManager._to_legacy(past_key_values)
        for li, (k, v) in enumerate(past_key_values):
            seq_len = k.shape[2]
            existing = sum(b.token_count for b in self._blocks[li])
            if seq_len <= existing:
                continue
            for pos in range(existing, seq_len, self.block_size):
                end = min(pos + self.block_size, seq_len)
                n = end - pos
                bid = self._next_block_id
                self._next_block_id += 1
                blk = KVBlock(block_id=bid, layer=li, token_start=pos,
                              token_count=n, capacity=self.block_size)
                data = torch.zeros(2, self.n_kv_heads, self.block_size, self.head_dim,
                                   dtype=self.dtype, device="cuda")
                data[0, :, :n, :] = k[0, :, pos:end, :]
                data[1, :, :n, :] = v[0, :, pos:end, :]
                blk.gpu_data = data
                blk.tier = TIER_GPU
                self._blocks[li].append(blk)
                self._stats["total_blocks"] += 1
        self._total_tokens = past_key_values[0][0].shape[2]

    def append_token(self, past_key_values):
        past_key_values = KVCacheManager._to_legacy(past_key_values)
        for li, (k, v) in enumerate(past_key_values):
            seq_len = k.shape[2]
            blocks = self._blocks[li]
            total = sum(b.token_count for b in blocks)
            if seq_len <= total:
                continue
            if blocks and blocks[-1].token_count < blocks[-1].capacity:
                blk = blocks[-1]
                if blk.tier != TIER_GPU:
                    blk.gpu_data = blk.dram_data.to("cuda", non_blocking=False)
                    blk.tier = TIER_GPU
                    self._stats["dram_to_gpu"] += 1
                idx = blk.token_count
                blk.gpu_data[0, :, idx, :] = k[0, :, -1, :]
                blk.gpu_data[1, :, idx, :] = v[0, :, -1, :]
                blk.token_count += 1
            else:
                bid = self._next_block_id
                self._next_block_id += 1
                blk = KVBlock(block_id=bid, layer=li, token_start=total,
                              token_count=1, capacity=self.block_size)
                data = torch.zeros(2, self.n_kv_heads, self.block_size, self.head_dim,
                                   dtype=self.dtype, device="cuda")
                data[0, :, 0, :] = k[0, :, -1, :]
                data[1, :, 0, :] = v[0, :, -1, :]
                blk.gpu_data = data
                self._blocks[li].append(blk)
                self._stats["total_blocks"] += 1
        self._total_tokens = past_key_values[0][0].shape[2]

    def build_past_kv(self, device: str = "cuda"):
        result = []
        for li in range(self.n_layers):
            blocks = self._blocks[li]
            if not blocks:
                result.append((
                    torch.zeros(1, self.n_kv_heads, 0, self.head_dim,
                                dtype=self.dtype, device=device),
                    torch.zeros(1, self.n_kv_heads, 0, self.head_dim,
                                dtype=self.dtype, device=device),
                ))
                continue
            total = sum(b.token_count for b in blocks)
            kout = torch.zeros(1, self.n_kv_heads, total, self.head_dim,
                               dtype=self.dtype, device=device)
            vout = torch.zeros_like(kout)
            pos = 0
            for blk in blocks:
                if blk.token_count == 0:
                    continue
                if blk.tier != TIER_GPU:
                    blk.gpu_data = blk.dram_data.to("cuda", non_blocking=False)
                    blk.tier = TIER_GPU
                    self._stats["dram_to_gpu"] += 1
                n = blk.token_count
                kout[0, :, pos:pos+n, :] = blk.gpu_data[0, :, :n, :]
                vout[0, :, pos:pos+n, :] = blk.gpu_data[1, :, :n, :]
                pos += n
            result.append((kout, vout))
        return KVCacheManager._from_legacy(tuple(result))

    def schedule(self) -> dict:
        """FIFO eviction: evict oldest GPU blocks when over budget."""
        if self.gpu_budget_bytes <= 0:
            return {}
        gpu_bytes = self.gpu_kv_bytes()
        evicted = 0
        if gpu_bytes > self.gpu_budget_bytes * 0.85:
            for li in range(self.n_layers):
                for blk in self._blocks[li]:
                    if gpu_bytes <= self.gpu_budget_bytes * 0.7:
                        break
                    if blk.tier == TIER_GPU and blk.gpu_data is not None:
                        blk.dram_data = torch.empty_like(
                            blk.gpu_data, device="cpu", pin_memory=True)
                        blk.dram_data.copy_(blk.gpu_data, non_blocking=False)
                        blk.gpu_data = None
                        blk.tier = TIER_DRAM
                        self._stats["gpu_to_dram"] += 1
                        gpu_bytes -= self._block_bytes
                        evicted += 1
        return {"evicted": evicted, "gpu_kv_bytes": self.gpu_kv_bytes()}

    def step_done(self):
        pass

    def report_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        pass

    def get_stats(self) -> dict:
        gpu = sum(1 for lb in self._blocks for b in lb if b.tier == TIER_GPU)
        dram = sum(1 for lb in self._blocks for b in lb if b.tier == TIER_DRAM)
        return {
            "blocks_gpu": gpu, "blocks_dram": dram, "blocks_ssd": 0,
            "blocks_total": gpu + dram,
            "gpu_kv_mb": gpu * self._block_bytes / (1 << 20),
            "migrations": dict(self._stats),
        }

    def destroy(self):
        pass

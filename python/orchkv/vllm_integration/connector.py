"""
D2: OrchKvCache OffloadingConnector for vLLM V1.

Implements vLLM's KVConnectorBase_V1 interface, routing KV cache
swap-out/swap-in through OrchKvCache's tiered storage (GPU → DRAM → NVM → SSD).

The connector has two roles:
  - SCHEDULER: decides which KV blocks to offload/load (runs in scheduler process)
  - WORKER:    performs actual GPU↔DRAM data transfers (runs in worker process)

Architecture:
  vLLM Attention Layer
      │ save_kv_layer()          ← after each layer's attention forward
      ▼
  OrchKvConnectorWorker
      │ cudaMemcpyAsync GPU→DRAM (pinned)
      │ orchkv_core C library manages tiered storage
      ▼
  OrchKvCache tiered_manager
      │ auto-demotes cold blocks DRAM→NVM→SSD
      │ auto-prefetches hot blocks back
      ▼
  start_load_kv() / wait_for_layer_load()
      │ promotes blocks from DRAM/NVM/SSD → GPU
      ▼
  vLLM Attention Layer (read KV)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

    class KVConnectorRole:
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        """Fallback stub when vLLM is not installed."""
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._connector_metadata = None
            self._role = role

        @property
        def role(self):
            return self._role

        def bind_connector_metadata(self, meta):
            self._connector_metadata = meta

        def clear_connector_metadata(self):
            self._connector_metadata = None

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVCacheConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext

try:
    import orchkv_core as _C
except ImportError:
    _C = None


@dataclass
class OrchKvConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each step."""
    blocks_to_save: dict[str, list[int]] = field(default_factory=dict)
    blocks_to_load: dict[str, list[int]] = field(default_factory=dict)
    req_block_map: dict[str, dict[int, int]] = field(default_factory=dict)


@dataclass
class _BlockRecord:
    """Tracks a single block's state in the connector."""
    vllm_block_id: int
    orchkv_block_id: int = -1
    tier: str = "gpu"
    saved: bool = False


class OrchKvConnectorWorker:
    """Worker-side logic: performs actual GPU↔DRAM transfers via orchkv_core."""

    def __init__(self, orchkv_cfg: dict[str, Any]):
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._save_stream = torch.cuda.Stream()
        self._load_stream = torch.cuda.Stream()
        self._pending_saves: list[tuple[str, int]] = []
        self._pending_loads: list[tuple[str, int]] = []
        self._load_events: dict[str, torch.cuda.Event] = {}

        self._dram_buffers: dict[tuple[str, int], torch.Tensor] = {}
        self._lock = threading.Lock()

        self._save_count = 0
        self._load_count = 0
        self._initialized = False
        self._orchkv_cfg = orchkv_cfg

    def _ensure_init(self):
        if self._initialized or _C is None:
            return
        cfg = _C.Config()
        cfg.gpu_pool_bytes = 0  # GPU memory is managed by vLLM
        cfg.dram_pool_bytes = self._orchkv_cfg.get(
            "dram_pool_bytes", 8 * (1 << 30))
        cfg.d_head = self._orchkv_cfg.get("d_head", 128)
        cfg.tokens_per_block = self._orchkv_cfg.get("tokens_per_block", 16)
        cfg.num_cuda_streams = self._orchkv_cfg.get("num_cuda_streams", 4)
        cfg.orchfs_io_workers = self._orchkv_cfg.get("io_workers", 4)
        _C.init(cfg)
        self._initialized = True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        self._ensure_init()
        logger.info("OrchKvConnector: registered %d KV cache layers",
                     len(kv_caches))

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        metadata: OrchKvConnectorMetadata,
        block_ids: list[int] | None = None,
    ):
        """Async save: GPU → pinned DRAM buffer (cudaMemcpyAsync)."""
        if block_ids is None:
            block_ids = metadata.blocks_to_save.get(layer_name, [])
        if not block_ids:
            return

        with torch.cuda.stream(self._save_stream):
            for bid in block_ids:
                src = kv_layer[bid]
                buf_key = (layer_name, bid)
                with self._lock:
                    if buf_key not in self._dram_buffers:
                        buf = torch.empty_like(src, device="cpu").pin_memory()
                        self._dram_buffers[buf_key] = buf
                    else:
                        buf = self._dram_buffers[buf_key]
                buf.copy_(src, non_blocking=True)
                self._pending_saves.append((layer_name, bid))
        self._save_count += len(block_ids)

    def wait_for_save(self):
        """Block until all pending saves are complete."""
        self._save_stream.synchronize()
        self._pending_saves.clear()

    def start_load_kv(self, metadata: OrchKvConnectorMetadata):
        """Kick off async loads: DRAM buffer → GPU KV cache."""
        for layer_name, block_ids in metadata.blocks_to_load.items():
            kv_layer = self._kv_caches.get(layer_name)
            if kv_layer is None:
                continue

            with torch.cuda.stream(self._load_stream):
                for bid in block_ids:
                    buf_key = (layer_name, bid)
                    with self._lock:
                        buf = self._dram_buffers.get(buf_key)
                    if buf is None:
                        logger.warning("OrchKv: no DRAM buffer for %s block %d",
                                        layer_name, bid)
                        continue
                    kv_layer[bid].copy_(buf, non_blocking=True)
                    self._pending_loads.append((layer_name, bid))

                event = self._load_stream.record_event()
                self._load_events[layer_name] = event
        self._load_count += sum(
            len(bids) for bids in metadata.blocks_to_load.values())

    def wait_for_layer_load(self, layer_name: str):
        """Block until a specific layer's load is complete."""
        event = self._load_events.pop(layer_name, None)
        if event is not None:
            event.synchronize()

    def free_blocks(self, layer_name: str, block_ids: list[int]):
        """Release DRAM buffers for freed blocks."""
        with self._lock:
            for bid in block_ids:
                self._dram_buffers.pop((layer_name, bid), None)

    def get_stats(self) -> dict[str, Any]:
        return {
            "save_count": self._save_count,
            "load_count": self._load_count,
            "dram_buffers": len(self._dram_buffers),
        }

    def shutdown(self):
        self._save_stream.synchronize()
        self._load_stream.synchronize()
        with self._lock:
            self._dram_buffers.clear()
        if self._initialized and _C is not None:
            _C.shutdown()
            self._initialized = False


class OrchKvConnectorScheduler:
    """Scheduler-side logic: tracks which blocks to offload/load."""

    def __init__(self, orchkv_cfg: dict[str, Any]):
        self._orchkv_cfg = orchkv_cfg
        self._cached_requests: dict[str, set[int]] = {}
        self._pending_offload: dict[str, list[int]] = {}
        self._pending_load: dict[str, list[int]] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check if we have cached KV data for this request's prefix."""
        req_id = request.request_id
        cached = self._cached_requests.get(req_id)
        if cached is None:
            return 0, False

        n_cached = len(cached)
        new_tokens = max(0, n_cached - num_computed_tokens)
        return new_tokens, (new_tokens > 0)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        """Track block allocation for a request."""
        pass

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput",
    ) -> OrchKvConnectorMetadata:
        """Build metadata for the worker to execute this step's transfers."""
        meta = OrchKvConnectorMetadata()
        meta.blocks_to_save = dict(self._pending_offload)
        meta.blocks_to_load = dict(self._pending_load)
        self._pending_offload.clear()
        self._pending_load.clear()
        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """When a request finishes, save its blocks for potential reuse."""
        req_id = request.request_id
        self._cached_requests[req_id] = set(block_ids)
        return False, None

    def handle_preemptions(self, preempted_req_ids: set[str]):
        """Mark preempted request blocks for offloading to DRAM."""
        for req_id in preempted_req_ids:
            cached = self._cached_requests.get(req_id)
            if cached:
                for layer_name in self._get_layer_names():
                    self._pending_offload.setdefault(
                        layer_name, []).extend(cached)

    def _get_layer_names(self) -> list[str]:
        n_layers = self._orchkv_cfg.get("n_layers", 32)
        return [f"layer_{i}" for i in range(n_layers)]


class OrchKvOffloadingConnector(KVConnectorBase_V1):
    """
    vLLM V1 KVConnectorBase_V1 implementation that routes KV cache
    offloading through OrchKvCache's tiered storage system.

    Replaces vLLM's default CPU-only offloading with a multi-tier
    DRAM → NVM → SSD pipeline managed by OrchKvCache's tiered_manager.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        extra_cfg = {}
        if hasattr(vllm_config, "kv_transfer_config"):
            ktc = vllm_config.kv_transfer_config
            if ktc is not None and hasattr(ktc, "kv_connector_extra_config"):
                extra_cfg = ktc.kv_connector_extra_config or {}

        orchkv_cfg = {
            "dram_pool_bytes": int(
                extra_cfg.get("dram_pool_gb", 8)) * (1 << 30),
            "d_head": getattr(
                getattr(vllm_config, "model_config", None),
                "head_dim", extra_cfg.get("d_head", 128)),
            "tokens_per_block": extra_cfg.get("tokens_per_block", 16),
            "num_cuda_streams": extra_cfg.get("num_cuda_streams", 4),
            "io_workers": extra_cfg.get("io_workers", 4),
            "n_layers": getattr(
                getattr(vllm_config, "model_config", None),
                "num_layers", extra_cfg.get("n_layers", 32)),
        }

        self._worker: OrchKvConnectorWorker | None = None
        self._scheduler: OrchKvConnectorScheduler | None = None

        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = OrchKvConnectorScheduler(orchkv_cfg)
        elif role == KVConnectorRole.WORKER:
            self._worker = OrchKvConnectorWorker(orchkv_cfg)

        logger.info("OrchKvOffloadingConnector initialized (role=%s)", role)

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        assert self._worker is not None
        meta = self._connector_metadata
        if isinstance(meta, OrchKvConnectorMetadata):
            self._worker.start_load_kv(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self._worker is not None
        self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        assert self._worker is not None
        meta = self._connector_metadata
        if isinstance(meta, OrchKvConnectorMetadata):
            self._worker.save_kv_layer(layer_name, kv_layer, meta)

    def wait_for_save(self):
        assert self._worker is not None
        self._worker.wait_for_save()

    def handle_preemptions(self, preempted_req_ids: set[str]):
        if self._scheduler is not None:
            self._scheduler.handle_preemptions(preempted_req_ids)

    def shutdown(self):
        if self._worker is not None:
            self._worker.shutdown()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)

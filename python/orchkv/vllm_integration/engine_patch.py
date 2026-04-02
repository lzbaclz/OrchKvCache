"""
D2: Register OrchKvCache backends into vLLM.

Two integration modes:

  1. KV Offloading Connector (V1 API) — routes KV save/load through
     OrchKvCache tiered storage.

  2. Block-Level Partial Swap — patches the scheduler to evict cold
     blocks instead of whole requests when GPU memory is tight.

Usage:
    import os
    os.environ["ORCHKV_SWAP"] = "1"          # enable attention-aware swap
    os.environ["ORCHKV_PARTIAL_SWAP"] = "1"   # enable block-level partial swap

    from orchkv.vllm_integration.engine_patch import (
        register_orchkv_backend,
        install_block_level_swap,
    )
    register_orchkv_backend()

    from vllm import LLM
    llm = LLM(model="meta-llama/Llama-2-7b-hf", swap_space=32, ...)

    # After engine creation, install partial swap on the scheduler:
    install_block_level_swap(llm.llm_engine.scheduler[0])
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_orchkv_backend():
    """
    Register OrchKvOffloadingConnector into vLLM's connector registry.

    Safe to call multiple times (idempotent).
    Must be called BEFORE creating the vLLM engine.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import base as v1_base
    except ImportError:
        logger.warning(
            "vLLM not installed or V1 connector API not available. "
            "OrchKvCache backend registration skipped."
        )
        return

    from orchkv.vllm_integration.connector import OrchKvOffloadingConnector

    registry = getattr(v1_base, "_CONNECTOR_REGISTRY", None)
    if registry is not None:
        registry["OrchKvOffloadingConnector"] = OrchKvOffloadingConnector
        logger.info("Registered OrchKvOffloadingConnector in vLLM V1 registry")
    else:
        try:
            from vllm.distributed.kv_transfer import kv_connector
            if hasattr(kv_connector, "register"):
                kv_connector.register(
                    "OrchKvOffloadingConnector", OrchKvOffloadingConnector)
                logger.info(
                    "Registered OrchKvOffloadingConnector via "
                    "kv_connector.register()"
                )
            else:
                import vllm.distributed.kv_transfer.kv_connector.v1.base as base_mod
                if not hasattr(base_mod, "_CONNECTOR_REGISTRY"):
                    base_mod._CONNECTOR_REGISTRY = {}
                base_mod._CONNECTOR_REGISTRY[
                    "OrchKvOffloadingConnector"] = OrchKvOffloadingConnector
                logger.info(
                    "Registered OrchKvOffloadingConnector "
                    "(injected _CONNECTOR_REGISTRY)"
                )
        except Exception as e:
            logger.error("Failed to register OrchKv backend: %s", e)
            return

    _REGISTERED = True


def install_block_level_swap(
    scheduler,
    tm_handle: Optional[int] = None,
    gpu_hwm: float = 0.90,
    max_evict_frac: float = 0.20,
    ema_lambda: float = 0.3,
    sink_blocks: int = 1,
):
    """Install block-level partial swap on a vLLM scheduler.

    This replaces vLLM's request-level preemption with block-level
    cold-block eviction. When GPU memory is tight, only the coldest
    blocks are swapped out (across multiple requests), instead of
    fully preempting one entire request.

    Args:
        scheduler: A vLLM ``Scheduler`` instance (e.g.
            ``llm.llm_engine.scheduler[0]``).
        tm_handle: orchkv_core tiered_manager handle, or None.
        gpu_hwm: GPU utilization threshold to trigger eviction.
        max_evict_frac: Maximum fraction of candidate blocks to evict.
        ema_lambda: EMA decay factor for hotness tracking.
        sink_blocks: Number of attention-sink blocks to protect per seq.

    Returns:
        The ``PartialSwapManager`` instance (useful for reporting stats).
    """
    from orchkv.vllm_integration.block_level_swap import install_partial_swap

    return install_partial_swap(
        scheduler=scheduler,
        tm_handle=tm_handle,
        gpu_hwm=gpu_hwm,
        max_evict_frac=max_evict_frac,
        ema_lambda=ema_lambda,
        sink_blocks=sink_blocks,
    )

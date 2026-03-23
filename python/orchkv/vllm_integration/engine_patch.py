"""
D2: Register OrchKvCache as a vLLM KV offloading backend.

Usage (programmatic):
    from orchkv.vllm_integration.engine_patch import register_orchkv_backend
    register_orchkv_backend()

    from vllm import LLM
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        kv_transfer_config={
            "kv_connector": "OrchKvOffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "dram_pool_gb": 8,
            },
        },
    )

Usage (CLI):
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-7b-hf \\
        --kv-transfer-config '{
            "kv_connector": "OrchKvOffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"dram_pool_gb": 8}
        }'
"""
from __future__ import annotations

import logging

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

"""
OrchKvCache — Tiered KV-Cache management for LLM inference.

Usage:
    import orchkv
    cfg = orchkv.Config()
    cfg.gpu_pool_bytes = 4 * (1 << 30)   # 4 GB
    orchkv.init(cfg)
    ...
    orchkv.shutdown()
"""

try:
    from orchkv_core import *  # noqa: F401,F403
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

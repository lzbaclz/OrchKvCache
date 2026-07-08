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

from orchkv.config import PAPER_DEFAULTS  # noqa: F401

try:
    from orchkv.config import make_config  # noqa: F401
except Exception:
    pass

try:
    from orchkv.kvcache_manager import KVCacheManager  # noqa: F401
except Exception:
    pass

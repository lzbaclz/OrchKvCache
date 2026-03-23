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
except ImportError as e:
    raise ImportError(
        "orchkv_core C extension not found. "
        "Build with: cmake --build build && cp build/bindings/orchkv_core*.so python/orchkv/"
    ) from e

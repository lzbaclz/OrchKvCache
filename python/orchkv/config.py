"""
Convenience wrappers for OrchKvCache configuration.
"""
from orchkv_core import Config as _Config


def make_config(
    gpu_pool_gb: float = 4.0,
    dram_pool_gb: float = 8.0,
    d_head: int = 128,
    tokens_per_block: int = 64,
    num_cuda_streams: int = 4,
    io_workers: int = 4,
) -> "_Config":
    """Create a Config with common settings in human-friendly units."""
    cfg = _Config()
    cfg.gpu_pool_bytes = int(gpu_pool_gb * (1 << 30))
    cfg.dram_pool_bytes = int(dram_pool_gb * (1 << 30))
    cfg.d_head = d_head
    cfg.tokens_per_block = tokens_per_block
    cfg.num_cuda_streams = num_cuda_streams
    cfg.orchfs_io_workers = io_workers
    return cfg

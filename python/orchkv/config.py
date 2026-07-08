"""
Convenience wrappers for OrchKvCache configuration.

Canonical paper parameters (SIGMETRICS 2027 submission):

    Hotness formula:  S = α·attn_ema + β·recency + γ·frequency
    EMA update:       ema_t = λ · ema_{t-1} + (1-λ) · raw_t
                      (λ is retention/decay factor; higher λ = more smoothing)

    Defaults:
        α (alpha)           = 0.7   # attention weight
        β (beta)            = 0.2   # recency weight
        γ (gamma)           = 0.1   # frequency weight
        λ (ema_lambda)      = 0.9   # EMA retention factor
        τ (recency_tau)     = 50.0  # recency decay half-life in steps
        prefetch_budget (K) = 8     # blocks prefetched per cycle
        gpu_hwm             = 0.9   # GPU high water mark
        gpu_lwm             = 0.7   # GPU low water mark
        dram_hwm            = 0.9
        dram_lwm            = 0.7
        cooldown_sec        = 0.5   # threshold adjustment cooldown
        adjust_step         = 0.02  # threshold increment per adjustment
        threshold_to_gpu    = 0.5   # score above which to promote to GPU
        threshold_to_dram   = 0.2   # score below which to demote from GPU

    These match the pybind defaults set in _init_tiered_manager().
"""
try:
    from orchkv_core import Config as _Config
except ImportError:
    _Config = None


PAPER_DEFAULTS = {
    "alpha": 0.7,
    "beta": 0.2,
    "gamma": 0.1,
    "ema_lambda": 0.9,
    "recency_tau": 50.0,
    "prefetch_budget": 8,
    "gpu_hwm": 0.9,
    "gpu_lwm": 0.7,
    "dram_hwm": 0.9,
    "dram_lwm": 0.7,
    "cooldown_sec": 0.5,
    "adjust_step": 0.02,
    "threshold_to_gpu": 0.5,
    "threshold_to_dram": 0.2,
    "max_blocks": 8192,
}


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

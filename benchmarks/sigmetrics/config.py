"""
Experiment configuration for SIGMETRICS 2027 evaluation.

Defines the full experiment grid: models, memory budgets, workloads,
baselines, and the metric set collected for every run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# ── Models ────────────────────────────────────────────────────────────

MODELS: dict[str, dict[str, Any]] = {
    "qwen2.5-7b": {
        "hf_name": "Qwen/Qwen2.5-7B",
        "n_layers": 28,
        "n_kv_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "max_position": 32768,
        "kv_bytes_per_token": 28 * 2 * 4 * 128 * 2,    # layers * 2(K+V) * heads * dim * fp16
    },
    "llama-3.1-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B",
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "max_position": 131072,
        "kv_bytes_per_token": 32 * 2 * 8 * 128 * 2,
    },
    "llama-2-7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "n_layers": 32,
        "n_kv_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
        "max_position": 4096,
        "kv_bytes_per_token": 32 * 2 * 32 * 128 * 2,
    },
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 32768,
        "max_position": 32768,
        "kv_bytes_per_token": 32 * 2 * 8 * 128 * 2,
    },
}

# ── Memory budgets (fraction of model's full KV capacity) ────────────

BUDGET_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75]


def budget_bytes(model_key: str, fraction: float, seq_len: int) -> int:
    """Compute GPU KV budget in bytes for a given model, fraction, and seq length."""
    bpt = MODELS[model_key]["kv_bytes_per_token"]
    return int(bpt * seq_len * fraction)


def budget_mb(model_key: str, fraction: float, seq_len: int) -> float:
    return budget_bytes(model_key, fraction, seq_len) / (1 << 20)


# ── Workloads ─────────────────────────────────────────────────────────

WORKLOADS: dict[str, dict[str, Any]] = {
    "sharegpt": {
        "description": "Multi-turn conversations from ShareGPT",
        "hf_dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "typical_prompt_len": (256, 2048),
        "typical_output_len": (64, 512),
    },
    "longbench": {
        "description": "Long-context multi-doc QA + summarization from LongBench",
        "hf_dataset": "THUDM/LongBench",
        "subsets": ["multifieldqa_en", "multi_news", "gov_report"],
        "typical_prompt_len": (2048, 16384),
        "typical_output_len": (64, 256),
    },
    "ruler": {
        "description": "Synthetic NIAH and multi-hop at controlled lengths",
        "synthetic": True,
        "lengths": [4096, 8192, 16384, 32768],
        "tasks": ["niah_single", "niah_multi", "multi_hop"],
    },
    "rag": {
        "description": "Multi-document retrieval-augmented prompts",
        "synthetic": True,
        "n_documents": [3, 5, 10, 20],
        "doc_length": (256, 1024),
    },
    "agentic": {
        "description": "Multi-step tool-use agent traces with interleaved reasoning",
        "synthetic": True,
        "n_turns": [3, 5, 10],
        "context_per_turn": (512, 2048),
    },
}

# ── Baselines ─────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    """Configuration for one baseline system variant."""
    name: str
    manager_cls: str              # Python class name or "none" for GPU-only
    report_attn: bool = False
    sample_interval: int = 0
    use_qk_proxy: bool = False    # QK-norm proxy instead of full attention
    extra_kwargs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


BASELINES: dict[str, BaselineConfig] = {
    "gpu_only": BaselineConfig(
        name="gpu_only",
        manager_cls="none",
        report_attn=False,
    ),
    "fifo_offload": BaselineConfig(
        name="fifo_offload",
        manager_cls="FastFIFOManager",
        report_attn=False,
    ),
    "orchkv": BaselineConfig(
        name="orchkv",
        manager_cls="FastKVCacheManager",
        report_attn=True,
        sample_interval=10,
    ),
    "orchkv_sampling": BaselineConfig(
        name="orchkv_sampling",
        manager_cls="FastKVCacheManager",
        report_attn=True,
        sample_interval=50,
        extra_kwargs={"description": "OrchKv with sparser attention sampling (N=50)"},
    ),
    "orchkv_qk_proxy": BaselineConfig(
        name="orchkv_qk_proxy",
        manager_cls="FastKVCacheManager",
        report_attn=True,
        sample_interval=10,
        use_qk_proxy=True,
        extra_kwargs={"description": "OrchKv with QK-norm proxy signal"},
    ),
}

# ── Metrics ───────────────────────────────────────────────────────────

METRIC_KEYS = [
    # Latency
    "ttft_ms",
    "tpot_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    # Throughput
    "throughput_tok_s",
    "goodput_under_slo",
    # Memory
    "gpu_mem_used_mb",
    "dram_mem_used_mb",
    # I/O
    "ssd_traffic_mb",
    # Migration
    "evictions",
    "promotions",
    "promotion_stall_count",
    "promotion_stall_p99_us",
    # Quality
    "bit_exact_match",
]

SLO_TARGETS_MS = {
    "ttft": 500.0,
    "tpot": 50.0,
    "itl_p99": 100.0,
}

# ── Full experiment point ─────────────────────────────────────────────

@dataclass
class ExperimentPoint:
    """One point in the full experiment grid."""
    model: str
    workload: str
    baseline: str
    budget_fraction: float
    num_prompts: int = 32
    max_new_tokens: int = 256
    seq_len: int = 2048
    seed: int = 42
    tag: str = ""

    @property
    def budget_bytes(self) -> int:
        return budget_bytes(self.model, self.budget_fraction, self.seq_len)

    @property
    def budget_mb(self) -> float:
        return self.budget_bytes / (1 << 20)

    @property
    def baseline_config(self) -> BaselineConfig:
        return BASELINES[self.baseline]

    @property
    def model_config(self) -> dict:
        return MODELS[self.model]

    def result_name(self) -> str:
        parts = [
            self.model,
            self.workload,
            self.baseline,
            f"b{int(self.budget_fraction * 100)}",
        ]
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def iter_experiment_grid(
    models: list[str] | None = None,
    workloads: list[str] | None = None,
    baselines: list[str] | None = None,
    budgets: list[float] | None = None,
):
    """Yield ExperimentPoint for each combination in the grid."""
    for m in (models or list(MODELS)):
        for w in (workloads or list(WORKLOADS)):
            for bl in (baselines or list(BASELINES)):
                for bf in (budgets or BUDGET_FRACTIONS):
                    yield ExperimentPoint(model=m, workload=w, baseline=bl,
                                          budget_fraction=bf)


def grid_size(
    models: list[str] | None = None,
    workloads: list[str] | None = None,
    baselines: list[str] | None = None,
    budgets: list[float] | None = None,
) -> int:
    nm = len(models or MODELS)
    nw = len(workloads or WORKLOADS)
    nb = len(baselines or BASELINES)
    nbf = len(budgets or BUDGET_FRACTIONS)
    return nm * nw * nb * nbf

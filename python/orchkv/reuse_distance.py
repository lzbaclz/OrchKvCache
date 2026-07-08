"""
Reuse-Distance Predictor for KV-Cache Blocks.

Core contribution for SIGMETRICS revision: models the temporal reuse pattern
of KV-cache blocks during autoregressive LLM decoding, enabling predictive
tiered placement decisions that overlap data movement with computation.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_GPU = 0
TIER_DRAM = 1
TIER_SSD = 2

_TIER_NAMES = {TIER_GPU: "GPU", TIER_DRAM: "DRAM", TIER_SSD: "SSD"}


class Decision(Enum):
    KEEP_GPU = auto()
    DEMOTE_DRAM = auto()
    DEMOTE_SSD = auto()
    PROMOTE_GPU = auto()


class SignalMode(Enum):
    FULL_ATTENTION = auto()
    PERIODIC_SAMPLING = auto()
    QK_PROXY = auto()


# ---------------------------------------------------------------------------
# BlockResidencyDecision
# ---------------------------------------------------------------------------


@dataclass
class BlockResidencyDecision:
    """Result of the reuse-distance-based placement decision for one block."""

    block_id: int
    predicted_reuse_distance: float
    promotion_latency_us: float
    overlap_slack_steps: float
    decision: Decision
    confidence: float
    critical_path_probability: float


# ---------------------------------------------------------------------------
# SignalAcquisition
# ---------------------------------------------------------------------------


class SignalAcquisition:
    """Acquires per-block importance signals with configurable cost/accuracy.

    Three modes trade off accuracy against overhead:
      FULL_ATTENTION   – raw softmax weights every step (oracle baseline)
      PERIODIC_SAMPLING – full attention every N steps, linear interpolation
      QK_PROXY         – Q·K_max proxy without materializing full softmax
                         (inspired by Quest: per-block key statistics)
    """

    def __init__(
        self,
        mode: SignalMode = SignalMode.FULL_ATTENTION,
        sampling_interval: int = 8,
        proxy_top_k: int = 4,
    ):
        self.mode = mode
        self.sampling_interval = sampling_interval
        self.proxy_top_k = proxy_top_k
        self._step = 0
        self._last_full: Optional[torch.Tensor] = None

    def step(self) -> None:
        self._step += 1

    def should_acquire_full(self) -> bool:
        if self.mode == SignalMode.FULL_ATTENTION:
            return True
        if self.mode == SignalMode.PERIODIC_SAMPLING:
            return (self._step % self.sampling_interval) == 0
        return False

    def compute_block_importance(
        self,
        query: torch.Tensor,
        key_blocks: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Return per-block importance scores [num_blocks].

        Args:
            query: [batch, heads, 1, d_head] — current step query.
            key_blocks: [batch, heads, seq_len, d_head] — all cached keys.
            attention_weights: [batch, heads, 1, seq_len] — precomputed attn
                (required for FULL mode, optional for others).
            block_size: tokens per KV block.
        """
        if self.mode == SignalMode.FULL_ATTENTION:
            return self._importance_from_attention(attention_weights, block_size)
        elif self.mode == SignalMode.PERIODIC_SAMPLING:
            return self._importance_sampling(
                query, key_blocks, attention_weights, block_size
            )
        else:
            return self._importance_qk_proxy(query, key_blocks, block_size)

    def _importance_from_attention(
        self, attn: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Aggregate per-token attention into per-block importance."""
        # attn: [B, H, 1, S] -> per-block max across tokens in block
        B, H, _, S = attn.shape
        n_blocks = math.ceil(S / block_size)
        padded_len = n_blocks * block_size
        if S < padded_len:
            attn = F.pad(attn, (0, padded_len - S), value=0.0)
        attn_blocks = attn.view(B, H, 1, n_blocks, block_size)
        importance = attn_blocks.max(dim=-1).values.squeeze(2)  # [B, H, n_blocks]
        importance = importance.mean(dim=(0, 1))  # [n_blocks]
        self._last_full = importance
        return importance

    def _importance_sampling(
        self,
        query: torch.Tensor,
        key_blocks: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        block_size: int,
    ) -> torch.Tensor:
        """Use full attention on sample steps, interpolate otherwise."""
        if self.should_acquire_full() and attention_weights is not None:
            return self._importance_from_attention(attention_weights, block_size)
        if self._last_full is not None:
            return self._last_full
        return self._importance_qk_proxy(query, key_blocks, block_size)

    def _importance_qk_proxy(
        self, query: torch.Tensor, key_blocks: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Cheap Q·K_max proxy: dot query with per-block max-norm key."""
        B, H, _, D = query.shape
        S = key_blocks.shape[2]
        n_blocks = math.ceil(S / block_size)
        padded_len = n_blocks * block_size

        if S < padded_len:
            key_blocks = F.pad(key_blocks, (0, 0, 0, padded_len - S), value=0.0)

        keys_reshaped = key_blocks.view(B, H, n_blocks, block_size, D)

        # Per-block representative: key with largest L2 norm
        norms = keys_reshaped.norm(dim=-1)  # [B, H, n_blocks, block_size]
        top_idx = norms.topk(min(self.proxy_top_k, block_size), dim=-1).indices
        top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        top_keys = keys_reshaped.gather(3, top_idx_expanded)  # [B,H,nblk,topk,D]

        # Score = max over top-k of (q · k)
        q = query.unsqueeze(3)  # [B, H, 1, 1, D]
        scores = (q * top_keys).sum(dim=-1) / math.sqrt(D)  # [B,H,nblk,topk]
        importance = scores.max(dim=-1).values.squeeze(2)  # [B, H, n_blocks]
        importance = importance.mean(dim=(0, 1))  # [n_blocks]
        return importance


# ---------------------------------------------------------------------------
# ReuseDistancePredictor
# ---------------------------------------------------------------------------


class ReuseDistancePredictor:
    """Predicts the number of decode steps until a KV block will next be
    critically accessed (reuse distance).

    The key insight: in lossless LLM decoding, all tokens participate in
    attention, but their contribution is highly skewed. A block's "reuse
    distance" is defined as the number of steps until its attention weight
    exceeds a criticality threshold (meaning it meaningfully impacts output).

    If predicted reuse distance > promotion_latency + scheduling_slack,
    the block can safely leave GPU.
    """

    def __init__(
        self,
        num_blocks: int,
        criticality_threshold: float = 0.01,
        ema_alpha: float = 0.3,
        history_window: int = 64,
        signal_mode: SignalMode = SignalMode.FULL_ATTENTION,
        sampling_interval: int = 8,
    ):
        self.num_blocks = num_blocks
        self.criticality_threshold = criticality_threshold
        self.ema_alpha = ema_alpha
        self.history_window = history_window

        # Per-block tracking
        self._ema_importance = torch.zeros(num_blocks)
        self._last_critical_step = torch.full((num_blocks,), -1, dtype=torch.long)
        self._reuse_intervals: Dict[int, deque] = {
            i: deque(maxlen=history_window) for i in range(num_blocks)
        }
        self._ema_reuse_distance = torch.full((num_blocks,), float("inf"))
        self._importance_history: Dict[int, deque] = {
            i: deque(maxlen=history_window) for i in range(num_blocks)
        }
        self._current_step: int = 0

        self.signal = SignalAcquisition(
            mode=signal_mode, sampling_interval=sampling_interval
        )

    @property
    def current_step(self) -> int:
        return self._current_step

    def observe(self, block_importance: torch.Tensor) -> None:
        """Record one decode step's per-block importance scores.

        Args:
            block_importance: [num_blocks] tensor of importance scores.
        """
        self._current_step += 1
        self.signal.step()

        n = min(block_importance.shape[0], self.num_blocks)
        imp = block_importance[:n].detach().cpu()

        # Update EMA of importance
        alpha = self.ema_alpha
        self._ema_importance[:n] = (
            alpha * imp + (1 - alpha) * self._ema_importance[:n]
        )

        # Detect critical accesses and update reuse intervals
        critical_mask = imp > self.criticality_threshold
        for bid in critical_mask.nonzero(as_tuple=True)[0].tolist():
            last = self._last_critical_step[bid].item()
            if last >= 0:
                interval = self._current_step - last
                self._reuse_intervals[bid].append(interval)
                # Update EMA of reuse distance
                prev_ema = self._ema_reuse_distance[bid].item()
                if math.isinf(prev_ema):
                    self._ema_reuse_distance[bid] = float(interval)
                else:
                    self._ema_reuse_distance[bid] = (
                        alpha * interval + (1 - alpha) * prev_ema
                    )
            self._last_critical_step[bid] = self._current_step

        # Store raw history
        for bid in range(n):
            self._importance_history[bid].append(imp[bid].item())

    def predict(self, block_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Predict reuse distance for specified blocks (or all).

        Returns:
            Tensor [len(block_ids)] of predicted steps until next critical access.
        """
        if block_ids is None:
            block_ids = list(range(self.num_blocks))

        predictions = torch.empty(len(block_ids))
        for i, bid in enumerate(block_ids):
            predictions[i] = self._predict_single(bid)
        return predictions

    def _predict_single(self, block_id: int) -> float:
        """Predict reuse distance for a single block."""
        ema_rd = self._ema_reuse_distance[block_id].item()

        if not math.isinf(ema_rd):
            # Use EMA prediction with recency correction
            steps_since_last = (
                self._current_step - self._last_critical_step[block_id].item()
            )
            # If we're already past the predicted interval, reuse is imminent
            remaining = max(0.0, ema_rd - steps_since_last)
            return remaining

        # No reuse history: fall back to importance-based heuristic
        ema_imp = self._ema_importance[block_id].item()
        if ema_imp < 1e-8:
            return float("inf")

        # Inverse relationship: low importance -> high predicted reuse distance
        return 1.0 / (ema_imp + 1e-8)

    def confidence(self, block_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Confidence in predictions based on history stability.

        Returns values in [0, 1] where 1 = highly stable reuse pattern.
        """
        if block_ids is None:
            block_ids = list(range(self.num_blocks))

        conf = torch.empty(len(block_ids))
        for i, bid in enumerate(block_ids):
            intervals = self._reuse_intervals[bid]
            if len(intervals) < 2:
                conf[i] = 0.0
                continue
            arr = torch.tensor(list(intervals), dtype=torch.float32)
            cv = arr.std() / (arr.mean() + 1e-8)  # coefficient of variation
            conf[i] = max(0.0, 1.0 - cv.item())
        return conf


# ---------------------------------------------------------------------------
# CriticalPathModel
# ---------------------------------------------------------------------------


class CriticalPathModel:
    """Models promotion latency vs. compute overlap for placement decisions.

    Core condition: offload block iff
        predicted_reuse_distance > L_promote(tier) / step_time + slack_steps
    """

    def __init__(
        self,
        promotion_latency_gpu_dram_us: float = 50.0,
        promotion_latency_ssd_dram_us: float = 500.0,
        promotion_latency_ssd_gpu_us: float = 600.0,
        decode_step_time_us: float = 1000.0,
        slack_steps: float = 2.0,
    ):
        self.latencies = {
            (TIER_DRAM, TIER_GPU): promotion_latency_gpu_dram_us,
            (TIER_SSD, TIER_DRAM): promotion_latency_ssd_dram_us,
            (TIER_SSD, TIER_GPU): promotion_latency_ssd_gpu_us,
            (TIER_GPU, TIER_DRAM): promotion_latency_gpu_dram_us * 0.8,
            (TIER_GPU, TIER_SSD): promotion_latency_ssd_gpu_us,
            (TIER_DRAM, TIER_SSD): promotion_latency_ssd_dram_us * 0.6,
        }
        self.decode_step_time_us = decode_step_time_us
        self.slack_steps = slack_steps

    def promotion_steps(self, source_tier: int, target_tier: int) -> float:
        """Steps of decode compute needed to hide a promotion from source to target."""
        latency = self.latencies.get(
            (source_tier, target_tier), self.decode_step_time_us
        )
        return latency / self.decode_step_time_us

    def overlap_slack(self, source_tier: int, target_tier: int) -> float:
        """Total slack in steps: promotion_steps + scheduling slack."""
        return self.promotion_steps(source_tier, target_tier) + self.slack_steps

    def offload_decision(
        self,
        predicted_reuse_dist: float,
        current_tier: int,
        target_tier: int,
    ) -> bool:
        """Return True if the block can safely be offloaded to target_tier.

        Offload iff predicted_R > L_promote(target->current) / step_time + slack
        """
        # Promotion path is from target back to current (for recall)
        recall_slack = self.overlap_slack(target_tier, current_tier)
        return predicted_reuse_dist > recall_slack

    def make_decision(
        self,
        block_id: int,
        predicted_reuse_dist: float,
        confidence: float,
        current_tier: int = TIER_GPU,
    ) -> BlockResidencyDecision:
        """Full placement decision for a block."""
        # Try demoting to DRAM first, then SSD
        if current_tier == TIER_GPU:
            dram_slack = self.overlap_slack(TIER_DRAM, TIER_GPU)
            ssd_slack = self.overlap_slack(TIER_SSD, TIER_GPU)

            if predicted_reuse_dist > ssd_slack and confidence > 0.5:
                decision = Decision.DEMOTE_SSD
                latency = self.latencies[(TIER_SSD, TIER_GPU)]
                slack = ssd_slack
            elif predicted_reuse_dist > dram_slack and confidence > 0.2:
                decision = Decision.DEMOTE_DRAM
                latency = self.latencies[(TIER_DRAM, TIER_GPU)]
                slack = dram_slack
            else:
                decision = Decision.KEEP_GPU
                latency = 0.0
                slack = 0.0
        else:
            # Block not on GPU — check if it needs promotion
            gpu_slack = self.overlap_slack(current_tier, TIER_GPU)
            if predicted_reuse_dist <= gpu_slack:
                decision = Decision.PROMOTE_GPU
                latency = self.latencies.get(
                    (current_tier, TIER_GPU), self.decode_step_time_us
                )
                slack = gpu_slack
            else:
                decision = Decision.KEEP_GPU  # keep where it is
                latency = 0.0
                slack = 0.0

        # Critical-path probability: chance the block will be needed before
        # the promotion can complete
        if predicted_reuse_dist > 0 and slack > 0:
            crit_prob = max(0.0, min(1.0, slack / predicted_reuse_dist))
        else:
            crit_prob = 1.0

        return BlockResidencyDecision(
            block_id=block_id,
            predicted_reuse_distance=predicted_reuse_dist,
            promotion_latency_us=latency,
            overlap_slack_steps=slack,
            decision=decision,
            confidence=confidence,
            critical_path_probability=crit_prob,
        )


# ---------------------------------------------------------------------------
# WorkloadCharacterizer
# ---------------------------------------------------------------------------


class WorkloadCharacterizer:
    """Computes workload statistics for paper figures and analysis.

    Tracks attention patterns across decode steps to characterize:
    - Attention concentration (Gini coefficient)
    - Hot-set stability (Jaccard similarity)
    - Reuse-distance distribution (CDF)
    - Per-layer/per-head breakdown
    """

    def __init__(self, num_blocks: int, num_layers: int = 1, num_heads: int = 1):
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads

        self._gini_history: List[float] = []
        self._jaccard_history: List[float] = []
        self._reuse_distances: List[float] = []
        self._prev_hot_set: Optional[set] = None
        self._per_layer_importance: List[List[float]] = [
            [] for _ in range(num_layers)
        ]
        self._per_head_importance: List[List[float]] = [
            [] for _ in range(num_heads)
        ]

    @staticmethod
    def gini_coefficient(values: torch.Tensor) -> float:
        """Compute Gini coefficient measuring attention concentration."""
        if values.numel() == 0:
            return 0.0
        sorted_vals = values.flatten().sort().values
        n = sorted_vals.shape[0]
        if n == 0:
            return 0.0
        cumsum = sorted_vals.cumsum(0)
        total = sorted_vals.sum()
        if total < 1e-12:
            return 0.0
        gini = 1.0 - 2.0 * cumsum.sum().item() / (n * total.item()) + 1.0 / n
        return max(0.0, min(1.0, gini))

    def observe_step(
        self,
        block_importance: torch.Tensor,
        hot_fraction: float = 0.2,
        layer_importance: Optional[torch.Tensor] = None,
        head_importance: Optional[torch.Tensor] = None,
    ) -> None:
        """Record statistics for one decode step."""
        # Gini
        gini = self.gini_coefficient(block_importance)
        self._gini_history.append(gini)

        # Hot-set Jaccard stability
        k = max(1, int(hot_fraction * block_importance.shape[0]))
        hot_ids = set(block_importance.topk(k).indices.tolist())
        if self._prev_hot_set is not None:
            intersection = len(hot_ids & self._prev_hot_set)
            union = len(hot_ids | self._prev_hot_set)
            jaccard = intersection / union if union > 0 else 1.0
            self._jaccard_history.append(jaccard)
        self._prev_hot_set = hot_ids

        # Per-layer/head stats
        if layer_importance is not None:
            for layer_idx in range(min(self.num_layers, layer_importance.shape[0])):
                self._per_layer_importance[layer_idx].append(
                    layer_importance[layer_idx].mean().item()
                )
        if head_importance is not None:
            for head_idx in range(min(self.num_heads, head_importance.shape[0])):
                self._per_head_importance[head_idx].append(
                    head_importance[head_idx].mean().item()
                )

    def observe_reuse_distances(self, distances: List[float]) -> None:
        """Record observed reuse distances for CDF computation."""
        self._reuse_distances.extend(distances)

    def reuse_distance_cdf(
        self, num_points: int = 100
    ) -> Tuple[List[float], List[float]]:
        """Compute empirical CDF of reuse distances.

        Returns (x_values, cdf_values) for plotting.
        """
        if not self._reuse_distances:
            return [], []
        sorted_rd = sorted(self._reuse_distances)
        n = len(sorted_rd)
        step = max(1, n // num_points)
        x_vals = [sorted_rd[i] for i in range(0, n, step)]
        cdf_vals = [(i + 1) / n for i in range(0, n, step)]
        return x_vals, cdf_vals

    def summary(self) -> Dict:
        """Return characterization summary dict."""
        gini_t = torch.tensor(self._gini_history) if self._gini_history else torch.zeros(1)
        jaccard_t = (
            torch.tensor(self._jaccard_history)
            if self._jaccard_history
            else torch.zeros(1)
        )
        rd_t = (
            torch.tensor(self._reuse_distances, dtype=torch.float32)
            if self._reuse_distances
            else torch.zeros(1)
        )

        return {
            "attention_concentration": {
                "gini_mean": gini_t.mean().item(),
                "gini_std": gini_t.std().item(),
                "gini_p50": gini_t.median().item(),
            },
            "hot_set_stability": {
                "jaccard_mean": jaccard_t.mean().item(),
                "jaccard_std": jaccard_t.std().item(),
            },
            "reuse_distance": {
                "mean": rd_t.mean().item(),
                "std": rd_t.std().item(),
                "p50": rd_t.median().item(),
                "p90": rd_t.quantile(0.9).item() if rd_t.numel() > 1 else 0.0,
                "p99": rd_t.quantile(0.99).item() if rd_t.numel() > 1 else 0.0,
            },
            "num_steps_observed": len(self._gini_history),
            "per_layer_mean_importance": [
                (sum(h) / len(h)) if h else 0.0
                for h in self._per_layer_importance
            ],
            "per_head_mean_importance": [
                (sum(h) / len(h)) if h else 0.0
                for h in self._per_head_importance
            ],
        }

    def export_json(self, path: str) -> None:
        """Export full characterization as JSON for paper figures."""
        rd_x, rd_cdf = self.reuse_distance_cdf()
        data = {
            "summary": self.summary(),
            "gini_trace": self._gini_history,
            "jaccard_trace": self._jaccard_history,
            "reuse_distance_cdf": {"x": rd_x, "y": rd_cdf},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# End-to-end integration helper
# ---------------------------------------------------------------------------


def run_predictor_step(
    predictor: ReuseDistancePredictor,
    critical_path: CriticalPathModel,
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None,
    block_size: int = 64,
    current_tiers: Optional[Dict[int, int]] = None,
) -> List[BlockResidencyDecision]:
    """Execute one full prediction + decision cycle.

    Returns a list of BlockResidencyDecision for each block.
    """
    importance = predictor.signal.compute_block_importance(
        query, key_blocks, attention_weights, block_size
    )
    predictor.observe(importance)

    predictions = predictor.predict()
    confidences = predictor.confidence()

    decisions = []
    for bid in range(predictor.num_blocks):
        tier = (current_tiers or {}).get(bid, TIER_GPU)
        dec = critical_path.make_decision(
            block_id=bid,
            predicted_reuse_dist=predictions[bid].item(),
            confidence=confidences[bid].item(),
            current_tier=tier,
        )
        decisions.append(dec)

    return decisions


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(42)

    NUM_BLOCKS = 32
    NUM_STEPS = 500
    BLOCK_SIZE = 64
    SEQ_LEN = NUM_BLOCKS * BLOCK_SIZE
    BATCH, HEADS, D_HEAD = 1, 8, 128

    print("=" * 70)
    print("Reuse-Distance Predictor Demo")
    print("=" * 70)

    # Initialize components
    predictor = ReuseDistancePredictor(
        num_blocks=NUM_BLOCKS,
        criticality_threshold=0.02,
        ema_alpha=0.3,
        signal_mode=SignalMode.QK_PROXY,
        sampling_interval=8,
    )
    critical_path = CriticalPathModel(
        promotion_latency_gpu_dram_us=80.0,
        promotion_latency_ssd_dram_us=400.0,
        promotion_latency_ssd_gpu_us=500.0,
        decode_step_time_us=800.0,
        slack_steps=2.0,
    )
    characterizer = WorkloadCharacterizer(num_blocks=NUM_BLOCKS)

    # Simulate decode steps with synthetic attention pattern:
    # blocks 0-3 are "sink" (always hot), blocks 4-7 are periodic (hot every
    # ~20 steps), remaining blocks are mostly cold with occasional bursts.
    key_cache = torch.randn(BATCH, HEADS, SEQ_LEN, D_HEAD)

    print(f"\nSimulating {NUM_STEPS} decode steps with {NUM_BLOCKS} blocks...")
    print(f"  Signal mode: {predictor.signal.mode.name}")
    print(f"  Block size: {BLOCK_SIZE} tokens")
    print(f"  Criticality threshold: {predictor.criticality_threshold}")
    print()

    decision_counts = {d: 0 for d in Decision}

    for step in range(NUM_STEPS):
        query = torch.randn(BATCH, HEADS, 1, D_HEAD)

        # Synthetic importance with three distinct access patterns:
        #   Blocks 0-3: "attention sinks" — always critically accessed
        #   Blocks 4-7: periodic — critical every 20 steps exactly
        #   Blocks 8-31: cold — only accessed on rare bursts
        importance = torch.zeros(NUM_BLOCKS)
        importance[0:4] = torch.rand(4) * 0.5 + 0.5
        if step % 20 == 0:
            importance[4:8] = torch.rand(4) * 0.3 + 0.1
        if step % 80 == 0:
            burst_ids = torch.randint(8, NUM_BLOCKS, (3,))
            importance[burst_ids] = torch.rand(3) * 0.2 + 0.05
        importance += torch.rand(NUM_BLOCKS) * 0.001

        predictor.observe(importance)
        characterizer.observe_step(importance)

        if step >= 100 and step % 10 == 0:
            predictions = predictor.predict()
            confidences = predictor.confidence()
            for bid in range(NUM_BLOCKS):
                dec = critical_path.make_decision(
                    block_id=bid,
                    predicted_reuse_dist=predictions[bid].item(),
                    confidence=confidences[bid].item(),
                )
                decision_counts[dec.decision] += 1

    # Collect reuse distances from predictor internals
    all_rd = []
    for bid in range(NUM_BLOCKS):
        all_rd.extend(predictor._reuse_intervals[bid])
    characterizer.observe_reuse_distances(all_rd)

    # Print results
    print("Decision distribution (sampled every 10 steps):")
    total_decisions = sum(decision_counts.values())
    for dec, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total_decisions if total_decisions > 0 else 0
        print(f"  {dec.name:<15} {count:>5} ({pct:5.1f}%)")

    print("\nPredicted reuse distances at final step:")
    final_pred = predictor.predict()
    final_conf = predictor.confidence()
    for bid in [0, 1, 4, 5, 10, 20, 31]:
        rd = final_pred[bid].item()
        c = final_conf[bid].item()
        rd_str = f"{rd:.1f}" if not math.isinf(rd) else "inf"
        print(f"  Block {bid:>2}: predicted_R={rd_str:>8}, confidence={c:.3f}")

    print("\nWorkload characterization:")
    summary = characterizer.summary()
    print(f"  Gini (attention concentration): {summary['attention_concentration']['gini_mean']:.3f}")
    print(f"  Jaccard (hot-set stability):    {summary['hot_set_stability']['jaccard_mean']:.3f}")
    if all_rd:
        print(f"  Reuse distance mean:            {summary['reuse_distance']['mean']:.1f} steps")
        print(f"  Reuse distance p90:             {summary['reuse_distance']['p90']:.1f} steps")

    print("\n" + "=" * 70)
    print("Demo complete. Module is importable as:")
    print("  from orchkv.reuse_distance import ReuseDistancePredictor, CriticalPathModel")
    print("=" * 70)

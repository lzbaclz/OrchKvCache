"""
InfiniGen-style cross-layer KV importance predictor.

Implements the core idea from InfiniGen (OSDI'24): attention patterns between
adjacent transformer layers exhibit high correlation (Jaccard ~0.5-0.7 for
top-K pages). This enables low-overhead cross-layer prefetch/eviction
decisions without extracting attention at every layer.

Usage:
    from orchkv.infinigen_predictor import InfiniGenPredictor
    predictor = InfiniGenPredictor(n_layers=28, n_kv_heads=4, block_size=16)
    # After observing layer L's attention:
    predictor.observe_layer(layer_idx=L, attn_weights=attn_L)
    # Predict hot blocks for layer L+1:
    predicted_hot = predictor.predict_next_layer(layer_idx=L)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerPredictionRecord:
    """Stores prediction vs actual for one layer at one step."""
    step: int
    layer: int
    predicted_hot: Set[int]
    actual_hot: Set[int]
    precision: float
    recall: float
    jaccard: float


@dataclass
class CrossLayerStats:
    """Accumulated cross-layer prediction statistics."""
    precision_at_k: List[float] = field(default_factory=list)
    recall_at_k: List[float] = field(default_factory=list)
    jaccard: List[float] = field(default_factory=list)
    layer_pair_jaccard: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InfiniGenPredictor
# ---------------------------------------------------------------------------


class InfiniGenPredictor:
    """InfiniGen-style cross-layer KV importance predictor.

    Predicts which KV blocks will be critical in the next layer based on
    the attention pattern observed in the current/previous layer.

    Key insight from InfiniGen (OSDI'24): attention patterns between adjacent
    layers have high correlation (Jaccard ~0.5-0.7 for top-K pages), enabling
    low-overhead cross-layer prefetch/eviction decisions.

    Args:
        n_layers: number of transformer layers
        n_kv_heads: number of KV attention heads
        block_size: tokens per KV block
        top_k_blocks: number of blocks to predict as "hot"
        history_window: steps of history to retain per layer
        ema_decay: decay factor for EMA smoothing of block scores
        rehearsal_heads: number of heads to use in rehearsal probe (0=all)
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        block_size: int = 16,
        top_k_blocks: int = 8,
        history_window: int = 32,
        ema_decay: float = 0.7,
        rehearsal_heads: int = 0,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.history_window = history_window
        self.ema_decay = ema_decay
        self.rehearsal_heads = rehearsal_heads if rehearsal_heads > 0 else n_kv_heads

        self._step = 0
        # Per-layer block importance scores (EMA-smoothed)
        self._layer_scores: List[Optional[torch.Tensor]] = [None] * n_layers
        # Per-layer hot sets from the most recent observation
        self._layer_hot_sets: List[Optional[Set[int]]] = [None] * n_layers
        # History of hot sets for stability analysis
        self._hot_set_history: List[deque] = [
            deque(maxlen=history_window) for _ in range(n_layers)
        ]
        # Cross-layer correlation tracking
        self._cross_layer_jaccard: Dict[Tuple[int, int], deque] = {}
        for l in range(n_layers - 1):
            self._cross_layer_jaccard[(l, l + 1)] = deque(maxlen=history_window)

        # Prediction log for accuracy computation
        self._prediction_log: List[LayerPredictionRecord] = []
        self._stats = CrossLayerStats()

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    # Observation: record attention patterns per layer
    # ------------------------------------------------------------------

    def observe_layer(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Record the attention pattern for a layer and extract block scores.

        Args:
            layer_idx: which transformer layer this attention is from
            attn_weights: [batch, n_heads, q_len, kv_len] attention weights
            seq_len: actual sequence length (if kv_len includes padding)

        Returns:
            Per-block importance scores [n_blocks].
        """
        with torch.no_grad():
            B, H, Q, S = attn_weights.shape
            if seq_len is not None:
                S = seq_len
                attn_weights = attn_weights[:, :, :, :S]

            n_blocks = math.ceil(S / self.block_size)
            block_scores = self._compute_block_scores(attn_weights, n_blocks, S)

            # EMA update
            if self._layer_scores[layer_idx] is None:
                self._layer_scores[layer_idx] = block_scores
            else:
                prev = self._layer_scores[layer_idx]
                if prev.shape[0] < n_blocks:
                    prev = F.pad(prev, (0, n_blocks - prev.shape[0]))
                elif prev.shape[0] > n_blocks:
                    prev = prev[:n_blocks]
                self._layer_scores[layer_idx] = (
                    self.ema_decay * prev + (1 - self.ema_decay) * block_scores
                )

            # Extract hot set
            k = min(self.top_k_blocks, n_blocks)
            hot_indices = self._layer_scores[layer_idx].topk(k).indices
            hot_set = set(hot_indices.tolist())
            self._layer_hot_sets[layer_idx] = hot_set
            self._hot_set_history[layer_idx].append(hot_set)

            # Update cross-layer Jaccard if previous layer was already observed
            if layer_idx > 0 and self._layer_hot_sets[layer_idx - 1] is not None:
                prev_hot = self._layer_hot_sets[layer_idx - 1]
                jacc = self._jaccard(prev_hot, hot_set)
                key = (layer_idx - 1, layer_idx)
                self._cross_layer_jaccard[key].append(jacc)

            return self._layer_scores[layer_idx]

    def _compute_block_scores(
        self, attn_weights: torch.Tensor, n_blocks: int, seq_len: int
    ) -> torch.Tensor:
        """Aggregate token-level attention into per-block scores.

        Uses mean-of-max strategy: for each head, take the max attention
        within each block, then average across heads.
        """
        B, H, Q, S = attn_weights.shape
        # Mean over batch and query dims -> [H, S]
        avg_attn = attn_weights.float().mean(dim=(0, 2))

        padded_len = n_blocks * self.block_size
        if S < padded_len:
            avg_attn = F.pad(avg_attn, (0, padded_len - S), value=0.0)

        # Reshape to [H, n_blocks, block_size]
        avg_attn = avg_attn[:, :padded_len].view(H, n_blocks, self.block_size)
        # Per-block score: sum within block, mean across heads
        block_scores = avg_attn.sum(dim=-1).mean(dim=0)  # [n_blocks]
        return block_scores.cpu()

    # ------------------------------------------------------------------
    # Prediction: cross-layer forecast
    # ------------------------------------------------------------------

    def predict_next_layer(
        self, layer_idx: int, k: Optional[int] = None
    ) -> Set[int]:
        """Predict hot blocks for layer (layer_idx + 1) based on layer_idx.

        This is the core InfiniGen insight: the hot set at layer L is a
        strong predictor of the hot set at layer L+1.

        Args:
            layer_idx: the layer whose pattern we use as the predictor
            k: number of blocks to predict (default: self.top_k_blocks)

        Returns:
            Set of predicted hot block indices for the next layer.
        """
        k = k or self.top_k_blocks
        if self._layer_hot_sets[layer_idx] is None:
            return set()

        hot_set = self._layer_hot_sets[layer_idx]
        # The prediction is simply: next layer's hot set ≈ this layer's hot set
        # (the cross-layer correlation property from InfiniGen)
        predicted = set(sorted(hot_set)[:k])
        return predicted

    def predict_with_scores(
        self, layer_idx: int, k: Optional[int] = None
    ) -> Tuple[Set[int], torch.Tensor]:
        """Predict hot blocks with associated confidence scores.

        Returns both the predicted hot set and per-block predicted scores
        for the next layer.
        """
        k = k or self.top_k_blocks
        scores = self._layer_scores[layer_idx]
        if scores is None:
            return set(), torch.zeros(0)

        top_k_actual = min(k, scores.shape[0])
        top_indices = scores.topk(top_k_actual).indices
        predicted = set(top_indices.tolist())
        return predicted, scores.clone()

    # ------------------------------------------------------------------
    # Rehearsal mode: lightweight Q·K probe
    # ------------------------------------------------------------------

    def rehearsal_probe(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        k: Optional[int] = None,
    ) -> Set[int]:
        """Lightweight Q·K probe using a subset of heads.

        Instead of running full attention, compute Q·K^T on a subset of
        heads to estimate which blocks are important. This is the "rehearsal"
        mechanism that reduces cross-layer prediction cost.

        Args:
            layer_idx: target layer for prediction
            query: [batch, n_heads, 1, head_dim] current query
            key_cache: [batch, n_kv_heads, seq_len, head_dim] cached keys
            k: number of hot blocks to return

        Returns:
            Set of predicted hot block indices.
        """
        k = k or self.top_k_blocks

        with torch.no_grad():
            B, Hq, _, D = query.shape
            _, Hk, S, _ = key_cache.shape

            # GQA: map query heads to KV heads
            gqa_ratio = Hq // Hk
            if gqa_ratio > 1:
                q = query[:, ::gqa_ratio, :, :]  # [B, Hk, 1, D]
            else:
                q = query

            # Use only a subset of heads for the probe
            n_probe_heads = min(self.rehearsal_heads, Hk)
            q_probe = q[:, :n_probe_heads, :, :]  # [B, n_probe, 1, D]
            k_probe = key_cache[:, :n_probe_heads, :, :]  # [B, n_probe, S, D]

            # Compute QK^T scores
            # [B, n_probe, 1, S]
            qk = torch.matmul(q_probe, k_probe.transpose(-1, -2))
            qk = qk / math.sqrt(D)
            # Take absolute max across batch and heads -> [S]
            qk_scores = qk.abs().mean(dim=(0, 1)).squeeze(0)  # [S]

            # Aggregate to block level
            n_blocks = math.ceil(S / self.block_size)
            padded_len = n_blocks * self.block_size
            if S < padded_len:
                qk_scores = F.pad(qk_scores, (0, padded_len - S), value=0.0)

            block_scores = qk_scores[:padded_len].view(n_blocks, self.block_size)
            block_scores = block_scores.sum(dim=-1)  # [n_blocks]

            top_k_actual = min(k, n_blocks)
            hot_indices = block_scores.topk(top_k_actual).indices
            return set(hot_indices.cpu().tolist())

    # ------------------------------------------------------------------
    # Evaluation: compare predictions against ground truth
    # ------------------------------------------------------------------

    def evaluate_prediction(
        self,
        layer_idx: int,
        predicted_hot: Set[int],
        actual_attn: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> LayerPredictionRecord:
        """Compare predicted hot set against ground truth attention.

        Args:
            layer_idx: the layer being predicted
            predicted_hot: set of block indices predicted to be hot
            actual_attn: [B, H, Q, S] actual attention weights for this layer
            seq_len: actual sequence length

        Returns:
            LayerPredictionRecord with precision, recall, jaccard.
        """
        with torch.no_grad():
            B, H, Q, S = actual_attn.shape
            if seq_len is not None:
                S = seq_len
                actual_attn = actual_attn[:, :, :, :S]

            n_blocks = math.ceil(S / self.block_size)
            actual_scores = self._compute_block_scores(actual_attn, n_blocks, S)

            k = min(self.top_k_blocks, n_blocks)
            actual_hot_indices = actual_scores.topk(k).indices
            actual_hot = set(actual_hot_indices.tolist())

        precision = self._precision(predicted_hot, actual_hot)
        recall = self._recall(predicted_hot, actual_hot)
        jaccard = self._jaccard(predicted_hot, actual_hot)

        record = LayerPredictionRecord(
            step=self._step,
            layer=layer_idx,
            predicted_hot=predicted_hot,
            actual_hot=actual_hot,
            precision=precision,
            recall=recall,
            jaccard=jaccard,
        )
        self._prediction_log.append(record)
        self._stats.precision_at_k.append(precision)
        self._stats.recall_at_k.append(recall)
        self._stats.jaccard.append(jaccard)

        pair_key = (layer_idx - 1, layer_idx) if layer_idx > 0 else (0, 0)
        if pair_key not in self._stats.layer_pair_jaccard:
            self._stats.layer_pair_jaccard[pair_key] = []
        self._stats.layer_pair_jaccard[pair_key].append(jaccard)

        return record

    def step_done(self) -> None:
        """Mark end of a decode step."""
        self._step += 1

    # ------------------------------------------------------------------
    # Accuracy metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _precision(predicted: Set[int], actual: Set[int]) -> float:
        if not predicted:
            return 0.0
        return len(predicted & actual) / len(predicted)

    @staticmethod
    def _recall(predicted: Set[int], actual: Set[int]) -> float:
        if not actual:
            return 1.0
        return len(predicted & actual) / len(actual)

    @staticmethod
    def _jaccard(set_a: Set[int], set_b: Set[int]) -> float:
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 1.0
        return len(set_a & set_b) / len(union)

    def get_accuracy_summary(self) -> Dict:
        """Return aggregated prediction accuracy metrics."""
        def _safe_stats(vals: List[float]) -> Dict:
            if not vals:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            t = torch.tensor(vals)
            return {
                "mean": t.mean().item(),
                "std": t.std().item() if len(vals) > 1 else 0.0,
                "min": t.min().item(),
                "max": t.max().item(),
                "count": len(vals),
            }

        per_pair = {}
        for (l1, l2), vals in self._stats.layer_pair_jaccard.items():
            per_pair[f"L{l1}->L{l2}"] = _safe_stats(vals)

        return {
            "precision_at_k": _safe_stats(self._stats.precision_at_k),
            "recall_at_k": _safe_stats(self._stats.recall_at_k),
            "jaccard": _safe_stats(self._stats.jaccard),
            "cross_layer_jaccard": self._get_cross_layer_summary(),
            "per_pair_jaccard": per_pair,
            "total_predictions": len(self._prediction_log),
            "top_k": self.top_k_blocks,
        }

    def _get_cross_layer_summary(self) -> Dict:
        """Summarize cross-layer Jaccard correlations."""
        all_jaccard = []
        for pair_deque in self._cross_layer_jaccard.values():
            all_jaccard.extend(pair_deque)
        if not all_jaccard:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        t = torch.tensor(all_jaccard)
        return {
            "mean": t.mean().item(),
            "std": t.std().item() if len(all_jaccard) > 1 else 0.0,
            "count": len(all_jaccard),
        }

    # ------------------------------------------------------------------
    # Trace-driven simulation
    # ------------------------------------------------------------------

    def simulate_trace(
        self,
        attention_trace: List[torch.Tensor],
        n_layers: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict:
        """Run trace-driven simulation over a sequence of attention matrices.

        Args:
            attention_trace: list of [n_layers, n_heads, q_len, kv_len] tensors,
                one per decode step. Or list of per-layer attention tensors.
            n_layers: override number of layers (auto-detected if None)
            k: top-K for hot set (default: self.top_k_blocks)

        Returns:
            Dict with per-step and aggregate accuracy metrics.
        """
        k = k or self.top_k_blocks
        per_step_results = []

        for step_idx, step_attn in enumerate(attention_trace):
            step_results = self._simulate_one_step(step_attn, step_idx, k)
            per_step_results.append(step_results)
            self.step_done()

        return {
            "per_step": per_step_results,
            "aggregate": self.get_accuracy_summary(),
            "n_steps": len(attention_trace),
        }

    def _simulate_one_step(
        self,
        step_attn: torch.Tensor,
        step_idx: int,
        k: int,
    ) -> Dict:
        """Simulate one decode step with cross-layer prediction.

        step_attn is either:
          - A list/tuple of per-layer attention tensors [B, H, Q, S]
          - A single tensor [n_layers, H, Q, S] (batch=1 assumed)
        """
        if isinstance(step_attn, (list, tuple)):
            layer_attns = step_attn
        elif step_attn.dim() == 4:
            layer_attns = [step_attn[i:i+1] for i in range(step_attn.shape[0])]
        else:
            layer_attns = [step_attn.unsqueeze(0)]

        n_layers = min(len(layer_attns), self.n_layers)
        step_records = []

        for l_idx in range(n_layers):
            attn_l = layer_attns[l_idx]
            if attn_l.dim() == 3:
                attn_l = attn_l.unsqueeze(0)

            # Observe this layer
            self.observe_layer(l_idx, attn_l)

            # If not the first layer, evaluate prediction from L-1 -> L
            if l_idx > 0 and self._layer_hot_sets[l_idx - 1] is not None:
                predicted = self.predict_next_layer(l_idx - 1, k=k)
                record = self.evaluate_prediction(l_idx, predicted, attn_l)
                step_records.append({
                    "layer": l_idx,
                    "precision": record.precision,
                    "recall": record.recall,
                    "jaccard": record.jaccard,
                })

        return {
            "step": step_idx,
            "layer_results": step_records,
            "mean_precision": (
                sum(r["precision"] for r in step_records) / len(step_records)
                if step_records else 0.0
            ),
            "mean_recall": (
                sum(r["recall"] for r in step_records) / len(step_records)
                if step_records else 0.0
            ),
            "mean_jaccard": (
                sum(r["jaccard"] for r in step_records) / len(step_records)
                if step_records else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Integration with KVCacheManager
    # ------------------------------------------------------------------

    def get_predicted_hot_blocks(
        self, layer_idx: int, k: Optional[int] = None
    ) -> List[int]:
        """Get predicted hot blocks for integration with KVCacheManager.

        Returns sorted list of block indices predicted to be hot for the
        given layer, suitable for prefetch/promotion decisions.
        """
        k = k or self.top_k_blocks
        if layer_idx == 0:
            # First layer: use its own scores if available
            if self._layer_scores[0] is not None:
                top_k_actual = min(k, self._layer_scores[0].shape[0])
                return self._layer_scores[0].topk(top_k_actual).indices.tolist()
            return []

        # Use previous layer's pattern to predict this layer
        predicted = self.predict_next_layer(layer_idx - 1, k=k)
        return sorted(predicted)

    def get_eviction_candidates(
        self, layer_idx: int, n_candidates: int = 8
    ) -> List[int]:
        """Get coldest blocks suitable for eviction.

        Returns block indices sorted by ascending predicted importance
        (coldest first), excluding the hot set.
        """
        scores = self._layer_scores[layer_idx]
        if scores is None:
            return []

        hot_set = self._layer_hot_sets[layer_idx] or set()
        n_blocks = scores.shape[0]

        # Sort by score ascending (coldest first)
        sorted_indices = scores.argsort()
        candidates = []
        for idx in sorted_indices.tolist():
            if idx not in hot_set:
                candidates.append(idx)
                if len(candidates) >= n_candidates:
                    break
        return candidates

    def reset(self) -> None:
        """Reset predictor state for a new sequence."""
        self._step = 0
        self._layer_scores = [None] * self.n_layers
        self._layer_hot_sets = [None] * self.n_layers
        self._hot_set_history = [
            deque(maxlen=self.history_window) for _ in range(self.n_layers)
        ]
        for key in self._cross_layer_jaccard:
            self._cross_layer_jaccard[key].clear()
        self._prediction_log.clear()
        self._stats = CrossLayerStats()


# ---------------------------------------------------------------------------
# OraclePredictor (Belady-like)
# ---------------------------------------------------------------------------


class OraclePredictor:
    """Oracle (Belady-like) predictor with future knowledge.

    Uses the full future attention trace to determine the optimal hot set
    at each step — i.e., the blocks that will be accessed soonest.
    Serves as an upper bound for prediction accuracy.

    Usage in trace mode:
        oracle = OraclePredictor(n_layers=28, block_size=16)
        oracle.load_trace(attention_trace)
        hot_set = oracle.get_optimal_hot_set(step=5, layer=3, k=8)
    """

    def __init__(
        self,
        n_layers: int,
        block_size: int = 16,
        top_k_blocks: int = 8,
        criticality_threshold: float = 0.01,
    ):
        self.n_layers = n_layers
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.criticality_threshold = criticality_threshold
        self._trace: Optional[List] = None
        self._block_scores_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def load_trace(self, attention_trace: List) -> None:
        """Load a full attention trace for oracle computation.

        Args:
            attention_trace: list of per-step attention data.
                Each element is either:
                  - A list of per-layer tensors [B, H, Q, S]
                  - A tensor [n_layers, H, Q, S]
        """
        self._trace = attention_trace
        self._block_scores_cache.clear()
        self._precompute_block_scores()

    def _precompute_block_scores(self) -> None:
        """Pre-compute per-block scores for every (step, layer)."""
        if self._trace is None:
            return

        for step_idx, step_attn in enumerate(self._trace):
            if isinstance(step_attn, (list, tuple)):
                layer_attns = step_attn
            elif isinstance(step_attn, torch.Tensor) and step_attn.dim() == 4:
                layer_attns = [step_attn[i:i+1] for i in range(step_attn.shape[0])]
            else:
                continue

            for l_idx, attn_l in enumerate(layer_attns):
                if l_idx >= self.n_layers:
                    break
                if isinstance(attn_l, torch.Tensor):
                    if attn_l.dim() == 3:
                        attn_l = attn_l.unsqueeze(0)
                    B, H, Q, S = attn_l.shape
                    n_blocks = math.ceil(S / self.block_size)
                    scores = self._compute_block_scores(attn_l, n_blocks, S)
                    self._block_scores_cache[(step_idx, l_idx)] = scores

    def _compute_block_scores(
        self, attn: torch.Tensor, n_blocks: int, seq_len: int
    ) -> torch.Tensor:
        """Same aggregation as InfiniGenPredictor for consistency."""
        B, H, Q, S = attn.shape
        avg_attn = attn.float().mean(dim=(0, 2))
        padded_len = n_blocks * self.block_size
        if S < padded_len:
            avg_attn = F.pad(avg_attn, (0, padded_len - S), value=0.0)
        avg_attn = avg_attn[:, :padded_len].view(H, n_blocks, self.block_size)
        block_scores = avg_attn.sum(dim=-1).mean(dim=0)
        return block_scores.cpu()

    def get_optimal_hot_set(
        self, step: int, layer: int, k: Optional[int] = None
    ) -> Set[int]:
        """Get the oracle-optimal hot set for a given step and layer.

        Uses Belady's algorithm: the optimal set is the K blocks with the
        shortest forward reuse distance (will be needed soonest).
        """
        k = k or self.top_k_blocks
        key = (step, layer)

        if key in self._block_scores_cache:
            scores = self._block_scores_cache[key]
            top_k_actual = min(k, scores.shape[0])
            return set(scores.topk(top_k_actual).indices.tolist())

        # Fallback: compute forward reuse distance
        return self._belady_hot_set(step, layer, k)

    def _belady_hot_set(self, step: int, layer: int, k: int) -> Set[int]:
        """Belady's optimal: select blocks with nearest future critical access."""
        if self._trace is None:
            return set()

        n_steps = len(self._trace)
        # Get current block count from this step
        current_key = (step, layer)
        if current_key not in self._block_scores_cache:
            return set()

        n_blocks = self._block_scores_cache[current_key].shape[0]
        # For each block, find next step where it's in the top-K
        forward_distance = torch.full((n_blocks,), float("inf"))

        for future_step in range(step + 1, min(step + 100, n_steps)):
            future_key = (future_step, layer)
            if future_key not in self._block_scores_cache:
                continue
            future_scores = self._block_scores_cache[future_key]
            n_future = min(future_scores.shape[0], n_blocks)
            critical = future_scores[:n_future] > self.criticality_threshold
            for bid in critical.nonzero(as_tuple=True)[0].tolist():
                if math.isinf(forward_distance[bid].item()):
                    forward_distance[bid] = future_step - step

        # Select K blocks with shortest forward distance
        top_k_actual = min(k, n_blocks)
        # Negate so that smallest distance = largest value for topk
        neg_dist = -forward_distance
        # Replace -inf with large negative so they sort last
        neg_dist[neg_dist == float("-inf")] = -1e9
        hot_indices = neg_dist.topk(top_k_actual).indices
        return set(hot_indices.tolist())

    def get_accuracy_vs_predictor(
        self, predictor_hot_sets: Dict[Tuple[int, int], Set[int]]
    ) -> Dict:
        """Compare a predictor's hot sets against oracle optimal.

        Args:
            predictor_hot_sets: dict mapping (step, layer) -> predicted hot set

        Returns:
            Accuracy comparison metrics.
        """
        precisions, recalls, jaccards = [], [], []

        for (step, layer), predicted in predictor_hot_sets.items():
            optimal = self.get_optimal_hot_set(step, layer)
            if not optimal:
                continue
            p = len(predicted & optimal) / len(predicted) if predicted else 0.0
            r = len(predicted & optimal) / len(optimal) if optimal else 1.0
            union = predicted | optimal
            j = len(predicted & optimal) / len(union) if union else 1.0
            precisions.append(p)
            recalls.append(r)
            jaccards.append(j)

        def _stats(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0, "count": 0}
            t = torch.tensor(vals)
            return {
                "mean": t.mean().item(),
                "std": t.std().item() if len(vals) > 1 else 0.0,
                "count": len(vals),
            }

        return {
            "precision_at_k": _stats(precisions),
            "recall_at_k": _stats(recalls),
            "jaccard": _stats(jaccards),
        }


# ---------------------------------------------------------------------------
# EMA Predictor (OrchKvCache baseline for comparison)
# ---------------------------------------------------------------------------


class EMAPredictor:
    """EMA-based block importance predictor (OrchKvCache's native approach).

    Uses exponential moving average of per-block attention scores,
    independently at each layer. No cross-layer correlation.
    """

    def __init__(
        self,
        n_layers: int,
        block_size: int = 16,
        top_k_blocks: int = 8,
        ema_decay: float = 0.9,
    ):
        self.n_layers = n_layers
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.ema_decay = ema_decay

        self._step = 0
        self._layer_scores: List[Optional[torch.Tensor]] = [None] * n_layers
        self._prediction_log: List[LayerPredictionRecord] = []
        self._stats = CrossLayerStats()

    @property
    def step(self) -> int:
        return self._step

    def observe_layer(
        self, layer_idx: int, attn_weights: torch.Tensor, seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Record attention and update EMA scores."""
        with torch.no_grad():
            B, H, Q, S = attn_weights.shape
            if seq_len is not None:
                S = seq_len
                attn_weights = attn_weights[:, :, :, :S]

            n_blocks = math.ceil(S / self.block_size)
            avg_attn = attn_weights.float().mean(dim=(0, 2))
            padded_len = n_blocks * self.block_size
            if S < padded_len:
                avg_attn = F.pad(avg_attn, (0, padded_len - S), value=0.0)
            avg_attn = avg_attn[:, :padded_len].view(H, n_blocks, self.block_size)
            block_scores = avg_attn.sum(dim=-1).mean(dim=0).cpu()

            if self._layer_scores[layer_idx] is None:
                self._layer_scores[layer_idx] = block_scores
            else:
                prev = self._layer_scores[layer_idx]
                if prev.shape[0] < n_blocks:
                    prev = F.pad(prev, (0, n_blocks - prev.shape[0]))
                elif prev.shape[0] > n_blocks:
                    prev = prev[:n_blocks]
                self._layer_scores[layer_idx] = (
                    self.ema_decay * prev + (1 - self.ema_decay) * block_scores
                )

            return self._layer_scores[layer_idx]

    def predict_hot_blocks(self, layer_idx: int, k: Optional[int] = None) -> Set[int]:
        """Predict hot blocks based on EMA scores for this layer."""
        k = k or self.top_k_blocks
        scores = self._layer_scores[layer_idx]
        if scores is None:
            return set()
        top_k_actual = min(k, scores.shape[0])
        return set(scores.topk(top_k_actual).indices.tolist())

    def evaluate_prediction(
        self, layer_idx: int, predicted_hot: Set[int], actual_attn: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> LayerPredictionRecord:
        """Evaluate prediction accuracy against actual attention."""
        with torch.no_grad():
            B, H, Q, S = actual_attn.shape
            if seq_len is not None:
                S = seq_len
                actual_attn = actual_attn[:, :, :, :S]
            n_blocks = math.ceil(S / self.block_size)
            avg_attn = actual_attn.float().mean(dim=(0, 2))
            padded_len = n_blocks * self.block_size
            if S < padded_len:
                avg_attn = F.pad(avg_attn, (0, padded_len - S), value=0.0)
            avg_attn = avg_attn[:, :padded_len].view(H, n_blocks, self.block_size)
            actual_scores = avg_attn.sum(dim=-1).mean(dim=0).cpu()

        k = min(self.top_k_blocks, n_blocks)
        actual_hot = set(actual_scores.topk(k).indices.tolist())

        precision = len(predicted_hot & actual_hot) / len(predicted_hot) if predicted_hot else 0.0
        recall = len(predicted_hot & actual_hot) / len(actual_hot) if actual_hot else 1.0
        union = predicted_hot | actual_hot
        jaccard = len(predicted_hot & actual_hot) / len(union) if union else 1.0

        record = LayerPredictionRecord(
            step=self._step, layer=layer_idx,
            predicted_hot=predicted_hot, actual_hot=actual_hot,
            precision=precision, recall=recall, jaccard=jaccard,
        )
        self._prediction_log.append(record)
        self._stats.precision_at_k.append(precision)
        self._stats.recall_at_k.append(recall)
        self._stats.jaccard.append(jaccard)
        return record

    def step_done(self) -> None:
        self._step += 1

    def get_accuracy_summary(self) -> Dict:
        """Return aggregated accuracy metrics."""
        def _safe_stats(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0, "count": 0}
            t = torch.tensor(vals)
            return {
                "mean": t.mean().item(),
                "std": t.std().item() if len(vals) > 1 else 0.0,
                "count": len(vals),
            }
        return {
            "precision_at_k": _safe_stats(self._stats.precision_at_k),
            "recall_at_k": _safe_stats(self._stats.recall_at_k),
            "jaccard": _safe_stats(self._stats.jaccard),
            "total_predictions": len(self._prediction_log),
            "top_k": self.top_k_blocks,
        }

    def reset(self) -> None:
        self._step = 0
        self._layer_scores = [None] * self.n_layers
        self._prediction_log.clear()
        self._stats = CrossLayerStats()

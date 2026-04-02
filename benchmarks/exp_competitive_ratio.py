#!/usr/bin/env python3
"""
Direction B: Empirical Competitive Ratio Analysis

Compares five eviction strategies on synthetic attention traces:
  1. FIFO    — evict oldest block
  2. LRU     — evict least recently used
  3. LFU     — evict least frequently used
  4. EMA     — evict lowest EMA attention score (OrchKvCache policy)
  5. OPT     — Belady's offline optimal (evict farthest future use)

Computes empirical competitive ratio = strategy_cost / OPT_cost,
where cost = total number of evictions (GPU→DRAM demotions).

Traces: Zipf-distributed attention with slowly shifting hot set,
matching observed LLM attention patterns (Gini ~0.9, Jaccard ~0.6).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Trace Generation (reuse from exp_attn_sampling)
# ═══════════════════════════════════════════════════════════════════

def generate_trace(n_blocks: int, n_steps: int, hot_frac: float = 0.10,
                   shift_every: int = 5, shift_frac: float = 0.15,
                   seed: int = 42) -> list[set[int]]:
    """
    Generate per-step "requested block" sets.
    A block is "requested" if its attention weight > threshold (simulating
    that the attention kernel needs this block on GPU).
    Hot blocks are always requested; warm blocks are requested 50% of the time;
    cold blocks 5%.
    """
    rng = random.Random(seed)
    n_hot = max(1, int(n_blocks * hot_frac))
    hot_set = set(range(n_hot))
    warm_pool = set(range(n_hot, n_hot + int(n_blocks * 0.15)))

    trace: list[set[int]] = []
    attention_scores: list[dict[int, float]] = []

    for step in range(n_steps):
        if step > 0 and step % shift_every == 0:
            n_shift = max(1, int(n_hot * shift_frac))
            removable = list(hot_set)
            to_remove = set(rng.sample(removable, min(n_shift, len(removable))))
            cold_cands = [b for b in range(n_blocks) if b not in hot_set]
            to_add = set(rng.sample(cold_cands, min(n_shift, len(cold_cands))))
            warm_pool |= to_remove
            hot_set = (hot_set - to_remove) | to_add

        requested = set()
        scores = {}
        for bid in range(n_blocks):
            if bid in hot_set:
                requested.add(bid)
                scores[bid] = max(0.0, rng.gauss(0.80, 0.10))
            elif bid in warm_pool:
                if rng.random() < 0.5:
                    requested.add(bid)
                scores[bid] = max(0.0, rng.gauss(0.15, 0.05))
            else:
                if rng.random() < 0.05:
                    requested.add(bid)
                scores[bid] = max(0.0, rng.gauss(0.02, 0.01))

        trace.append(requested)
        attention_scores.append(scores)

    return trace, attention_scores


# ═══════════════════════════════════════════════════════════════════
#  Eviction Strategies
# ═══════════════════════════════════════════════════════════════════

class CacheSimulator:
    """Base class for cache eviction simulation."""

    def __init__(self, capacity: int, n_blocks: int):
        self.capacity = capacity
        self.n_blocks = n_blocks
        self.gpu_set: set[int] = set()
        self.evictions = 0
        self.promotions = 0

    def step(self, requested: set[int], scores: dict[int, float], step_idx: int,
             future_trace: list[set[int]] | None = None):
        for bid in requested:
            if bid not in self.gpu_set:
                if len(self.gpu_set) >= self.capacity:
                    victim = self._select_victim(requested, scores, step_idx,
                                                  future_trace)
                    if victim is not None:
                        self.gpu_set.discard(victim)
                        self.evictions += 1
                self.gpu_set.add(bid)
                self.promotions += 1
        self._post_step(requested, scores, step_idx)

    def _select_victim(self, requested, scores, step_idx, future_trace):
        raise NotImplementedError

    def _post_step(self, requested, scores, step_idx):
        pass


class FIFOCache(CacheSimulator):
    def __init__(self, capacity, n_blocks):
        super().__init__(capacity, n_blocks)
        self._order: list[int] = []

    def step(self, requested, scores, step_idx, future_trace=None):
        for bid in requested:
            if bid not in self.gpu_set:
                if len(self.gpu_set) >= self.capacity:
                    victim = self._select_victim(requested, scores, step_idx, None)
                    if victim is not None:
                        self.gpu_set.discard(victim)
                        self._order.remove(victim)
                        self.evictions += 1
                self.gpu_set.add(bid)
                self._order.append(bid)
                self.promotions += 1

    def _select_victim(self, requested, scores, step_idx, future_trace):
        for bid in self._order:
            if bid not in requested:
                return bid
        return self._order[0] if self._order else None


class LRUCache(CacheSimulator):
    def __init__(self, capacity, n_blocks):
        super().__init__(capacity, n_blocks)
        self._lru = OrderedDict()

    def step(self, requested, scores, step_idx, future_trace=None):
        for bid in requested:
            if bid in self._lru:
                self._lru.move_to_end(bid)
            else:
                if len(self._lru) >= self.capacity:
                    victim = self._find_victim(requested)
                    if victim is not None:
                        del self._lru[victim]
                        self.gpu_set.discard(victim)
                        self.evictions += 1
                self._lru[bid] = True
                self.gpu_set.add(bid)
                self.promotions += 1

    def _find_victim(self, requested):
        for bid in self._lru:
            if bid not in requested:
                return bid
        return next(iter(self._lru))

    def _select_victim(self, *args):
        pass


class LFUCache(CacheSimulator):
    def __init__(self, capacity, n_blocks):
        super().__init__(capacity, n_blocks)
        self._freq: dict[int, int] = defaultdict(int)

    def step(self, requested, scores, step_idx, future_trace=None):
        for bid in requested:
            self._freq[bid] += 1
            if bid not in self.gpu_set:
                if len(self.gpu_set) >= self.capacity:
                    victim = self._select_victim(requested, scores, step_idx, None)
                    if victim is not None:
                        self.gpu_set.discard(victim)
                        self.evictions += 1
                self.gpu_set.add(bid)
                self.promotions += 1

    def _select_victim(self, requested, scores, step_idx, future_trace):
        candidates = [(self._freq.get(b, 0), b) for b in self.gpu_set
                       if b not in requested]
        if not candidates:
            candidates = [(self._freq.get(b, 0), b) for b in self.gpu_set]
        candidates.sort()
        return candidates[0][1] if candidates else None


class EMACache(CacheSimulator):
    """OrchKvCache's EMA-based policy."""

    def __init__(self, capacity, n_blocks, ema_lambda=0.9,
                 alpha=0.7, beta=0.2, gamma=0.1, tau=50.0):
        super().__init__(capacity, n_blocks)
        self._lambda = ema_lambda
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._tau = tau
        self._ema: dict[int, float] = defaultdict(float)
        self._last_access: dict[int, int] = {}
        self._freq: dict[int, int] = defaultdict(int)
        self._step = 0

    def step(self, requested, scores, step_idx, future_trace=None):
        for bid in requested:
            raw = scores.get(bid, 0.0)
            old = self._ema[bid]
            self._ema[bid] = self._lambda * old + (1 - self._lambda) * raw
            self._last_access[bid] = step_idx
            self._freq[bid] += 1

        for bid in self._ema:
            if bid not in requested:
                self._ema[bid] *= self._lambda

        for bid in requested:
            if bid not in self.gpu_set:
                if len(self.gpu_set) >= self.capacity:
                    victim = self._select_victim(requested, scores, step_idx, None)
                    if victim is not None:
                        self.gpu_set.discard(victim)
                        self.evictions += 1
                self.gpu_set.add(bid)
                self.promotions += 1

        self._step = step_idx

    def _select_victim(self, requested, scores, step_idx, future_trace):
        max_ema = max(self._ema.values()) if self._ema else 1e-9
        max_ema = max(max_ema, 1e-9)
        max_freq = max(self._freq.values()) if self._freq else 1

        best_victim = None
        best_score = float("inf")

        for bid in self.gpu_set:
            if bid in requested:
                continue
            norm_ema = self._ema.get(bid, 0.0) / max_ema
            dt = step_idx - self._last_access.get(bid, 0)
            recency = math.exp(-dt / self._tau) if dt < 500 else 0.0
            freq = min(self._freq.get(bid, 0) / max(max_freq, 1), 1.0)
            hotness = self._alpha * norm_ema + self._beta * recency + self._gamma * freq
            if hotness < best_score:
                best_score = hotness
                best_victim = bid

        if best_victim is None:
            candidates = list(self.gpu_set)
            return candidates[0] if candidates else None
        return best_victim


class BeladyOPT(CacheSimulator):
    """Offline optimal: evict the block used farthest in the future."""

    def step(self, requested, scores, step_idx, future_trace=None):
        for bid in requested:
            if bid not in self.gpu_set:
                if len(self.gpu_set) >= self.capacity:
                    victim = self._select_victim(requested, scores, step_idx,
                                                  future_trace)
                    if victim is not None:
                        self.gpu_set.discard(victim)
                        self.evictions += 1
                self.gpu_set.add(bid)
                self.promotions += 1

    def _select_victim(self, requested, scores, step_idx, future_trace):
        if future_trace is None:
            return list(self.gpu_set)[0]

        farthest_bid = None
        farthest_dist = -1

        for bid in self.gpu_set:
            if bid in requested:
                continue
            next_use = float("inf")
            for future_step in range(step_idx + 1, len(future_trace)):
                if bid in future_trace[future_step]:
                    next_use = future_step
                    break
            if next_use > farthest_dist:
                farthest_dist = next_use
                farthest_bid = bid

        return farthest_bid


# ═══════════════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(n_blocks, n_steps, capacity_ratio, n_runs, seed):
    capacity = max(1, int(n_blocks * capacity_ratio))
    strategies = {
        "FIFO": lambda: FIFOCache(capacity, n_blocks),
        "LRU": lambda: LRUCache(capacity, n_blocks),
        "LFU": lambda: LFUCache(capacity, n_blocks),
        "EMA": lambda: EMACache(capacity, n_blocks),
        "OPT": lambda: BeladyOPT(capacity, n_blocks),
    }

    results = {name: {"evictions": [], "promotions": []} for name in strategies}

    for run in range(n_runs):
        run_seed = seed + run * 1000
        trace, scores = generate_trace(n_blocks, n_steps, seed=run_seed)

        for name, factory in strategies.items():
            cache = factory()

            for step_idx in range(n_steps):
                cache.step(trace[step_idx], scores[step_idx], step_idx,
                           future_trace=trace if name == "OPT" else None)

            results[name]["evictions"].append(cache.evictions)
            results[name]["promotions"].append(cache.promotions)

    return results


def main():
    ap = argparse.ArgumentParser(description="Empirical Competitive Ratio")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    configs = [
        (64, 500, 0.30),
        (128, 500, 0.30),
        (256, 500, 0.30),
        (512, 300, 0.30),
    ]

    print(f"{'='*70}")
    print(f"  Empirical Competitive Ratio Analysis")
    print(f"  Strategies: FIFO, LRU, LFU, EMA (OrchKvCache), OPT (Belady)")
    print(f"{'='*70}")

    all_results = []

    for n_blocks, n_steps, cap_ratio in configs:
        capacity = int(n_blocks * cap_ratio)
        print(f"\n--- n_blocks={n_blocks}, capacity={capacity} ({cap_ratio:.0%}), "
              f"steps={n_steps} ---")

        results = run_experiment(n_blocks, n_steps, cap_ratio, args.n_runs, args.seed)

        opt_evict = statistics.mean(results["OPT"]["evictions"])
        opt_evict = max(opt_evict, 1)

        row = {"n_blocks": n_blocks, "capacity": capacity,
               "capacity_ratio": cap_ratio, "n_steps": n_steps}

        print(f"  {'Strategy':<8s} {'Evictions':>10s} {'CR vs OPT':>10s} {'Promotions':>11s}")
        for name in ["OPT", "EMA", "LFU", "LRU", "FIFO"]:
            avg_ev = statistics.mean(results[name]["evictions"])
            avg_pr = statistics.mean(results[name]["promotions"])
            cr = avg_ev / opt_evict
            row[f"{name}_evictions"] = round(avg_ev, 1)
            row[f"{name}_CR"] = round(cr, 2)
            row[f"{name}_promotions"] = round(avg_pr, 1)
            marker = " *" if name == "EMA" else ""
            print(f"  {name:<8s} {avg_ev:>10.1f} {cr:>10.2f}x {avg_pr:>11.1f}{marker}")

        all_results.append(row)

    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPETITIVE RATIO SUMMARY")
    print(f"{'='*70}")
    print(f"  {'n_blocks':>8s} {'FIFO CR':>8s} {'LRU CR':>8s} {'LFU CR':>8s} "
          f"{'EMA CR':>8s} {'OPT':>8s}")
    for r in all_results:
        print(f"  {r['n_blocks']:>8d} {r['FIFO_CR']:>8.2f} {r['LRU_CR']:>8.2f} "
              f"{r['LFU_CR']:>8.2f} {r['EMA_CR']:>8.2f} {1.00:>8.2f}")

    out = {"results": all_results, "config": {
        "n_runs": args.n_runs, "seed": args.seed, "configs": configs,
    }}
    out_path = RESULTS_DIR / "exp_competitive_ratio.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

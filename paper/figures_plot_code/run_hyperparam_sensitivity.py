#!/usr/bin/env python3
"""
E11: Hyperparameter Sensitivity — pure Python reimplementation.

No orchkv_core dependency. Simulates EMA-based hot/cold classification
with sweeps over λ, τ, cooldown, and joint λ×τ.

Usage:
    python run_hyperparam_sensitivity.py

Outputs (in same directory):
    exp_e11_hyperparam.json
    fig_hyperparam_sensitivity.pdf / .png
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent

# ─── Configuration ───────────────────────────────────────────────
N_BLOCKS = 256
N_STEPS  = 200
N_RUNS   = 3
SEED     = 42

ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2
DEFAULT_LAMBDA  = 0.9
DEFAULT_TAU     = 50.0
DEFAULT_COOLDOWN = 0.5

LAMBDA_VALUES   = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
TAU_VALUES      = [5, 10, 25, 50, 100, 200]
COOLDOWN_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

HOT_THRESH  = 0.50
WARM_THRESH = 0.15
THETA_HOT   = 0.40
THETA_COLD  = 0.15
ADJUST_STEP = 0.02
GPU_HWM, GPU_LWM = 0.80, 0.60


# ─── Trace generator (matches exp_attn_sampling.py) ─────────────
def generate_trace(n_blocks, n_steps, seed=42,
                   hot_frac=0.10, warm_frac=0.15,
                   shift_every=5, shift_frac=0.15):
    rng = random.Random(seed)
    n_hot = max(1, int(n_blocks * hot_frac))
    n_warm_pool = max(1, int(n_blocks * warm_frac * 2))
    hot_set = set(range(n_hot))
    warm_pool = set(range(n_hot, n_hot + n_warm_pool))

    trace = []
    for step in range(n_steps):
        if step > 0 and step % shift_every == 0:
            n_shift = max(1, int(n_hot * shift_frac))
            to_remove = set(rng.sample(list(hot_set), min(n_shift, len(hot_set))))
            cold_cands = [b for b in range(n_blocks) if b not in hot_set]
            to_add = set(rng.sample(cold_cands, min(n_shift, len(cold_cands))))
            hot_set = (hot_set - to_remove) | to_add
            warm_pool |= to_remove

        step_data = []
        for bid in range(n_blocks):
            if bid in hot_set:
                w = max(0.0, rng.gauss(0.80, 0.10))
            elif bid in warm_pool:
                w = max(0.0, rng.gauss(0.15, 0.05))
            else:
                w = max(0.0, rng.gauss(0.02, 0.01))
            step_data.append((bid, w))
        trace.append(step_data)
    return trace


def ground_truth(trace, n_blocks):
    gt = []
    for step_data in trace:
        ws = {bid: w for bid, w in step_data}
        n_hot = sum(1 for b in range(n_blocks) if ws.get(b, 0) >= HOT_THRESH)
        n_warm = sum(1 for b in range(n_blocks) if WARM_THRESH <= ws.get(b, 0) < HOT_THRESH)
        n_cold = n_blocks - n_hot - n_warm
        gt.append({"n_hot": n_hot, "n_warm": n_warm, "n_cold": n_cold})
    return gt


# ─── Pure-Python EMA classifier ─────────────────────────────────
class EMAClassifier:
    def __init__(self, n_blocks, ema_lambda=0.9, recency_tau=50.0,
                 cooldown_sec=0.5):
        self.n = n_blocks
        self.lam = ema_lambda
        self.tau = recency_tau
        self.cooldown = cooldown_sec

        self.ema = [0.0] * n_blocks
        self.last_access = [0] * n_blocks
        self.freq = [0] * n_blocks
        self.step = 0

        self.theta_hot = THETA_HOT
        self.theta_cold = THETA_COLD
        self.last_adj_step = -9999

    def report(self, step_data):
        self.step += 1
        for bid, w in step_data:
            if w >= 0.05:
                self.ema[bid] = self.lam * w + (1 - self.lam) * self.ema[bid]
                self.last_access[bid] = self.step
                self.freq[bid] += 1
        for bid in range(self.n):
            accessed = any(bid == b and w >= 0.05 for b, w in step_data)
            if not accessed:
                self.ema[bid] *= (1 - self.lam) * 0.95 + self.lam

    def classify(self):
        max_ema = max(self.ema) if max(self.ema) > 1e-9 else 1.0
        max_freq = max(self.freq) if max(self.freq) > 0 else 1

        counts = {"n_hot": 0, "n_warm": 0, "n_cold": 0}
        for bid in range(self.n):
            a_hat = self.ema[bid] / max_ema
            dt = self.step - self.last_access[bid]
            R = math.exp(-dt / max(self.tau, 0.01))
            F = min(self.freq[bid] / max_freq, 1.0)
            S = ALPHA * a_hat + BETA * R + GAMMA * F

            if S >= self.theta_hot:
                counts["n_hot"] += 1
            elif S < self.theta_cold:
                counts["n_cold"] += 1
            else:
                counts["n_warm"] += 1
        return counts

    def maybe_adjust(self, gpu_util=0.85):
        steps_since = self.step - self.last_adj_step
        effective_cooldown_steps = max(1, int(self.cooldown * 20))
        if steps_since < effective_cooldown_steps:
            return
        if gpu_util > GPU_HWM:
            self.theta_hot = max(0.05, self.theta_hot - ADJUST_STEP)
        elif gpu_util < GPU_LWM:
            self.theta_hot = min(0.95, self.theta_hot + ADJUST_STEP)
        self.last_adj_step = self.step


# ─── Simulation ──────────────────────────────────────────────────
def run_sim(trace, n_blocks, ema_lambda=0.9, recency_tau=50.0, cooldown=0.5):
    cls = EMAClassifier(n_blocks, ema_lambda, recency_tau, cooldown)
    records = []
    for step, step_data in enumerate(trace):
        cls.report(step_data)
        if step % 5 == 0:
            cls.maybe_adjust(gpu_util=0.85)
        c = cls.classify()
        records.append(c)
    return records


def accuracy(records, gt, n_blocks):
    n = min(len(records), len(gt))
    total = 0.0
    for rec, ref in zip(records[:n], gt[:n]):
        diff = (abs(rec["n_hot"] - ref["n_hot"])
                + abs(rec["n_warm"] - ref["n_warm"])
                + abs(rec["n_cold"] - ref["n_cold"]))
        total += max(0.0, 1.0 - diff / (2 * n_blocks))
    return total / max(n, 1)


def oscillations(records):
    hot = [r["n_hot"] for r in records]
    osc = 0
    for i in range(2, len(hot)):
        d1 = hot[i-1] - hot[i-2]
        d2 = hot[i] - hot[i-1]
        if d1 * d2 < 0 and abs(d1) >= 2 and abs(d2) >= 2:
            osc += 1
    return osc


def migrations(records):
    total = 0
    for i in range(1, len(records)):
        total += abs(records[i]["n_hot"] - records[i-1]["n_hot"])
    return total


# ─── Sweep functions ─────────────────────────────────────────────
def sweep_one(name, values, default_params, n_blocks, n_steps, n_runs, seed):
    results = []
    for val in values:
        accs, oscs, migs = [], [], []
        for rid in range(n_runs):
            tr = generate_trace(n_blocks, n_steps, seed=seed + rid * 1000)
            gt = ground_truth(tr, n_blocks)
            params = dict(default_params)
            params[name] = val
            recs = run_sim(tr, n_blocks, **params)
            accs.append(accuracy(recs, gt, n_blocks))
            oscs.append(oscillations(recs))
            migs.append(migrations(recs))
        results.append({
            "param": name, "value": val,
            "accuracy": round(statistics.mean(accs), 4),
            "oscillations": round(statistics.mean(oscs), 1),
            "migrations": round(statistics.mean(migs), 1),
        })
        print(f"    {name}={val:<8}  acc={results[-1]['accuracy']:.3f}  "
              f"osc={results[-1]['oscillations']:.0f}  mig={results[-1]['migrations']:.0f}")
    return results


def sweep_combined(n_blocks, n_steps, n_runs, seed):
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    taus = [5, 10, 25, 50, 100]
    results = []
    for lam in lambdas:
        for tau in taus:
            accs = []
            for rid in range(n_runs):
                tr = generate_trace(n_blocks, n_steps, seed=seed + rid * 1000)
                gt = ground_truth(tr, n_blocks)
                recs = run_sim(tr, n_blocks, ema_lambda=lam,
                               recency_tau=tau, cooldown=DEFAULT_COOLDOWN)
                accs.append(accuracy(recs, gt, n_blocks))
            results.append({
                "ema_lambda": lam, "recency_tau": tau,
                "accuracy": round(statistics.mean(accs), 4),
            })
            print(f"    λ={lam:.2f} τ={tau:>4}  acc={statistics.mean(accs):.3f}")
    return results


# ─── Plot ────────────────────────────────────────────────────────
def plot_all(lam_res, cooldown_res, tau_res, combined_res):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    BAR_KW = dict(width=0.6, edgecolor="white", linewidth=0.5)

    # (a) λ sweep
    ax = axes[0, 0]
    vals = [r["value"] for r in lam_res]
    accs = [r["accuracy"] for r in lam_res]
    ax.bar(range(len(vals)), accs, color="#9CC0D8", edgecolor="#7AA8C4", **BAR_KW)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f"{v}" for v in vals])
    ax.set_xlabel(r"EMA decay $\lambda$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(r"(a) $\lambda$ sweep")
    ax.set_ylim(min(accs) * 0.92, min(max(accs) * 1.05, 1.0))
    ax.axhline(y=statistics.mean(accs), color="#C07868", ls="--", lw=0.8,
               label=f"mean={statistics.mean(accs):.3f}")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.2)

    # (b) cooldown sweep
    ax = axes[0, 1]
    vals = [r["value"] for r in cooldown_res]
    oscs = [r["oscillations"] for r in cooldown_res]
    ax.bar(range(len(vals)), oscs, color="#E8B4A8", edgecolor="#C89888", **BAR_KW)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f"{v}" for v in vals])
    ax.set_xlabel("Cooldown (s)")
    ax.set_ylabel("Oscillation count")
    ax.set_title("(b) Threshold cooldown")
    ax.grid(axis="y", alpha=0.2)

    # (c) τ sweep
    ax = axes[1, 0]
    vals = [r["value"] for r in tau_res]
    accs = [r["accuracy"] for r in tau_res]
    ax.bar(range(len(vals)), accs, color="#A8D5BA", edgecolor="#88BDA0", **BAR_KW)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([str(v) for v in vals])
    ax.set_xlabel(r"Recency $\tau$ (steps)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(r"(c) $\tau$ sweep")
    ax.set_ylim(min(accs) * 0.92, min(max(accs) * 1.05, 1.0))
    ax.grid(axis="y", alpha=0.2)

    # (d) λ×τ heatmap
    ax = axes[1, 1]
    lambdas = sorted(set(r["ema_lambda"] for r in combined_res))
    taus = sorted(set(r["recency_tau"] for r in combined_res))
    mat = np.zeros((len(lambdas), len(taus)))
    for r in combined_res:
        i = lambdas.index(r["ema_lambda"])
        j = taus.index(r["recency_tau"])
        mat[i, j] = r["accuracy"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "soft", ["#F5E6D8", "#E8C8A0", "#D4A878", "#B88858", "#966840"])
    im = ax.imshow(mat, cmap=cmap, aspect="auto",
                   vmin=mat.min() * 0.95, vmax=min(mat.max() * 1.02, 1.0))
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([str(t) for t in taus])
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels([f"{l:.1f}" for l in lambdas])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"(d) Joint $\lambda \times \tau$")
    for i in range(len(lambdas)):
        for j in range(len(taus)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="#333333")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_hyperparam_sensitivity.{ext}", dpi=300)
    plt.close(fig)
    print(f"\nSaved fig_hyperparam_sensitivity.pdf/png to {HERE}")


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  E11: Hyperparameter Sensitivity (pure Python)")
    print(f"  n_blocks={N_BLOCKS}  n_steps={N_STEPS}  n_runs={N_RUNS}")
    print("=" * 60)

    defaults = {"ema_lambda": DEFAULT_LAMBDA, "recency_tau": DEFAULT_TAU,
                "cooldown": DEFAULT_COOLDOWN}

    print("\n  --- λ sweep ---")
    lam_res = sweep_one("ema_lambda", LAMBDA_VALUES, defaults,
                        N_BLOCKS, N_STEPS, N_RUNS, SEED)

    print("\n  --- τ sweep ---")
    tau_res = sweep_one("recency_tau", TAU_VALUES, defaults,
                        N_BLOCKS, N_STEPS, N_RUNS, SEED)

    print("\n  --- cooldown sweep ---")
    cooldown_res = sweep_one("cooldown", COOLDOWN_VALUES, defaults,
                             N_BLOCKS, N_STEPS, N_RUNS, SEED)

    print("\n  --- combined λ×τ ---")
    combined_res = sweep_combined(N_BLOCKS, N_STEPS, N_RUNS, SEED)

    output = {
        "config": {"n_blocks": N_BLOCKS, "n_steps": N_STEPS,
                    "n_runs": N_RUNS, "seed": SEED},
        "ema_lambda": lam_res,
        "recency_tau": tau_res,
        "cooldown": cooldown_res,
        "combined": combined_res,
    }
    out_path = HERE / "exp_e11_hyperparam.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    plot_all(lam_res, cooldown_res, tau_res, combined_res)


if __name__ == "__main__":
    main()

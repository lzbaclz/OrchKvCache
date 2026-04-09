#!/usr/bin/env python3
"""
P0: 16-request signal ablation (Full EMA vs No-attn)
W4: 32K context experiment on Qwen2.5-7B SDPA fallback

Usage:
    python benchmarks/exp_p0_w4_final.py --task p0 --device cuda:0
    python benchmarks/exp_p0_w4_final.py --task w4 --device cuda:1
    python benchmarks/exp_p0_w4_final.py --task all --device cuda:0
"""
from __future__ import annotations
import argparse, gc, os, sys, time, json
from pathlib import Path
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

try:
    import orchkv_core as _C
except ImportError:
    _C = None
    print("WARNING: orchkv_core not found, using Python-only fallback")

from orchkv.kvcache_manager import KVCacheManager, NaiveOffloadManager

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(name, attn_impl="eager", device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation=attn_impl,
        local_files_only=True)
    mdl.eval()
    return mdl, tok


class ConfigurableKVCacheManager(KVCacheManager):
    """KVCacheManager with configurable alpha/beta/gamma."""
    def __init__(self, *, alpha=0.7, beta=0.2, gamma=0.1, **kwargs):
        self._custom_alpha = alpha
        self._custom_beta = beta
        self._custom_gamma = gamma
        super().__init__(**kwargs)

    def _init_tiered_manager(self):
        if _C is None:
            return
        self._tm_handle = _C.tm_create(
            alpha=self._custom_alpha,
            beta=self._custom_beta,
            gamma=self._custom_gamma,
            ema_lambda=0.9,
            recency_tau=50.0,
            cooldown_sec=0.5,
            threshold_to_gpu=0.6,
            threshold_to_dram=0.15,
        )


def run_decode_single(model, input_ids, max_new, manager=None, attn_every=10,
                      collect_attn=True):
    generated = []
    cur_ids = input_ids.clone()
    past_kv = None
    tpots = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_new):
        want_attn = (collect_attn and manager is not None
                     and isinstance(manager, KVCacheManager)
                     and step % attn_every == 0)
        with torch.no_grad():
            out = model(cur_ids, past_key_values=past_kv, use_cache=True,
                        output_attentions=want_attn)

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())

        step_start = time.perf_counter()

        if manager is not None:
            new_past = out.past_key_values
            if step == 0:
                manager.ingest_step(new_past)
            else:
                manager.append_token(new_past)
            if hasattr(out, 'attentions') and out.attentions is not None:
                for li, attn in enumerate(out.attentions):
                    manager.report_attention(li, attn)
            manager.step_done()
            manager.schedule()
            past_kv = manager.build_past_kv()
        else:
            past_kv = out.past_key_values
        cur_ids = next_tok

        tpots.append(time.perf_counter() - step_start)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tok = input_ids.shape[1] + len(generated)
    return {
        "generated": generated,
        "prompt_len": input_ids.shape[1],
        "elapsed_s": round(elapsed, 3),
        "throughput": round(total_tok / elapsed, 1),
    }


# =====================================================================
# P0: 16-request Signal Ablation
# =====================================================================
def run_p0(device="cuda:0"):
    print("=" * 70)
    print("  P0: 16-Request Signal Ablation (Full EMA vs No-attn)")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-7B"
    seq_len = 2048
    budget_mb = 50
    max_new = 64

    configs = [
        {"name": "full_ema", "alpha": 0.7, "beta": 0.2, "gamma": 0.1,
         "collect_attn": True, "desc": "Full EMA (α=0.7)"},
        {"name": "no_attn", "alpha": 0.0, "beta": 0.6, "gamma": 0.4,
         "collect_attn": False, "desc": "No-attn (α=0.0, SDPA fallback)"},
    ]
    nreq_list = [4, 8, 16]

    model, tok = load_model(model_name, "eager", device)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    text = "Artificial intelligence is transforming every aspect of modern life. " * (seq_len // 8 + 1)
    input_ids = tok(text, return_tensors="pt", truncation=True,
                    max_length=seq_len)["input_ids"].to(device)
    actual_len = input_ids.shape[1]
    print(f"  Model: {model_name}, seq={actual_len}, budget={budget_mb}MB, gen={max_new}")

    results = []
    for nreq in nreq_list:
        per_req_budget = (budget_mb * (1 << 20)) // nreq
        print(f"\n  --- nreq={nreq}, per_req_budget={per_req_budget/(1<<20):.1f}MB ---")

        for c in configs:
            print(f"    {c['desc']}:", end=" ", flush=True)
            total_evict = 0
            req_tps = []

            for ri in range(nreq):
                mgr = ConfigurableKVCacheManager(
                    alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"],
                    n_layers=n_layers, n_kv_heads=n_kv_heads,
                    head_dim=head_dim, block_size=16, dtype=torch.float16,
                    gpu_budget_bytes=per_req_budget,
                )
                try:
                    res = run_decode_single(model, input_ids, max_new, manager=mgr,
                                            attn_every=10, collect_attn=c["collect_attn"])
                    stats = mgr.get_stats()
                    mig = stats.get("migrations", {})
                    evict = mig.get("gpu_to_dram", 0) + mig.get("dram_to_ssd", 0)
                    total_evict += evict
                    req_tps.append(res["throughput"])
                except Exception as e:
                    print(f"req {ri} failed: {e}")
                    req_tps.append(0)
                finally:
                    mgr.destroy()
                    gc.collect(); torch.cuda.empty_cache()

            avg_tps = sum(req_tps) / max(len(req_tps), 1)
            row = {
                "config": c["name"], "desc": c["desc"],
                "alpha": c["alpha"], "beta": c["beta"], "gamma": c["gamma"],
                "nreq": nreq, "avg_tok_s": round(avg_tps, 1),
                "total_evictions": total_evict,
                "per_req_budget_mb": round(per_req_budget / (1 << 20), 2),
            }
            results.append(row)
            print(f"avg={avg_tps:.1f} tok/s, evict={total_evict}")

    del model; gc.collect(); torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "exp_p0_16req_ablation.json"
    with open(out_path, "w") as f:
        json.dump({"task": "p0", "results": results}, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n  === P0 SUMMARY ===")
    print(f"  {'Config':<30s} {'nreq':>5s} {'tok/s':>8s} {'Evictions':>10s}")
    for r in results:
        print(f"  {r['desc']:<30s} {r['nreq']:>5d} {r['avg_tok_s']:>8.1f} {r['total_evictions']:>10d}")

    return results


# =====================================================================
# W4: 32K Context Experiment
# =====================================================================
def run_w4(device="cuda:0"):
    print("\n" + "=" * 70)
    print("  W4: 32K Context Experiment (Qwen2.5-7B, SDPA fallback)")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-7B"
    max_new = 32
    budget_mb = 50
    configs = [
        (8192, 50),
        (16384, 50),
        (32768, 50),
    ]

    results = []
    for seq_len, bm in configs:
        print(f"\n  --- seq={seq_len}, budget={bm}MB ---")

        model, tok = load_model(model_name, "sdpa", device)
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 8 + 1)
        input_ids = tok(text, return_tensors="pt", truncation=True,
                        max_length=seq_len)["input_ids"].to(device)
        actual_len = input_ids.shape[1]
        print(f"    actual tokens: {actual_len}")

        row = {"model": "Qwen2.5-7B", "seq_len": seq_len, "actual_len": actual_len,
               "budget_mb": bm, "max_new": max_new, "attn_impl": "sdpa"}

        # GPU-Only (SDPA)
        gc.collect(); torch.cuda.empty_cache()
        try:
            cur, past = input_ids.clone(), None
            tokens_gpu = []
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(max_new):
                with torch.no_grad():
                    out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens_gpu.append(cur.item())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tps_gpu = (actual_len + max_new) / elapsed
            row["gpu_only_tok_s"] = round(tps_gpu, 1)
            print(f"    GPU-Only:   {tps_gpu:.1f} tok/s")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            row["gpu_only_tok_s"] = "OOM"
            tokens_gpu = []
            print(f"    GPU-Only:   OOM ({str(e)[:60]})")

        # FIFO
        gc.collect(); torch.cuda.empty_cache()
        try:
            mgr = NaiveOffloadManager(
                n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=bm * (1 << 20),
            )
            res = run_decode_single(model, input_ids, max_new, manager=mgr,
                                    collect_attn=False)
            stats = mgr.get_stats()
            evict_fifo = stats.get("gpu_to_dram", 0)
            match_fifo = sum(a == b for a, b in zip(tokens_gpu, res["generated"])) / max(len(tokens_gpu), 1) if tokens_gpu else -1
            row["fifo_tok_s"] = res["throughput"]
            row["fifo_evictions"] = evict_fifo
            row["fifo_match"] = round(match_fifo, 4) if match_fifo >= 0 else "N/A"
            mgr.destroy()
            print(f"    FIFO:       {res['throughput']:.1f} tok/s, evict={evict_fifo}, match={match_fifo:.2%}" if match_fifo >= 0
                  else f"    FIFO:       {res['throughput']:.1f} tok/s, evict={evict_fifo}")
        except Exception as e:
            row["fifo_tok_s"] = "ERROR"
            print(f"    FIFO:       ERROR ({str(e)[:60]})")

        # OrchKvCache (SDPA fallback: alpha=0.0)
        gc.collect(); torch.cuda.empty_cache()
        try:
            mgr = ConfigurableKVCacheManager(
                alpha=0.0, beta=0.6, gamma=0.4,
                n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
                block_size=16, dtype=torch.float16,
                gpu_budget_bytes=bm * (1 << 20),
            )
            res = run_decode_single(model, input_ids, max_new, manager=mgr,
                                    collect_attn=False)
            stats = mgr.get_stats()
            mig = stats.get("migrations", {})
            evict = mig.get("gpu_to_dram", 0)
            match = sum(a == b for a, b in zip(tokens_gpu, res["generated"])) / max(len(tokens_gpu), 1) if tokens_gpu else -1
            row["orchkv_tok_s"] = res["throughput"]
            row["orchkv_evictions"] = evict
            row["orchkv_match"] = round(match, 4) if match >= 0 else "N/A"
            speedup = res["throughput"] / row["fifo_tok_s"] if isinstance(row.get("fifo_tok_s"), (int, float)) and row["fifo_tok_s"] > 0 else "N/A"
            row["orch_vs_fifo"] = round(speedup, 2) if isinstance(speedup, float) else speedup
            mgr.destroy()
            label = f"match={match:.2%}" if match >= 0 else ""
            print(f"    OrchKv:     {res['throughput']:.1f} tok/s, evict={evict}, {label}, vs FIFO={speedup:.2f}x" if isinstance(speedup, float)
                  else f"    OrchKv:     {res['throughput']:.1f} tok/s, evict={evict}")
        except Exception as e:
            row["orchkv_tok_s"] = "ERROR"
            print(f"    OrchKv:     ERROR ({str(e)[:60]})")

        results.append(row)
        del model; gc.collect(); torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "exp_w4_32k_context.json"
    with open(out_path, "w") as f:
        json.dump({"task": "w4", "results": results}, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n  === W4 SUMMARY ===")
    print(f"  {'Seq':>6s} {'GPU-Only':>10s} {'FIFO':>10s} {'OrchKv':>10s} {'Orch/FIFO':>10s} {'Match':>7s}")
    for r in results:
        gpu = r.get('gpu_only_tok_s', 'N/A')
        fifo = r.get('fifo_tok_s', 'N/A')
        orch = r.get('orchkv_tok_s', 'N/A')
        vs = r.get('orch_vs_fifo', 'N/A')
        m = r.get('orchkv_match', 'N/A')
        print(f"  {r['seq_len']:>6d} {str(gpu):>10s} {str(fifo):>10s} {str(orch):>10s} {str(vs):>10s} {str(m):>7s}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["p0", "w4", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.task in ("p0", "all"):
        run_p0(args.device)
    if args.task in ("w4", "all"):
        run_w4(args.device)


if __name__ == "__main__":
    main()

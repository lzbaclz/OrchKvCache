#!/usr/bin/env python3
"""
P2: Prefetch Utility Metrics from E2E inference.

Runs OrchKvCache on Qwen2.5-7B and LLaMA-2-7B, collects:
  - prefetches_dispatched
  - prefetch_hits
  - prefetch_wasted
  - prefetch_hit_rate
  - gpu_demotes (demand misses proxy)
"""
from __future__ import annotations
import gc, os, sys, json, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, RESULTS_DIR
from orchkv.kvcache_manager import KVCacheManager

try:
    import orchkv_core as _C
except ImportError:
    _C = None


def run_prefetch_experiment(model_name, seq_len=2048, max_new=64, budget_mb=50, n_req=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    short = model_name.split("/")[-1]
    print(f"\n  {short}  seq={seq_len}  gen={max_new}  budget={budget_mb}MB  nreq={n_req}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True)
    mdl.eval()
    cfg = mdl.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    per_req_budget = (budget_mb * (1 << 20)) // max(n_req, 1)
    all_req_stats = []

    for ri in range(n_req):
        text = "Artificial intelligence is transforming every aspect of modern life. " * (seq_len // 10 + 1)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"].to("cuda:0")

        mgr = KVCacheManager(
            n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
            block_size=16, dtype=torch.float16, gpu_budget_bytes=per_req_budget,
        )

        cur, past = ids.clone(), None
        for s in range(max_new):
            want_attn = s % 10 == 0
            with torch.no_grad():
                out = mdl(cur, past_key_values=past, use_cache=True,
                          output_attentions=want_attn)
            new_past = out.past_key_values
            if s == 0:
                mgr.ingest_step(new_past)
            else:
                mgr.append_token(new_past)
            if hasattr(out, 'attentions') and out.attentions is not None:
                for li, attn in enumerate(out.attentions):
                    mgr.report_attention(li, attn)
            mgr.step_done()
            mgr.schedule()
            past = mgr.build_past_kv()
            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        py_stats = mgr.get_stats()
        mig = py_stats.get("migrations", {})

        tm_stats = {}
        if _C and mgr._tm_handle is not None:
            tm_stats = _C.tm_get_stats(mgr._tm_handle)

        mgr.destroy()

        req_stats = {
            "request": ri,
            "gpu_to_dram": mig.get("gpu_to_dram", 0),
            "dram_to_gpu": mig.get("dram_to_gpu", 0),
            "prefetches_dispatched": tm_stats.get("prefetches_dispatched", 0),
            "prefetch_hits": tm_stats.get("prefetch_hits", 0),
            "prefetch_wasted": tm_stats.get("prefetch_wasted", 0),
            "prefetch_hit_rate": round(tm_stats.get("prefetch_hit_rate", 0), 4),
            "schedule_cycles": tm_stats.get("schedule_cycles", 0),
            "n_hot": tm_stats.get("n_hot", 0),
            "n_warm": tm_stats.get("n_warm", 0),
            "n_cold": tm_stats.get("n_cold", 0),
        }

        all_req_stats.append(req_stats)
        print(f"    req {ri}: demotes={req_stats['gpu_to_dram']}, "
              f"dispatched={req_stats['prefetches_dispatched']}, "
              f"hits={req_stats['prefetch_hits']}, "
              f"wasted={req_stats['prefetch_wasted']}, "
              f"hit_rate={req_stats['prefetch_hit_rate']:.2%}")

        gc.collect(); torch.cuda.empty_cache()

    del mdl; gc.collect(); torch.cuda.empty_cache()

    total_demotes = sum(r["gpu_to_dram"] for r in all_req_stats)
    total_promotes = sum(r["dram_to_gpu"] for r in all_req_stats)
    total_dispatched = sum(r["prefetches_dispatched"] for r in all_req_stats)
    total_hits = sum(r["prefetch_hits"] for r in all_req_stats)
    total_wasted = sum(r["prefetch_wasted"] for r in all_req_stats)
    hit_rate = total_hits / total_dispatched if total_dispatched > 0 else 0

    return {
        "model": short,
        "seq_len": seq_len,
        "n_requests": n_req,
        "budget_mb": budget_mb,
        "total_demotes": total_demotes,
        "total_promotes": total_promotes,
        "total_prefetches_dispatched": total_dispatched,
        "total_prefetch_hits": total_hits,
        "total_prefetch_wasted": total_wasted,
        "prefetch_hit_rate": round(hit_rate, 4),
        "useful_prefetch_ratio": round(total_hits / max(total_dispatched, 1), 4),
        "per_request": all_req_stats,
    }


def main():
    models = ["Qwen/Qwen2.5-7B"]
    results = []
    for m in models:
        r = run_prefetch_experiment(m, seq_len=2048, max_new=64, budget_mb=50, n_req=4)
        if r:
            results.append(r)

    out_path = RESULTS_DIR / "exp_prefetch_utility.json"
    save_json(results, out_path)
    print(f"\nSaved to {out_path}")

    for r in results:
        print(f"\n{r['model']}:")
        print(f"  demotes={r['total_demotes']}, promotes={r['total_promotes']}")
        print(f"  prefetch dispatched={r['total_prefetches_dispatched']}, "
              f"hits={r['total_prefetch_hits']}, wasted={r['total_prefetch_wasted']}")
        print(f"  hit_rate={r['prefetch_hit_rate']:.2%}, "
              f"useful_ratio={r['useful_prefetch_ratio']:.2%}")


if __name__ == "__main__":
    main()

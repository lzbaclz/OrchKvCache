#!/usr/bin/env python3
"""
Trace-Driven Competitive Ratio & Signal Ablation (v4)

Uses MULTI-REQUEST traces: 4 different prompts interleaved round-robin,
each with its own hot set, competing for shared GPU capacity.
This matches OrchKvCache's actual deployment scenario.

Standard online paging: each step requests certain blocks; miss = evict + fetch.
"""
from __future__ import annotations
import gc, json, math, os, sys
from collections import defaultdict, OrderedDict, deque
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import torch


DIVERSE_PROMPTS = [
    "The history of artificial intelligence begins in the mid-twentieth century when researchers first proposed that machines could be made to simulate human thought processes. " * 30,
    "In quantum computing, qubits can exist in superposition states, enabling parallel computation across exponentially many states simultaneously. " * 30,
    "Climate change is driven primarily by greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes that alter Earth's energy balance. " * 30,
    "The human genome contains approximately three billion base pairs of DNA organized into twenty-three pairs of chromosomes encoding roughly twenty thousand protein-coding genes. " * 30,
]


def collect_multi_request_trace(model_name: str, seq_len: int = 512,
                                n_requests: int = 4, steps_per_req: int = 16,
                                block_size: int = 16, device: str = "cuda:0"):
    """Collect interleaved attention traces from multiple requests."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Collecting multi-request trace: {model_name.split('/')[-1]}")
    print(f"    {n_requests} requests × {steps_per_req} steps, seq_len={seq_len}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True)
    mdl.eval()

    # Prepare inputs for each request
    request_ids = []
    for i in range(n_requests):
        ids = tok(DIVERSE_PROMPTS[i % len(DIVERSE_PROMPTS)],
                  return_tensors="pt", truncation=True,
                  max_length=seq_len)["input_ids"].to(device)
        request_ids.append(ids)

    # Collect traces per request
    per_req_traces = []
    for ri in range(n_requests):
        ids = request_ids[ri]
        actual_len = ids.shape[1]
        cur, past = ids.clone(), None
        req_trace = []
        for step in range(steps_per_req):
            with torch.no_grad():
                out = mdl(cur, past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values
            n_blocks = (actual_len + step + 1 + block_size - 1) // block_size

            block_scores = {}
            if out.attentions is not None:
                for li, attn_w in enumerate(out.attentions):
                    avg = attn_w.float().mean(dim=(0, 2)).squeeze(0).cpu().numpy()
                    for bid in range(n_blocks):
                        s = bid * block_size
                        e = min(s + block_size, len(avg))
                        if s >= len(avg): break
                        block_scores[bid] = block_scores.get(bid, 0.0) + float(avg[s:e].sum())
            req_trace.append({"n_blocks": n_blocks, "scores": block_scores})
            cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        per_req_traces.append(req_trace)

    del mdl; gc.collect(); torch.cuda.empty_cache()

    # Interleave: round-robin across requests, offset block IDs per request
    interleaved = []
    block_offsets = []
    offset = 0
    for ri in range(n_requests):
        block_offsets.append(offset)
        max_blocks = max(t["n_blocks"] for t in per_req_traces[ri])
        offset += max_blocks

    total_blocks = offset
    total_steps = steps_per_req

    for step in range(total_steps):
        for ri in range(n_requests):
            t = per_req_traces[ri][step]
            off = block_offsets[ri]
            global_scores = {}
            for bid, sc in t["scores"].items():
                global_scores[bid + off] = sc
            interleaved.append({
                "request": ri, "step": step,
                "n_blocks_total": total_blocks,
                "scores": global_scores,
            })

    print(f"    Total blocks across all requests: {total_blocks}")
    print(f"    Interleaved steps: {len(interleaved)}")
    return interleaved, total_blocks


def extract_requests(trace, hot_frac=0.25):
    requests = []
    for t in trace:
        scores = t["scores"]
        if not scores:
            requests.append([])
            continue
        k = max(1, int(len(scores) * hot_frac))
        ranked = sorted(scores.keys(), key=lambda b: scores.get(b, 0), reverse=True)
        requests.append(ranked[:k])
    return requests


# ======================================================================
# 5 policies (same as v3 but tested)
# ======================================================================

def sim_fifo(requests, n_pages, cache_size):
    cache = set(); q = deque(); misses = 0
    for req in requests:
        for page in req:
            if page in cache: continue
            misses += 1
            if len(cache) >= cache_size:
                while q:
                    v = q.popleft()
                    if v in cache: cache.discard(v); break
            cache.add(page); q.append(page)
    return misses

def sim_lru(requests, n_pages, cache_size):
    cache = OrderedDict(); misses = 0
    for req in requests:
        for page in req:
            if page in cache: cache.move_to_end(page); continue
            misses += 1
            if len(cache) >= cache_size: cache.popitem(last=False)
            cache[page] = True
    return misses

def sim_lfu(requests, n_pages, cache_size):
    cache = set(); freq = defaultdict(int); misses = 0
    for req in requests:
        for page in req:
            freq[page] += 1
            if page in cache: continue
            misses += 1
            if len(cache) >= cache_size:
                victim = min(cache, key=lambda p: freq[p])
                cache.discard(victim)
            cache.add(page)
    return misses

def sim_ema(requests, trace, n_pages, cache_size,
            alpha=0.7, beta=0.2, gamma=0.1, lam=0.3):
    cache = set(); misses = 0
    ema = defaultdict(float); last_hit = {}; hit_count = defaultdict(int)
    for step, req in enumerate(requests):
        for bid, sc in trace[step]["scores"].items():
            ema[bid] = lam * sc + (1 - lam) * ema[bid]
            last_hit[bid] = step; hit_count[bid] += 1
        for bid in list(ema.keys()):
            if bid not in trace[step]["scores"]: ema[bid] *= (1 - lam)
        max_e = max(ema.values()) if ema else 1e-9
        max_f = max(hit_count.values()) if hit_count else 1
        def score(b):
            a = ema.get(b, 0) / max(max_e, 1e-9)
            dt = step - last_hit.get(b, 0)
            r = math.exp(-dt / 50.0) if dt < 256 else 0.0
            f = min(hit_count.get(b, 0) / max(max_f, 1), 1.0)
            return alpha * a + beta * r + gamma * f
        for page in req:
            if page in cache: continue
            misses += 1
            if len(cache) >= cache_size:
                victim = min(cache, key=score)
                cache.discard(victim)
            cache.add(page)
    return misses

def sim_opt(requests, n_pages, cache_size):
    next_use = {}; last = {}
    for step in range(len(requests) - 1, -1, -1):
        for page in requests[step]:
            if page in last: next_use[(step, page)] = last[page]
            last[page] = step
    cache = set(); misses = 0
    for step, req in enumerate(requests):
        for page in req:
            if page in cache: continue
            misses += 1
            if len(cache) >= cache_size:
                victim = max(cache, key=lambda p: next_use.get((step, p), len(requests) + 1))
                cache.discard(victim)
            cache.add(page)
    return misses


def main():
    print("=" * 70)
    print("  Trace-Driven CR (v4): Multi-Request Interleaved Traces")
    print("=" * 70)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    models = [("Qwen/Qwen2.5-7B", 512), ("meta-llama/Llama-2-7b-hf", 512)]
    cap_fracs = [0.10, 0.20, 0.30, 0.50, 0.70]
    all_cr = []; all_abl = []

    for model_name, seq_len in models:
        short = model_name.split("/")[-1]
        trace, n_blocks = collect_multi_request_trace(
            model_name, seq_len=seq_len, n_requests=4, steps_per_req=16)
        requests = extract_requests(trace, hot_frac=0.30)
        print(f"  Avg request size: {np.mean([len(r) for r in requests]):.1f} blocks/step")

        for cf in cap_fracs:
            cs = max(2, int(n_blocks * cf))
            m_fifo = sim_fifo(requests, n_blocks, cs)
            m_lru  = sim_lru(requests, n_blocks, cs)
            m_lfu  = sim_lfu(requests, n_blocks, cs)
            m_ema  = sim_ema(requests, trace, n_blocks, cs)
            m_opt  = sim_opt(requests, n_blocks, cs)
            opt = max(m_opt, 1)
            row = {"model": short, "n_blocks": n_blocks, "capacity_frac": cf,
                   "gpu_cap": cs,
                   "FIFO_misses": m_fifo, "LRU_misses": m_lru,
                   "LFU_misses": m_lfu, "EMA_misses": m_ema, "OPT_misses": m_opt,
                   "FIFO_cr": round(m_fifo / opt, 2),
                   "LRU_cr": round(m_lru / opt, 2),
                   "LFU_cr": round(m_lfu / opt, 2),
                   "EMA_cr": round(m_ema / opt, 2),
                   "OPT_cr": 1.00}
            all_cr.append(row)
            print(f"  cap={cf*100:>3.0f}%  FIFO={row['FIFO_cr']:.2f}  LRU={row['LRU_cr']:.2f}  "
                  f"LFU={row['LFU_cr']:.2f}  EMA={row['EMA_cr']:.2f}  OPT=1.00  "
                  f"(miss: {m_fifo}/{m_lru}/{m_lfu}/{m_ema}/{m_opt})")

        # Signal ablation
        cs30 = max(2, int(n_blocks * 0.30))
        m_opt30 = max(sim_opt(requests, n_blocks, cs30), 1)
        print(f"\n  Signal ablation (30% cap):")
        for name, a, b, g in [("Full EMA", 0.7, 0.2, 0.1),
                               ("No-attn", 0.0, 0.6, 0.4),
                               ("Recency-only", 0.0, 1.0, 0.0),
                               ("Freq-only", 0.0, 0.0, 1.0)]:
            m = sim_ema(requests, trace, n_blocks, cs30, a, b, g)
            cr = round(m / m_opt30, 3)
            all_abl.append({"model": short, "config": name,
                            "alpha": a, "beta": b, "gamma": g,
                            "misses": m, "cr": cr})
            print(f"    {name:<16s}  misses={m:>5d}  CR={cr:.3f}")

    for data, name in [(all_cr, "exp_trace_driven_cr"), (all_abl, "exp_signal_ablation")]:
        path = RESULTS_DIR / f"{name}.json"
        with open(path, "w") as f: json.dump(data, f, indent=2)
        paper = Path(__file__).parent / ".." / "paper" / "plot_figures_code_data"
        if paper.exists():
            import shutil; shutil.copy(path, paper / f"{name}.json")

    print(f"\n{'='*70}")
    print(f"  CR SUMMARY (multi-request real traces)")
    print(f"{'='*70}")
    print(f"  {'Model':<18s} {'Cap':>4s} {'FIFO':>6s} {'LRU':>6s} {'LFU':>6s} {'EMA':>6s} {'OPT':>6s}")
    for r in all_cr:
        print(f"  {r['model']:<18s} {r['capacity_frac']*100:>3.0f}% "
              f"{r['FIFO_cr']:>6.2f} {r['LRU_cr']:>6.2f} "
              f"{r['LFU_cr']:>6.2f} {r['EMA_cr']:>6.2f} {'1.00':>6s}")
    print(f"\n  ABLATION")
    for a in all_abl:
        print(f"  {a['model']:<18s} {a['config']:<16s} miss={a['misses']:>5d} CR={a['cr']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
P0.3: Large-scale SSD ablation — MB-level SSD traffic.

Uses LLaMA-2-7B (512 KB/tok) at seq=2048, budget=10MB to generate
substantial SSD write volume, where batching effects become observable.

Compares: GPU+DRAM only  vs  GPU+DRAM+SSD per-block  vs  GPU+DRAM+SSD batched-8
"""
import gc, os, sys, json, time, shutil, tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bench_utils import save_json, RESULTS_DIR
from orchkv.kvcache_manager import KVCacheManager

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
SHORT_NAME = "LLaMA-2-7B"
SEQ_LEN = 2048
MAX_NEW = 64
BUDGET_MB = 10


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True)
    mdl.eval()
    cfg = mdl.config
    mc = {
        "n_layers": cfg.num_hidden_layers,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
    }
    kv_per_tok = 2 * mc["n_layers"] * mc["n_kv_heads"] * mc["head_dim"] * 2
    print(f"  {SHORT_NAME}: {mc['n_layers']}L, {mc['n_kv_heads']}KV, "
          f"d={mc['head_dim']}, {kv_per_tok//1024}KB/tok")
    return mdl, tok, mc


def run_e2e(model, tokenizer, mc, ssd_dir=None, batch_ssd=1):
    text = "The study of artificial intelligence and machine learning " * (SEQ_LEN // 8 + 1)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=SEQ_LEN)["input_ids"].to("cuda:0")

    mgr = KVCacheManager(
        n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
        head_dim=mc["head_dim"], block_size=16, dtype=torch.float16,
        gpu_budget_bytes=BUDGET_MB * (1 << 20),
        ssd_dir=ssd_dir,
    )

    if batch_ssd > 1 and ssd_dir:
        mgr._ssd_batch_size = batch_ssd

    cur, past = ids.clone(), None
    t0 = time.perf_counter()

    for s in range(MAX_NEW):
        want_attn = s % 10 == 0
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True,
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

    elapsed = time.perf_counter() - t0
    total_tok = ids.shape[1] + MAX_NEW
    tok_s = total_tok / elapsed

    stats = mgr.get_stats()
    mig = stats.get("migrations", {})

    mgr.destroy()
    gc.collect(); torch.cuda.empty_cache()

    ssd_files = 0
    ssd_bytes = 0
    if ssd_dir and os.path.exists(ssd_dir):
        for f in os.listdir(ssd_dir):
            fp = os.path.join(ssd_dir, f)
            if os.path.isfile(fp):
                ssd_files += 1
                ssd_bytes += os.path.getsize(fp)

    return {
        "tok_s": round(tok_s, 1),
        "elapsed_s": round(elapsed, 3),
        "gpu_to_dram": mig.get("gpu_to_dram", 0),
        "dram_to_ssd": mig.get("dram_to_ssd", 0),
        "ssd_to_dram": mig.get("ssd_to_dram", 0),
        "ssd_files": ssd_files,
        "ssd_bytes_kb": round(ssd_bytes / 1024, 1),
    }


def main():
    model, tokenizer, mc = load_model()

    configs = [
        ("GPU+DRAM only", None, 1),
        ("GPU+DRAM+SSD (per-block)", "per_block", 1),
        ("GPU+DRAM+SSD (batched-8)", "batched_8", 8),
    ]

    results = []
    for name, ssd_mode, batch_sz in configs:
        print(f"\n{'='*50}")
        print(f"  Config: {name}")

        ssd_dir = None
        if ssd_mode:
            ssd_dir = tempfile.mkdtemp(prefix=f"orchkv_ssd_large_{ssd_mode}_")

        trials = []
        for trial in range(3):
            if ssd_dir and os.path.exists(ssd_dir):
                for f in os.listdir(ssd_dir):
                    os.remove(os.path.join(ssd_dir, f))

            r = run_e2e(model, tokenizer, mc, ssd_dir=ssd_dir, batch_ssd=batch_sz)
            trials.append(r)
            print(f"    trial {trial}: {r['tok_s']} tok/s, "
                  f"G→D={r['gpu_to_dram']}, D→S={r['dram_to_ssd']}, "
                  f"S→D={r['ssd_to_dram']}, "
                  f"SSD={r['ssd_bytes_kb']:.0f}KB")

        if ssd_dir and os.path.exists(ssd_dir):
            shutil.rmtree(ssd_dir)

        avg = lambda k: round(sum(t[k] for t in trials) / len(trials), 1)
        row = {
            "config": name,
            "model": SHORT_NAME,
            "seq_len": SEQ_LEN,
            "budget_mb": BUDGET_MB,
            "batch_size": batch_sz,
            "avg_tok_s": avg("tok_s"),
            "avg_gpu_to_dram": int(avg("gpu_to_dram")),
            "avg_dram_to_ssd": int(avg("dram_to_ssd")),
            "avg_ssd_to_dram": int(avg("ssd_to_dram")),
            "avg_ssd_bytes_kb": avg("ssd_bytes_kb"),
            "trials": trials,
        }
        results.append(row)
        print(f"  AVG: {row['avg_tok_s']} tok/s, "
              f"SSD write={row['avg_ssd_bytes_kb']:.0f}KB")

    print(f"\n{'='*50}")
    print(f"  SUMMARY ({SHORT_NAME}, seq={SEQ_LEN}, budget={BUDGET_MB}MB)")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['config']:<30s} {r['avg_tok_s']:>8.1f} tok/s  "
              f"SSD={r['avg_ssd_bytes_kb']:>8.0f}KB")

    out = RESULTS_DIR / "exp_ssd_ablation_large.json"
    save_json(results, out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

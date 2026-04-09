#!/usr/bin/env python3
"""Sub-batch rotation on Mistral-7B (GQA, 128KB/tok)."""
import gc, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from orchkv.fast_kvcache_manager import FastKVCacheManager

MODEL = "/raid/models/Mistral-7B-v0.1"
SEQ, GEN, BUD, NREQ = 1024, 64, 50, 8
PROMPTS = [f"Topic {i}: " + "AI systems research. " * 60 for i in range(16)]

def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True, attn_implementation="eager")
    m.eval()
    c = m.config
    mc = {"n_layers": c.num_hidden_layers,
          "n_kv_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
          "head_dim": c.hidden_size // c.num_attention_heads}
    kv = 2 * mc["n_layers"] * mc["n_kv_heads"] * mc["head_dim"] * 2
    print(f"Mistral: {mc}, KV/tok={kv//1024}KB")
    return m, t, mc

def run(mdl, tok, mc, sub_k):
    ids_list = [tok(p, return_tensors="pt", truncation=True,
                    max_length=SEQ)["input_ids"].to("cuda:0") for p in PROMPTS[:NREQ]]
    per_b = BUD * (1 << 20) // sub_k if sub_k < NREQ else BUD * (1 << 20) // NREQ
    mgrs = [FastKVCacheManager(n_layers=mc["n_layers"], n_kv_heads=mc["n_kv_heads"],
            head_dim=mc["head_dim"], block_size=16, dtype=torch.float16,
            gpu_budget_bytes=per_b, max_seq_len=SEQ + GEN + 64) for _ in range(NREQ)]
    curs = [x.clone() for x in ids_list]; pasts = [None] * NREQ
    for ri in range(NREQ):
        with torch.no_grad():
            out = mdl(curs[ri], past_key_values=pasts[ri], use_cache=True)
        mgrs[ri].ingest_step(out.past_key_values)
        mgrs[ri].step_done(); mgrs[ri].schedule()
        pasts[ri] = mgrs[ri].build_past_kv()
        curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    nsb = (NREQ + sub_k - 1) // sub_k
    torch.cuda.synchronize(); t0 = time.perf_counter()
    slats = []
    for step in range(GEN):
        for sb in range(nsb):
            s, e = sb * sub_k, min((sb + 1) * sub_k, NREQ)
            for ri in range(s, e):
                torch.cuda.synchronize(); ts = time.perf_counter()
                wa = step % 10 == 0
                with torch.no_grad():
                    out = mdl(curs[ri], past_key_values=pasts[ri],
                              use_cache=True, output_attentions=wa)
                mgrs[ri].append_token(out.past_key_values)
                if wa and getattr(out, "attentions", None):
                    for li, a in enumerate(out.attentions):
                        mgrs[ri].report_attention(li, a)
                mgrs[ri].step_done(); mgrs[ri].schedule()
                pasts[ri] = mgrs[ri].build_past_kv()
                curs[ri] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
                slats.append((time.perf_counter() - ts) * 1000)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
    ttok = sum(x.shape[1] for x in ids_list) + GEN * NREQ
    evict = sum(m.get_stats()["migrations"]["gpu_to_dram"] for m in mgrs)
    for m in mgrs: m.destroy()
    gc.collect(); torch.cuda.empty_cache()
    slats.sort(); n = len(slats)
    return {"K": sub_k, "tok_s": round(ttok / elapsed, 1), "evict": evict,
            "p50": round(slats[n // 2], 1), "p99": round(slats[int(n * 0.99)], 1)}

mdl, tok, mc = load()
_ = run(mdl, tok, mc, 1); gc.collect(); torch.cuda.empty_cache()

results = []
for k in [1, 2, 4, 8]:
    r = run(mdl, tok, mc, k); results.append(r)
    print(f"K={k}: {r['tok_s']} tok/s, evict={r['evict']}, p50={r['p50']}ms")

base = results[-1]["tok_s"]
print(f"\nSUMMARY (Mistral-7B, {NREQ} req, seq={SEQ}, {BUD}MB)")
for r in results:
    sp = r["tok_s"] / base
    print(f"  K={r['K']}: {r['tok_s']} tok/s, {sp:.2f}x, evict={r['evict']}, p50={r['p50']}ms")

with open(os.path.join(os.path.dirname(__file__), "results", "exp_subbatch_mistral.json"), "w") as f:
    json.dump(results, f, indent=2)

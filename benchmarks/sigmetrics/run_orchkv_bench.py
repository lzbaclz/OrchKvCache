"""Phase 2: OrchKvCache vs vLLM baseline comparison on Qwen2.5-7B."""
import torch
import traceback
import time
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from orchkv.kvcache_manager import KVCacheManager

MODEL = '/public/model_zoo/Qwen2.5-7B'
OUTPUT_DIR = 'benchmarks/sigmetrics/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Qwen2.5-7B (eager attention)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map='cuda',
    trust_remote_code=True, attn_implementation='eager')
model.eval()

n_layers = model.config.num_hidden_layers
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // model.config.num_attention_heads
print(f"Config: {n_layers}L, {n_kv_heads}KV, d={head_dim}")

PROMPT_LEN = 512
GEN_LEN = 32
seq_kv_bytes = 2 * n_layers * n_kv_heads * (PROMPT_LEN + GEN_LEN) * head_dim * 2
budget = int(seq_kv_bytes * 0.5)

mgr = KVCacheManager(
    n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
    block_size=16, dtype=torch.float16,
    gpu_budget_bytes=budget,
    ssd_dir='/tmp/orchkv_bench_ssd', sink_tokens=4,
)

prompt = "Explain general relativity and quantum mechanics in detail. " * 64
input_ids = tokenizer.encode(prompt, return_tensors='pt',
                             max_length=PROMPT_LEN, truncation=True).cuda()
print(f"Prompt: {input_ids.shape[1]} tokens, budget={budget} bytes")

# Prefill
with torch.no_grad():
    out = model(input_ids, use_cache=True, output_attentions=True)
mgr.ingest_step(out.past_key_values)
for li in range(n_layers):
    mgr.report_attention(li, out.attentions[li])
mgr.step_done()
sched = mgr.schedule()
print(f"Prefill done: evicted={sched.get('evicted', 0)}")

# Decode
next_token = out.logits[:, -1:, :].argmax(dim=-1)
t0 = time.time()
gen_tokens = 0

for step in range(GEN_LEN):
    try:
        with torch.no_grad():
            past_kv = mgr.build_past_kv()
            out = model(next_token, past_key_values=past_kv,
                        use_cache=True, output_attentions=True)
        mgr.ingest_step(out.past_key_values)
        for li in range(n_layers):
            mgr.report_attention(li, out.attentions[li])
        mgr.step_done()
        mgr.schedule()
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        gen_tokens += 1
        if next_token.item() == tokenizer.eos_token_id:
            break
    except Exception as e:
        traceback.print_exc()
        print(f"DECODE ERROR at step {step}: {e}")
        break

elapsed = time.time() - t0
if gen_tokens > 0:
    throughput = gen_tokens / elapsed
    tpot = (elapsed * 1000) / gen_tokens
    promo = mgr.get_promotion_latency_stats()

    print(f"\n=== RESULTS ===")
    print(f"Generated: {gen_tokens} tokens in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"TPOT: {tpot:.1f} ms")
    print(f"Evictions: {mgr._stats['gpu_to_dram']}")
    print(f"Promotions: {mgr._stats['dram_to_gpu']}")
    print(f"SSD spills: {mgr._stats.get('dram_to_ssd', 0)}")
    print(f"Promotion P50: {promo['p50']:.0f} us")
    print(f"Promotion P99: {promo['p99']:.0f} us")
    print(f"Decision log: {len(mgr.get_decision_log())} entries")

    results = {
        'model': 'Qwen2.5-7B', 'engine': 'OrchKvCache-HF-eager',
        'prompt_tokens': int(input_ids.shape[1]),
        'gen_tokens': gen_tokens,
        'budget_bytes': budget,
        'budget_pct': 50,
        'throughput_tok_s': round(throughput, 2),
        'tpot_ms': round(tpot, 2),
        'evictions': mgr._stats['gpu_to_dram'],
        'promotions': mgr._stats['dram_to_gpu'],
        'ssd_spills': mgr._stats.get('dram_to_ssd', 0),
        'promo_p50_us': round(promo['p50'], 1),
        'promo_p99_us': round(promo['p99'], 1),
        'decisions': len(mgr.get_decision_log()),
    }
    outpath = os.path.join(OUTPUT_DIR, 'orchkv_hf_qwen7b.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {outpath}")
else:
    print("NO TOKENS GENERATED")

mgr.destroy()
print("ORCHKV BENCHMARK COMPLETE")

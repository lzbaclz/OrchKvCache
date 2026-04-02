#!/usr/bin/env python3
"""
Fix 3: vLLM native baseline — real-world reference throughput.
Runs vLLM with default settings on the same models and prompt lengths.
"""
from __future__ import annotations
import gc, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from bench_utils import save_json, RESULTS_DIR


def bench_vllm(model_name, prompt_len=1024, max_new=64, n_prompts=4):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, max_model_len=prompt_len + max_new + 128,
              dtype="float16", gpu_memory_utilization=0.90,
              enforce_eager=False)
    sp = SamplingParams(max_tokens=max_new, temperature=0)

    prompts = ["The transformer architecture uses self-attention to process sequences efficiently. " * (prompt_len // 12)] * n_prompts

    _ = llm.generate(prompts[:1], sp)

    import torch; torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    import torch; torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_input = sum(len(o.prompt_token_ids) for o in outputs)
    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tok = total_input + total_output
    throughput = total_tok / elapsed

    del llm
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "model": model_name,
        "n_prompts": n_prompts,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(throughput, 1),
    }


def main():
    models = [
        "Qwen/Qwen2.5-7B",
        "meta-llama/Llama-2-7b-hf",
    ]
    results = []

    for m in models:
        short = m.split("/")[-1]
        print(f"\n{'='*50}")
        print(f"  vLLM native: {short}")
        print(f"{'='*50}")

        try:
            r = bench_vllm(m, prompt_len=1024, max_new=64, n_prompts=4)
            results.append(r)
            print(f"  Throughput: {r['throughput_tok_s']:.1f} tok/s")
            print(f"  Input: {r['total_input_tokens']} tokens, Output: {r['total_output_tokens']} tokens")
            print(f"  Elapsed: {r['elapsed_s']:.2f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"model": m, "error": str(e)})

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    for r in results:
        if "error" in r:
            print(f"  {r['model']}: FAILED")
        else:
            print(f"  {r['model'].split('/')[-1]:<20s}  {r['throughput_tok_s']:>8.1f} tok/s")

    save_json(results, "exp_vllm_native_baseline")
    print(f"\nSaved to {RESULTS_DIR}/exp_vllm_native_baseline.json")


if __name__ == "__main__":
    main()

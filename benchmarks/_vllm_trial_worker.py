#!/usr/bin/env python3
"""Subprocess worker for isolated vLLM trial execution.

Called by exp_vllm_block_scoring.py with a JSON config on argv[1].
Prints JSON result on the last line of stdout.
"""
from __future__ import annotations

import gc
import json
import sys
import time


def main():
    cfg = json.loads(sys.argv[1])
    model = cfg["model"]
    gpu_util = cfg["gpu_util"]
    num_prompts = cfg["num_prompts"]
    max_tokens = cfg["max_tokens"]
    swap_space = cfg["swap_space"]
    prompt_len = cfg["prompt_len"]
    strategy = cfg["strategy"]

    try:
        import vllm.core.scheduler as sched_module
        from orchkv.vllm_integration.block_level_swap import (
            apply_scheduler_patch,
        )
        apply_scheduler_patch(sched_module.__file__)
    except Exception:
        pass

    import torch
    from vllm import LLM, SamplingParams

    max_model_len = prompt_len + max_tokens + 128
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_util,
        swap_space=swap_space,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="float16",
        trust_remote_code=True,
    )

    sp = SamplingParams(max_tokens=max_tokens, temperature=0)

    base_prompt = (
        "Explain the concept of attention mechanisms in transformer models "
        "and how they enable efficient sequence processing in modern large "
        "language model architectures for various applications. "
    )
    prompts = [base_prompt * max(1, prompt_len // len(base_prompt.split()))] * num_prompts

    _ = llm.generate(prompts[:2], sp)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_in = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = (total_in + total_out) / elapsed

    preempt_count = -1
    try:
        preempt_count = llm.llm_engine.scheduler[0].num_cumulative_preemption
    except Exception:
        pass

    scorer_stats = {}
    try:
        scorer = getattr(llm.llm_engine.scheduler[0],
                         '_orchkv_block_scorer', None)
        if scorer is not None:
            scorer_stats = scorer.get_stats()
    except Exception:
        pass

    result = {
        "strategy": strategy,
        "gpu_util": gpu_util,
        "num_prompts": num_prompts,
        "throughput": round(throughput, 1),
        "total_input": total_in,
        "total_output": total_out,
        "elapsed": round(elapsed, 3),
        "preemptions": preempt_count,
        "scorer_stats": scorer_stats,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps(result))


if __name__ == "__main__":
    main()

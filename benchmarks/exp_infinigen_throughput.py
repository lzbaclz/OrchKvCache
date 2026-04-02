#!/usr/bin/env python3
"""
InfiniGen vs FlexGen: End-to-End Throughput Comparison on OPT-1.3B

Runs FlexGen Original, InfiniGen, and H2O on OPT-1.3B to provide
throughput comparison under KV-cache offloading (weights on GPU, KV on CPU).

Prerequisites:
  - conda env "infinigen" with torch 2.0.x, numpy<2
  - InfiniGen repo at /home/lzq/codes/InfiniGen/speedup
  - FlexGen installed in the infinigen env

Usage:
  /home/lzq/miniconda3/envs/infinigen/bin/python benchmarks/exp_infinigen_throughput.py
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INFINIGEN_SPEEDUP_DIR = Path("/home/lzq/codes/InfiniGen/speedup")
FLEXGEN_DIR = INFINIGEN_SPEEDUP_DIR / "flexgen"
INFINIGEN_PYTHON = "/home/lzq/miniconda3/envs/infinigen/bin/python"


def switch_flexgen_scheme(scheme: str):
    """Switch FlexGen to use a specific variant (original/infinigen/h2o)."""
    target_dir = FLEXGEN_DIR / "flexgen"
    src_dir = FLEXGEN_DIR / scheme

    for fname in ["flex_opt.py", "pytorch_backend.py"]:
        target = target_dir / fname
        source = src_dir / fname
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
    print(f"  Switched FlexGen to scheme: {scheme}")


def parse_flexgen_output(output: str) -> dict:
    """Parse Total/Prefill/Decode from FlexGen output."""
    result = {}
    for line in output.split("\n"):
        m = re.search(r"Total:\s+([\d.]+)\s+Prefill:\s+([\d.]+)\s+Decode:\s+([\d.]+)", line)
        if m:
            result["total_s"] = float(m.group(1))
            result["prefill_s"] = float(m.group(2))
            result["decode_s"] = float(m.group(3))
    return result


def run_flexgen_opt(model: str, scheme: str, prompt_len: int, gen_len: int,
                    batch_size: int, extra_args: str = "") -> dict:
    """Run FlexGen with a given scheme and measure throughput."""
    switch_flexgen_scheme(scheme)

    warmup_path = FLEXGEN_DIR / "pg19_firstbook.txt"
    if not warmup_path.exists():
        warmup_path = INFINIGEN_SPEEDUP_DIR / "pg19_firstbook.txt"

    cmd = (
        f"cd {FLEXGEN_DIR} && {INFINIGEN_PYTHON} -u -m flexgen.flex_opt "
        f"--model {model} "
        f"--percent 100 0 0 100 100 0 "
        f"--overlap false "
        f"--gpu-batch-size {batch_size} "
        f"--num-gpu-batches 1 "
        f"--prompt-len {prompt_len} "
        f"--gen-len {gen_len} "
    )

    if warmup_path.exists():
        cmd += f"--warmup-input-path {warmup_path} --test-input-path {warmup_path} "

    if extra_args:
        cmd += extra_args

    print(f"  Running: {scheme} on {model} (bs={batch_size}, prompt={prompt_len}, gen={gen_len})")

    infinigen_pypath = str(INFINIGEN_SPEEDUP_DIR / "infinigen")
    existing_pypath = os.environ.get("PYTHONPATH", "")
    merged_pypath = f"{infinigen_pypath}:{existing_pypath}" if existing_pypath else infinigen_pypath

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": merged_pypath},
        )
        output = result.stdout + result.stderr
        parsed = parse_flexgen_output(output)

        gen_tokens = batch_size * gen_len
        throughput = gen_tokens / parsed["decode_s"] if parsed.get("decode_s") else None

        return {
            "scheme": scheme,
            "model": model,
            "batch_size": batch_size,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            **parsed,
            "gen_throughput_tok_s": round(throughput, 2) if throughput else None,
            "returncode": result.returncode,
            "output_last_10": output.strip().split("\n")[-10:],
        }
    except subprocess.TimeoutExpired:
        return {"scheme": scheme, "model": model, "error": "timeout"}
    except Exception as e:
        return {"scheme": scheme, "model": model, "error": str(e)}


def main():
    model = "facebook/opt-1.3b"
    gen_len = 128
    batch_size = 4
    prompt_lens = [512, 1024, 1536]

    all_results = []

    for prompt_len in prompt_lens:
        print(f"\n{'='*65}")
        print(f"  Config: {model}  prompt={prompt_len}  gen={gen_len}  bs={batch_size}")
        print(f"{'='*65}")

        config_results = {"config": f"OPT-1.3B, bs={batch_size}, prompt={prompt_len}, gen={gen_len}, KV-on-CPU",
                          "results": []}

        # 1. FlexGen original baseline
        print(f"\n>>> [1/3] FlexGen Original ...")
        r1 = run_flexgen_opt(model, "original", prompt_len, gen_len, batch_size)
        config_results["results"].append(r1)
        thr = r1.get("gen_throughput_tok_s")
        print(f"    {'OK: ' + str(thr) + ' tok/s' if thr else 'FAILED: ' + str(r1.get('output_last_10', [])[-3:])}")

        gc.collect()

        # 2. InfiniGen
        print(f"\n>>> [2/3] InfiniGen ...")
        r2 = run_flexgen_opt(
            model, "infinigen", prompt_len, gen_len, batch_size,
            extra_args="--alpha 4 --partial-weight-ratio 0.2 --max-num-kv 200 ",
        )
        config_results["results"].append(r2)
        thr = r2.get("gen_throughput_tok_s")
        print(f"    {'OK: ' + str(thr) + ' tok/s' if thr else 'FAILED: ' + str(r2.get('output_last_10', [])[-3:])}")

        gc.collect()

        # 3. H2O
        print(f"\n>>> [3/3] H2O ...")
        r3 = run_flexgen_opt(
            model, "h2o", prompt_len, gen_len, batch_size,
            extra_args="--max-num-kv 400 --hh-ratio 0.1 --hh-all ",
        )
        config_results["results"].append(r3)
        thr = r3.get("gen_throughput_tok_s")
        print(f"    {'OK: ' + str(thr) + ' tok/s' if thr else 'FAILED: ' + str(r3.get('output_last_10', [])[-3:])}")

        # Summary for this config
        baseline_decode = r1.get("decode_s", 0)
        print(f"\n  SUMMARY (prompt={prompt_len}):")
        for r in config_results["results"]:
            t = r.get("gen_throughput_tok_s")
            d = r.get("decode_s")
            if t and baseline_decode and d:
                speedup = baseline_decode / d
                r["speedup_vs_original"] = round(speedup, 2)
                print(f"    {r['scheme']:<15s}  {t:>7.1f} tok/s  {speedup:>5.2f}x")
            elif t:
                print(f"    {r['scheme']:<15s}  {t:>7.1f} tok/s")
            else:
                print(f"    {r['scheme']:<15s}  FAILED")

        all_results.append(config_results)

    out_path = RESULTS_DIR / "exp_infinigen_throughput.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

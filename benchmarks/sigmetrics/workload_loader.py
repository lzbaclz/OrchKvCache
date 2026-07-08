#!/usr/bin/env python3
"""
Workload loader for SIGMETRICS 2027 evaluation.

Loads and prepares prompts for each workload type:
  - ShareGPT:  multi-turn conversations from HuggingFace
  - LongBench: long-context QA + summarisation
  - RULER:     synthetic NIAH / multi-hop at controlled lengths
  - RAG:       multi-document retrieval prompts
  - Agentic:   multi-step tool-use agent traces

Each loader returns a list of dicts:
    {"prompt": str, "prompt_tokens": int, "expected_output_len": int, "metadata": {...}}

Usage:
    python -m benchmarks.sigmetrics.workload_loader --workload sharegpt --num 64 --stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

# ── path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "build" / "bindings"))
sys.path.insert(0, str(ROOT / "python"))

from benchmarks.sigmetrics.config import WORKLOADS


# ── tokeniser cache ───────────────────────────────────────────────────

_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model_name: str = "Qwen/Qwen2.5-7B"):
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
    return _TOKENIZER_CACHE[model_name]


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return len(text.split()) * 4 // 3      # rough estimate
    return len(tokenizer.encode(text, add_special_tokens=False))


# =====================================================================
#  ShareGPT
# =====================================================================

def load_sharegpt(
    num_prompts: int = 64,
    max_prompt_tokens: int = 2048,
    tokenizer_name: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load multi-turn conversations from ShareGPT via HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        _missing("datasets", "pip install datasets")
        return []

    ds_name = WORKLOADS["sharegpt"]["hf_dataset"]
    try:
        ds = load_dataset(ds_name, split="train")
    except Exception as exc:
        print(f"[workload] Cannot load {ds_name}: {exc}")
        print(f"[workload] Download with: python -c \"from datasets import load_dataset; "
              f"load_dataset('{ds_name}')\"")
        return []

    tokenizer = _get_tokenizer(tokenizer_name) if tokenizer_name else None
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    for idx in indices:
        if len(prompts) >= num_prompts:
            break
        conv = ds[idx].get("conversations", [])
        if not conv or len(conv) < 2:
            continue

        user_turns = [t["value"] for t in conv if t.get("from") == "human"]
        assistant_turns = [t["value"] for t in conv if t.get("from") == "gpt"]
        if not user_turns:
            continue

        prompt_text = "\n\n".join(user_turns)
        n_tok = _count_tokens(prompt_text, tokenizer)
        if n_tok > max_prompt_tokens:
            words = prompt_text.split()
            prompt_text = " ".join(words[:int(max_prompt_tokens * 0.75)])
            n_tok = _count_tokens(prompt_text, tokenizer)

        expected_out = max(64, min(512, sum(len(a.split()) for a in assistant_turns)))
        prompts.append({
            "prompt": prompt_text,
            "prompt_tokens": n_tok,
            "expected_output_len": expected_out,
            "metadata": {"source": "sharegpt", "turns": len(user_turns), "idx": idx},
        })

    print(f"[workload] sharegpt: loaded {len(prompts)}/{num_prompts} prompts")
    return prompts


# =====================================================================
#  LongBench
# =====================================================================

def load_longbench(
    num_prompts: int = 64,
    subsets: list[str] | None = None,
    tokenizer_name: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load long-context tasks from THUDM/LongBench."""
    try:
        from datasets import load_dataset
    except ImportError:
        _missing("datasets", "pip install datasets")
        return []

    subsets = subsets or WORKLOADS["longbench"]["subsets"]
    tokenizer = _get_tokenizer(tokenizer_name) if tokenizer_name else None
    rng = random.Random(seed)
    prompts: list[dict] = []

    for subset in subsets:
        try:
            ds = load_dataset("THUDM/LongBench", subset, split="test")
        except Exception as exc:
            print(f"[workload] Cannot load LongBench/{subset}: {exc}")
            print(f"[workload] Download with: python -c \"from datasets import load_dataset; "
                  f"load_dataset('THUDM/LongBench', '{subset}')\"")
            continue

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        per_subset = max(1, num_prompts // len(subsets))

        for idx in indices[:per_subset]:
            row = ds[idx]
            context = row.get("context", "")
            question = row.get("input", "")
            prompt_text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

            n_tok = _count_tokens(prompt_text, tokenizer)
            prompts.append({
                "prompt": prompt_text,
                "prompt_tokens": n_tok,
                "expected_output_len": 128,
                "metadata": {"source": "longbench", "subset": subset, "idx": idx},
            })

    rng.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"[workload] longbench: loaded {len(prompts)}/{num_prompts} prompts")
    return prompts


# =====================================================================
#  RULER – Synthetic needle-in-a-haystack and multi-hop
# =====================================================================

_RULER_HAYSTACK_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
)

def _generate_ruler_niah_single(
    seq_len: int, seed: int = 42,
) -> dict:
    """Generate a single-needle NIAH task at the given sequence length."""
    rng = random.Random(seed)
    needle_key = f"NEEDLE-{rng.randint(1000, 9999)}"
    needle_value = f"answer-{rng.randint(100, 999)}"
    needle = f"The secret key for {needle_key} is {needle_value}."

    target_words = seq_len * 3 // 4
    haystack_words = _RULER_HAYSTACK_SENTENCE.split()
    n_repeats = max(1, target_words // len(haystack_words))
    hay_text = " ".join(haystack_words * n_repeats)

    insert_pos = rng.randint(len(hay_text) // 4, 3 * len(hay_text) // 4)
    context = hay_text[:insert_pos] + " " + needle + " " + hay_text[insert_pos:]
    question = f"What is the secret key for {needle_key}?"
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    return {
        "prompt": prompt,
        "prompt_tokens": len(prompt.split()) * 4 // 3,
        "expected_output_len": 16,
        "metadata": {
            "source": "ruler", "task": "niah_single",
            "seq_len": seq_len, "needle_value": needle_value,
            "needle_depth": insert_pos / max(len(hay_text), 1),
        },
    }


def _generate_ruler_niah_multi(
    seq_len: int, n_needles: int = 3, seed: int = 42,
) -> dict:
    """Generate a multi-needle NIAH task."""
    rng = random.Random(seed)
    needles = {}
    for i in range(n_needles):
        key = f"KEY-{rng.randint(1000, 9999)}-{i}"
        val = f"val-{rng.randint(100, 999)}"
        needles[key] = val

    target_words = seq_len * 3 // 4
    haystack_words = _RULER_HAYSTACK_SENTENCE.split()
    hay_text = " ".join(haystack_words * max(1, target_words // len(haystack_words)))

    segments = sorted(rng.sample(range(len(hay_text) // 4, 3 * len(hay_text) // 4),
                                  min(n_needles, len(hay_text) // 4)))
    offset = 0
    for seg, (k, v) in zip(segments, needles.items()):
        sentence = f" The value for {k} is {v}. "
        pos = seg + offset
        hay_text = hay_text[:pos] + sentence + hay_text[pos:]
        offset += len(sentence)

    ask_key = rng.choice(list(needles.keys()))
    prompt = f"{hay_text}\n\nQuestion: What is the value for {ask_key}?\nAnswer:"

    return {
        "prompt": prompt,
        "prompt_tokens": len(prompt.split()) * 4 // 3,
        "expected_output_len": 16,
        "metadata": {
            "source": "ruler", "task": "niah_multi",
            "seq_len": seq_len, "n_needles": n_needles,
            "expected": needles[ask_key],
        },
    }


def _generate_ruler_multi_hop(
    seq_len: int, hops: int = 3, seed: int = 42,
) -> dict:
    """Generate a multi-hop reasoning chain at the given length."""
    rng = random.Random(seed)
    entities = [f"Entity-{rng.randint(100, 999)}" for _ in range(hops + 1)]
    relations = [f"connects-to" for _ in range(hops)]
    facts = [f"{entities[i]} {relations[i]} {entities[i+1]}." for i in range(hops)]

    target_words = seq_len * 3 // 4
    haystack_words = _RULER_HAYSTACK_SENTENCE.split()
    hay_text = " ".join(haystack_words * max(1, target_words // len(haystack_words)))

    for fact in facts:
        pos = rng.randint(0, max(1, len(hay_text) - 1))
        hay_text = hay_text[:pos] + f" {fact} " + hay_text[pos:]

    question = (f"Starting from {entities[0]}, following the 'connects-to' chain "
                f"{hops} times, what entity do you reach?")
    prompt = f"{hay_text}\n\nQuestion: {question}\nAnswer:"

    return {
        "prompt": prompt,
        "prompt_tokens": len(prompt.split()) * 4 // 3,
        "expected_output_len": 32,
        "metadata": {
            "source": "ruler", "task": "multi_hop",
            "seq_len": seq_len, "hops": hops,
            "expected": entities[-1],
        },
    }


def load_ruler(
    num_prompts: int = 64,
    lengths: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic RULER tasks at controlled lengths."""
    lengths = lengths or WORKLOADS["ruler"]["lengths"]
    tasks = WORKLOADS["ruler"]["tasks"]
    rng = random.Random(seed)

    prompts = []
    per_combo = max(1, num_prompts // (len(lengths) * len(tasks)))
    for seq_len in lengths:
        for task in tasks:
            for i in range(per_combo):
                s = seed + hash((seq_len, task, i)) % (2**31)
                if task == "niah_single":
                    prompts.append(_generate_ruler_niah_single(seq_len, seed=s))
                elif task == "niah_multi":
                    prompts.append(_generate_ruler_niah_multi(seq_len, seed=s))
                elif task == "multi_hop":
                    prompts.append(_generate_ruler_multi_hop(seq_len, seed=s))

    rng.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"[workload] ruler: generated {len(prompts)}/{num_prompts} prompts")
    return prompts


# =====================================================================
#  RAG – Multi-document retrieval prompts
# =====================================================================

_RAG_DOC_TEMPLATE = (
    "Document {idx} (Title: {title}):\n{body}\n"
)

_RAG_TOPICS = [
    "quantum computing", "climate change mitigation", "transformer architectures",
    "protein folding", "urban planning", "renewable energy storage",
    "autonomous vehicles", "large language models", "supply chain optimization",
    "semiconductor fabrication", "gene therapy", "space exploration",
]


def load_rag(
    num_prompts: int = 64,
    n_documents: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate multi-document RAG prompts."""
    n_documents = n_documents or WORKLOADS["rag"]["n_documents"]
    doc_len_range = WORKLOADS["rag"]["doc_length"]
    rng = random.Random(seed)

    prompts = []
    per_ndoc = max(1, num_prompts // len(n_documents))

    for ndoc in n_documents:
        for i in range(per_ndoc):
            topic = rng.choice(_RAG_TOPICS)
            docs = []
            for d in range(ndoc):
                doc_words = rng.randint(doc_len_range[0] // 4, doc_len_range[1] // 4)
                body = " ".join(
                    rng.choice(_RULER_HAYSTACK_SENTENCE.split())
                    for _ in range(doc_words)
                )
                answer_fact = f"The key finding about {topic} is result-{rng.randint(100, 999)}."
                if d == rng.randint(0, ndoc - 1):
                    body = body + " " + answer_fact

                docs.append(_RAG_DOC_TEMPLATE.format(
                    idx=d + 1,
                    title=f"{topic} - source {d + 1}",
                    body=body,
                ))

            context = "\n".join(docs)
            question = f"Based on the documents above, what is the key finding about {topic}?"
            prompt_text = f"{context}\nQuestion: {question}\nAnswer:"

            prompts.append({
                "prompt": prompt_text,
                "prompt_tokens": len(prompt_text.split()) * 4 // 3,
                "expected_output_len": 128,
                "metadata": {
                    "source": "rag", "n_documents": ndoc,
                    "topic": topic, "idx": i,
                },
            })

    rng.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"[workload] rag: generated {len(prompts)}/{num_prompts} prompts")
    return prompts


# =====================================================================
#  Agentic – Multi-step tool-use agent traces
# =====================================================================

_AGENT_TOOLS = [
    "search(query)", "calculate(expression)", "lookup_database(table, key)",
    "read_file(path)", "write_file(path, content)", "api_call(endpoint, params)",
]

_AGENT_THOUGHT_TEMPLATE = (
    "Step {step}: I need to {action}.\n"
    "Tool call: {tool}\n"
    "Result: {result}\n"
    "Observation: The result shows {observation}.\n\n"
)


def load_agentic(
    num_prompts: int = 64,
    n_turns: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate multi-step agentic traces with interleaved reasoning."""
    n_turns = n_turns or WORKLOADS["agentic"]["n_turns"]
    ctx_range = WORKLOADS["agentic"]["context_per_turn"]
    rng = random.Random(seed)

    prompts = []
    per_nturn = max(1, num_prompts // len(n_turns))

    for nt in n_turns:
        for i in range(per_nturn):
            system = "You are a helpful assistant with access to tools.\n\n"
            trace = ""
            for step in range(nt):
                tool = rng.choice(_AGENT_TOOLS)
                ctx_words = rng.randint(ctx_range[0] // 4, ctx_range[1] // 4)
                result = " ".join(
                    rng.choice(_RULER_HAYSTACK_SENTENCE.split())
                    for _ in range(ctx_words)
                )
                trace += _AGENT_THOUGHT_TEMPLATE.format(
                    step=step + 1,
                    action=f"use {tool.split('(')[0]} to gather information",
                    tool=tool,
                    result=result,
                    observation=f"data point {rng.randint(1, 100)} is relevant",
                )

            prompt_text = system + trace + "Based on all the above, provide a final answer:\n"
            prompts.append({
                "prompt": prompt_text,
                "prompt_tokens": len(prompt_text.split()) * 4 // 3,
                "expected_output_len": 256,
                "metadata": {"source": "agentic", "n_turns": nt, "idx": i},
            })

    rng.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"[workload] agentic: generated {len(prompts)}/{num_prompts} prompts")
    return prompts


# =====================================================================
#  Unified loader + statistics
# =====================================================================

LOADERS = {
    "sharegpt": load_sharegpt,
    "longbench": load_longbench,
    "ruler": load_ruler,
    "rag": load_rag,
    "agentic": load_agentic,
}


def load_workload(
    name: str,
    num_prompts: int = 64,
    tokenizer_name: str | None = None,
    seed: int = 42,
    **kwargs,
) -> list[dict]:
    """Load a workload by name. Extra kwargs forwarded to the specific loader."""
    if name not in LOADERS:
        raise ValueError(f"Unknown workload '{name}'. Choose from: {list(LOADERS)}")
    loader = LOADERS[name]
    kw = {"num_prompts": num_prompts, "seed": seed}
    if tokenizer_name and name in ("sharegpt", "longbench"):
        kw["tokenizer_name"] = tokenizer_name
    kw.update(kwargs)
    return loader(**kw)


def workload_stats(prompts: list[dict]) -> dict:
    """Compute length CDF and distribution stats for a loaded workload."""
    if not prompts:
        return {"count": 0}

    lengths = sorted(p["prompt_tokens"] for p in prompts)
    out_lens = sorted(p["expected_output_len"] for p in prompts)

    cdf_points = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    length_cdf = {}
    for q in cdf_points:
        idx = min(int(q * len(lengths)), len(lengths) - 1)
        length_cdf[f"p{int(q*100)}"] = lengths[idx]

    return {
        "count": len(prompts),
        "prompt_tokens": {
            "min": lengths[0],
            "max": lengths[-1],
            "mean": round(statistics.mean(lengths), 1),
            "median": lengths[len(lengths) // 2],
            "stdev": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
            "cdf": length_cdf,
        },
        "output_len": {
            "min": out_lens[0],
            "max": out_lens[-1],
            "mean": round(statistics.mean(out_lens), 1),
        },
        "sources": dict(
            sorted(
                _count_by(prompts, lambda p: p["metadata"].get("source", "?")).items()
            )
        ),
    }


def _count_by(items: list[dict], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _missing(package: str, install_cmd: str):
    print(f"[workload] Missing '{package}'. Install with: {install_cmd}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect SIGMETRICS workloads")
    parser.add_argument("--workload", type=str, required=True,
                        choices=list(LOADERS), help="Workload name")
    parser.add_argument("--num", type=int, default=32, help="Number of prompts")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HF tokenizer for accurate token counts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats", action="store_true",
                        help="Print distribution statistics")
    parser.add_argument("--save", type=str, default=None,
                        help="Save prompts to JSON file")
    args = parser.parse_args()

    prompts = load_workload(args.workload, args.num,
                            tokenizer_name=args.tokenizer, seed=args.seed)

    if args.stats:
        stats = workload_stats(prompts)
        print(json.dumps(stats, indent=2))

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"[workload] Saved {len(prompts)} prompts to {out_path}")

    if not args.stats and not args.save:
        for i, p in enumerate(prompts[:5]):
            print(f"\n--- Prompt {i+1} ({p['prompt_tokens']} tokens) ---")
            print(p["prompt"][:300] + "..." if len(p["prompt"]) > 300 else p["prompt"])


if __name__ == "__main__":
    main()

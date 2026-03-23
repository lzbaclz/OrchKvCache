#!/usr/bin/env python3
"""
Analyze experiment results and generate a structured report.

Reads JSON from benchmarks/results/ and outputs:
  - Console summary of key findings
  - benchmarks/results/analysis_report.json  (machine-readable)
  - benchmarks/results/analysis_report.md    (human-readable)

Usage:
    python benchmarks/analyze_results.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

def load(name):
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def analyze_e5():
    data = load("benchmark_e5_policy_sweep")
    if not data:
        return None

    findings = {}

    fixed = [r for r in data if r["pattern"] == "fixed"]
    best = max(fixed, key=lambda r: r["avg_n_cold"])
    worst = min(fixed, key=lambda r: r["avg_n_cold"])

    findings["best_config_fixed"] = {
        "alpha": best["alpha"], "beta": best["beta"], "gamma": best["gamma"],
        "hot": best["avg_n_hot"], "warm": best["avg_n_warm"],
        "cold": best["avg_n_cold"],
    }
    findings["worst_config_fixed"] = {
        "alpha": worst["alpha"], "beta": worst["beta"], "gamma": worst["gamma"],
        "hot": worst["avg_n_hot"], "warm": worst["avg_n_warm"],
        "cold": worst["avg_n_cold"],
    }

    for pattern in ["fixed", "shift", "zipf"]:
        subset = [r for r in data if r["pattern"] == pattern]
        high_alpha = [r for r in subset if r["alpha"] >= 0.7]
        low_alpha = [r for r in subset if r["alpha"] <= 0.3]
        findings[f"high_alpha_avg_hot_{pattern}"] = round(
            sum(r["avg_n_hot"] for r in high_alpha) / max(len(high_alpha), 1), 1)
        findings[f"low_alpha_avg_hot_{pattern}"] = round(
            sum(r["avg_n_hot"] for r in low_alpha) / max(len(low_alpha), 1), 1)

    findings["n_configs"] = len(data)
    findings["n_patterns"] = len(set(r["pattern"] for r in data))
    return findings


def analyze_e7():
    data = load("benchmark_e7_prefetch")
    if not data:
        return None

    ok = [r for r in data if r.get("status") != "skip"]
    if not ok:
        return None

    findings = {}
    findings["budget_vs_dispatches"] = [
        {"budget": r["prefetch_budget"],
         "dispatched": r["avg_prefetches_dispatched"],
         "sched_us": r["avg_schedule_us"]}
        for r in ok
    ]

    if len(ok) >= 2:
        max_dispatched = max(r["avg_prefetches_dispatched"] for r in ok)
        findings["saturation_budget"] = next(
            (r["prefetch_budget"] for r in ok
             if r["avg_prefetches_dispatched"] >= max_dispatched * 0.95), None)

    findings["sched_overhead_range_us"] = {
        "min": min(r["avg_schedule_us"] for r in ok),
        "max": max(r["avg_schedule_us"] for r in ok),
    }
    return findings


def analyze_e8():
    data = load("benchmark_e8_storage_bw")
    if not data:
        return None

    findings = {}

    gd = data.get("gpu_dram", [])
    if gd:
        peak_d2h = max(r["d2h_gbps"] for r in gd)
        peak_h2d = max(r["h2d_gbps"] for r in gd)
        peak_d2h_size = next(r["size_mb"] for r in gd if r["d2h_gbps"] == peak_d2h)
        peak_h2d_size = next(r["size_mb"] for r in gd if r["h2d_gbps"] == peak_h2d)
        findings["gpu_dram"] = {
            "peak_d2h_gbps": peak_d2h,
            "peak_d2h_at_mb": peak_d2h_size,
            "peak_h2d_gbps": peak_h2d,
            "peak_h2d_at_mb": peak_h2d_size,
            "asymmetry_ratio": round(peak_h2d / max(peak_d2h, 0.01), 2),
        }

    ds = data.get("dram_storage", [])
    if ds:
        peak_w = max(r["write_gbps"] for r in ds)
        peak_r = max(r["read_gbps"] for r in ds)
        findings["dram_tmpfs"] = {
            "peak_write_gbps": peak_w,
            "peak_read_gbps": peak_r,
            "read_write_ratio": round(peak_r / max(peak_w, 0.01), 2),
        }

    if gd and ds:
        tier_gap = max(r["h2d_gbps"] for r in gd) / max(r["read_gbps"] for r in ds)
        findings["tier_gap_ratio"] = round(tier_gap, 1)

    return findings


def analyze_e9():
    data = load("benchmark_e9_scalability")
    if not data:
        return None

    ok = [r for r in data if r.get("status") != "skip"]
    if not ok:
        return None

    findings = {}

    small = ok[0]
    large = ok[-1]
    scale_factor = large["n_blocks"] / small["n_blocks"]
    latency_factor = large["avg_schedule_us"] / max(small["avg_schedule_us"], 0.01)

    findings["scaling"] = {
        "block_range": f"{small['n_blocks']}→{large['n_blocks']}",
        "scale_factor": round(scale_factor, 1),
        "latency_factor": round(latency_factor, 1),
        "scaling_exponent": round(
            statistics.log2(latency_factor) / statistics.log2(scale_factor)
            if scale_factor > 1 else 0, 3),
    }

    findings["at_4096_blocks"] = {
        "avg_us": large["avg_schedule_us"],
        "p99_us": large["p99_schedule_us"],
        "within_100us": large["p99_schedule_us"] < 100,
    }

    findings["per_block_ns"] = round(
        large["avg_schedule_us"] * 1000 / large["n_blocks"], 2)

    return findings


def generate_report():
    report = {}
    sections = {
        "E5_policy_sweep": analyze_e5,
        "E7_prefetch": analyze_e7,
        "E8_storage_bandwidth": analyze_e8,
        "E9_scalability": analyze_e9,
    }

    print("=" * 65)
    print("  OrchKvCache — Experiment Analysis Report")
    print("=" * 65)

    for name, analyzer in sections.items():
        result = analyzer()
        if result:
            report[name] = result
            print(f"\n  [{name}]")
            for k, v in result.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        print(f"      {k2}: {v2}")
                elif isinstance(v, list):
                    print(f"    {k}: [{len(v)} entries]")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"\n  [{name}] — no data available")

    # Save JSON
    json_path = RESULTS / "analysis_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved → {json_path}")

    # Save Markdown
    md_lines = ["# OrchKvCache Experiment Analysis Report\n"]

    if "E5_policy_sweep" in report:
        e5 = report["E5_policy_sweep"]
        md_lines.append("## E5: Hot/Cold Policy Sweep\n")
        best = e5["best_config_fixed"]
        md_lines.append(f"- **Best config (fixed pattern)**: "
                         f"α={best['alpha']} β={best['beta']} γ={best['gamma']} "
                         f"→ Hot={best['hot']:.0f}, Warm={best['warm']:.0f}, "
                         f"Cold={best['cold']:.0f}\n")
        md_lines.append(f"- **High α (≥0.7) avg hot (fixed)**: "
                         f"{e5.get('high_alpha_avg_hot_fixed', 'N/A')}\n")
        md_lines.append(f"- **Low α (≤0.3) avg hot (fixed)**: "
                         f"{e5.get('low_alpha_avg_hot_fixed', 'N/A')}\n")
        md_lines.append(f"- Total configs tested: {e5['n_configs']}\n")

    if "E7_prefetch" in report:
        e7 = report["E7_prefetch"]
        md_lines.append("\n## E7: Prefetch Effectiveness\n")
        if "saturation_budget" in e7:
            md_lines.append(f"- **Saturation budget**: {e7['saturation_budget']}\n")
        oh = e7["sched_overhead_range_us"]
        md_lines.append(f"- **Scheduling overhead**: "
                         f"{oh['min']:.1f}~{oh['max']:.1f} μs\n")

    if "E8_storage_bandwidth" in report:
        e8 = report["E8_storage_bandwidth"]
        md_lines.append("\n## E8: Storage Bandwidth\n")
        if "gpu_dram" in e8:
            gd = e8["gpu_dram"]
            md_lines.append(f"- **GPU↔DRAM**: D2H={gd['peak_d2h_gbps']}GB/s, "
                             f"H2D={gd['peak_h2d_gbps']}GB/s\n")
        if "dram_tmpfs" in e8:
            ds = e8["dram_tmpfs"]
            md_lines.append(f"- **DRAM↔tmpfs**: "
                             f"Write={ds['peak_write_gbps']}GB/s, "
                             f"Read={ds['peak_read_gbps']}GB/s\n")
        if "tier_gap_ratio" in e8:
            md_lines.append(f"- **Tier gap**: GPU↔DRAM is "
                             f"{e8['tier_gap_ratio']}× faster than "
                             f"DRAM↔tmpfs\n")

    if "E9_scalability" in report:
        e9 = report["E9_scalability"]
        md_lines.append("\n## E9: Scheduling Scalability\n")
        sc = e9["scaling"]
        md_lines.append(f"- **Range**: {sc['block_range']} "
                         f"({sc['scale_factor']}× blocks → "
                         f"{sc['latency_factor']}× latency)\n")
        md_lines.append(f"- **Scaling exponent**: {sc['scaling_exponent']} "
                         f"(1.0 = linear)\n")
        a4k = e9["at_4096_blocks"]
        md_lines.append(f"- **@4096 blocks**: avg={a4k['avg_us']}μs, "
                         f"p99={a4k['p99_us']}μs "
                         f"({'PASS' if a4k['within_100us'] else 'FAIL'} "
                         f"< 100μs)\n")
        md_lines.append(f"- **Per-block cost**: {e9['per_block_ns']}ns\n")

    md_path = RESULTS / "analysis_report.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)
    print(f"  Saved → {md_path}")

    return report


if __name__ == "__main__":
    import math
    statistics.log2 = lambda x: math.log2(x)
    generate_report()

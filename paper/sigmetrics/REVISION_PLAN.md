# OrchKvCache → SIGMETRICS 2027 Fall Revision Plan

**Target**: ACM SIGMETRICS 2027, Fall deadline (Oct 9, 2026 AoE)
**Branch**: `submission/sigmetrics`

---

## Executive Summary

Transform OrchKvCache from an "attention-aware KV-cache offloading system" (SC framing)
into a **measurement-driven lossless KV residency scheduler** (SIGMETRICS framing).

**New one-sentence pitch**:
> OrchKvCache shows that lossless KV offloading is not about dropping low-attention
> tokens; it is about predicting block reuse distance and moving KV blocks only when
> promotion latency can be hidden off the token-generation critical path.

---

## 1. Core Repositioning

### OLD claim (SC, rejected):
"Existing systems either FIFO/static/lossy → we do attention-aware hotness → 139-597× fewer migrations"

### NEW claim (SIGMETRICS):
"We measure KV block reuse distance under real workloads, model the critical-path
condition for lossless offloading, and build a residency scheduler that only moves
blocks when promotion can be hidden — achieving [X] goodput improvement under memory
pressure while preserving bit-exact outputs."

### Four contributions:
1. **C1: Workload Characterization** — measure attention concentration, reuse distance,
   hot-set stability, promotion criticality across models and workloads
2. **C2: Critical-Path Residency Model** — Offload(b,tier) iff R̂(b) > L_promote(tier)/step_time + slack
3. **C3: Low-Overhead Signal Acquisition** — QK proxy + periodic sampling + dynamic bypass
4. **C4: Production-Compatible Residency Scheduler** — OrchKvCache as vLLM/LMCache-compatible
   block-level GPU/DRAM/SSD placement policy

---

## 2. P0 Code Fixes (DONE)

| Fix | File | Status |
|-----|------|--------|
| tm_register_block_id parameter order | kvcache_manager.py:161 | ✅ Fixed |
| Eviction uses hotness score not access_count | kvcache_manager.py:_evict_cold_blocks | ✅ Fixed |
| Config alignment (α=0.7,β=0.2,γ=0.1) | kvcache_manager.py:_init_tiered_manager | ✅ Fixed |
| Async DMA (non_blocking=True + sync) | kvcache_manager.py:_demote/_promote | ✅ Fixed |
| Promotion latency instrumentation | kvcache_manager.py:_record_promotion_latency | ✅ Added |
| Per-block decision log | kvcache_manager.py:_record_decision | ✅ Added |
| Promote uses hotness score | kvcache_manager.py:_promote_warm_blocks | ✅ Fixed |
| SSD spill uses hotness score | kvcache_manager.py:_spill_dram_to_ssd | ✅ Fixed |

---

## 3. New Modules

| Module | Path | Purpose |
|--------|------|---------|
| Reuse-distance predictor | python/orchkv/reuse_distance.py | C1+C2: predict R̂(b), model critical path |
| Baseline harness | benchmarks/sigmetrics/ | Full experiment infrastructure |
| SIGMETRICS paper | paper/sigmetrics/main.tex | 20-page acmsmall format |

---

## 4. Experiment Matrix

### 4.1 Models
- Qwen2.5-7B (GQA, 7B, primary)
- Llama-3.1-8B (GQA, 128K context)
- LLaMA-2-7B (MHA, legacy comparison)
- Mistral-7B-v0.3 (GQA, sliding window)

### 4.2 Workloads
- ShareGPT (real conversation traces)
- LongBench (multi-doc QA, summarization)
- RULER (synthetic NIAH, multi-hop, controlled length)
- RAG (multi-document retrieval)
- AgentBench-derived (multi-turn tool-call)

### 4.3 Baselines
- vLLM native (continuous batching)
- vLLM CPU offload / swap
- LMCache + vLLM
- FlexGen (offline three-tier)
- InfiniGen-style predictor (trace-level)
- FIFO / LRU / LFU / Belady (policy simulation)
- OrchKvCache (full/sampling/QK-proxy)

### 4.4 Memory Budget Sweep
- 5%, 10%, 25%, 50%, 75% of available KV capacity

### 4.5 Metrics
| Category | Metrics |
|----------|---------|
| Serving | TTFT, TPOT/ITL, goodput@SLO, throughput, P50/P95/P99 |
| Memory | GPU used, DRAM used, SSD traffic, max context/concurrency |
| Migration | evictions, promotions, bytes moved, amplification |
| Critical path | stall count, stall time P50/P95/P99, SSD critical-path rate |
| Signal overhead | full-attn overhead, sampling overhead, QK-proxy overhead |
| Quality | bit-exact token match, logit max-diff, LongBench/RULER score |
| Robustness | sensitivity to α/β/γ/λ/τ, confidence intervals |

---

## 5. Timeline (98 days to Oct 9)

### Phase 1: Jul 8–14 — Code correctness + infrastructure
- [x] P0 fixes
- [ ] Reuse-distance module
- [ ] Baseline harness skeleton
- [ ] SIGMETRICS template
- [ ] Build & verify all tests pass

### Phase 2: Jul 15–28 — Baseline integration
- [ ] vLLM native baseline running
- [ ] vLLM CPU offload baseline
- [ ] LMCache integration attempt
- [ ] FlexGen reproduction
- [ ] InfiniGen-style trace predictor
- [ ] Unified workload runner verified

### Phase 3: Jul 29–Aug 11 — Attention overhead + proxy
- [ ] Full attention oracle instrumented
- [ ] N-step sampling (N=5/10/20)
- [ ] QK/page-stat proxy implementation
- [ ] Dynamic bypass logic
- [ ] TTFT/TPOT overhead decomposition
- [ ] Fig 5 data collected

### Phase 4: Aug 12–25 — Measurement + model
- [ ] Reuse-distance CDF (all workloads × models)
- [ ] Hot-set Jaccard stability
- [ ] Promotion latency measurement
- [ ] Overlap slack analysis
- [ ] Critical-path model validation
- [ ] Fig 2/3/4 data collected

### Phase 5: Aug 26–Sep 8 — Production integration
- [ ] vLLM continuous batching goodput
- [ ] SSD backend ablation (POSIX/direct/GDS-sim)
- [ ] Memory pressure sweep (5-75%)
- [ ] Table 2 + Fig 7/8/9

### Phase 6: Sep 9–22 — Large-scale validation
- [ ] 1000+ prompt correctness suite
- [ ] LongBench + RULER evaluation
- [ ] RAG + agentic workload
- [ ] Model sweep (all 4 models)

### Phase 7: Sep 23–Oct 1 — Writing + polish
- [ ] Full 20-page draft complete
- [ ] All figures finalized
- [ ] Internal review (3 reviewers)
- [ ] Artifact reproducibility statement

### Phase 8: Oct 2–9 — Submission
- [ ] Abstract registration (Oct 2)
- [ ] Final polish
- [ ] Anonymous compliance
- [ ] Submit (Oct 9)

---

## 6. Go/No-Go Gates

### Gate 1 (Jul 28): Baselines running?
If LMCache or vLLM baseline not working → consider Winter deadline.

### Gate 2 (Aug 11): Attention overhead acceptable?
If QK/sampling can't reduce overhead to <10% → change main claim to "policy insight".

### Gate 3 (Sep 8): Production gain exists?
If no positive metric under vLLM → pivot to memory-pressure/OOM-avoidance framing.

---

## 7. Key Design Decisions

### EMA Formula (unified):
```
ema_t = λ · ema_{t-1} + (1-λ) · raw_t
```
Where λ is the **retention/decay factor** (default 0.9 = 90% old + 10% new).
This matches the C code semantics. Paper formula must use this convention.

### Block Geometry:
- C core: 64 tokens/head, single KV-head → 32KB
- Python prototype: 16 tokens, all KV-heads → 32KB (for GQA)
- Paper reports block_size in tokens and notes the two implementations

### Three-Path Claim Separation:
1. **C/CUDA microbenchmark**: proves scheduling overhead <60μs, async DMA works
2. **Python prototype**: proves policy quality, lossless integrity, measurement
3. **vLLM integration**: proves production compatibility, reports serving metrics

Never conflate claims across paths.

---

## 8. SOTA Coverage (must discuss)

| Work | Relationship | Treatment |
|------|-------------|-----------|
| LMCache | Cross-query KV layer | Complementary; we do intra-request residency |
| InfiniGen | Cross-layer prefetch | Compare predictor quality; ours adds SSD tier |
| FlexGen | Offline 3-tier | We do online; FlexGen as static baseline |
| Tutti | GPU-centric SSD KV store | Discuss; compare if code available |
| KVDrive | Multi-tier KV management | Closest competitor; differentiate via model |
| FlexiCache | Head temporal stability | Our signal is finer-grained (block-level) |
| Quest | QK page selection | Borrow proxy idea; ours outputs residency not sparsity |
| H2O/ScissorHands | Lossy eviction | We preserve all KV; quality-throughput comparison |
| BaM/GDS/DUAL-BLADE | GPU-SSD direct | Storage backend is pluggable; ablation |
| KIVI/TurboQuant | KV compression | Complementary; residency + compression stackable |

---

## 9. Minimum Acceptance Criteria

1. No false novelty claim ("first attention-aware KV cache")
2. At least vLLM native + LMCache + FlexGen + InfiniGen compared
3. Main path NOT full eager attention
4. Reuse-distance + critical-path model present
5. TTFT/TPOT/P99/goodput reported (not just throughput)
6. Real workloads: LongBench + RULER + ShareGPT
7. 1000+ prompt lossless verification
8. Paper-code parameter consistency verified

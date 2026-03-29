# OrchKvCache: Heterogeneous Storage-Orchestrated Tiered KV-Cache Management for Efficient LLM Inference

> **Target venue: SC 2025 — The International Conference for High Performance Computing, Networking, Storage, and Analysis**
> **Format: 10 pages + references (IEEE format)**

---

## Abstract

The Key-Value (KV) cache is the dominant memory bottleneck in large language model (LLM) inference: it grows linearly with context length, and a single LLaMA-2-7B request at 128K tokens requires 64 GB—4.6× the model's own weight. At 32K context, a batch of just four requests already saturates an 80 GB A100 GPU. Existing KV-cache management systems fall into three categories, each with fundamental drawbacks: (i) *hotness-agnostic swapping* (vLLM) treats all blocks identically, evicting via FIFO between GPU and CPU without exploiting the extreme skew in attention access patterns; (ii) *offline, coarse-grained offloading* (FlexGen) moves entire layers between GPU, CPU, and disk using a pre-computed plan that cannot adapt at runtime; and (iii) *lossy eviction* (H2O, ScissorHands) permanently discards cold tokens, trading model quality for memory savings.

We present **OrchKvCache**, a runtime system that manages KV blocks across GPU HBM, host DRAM, and SSD, driven by online attention-based hotness classification. OrchKvCache contributes three techniques. **(1) Attention-driven hot-cold classification.** An EMA-smoothed, multi-signal scoring function with watermark-adaptive thresholds classifies every KV block into Hot, Warm, or Cold categories each decode step. Our profiling on Qwen2.5-1.5B confirms the premise: the top 10\% of tokens concentrate 90–97\% of attention weight (Gini 0.87–0.97), while initial "sink" tokens absorb up to 77\% per layer. **(2) Multi-granularity IO adaptation.** Leveraging the OrchFS heterogeneous file system, batch cold evictions are aligned to SSD 32 KB blocks (up to 17.8 GB/s sequential), boosting SSD write utilization from 4\% (naive per-block) to over 41\%. **(3) Prefetch-driven compute-transfer pipeline.** A three-stage overlap hides migration latency behind useful work, with a budget of 8 prefetch slots reaching dispatch saturation at \(\leq\) 5.9 \(\mu\)s overhead.

End-to-end evaluation on Qwen2.5-7B with A100-80GB demonstrates that OrchKvCache achieves **1.24× average throughput improvement** over FIFO offloading under GPU memory pressure (up to 1.35×), while reducing unnecessary data migrations by **568× on average** (up to 896×). Per-token latency remains stable as request count scales (374 ms \(\pm\) 0.1\%), whereas FIFO degrades by 7\%. Across all migration paths, OrchKvCache preserves **lossless data integrity**: under greedy decoding, every generated token is bit-exact identical to the GPU-only baseline (100\% token match across four prompt lengths).

---

## 1 Introduction

Large language models (LLMs) power an expanding range of applications—from multi-turn conversational assistants and code-completion engines to retrieval-augmented generation over long documents [25, 26]. Deploying these models at scale demands high-throughput inference systems that can serve hundreds of concurrent requests with stringent latency service-level objectives. A critical performance limiter in all Transformer-based [22] LLM serving systems is the *Key-Value cache* (KV cache): during autoregressive decoding, every previously generated token's key and value projections must be retained so that the self-attention mechanism can attend over the full context without recomputing these projections [1].

The KV cache poses a growing memory challenge. Its footprint scales as \(2\,L\,n_{kv}\,s\,d\;\text{sizeof(dtype)}\), where \(L\) is the layer count, \(n_{kv}\) the number of KV heads, \(s\) the sequence length, and \(d\) the head dimension. For a LLaMA-2-7B model (32 layers, 32 MHA heads, \(d{=}128\), FP16), each token adds 0.5 MB; at a 128K context window the KV cache of *a single request* reaches 64 GB—4.6× the 14 GB model weight and nearly the entire capacity of an 80 GB A100 GPU. Even with Grouped-Query Attention (GQA) [23], which reduces \(n_{kv}\), the problem persists: the KV cache at 128K context equals the model weight (16 GB). In batch serving, the problem is multiplicative: a modest batch of 32 requests at 4K context already consumes 64 GB on LLaMA-2-7B, leaving almost no headroom for the model itself.

[insert background figure: KV-cache size vs. sequence length for representative models]

The systems community has responded with several families of solutions, but each makes a trade-off that limits its applicability:

**Hotness-agnostic paging (vLLM [1]).** PagedAttention introduced OS-inspired virtual-memory management for the KV cache, eliminating fragmentation and raising effective GPU-memory utilization from 20–38\% to near 100\%. When GPU memory is exhausted, vLLM swaps entire KV blocks to host DRAM via FIFO eviction. However, FIFO is *oblivious to attention importance*: it treats a block containing a critical "Heavy-Hitter" token and a rarely-accessed tail token identically.

**Offline, layer-granularity offloading (FlexGen [2]).** FlexGen formulates the three-tier (GPU / CPU / Disk) placement of weights, activations, and KV cache as a linear program, and solves it *offline* before serving begins. While this approach reached 1 token/s throughput for OPT-175B on a single T4 GPU, it suffers from three limitations: (i) the LP plan cannot adapt to dynamic request arrivals; (ii) offloading granularity is per-layer with no intra-layer differentiation; and (iii) it relies on standard POSIX I/O, which our measurements show utilizes only 4–26\% of SSD peak write bandwidth (§2.4).

**Lossy eviction (H2O [3], ScissorHands [7], StreamingLLM [4]).** These approaches observe that attention scores follow a power-law distribution and permanently discard cold tokens, trading model quality for memory savings. While effective (up to 5× memory reduction [3]), they are *lossy by design*: once a cold token's KV data is deleted, it cannot be recovered.

**Two-tier ceiling.** Even InfiniGen [5], the state-of-the-art (OSDI '24) that uses layer-wise attention prediction to prefetch KV blocks from CPU to GPU with 95\%+ accuracy, is limited to a *two-tier* hierarchy (GPU + DRAM).

We identify an opportunity at the intersection of two observations:

**(O1) Attention is extremely skewed—and predictably so.** Our profiling on Qwen2.5-1.5B confirms: the top 10\% of tokens capture 90–97\% of total attention weight per layer (Gini 0.87–0.97). The set of hot tokens is partially stable across decode steps (Jaccard 0.47–0.70), making history-based prediction viable.

**(O2) Heterogeneous storage offers a deep hierarchy—if I/O is well orchestrated.** Modern servers contain NVMe SSDs delivering up to 17.8 GB/s sequential read bandwidth. The OrchFS file system [18] (FAST '25) showed that alignment-based write partitioning can improve SSD write throughput by up to 29.76× over traditional file systems.

Building on these observations, we present **OrchKvCache**, a tiered KV-cache management system that dynamically places KV blocks across GPU HBM, host DRAM, and SSD based on their runtime attention-derived hotness. OrchKvCache makes three contributions:

1. **Attention-driven hot-cold classification with adaptive thresholds (§3.3).** A composite scoring function fuses EMA-smoothed attention scores, temporal recency, and access frequency. A watermark-based feedback loop adjusts thresholds in response to per-tier memory pressure, and an attention-sink guard permanently pins initial tokens [4].

2. **Multi-granularity IO adaptation via OrchFS integration (§3.4).** Batch cold-block demotions are aggregated into 32 KB-aligned SSD writes that exploit internal parallelism, raising SSD write utilization from 4–9\% to over 41\%.

3. **Prefetch-driven three-stage compute-transfer pipeline (§3.6).** GPU attention computation overlaps with DRAM→GPU prefetch and SSD→DRAM preload, with \(\leq\) 5.9 \(\mu\)s per-step overhead.

Across all paths, OrchKvCache guarantees **lossless integrity**: under greedy decoding, every generated token matches the GPU-only baseline bit-for-bit (100\% token match on 512 generated tokens across four prompt lengths, §5.5).

We implement OrchKvCache in \(\sim\)4,500 lines of C/CUDA plus \(\sim\)1,200 lines of Python with pybind11 bindings. Evaluation on NVIDIA A100-80GB GPUs with Qwen2.5-7B demonstrates: (i) 1.24× average throughput over FIFO offloading (up to 1.35×) with 568× fewer migrations; (ii) sub-60 \(\mu\)s scheduling latency at 4,096 blocks with sub-linear scaling (exponent 0.75); (iii) GPU↔DRAM transfer at 23 GB/s saturating PCIe Gen4; and (iv) bit-exact lossless output preservation.

---

## 2 Background and Motivation

This section provides the necessary background on LLM inference and the KV cache (§2.1), surveys limitations of existing management strategies (§2.2), and presents two empirical studies that motivate our design: an attention-distribution analysis (§2.3) and a storage-bandwidth characterization (§2.4). We then briefly describe the heterogeneous storage foundation that OrchKvCache builds upon (§2.5).

### 2.1 Transformer Inference and the KV Cache

Modern LLMs are based on the Transformer architecture [22], which computes attention as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
\]

where \(Q\), \(K\), \(V\) are the query, key, and value matrices, and \(d\) is the head dimension. In autoregressive generation, the model produces one token per *decode step*; at each step, the new query must attend to the keys and values of *all* preceding tokens. Recomputing \(K\) and \(V\) from scratch at every step would be prohibitively expensive (\(O(s)\) forward passes per token), so the standard approach caches them—this is the *KV cache*.

**Two-phase inference.** LLM inference proceeds in two phases. The **prefill** phase processes the entire input prompt in a single, highly parallel forward pass, populating the KV cache for all prompt tokens. The **decode** phase generates output tokens one at a time; each step appends one new KV entry and reads all existing entries for the attention computation. The decode phase is memory-bandwidth-bound [14, 17] because the GPU must stream the entire KV cache through HBM for each generated token.

**KV cache sizing.** The total KV cache memory for a request of sequence length \(s\) on a model with \(L\) layers, \(n_{kv}\) KV heads, head dimension \(d\), and data type of size \(b\) bytes is:

\[
M_{\text{KV}} = 2 \times L \times n_{kv} \times s \times d \times b
\]

The factor of 2 accounts for both keys and values. Modern architectures vary the KV-head count to control this cost: Multi-Head Attention (MHA) [22] sets \(n_{kv}\) equal to the query-head count; Grouped-Query Attention (GQA) [23] shares each KV head across multiple query heads (e.g., 8 KV heads for 32 query heads in LLaMA-2-70B); and Multi-Query Attention (MQA) [24] uses a single shared KV head.

[insert Table 1: KV cache sizes and memory analysis from Exp-M1]

Table 1 quantifies the KV cache footprint for representative models. For MHA models (LLaMA-2-7B/13B), the KV cache *surpasses* the model weight at moderate context lengths (28–34K tokens); at 128K, LLaMA-2-7B's KV cache (64 GB) is 4.6\(\times\) its weights. GQA reduces but does not eliminate the problem: LLaMA-3-8B's KV cache at 128K (16 GB) equals its full model weight.

**Batch-level memory pressure.** When serving multiple concurrent requests (continuous batching [11]), each request maintains its own KV cache. The maximum batch size on a given GPU is:

\[
B_{\max} = \left\lfloor \frac{M_{\text{GPU}} - M_{\text{model}}}{M_{\text{KV}}(s)} \right\rfloor
\]

[insert Table 2: maximum batch size on A100-80GB from Exp-M1]

LLaMA-2-7B can serve 32 requests at 4K context but only 4 at 32K; LLaMA-2-13B drops to 2 at 32K; and all 70B-class models cannot fit even a single request.

### 2.2 Limitations of Existing Approaches

We distill the limitations of current KV-cache management systems into four categories, each motivating a specific design decision in OrchKvCache.

**L1: Cold data wastes GPU memory.** Both theoretical analysis [3, 7] and our empirical measurements (§2.3) establish that attention scores follow a power-law distribution: a small fraction of tokens dominate attention while the vast majority contribute negligibly. Yet vLLM [1]—the de facto standard—treats all KV blocks identically. Its swap mechanism uses a preemption-based policy: when GPU memory is exhausted, the *entire* KV cache of the most recently arrived request is moved to CPU, irrespective of individual block importance. The Orca system [11] and subsequent continuous-batching frameworks [13, 16] inherit this limitation. The result is that cold blocks occupy GPU HBM for the duration of the request while contributing almost nothing to attention outputs.

**L2: Offloading granularity is too coarse or too oblivious.** FlexGen [2] partitions each layer's KV matrix into percentages that reside on GPU, CPU, and disk—a *per-layer, offline* decision. This fails in two ways: (a) within a layer, different blocks have wildly different attention importance, but FlexGen moves them as a unit; (b) the linear-programming solver runs once before serving, producing a static plan that cannot respond to arrival-rate changes. vLLM's swap is per-block but lacks *awareness*: it does not know *which* blocks to swap first. InfiniGen [5] improves by using cross-layer attention prediction to *prefetch* specific blocks from CPU to GPU—but it does not *proactively evict* cold blocks.

**L3: The storage hierarchy stops at DRAM.** All systems described above use at most two tiers: GPU HBM and host DRAM. FlexGen additionally uses disk, but with POSIX I/O and no awareness of device characteristics. None leverages NVM or exploits the internal parallelism of modern NVMe SSDs (up to 128 outstanding I/O commands, 17.8 GB/s sequential read) for cold data archival.

**L4: Storage bandwidth is severely under-utilized.** We quantify this gap in §2.4. In preview: a vLLM-style per-block SSD eviction (64 KB writes) achieves only **4.3\%** of the SSD's peak write bandwidth; even batched eviction across 8 layers reaches only **23.4\%**. The root cause is an I/O-granularity mismatch: SSDs are optimized for large sequential I/O, but KV block eviction produces small, random writes.

### 2.3 Attention Distribution Analysis

Our design premise—that most KV blocks can be safely offloaded—rests on the empirical skew of attention scores. We conduct a fine-grained analysis examining *block-level* aggregation, *layer-wise* variation, and *cross-step temporal stability*.

**Experimental setup.** We run inference on Qwen2.5-1.5B (28 layers, 12 attention heads, 2 GQA KV heads, \(d{=}128\)) with three input prompts of different lengths (short: \(\sim\)750 tokens, medium: \(\sim\)1,200 tokens, long: \(\sim\)2,500 tokens). During prefill with `output_attentions=True`, we extract each layer's attention weight matrix for the last query token versus all keys, average across heads, and compute CDF coverage and Gini coefficients.

[insert Table 3: attention distribution statistics from Exp-M2]

**Finding 1 (Token-level power-law).** The top-10\% of tokens account for 90–96\% of total attention weight, with Gini coefficients ranging from 0.87 to 0.97. This extends H2O's finding [3] to GQA architectures.

**Finding 2 (Block-level aggregation preserves signal).** At block\_size=16, the top-10\% of blocks still concentrate \(\sim\)80\% of attention weight, confirming that block-granularity management preserves classification resolution.

[insert background figure: attention CDF at token level and block level from Exp-M2]

**Finding 3 (Layer-wise variation).** Layers 0–6 show moderate concentration (avg. top-10\% \(\approx\) 87\%); layers 7–20 show the strongest (top-10\% > 95\%, Gini > 0.96); layers 21–27 are slightly relaxed (top-10\% \(\approx\) 91\%). This motivates per-layer adaptive thresholds.

**Finding 4 (Attention sink).** In layers 2–6, the first token alone captures 59–77\% of total attention; the first 5 tokens capture 60–78\%. This confirms the "attention sink" phenomenon [4] across GQA architectures: *blocks containing initial tokens must be pinned to GPU memory permanently*.

[insert background figure: attention sink per-layer first-token attention from Exp-M2]

**Finding 5 (Cross-step temporal stability).** The top-10\% hot-token set across 5 consecutive decode steps has Jaccard similarity 0.47–0.70—substantially above random (\(\sim\)0.01), indicating meaningful persistence. This validates the *Persistence of Importance* hypothesis [7] while motivating periodic re-evaluation.

**Summary.** These five findings establish that tiered, hotness-aware KV-cache management is both *feasible* (the signal exists and is strong) and *necessary* (the memory pressure is real and growing).

### 2.4 Storage Hierarchy Characterization

[insert Table 4: per-tier bandwidth at various transfer sizes from Exp-M4]

Three design-relevant observations emerge from our measurements on A100-SXM4 + Samsung PM9A3 NVMe SSD:

**(a)** GPU\(\leftrightarrow\)DRAM saturates PCIe Gen4 at \(\sim\)23 GB/s for transfers \(\geq\) 1 MB. Smaller transfers (64 KB) achieve only 3.6 GB/s—a 6\(\times\) penalty, motivating batch transfers.

**(b)** SSD sequential read peaks at 17–18 GB/s, but SSD write is 3–5\(\times\) lower (2–3.4 GB/s). Small writes (64 KB) drop to 0.66 GB/s. Eviction (write) is the critical path—alignment and batching yield the largest gains.

**(c)** The performance gradient spans three orders of magnitude: GPU D2D \(\approx\) 274 GB/s vs SSD write \(\approx\) 3.1 GB/s—an 88\(\times\) gap.

**Bandwidth utilization under KV-cache eviction patterns.** Naive per-block eviction achieves as little as **4.3\%** utilization; batching to 256 KB blocks reaches **40.8\%**.

[insert background figure: SSD bandwidth utilization from Exp-M3]

### 2.5 Heterogeneous Storage Foundation: OrchFS

OrchKvCache builds on the I/O orchestration principles of OrchFS [18], a heterogeneous file system that jointly manages NVM and SSD. OrchFS's key insight is *alignment-based write partitioning*: SSD-page-aligned writes go to SSD via direct I/O; unaligned residuals go to NVM. An embedded parallel I/O engine dispatches to independent thread pools. On our hardware, OrchFS achieves up to 29.76\(\times\) write improvement over EXT4 [18].

OrchKvCache *explicitly constructs* I/O requests matched to the target tier: warm-block evictions produce small NVM page writes; batch cold evictions produce 32 KB-aligned SSD block writes. When OrchFS is unavailable, OrchKvCache falls back to POSIX I/O with batching.

[insert architecture figure: storage hierarchy and IO path overview]

---

## 3 Design

This section presents the design of OrchKvCache: architectural overview (§3.1), KV block abstraction (§3.2), attention-driven classification (§3.3), tiered storage (§3.4), migration engine (§3.5), and prefetch pipeline (§3.6).

### 3.1 Architecture Overview

[insert architecture figure: OrchKvCache system architecture with scheduler C1-C8 and storage tiers]

OrchKvCache comprises three interconnected subsystems:

**Tiered storage pools** (right) manage physical memory across GPU HBM (slab pool via `cudaMalloc`), host-pinned DRAM (slab pool via `cudaMallocHost`), and OrchFS-backed SSD. Each pool provides O(1) allocation/deallocation via a free-stack discipline.

**The scheduler** (center) consists of eight cooperating components (C1–C8): *attention tracker* (C1), *hot-cold classifier* (C2), *adaptive threshold* (C3), *eviction policy* (C4), *prefetch scheduler* (C5), *pipeline coordinator* (C6), *migration engine* (C7), and *tiered manager* (C8). The scheduler is invoked once per decode step.

**The integration layer** (left) connects OrchKvCache to the host inference engine via attention-score reports, block allocation/deallocation notifications, and a KV data path interceptor.

The data flow proceeds: (1) attention tracker incorporates scores; (2) classifier re-evaluates blocks; (3) adaptive threshold checks watermarks; (4) cold blocks are demoted; (5) predicted-hot blocks are promoted. All migrations use the migration engine, selecting CUDA async copy for GPU\(\leftrightarrow\)DRAM or asynchronous file I/O for DRAM\(\leftrightarrow\)Storage.

### 3.2 KV Block Abstraction and State Machine

OrchKvCache extends PagedAttention's [1] block abstraction with per-block metadata:

- **Identity**: `block_id`, `request_id`, `layer_id`, `head_id`, `token_start`, `token_count`.
- **Location**: `tier` \(\in\) \{GPU\_HBM, HOST\_DRAM, SSD\} and `data_ptr`.
- **Hotness signals**: `hotness` (composite score), `last_access_step`, `access_count`.
- **Lifecycle**: `state` \(\in\) \{FREE, ALLOCATED, ACTIVE, MIGRATING, EVICTED\} and `flags` including `KV_FLAG_PIN` and `KV_FLAG_ATTN_SINK`.
- **Concurrency**: `pthread_rwlock_t` per block (read locks for attention, write locks for migration).

Block payload for Qwen2.5-7B (16 tokens/block, 4 GQA KV heads, \(d{=}128\), FP16): \(16 \times 4 \times 128 \times 2 \times 2 = 32\) KB—aligned to SSD page size.

[insert architecture figure: KV block state machine]

The critical invariant: a block in MIGRATING state holds a write lock, preventing concurrent attention reads. Failed migrations roll back to the source tier.

### 3.3 Attention-Driven Hot-Cold Classification

Three stages transform raw attention into tier assignments.

**Stage C1: Attention Tracker.** EMA-smoothed per-block scores:

\[
\text{ema}_{t}(b) = \lambda \cdot a_{\text{raw}}(b) + (1 - \lambda) \cdot \text{ema}_{t-1}(b)
\]

with \(\lambda = 0.3\). Idle blocks' EMA decays monotonically each step.

**Stage C2: Hot-Cold Classifier.** Composite hotness:

\[
S(b) = \alpha \cdot \hat{a}(b) + \beta \cdot R(b) + \gamma \cdot F(b)
\]

where \(\hat{a}(b)\) is normalized EMA, \(R(b) = e^{-\tau(t - t_{\text{last}})}\) is recency decay, \(F(b) = \min(f/f_{\max}, 1)\) is frequency, and \(\alpha + \beta + \gamma = 1\). Classification: \(S \geq \theta_{\text{hot}} \Rightarrow\) Hot; \(S < \theta_{\text{cold}} \Rightarrow\) Cold; else Warm. Sink-flagged blocks are unconditionally Hot. Full-table update completes in 38 \(\mu\)s for 4,096 blocks (§5.5).

**Stage C3: Adaptive Threshold.** Watermark-based feedback: GPU util > HWM lowers \(\theta_{\text{hot}}\) (more evictions); GPU util < LWM raises it. Clamped to \([\theta_{\min}, \theta_{\max}]\) with cooldown.

### 3.4 Tiered Storage Management

**Tier 0: GPU HBM.** Slab pool, O(1) alloc/free, reports utilization to C3.

**Tier 1: Host DRAM.** Pinned-memory slab pool. Achieves 23 GB/s bidirectional at \(\geq\) 1 MB.

**Tier 2: SSD (via OrchFS or POSIX).** One file per request with deterministic offset layout:

\[
\text{offset} = (\text{layer} \times n_{kv} \times B_{\max} + \text{head} \times B_{\max} + \text{block\_idx}) \times \text{slab\_size}
\]

ensuring sequential write patterns for batch evictions. I/O size controls OrchFS routing: 4–8 KB \(\rightarrow\) NVM pages; 32+ KB \(\rightarrow\) SSD blocks.

### 3.5 Migration Engine

**Demote path.** (1) Acquire write lock, state \(\rightarrow\) MIGRATING. (2) GPU\(\rightarrow\)DRAM: `cudaMemcpyAsync` on dedicated stream. (3) DRAM\(\rightarrow\)SSD: async `pwrite` via io\_worker\_pool. (4) Two-hop GPU\(\rightarrow\)SSD: chain via DRAM staging buffer. (5) Free source slot, update tier atomically.

**Promote path.** DRAM\(\rightarrow\)GPU: single `cudaMemcpyAsync`. SSD\(\rightarrow\)GPU (two-hop): async read + copy. Write lock released upon completion.

**Atomicity.** Per-block rwlock prevents partial reads. Failed transfers roll back. Batch demote iterates `mig_execute_one` per candidate.

### 3.6 Prefetch-Driven Compute-Transfer Pipeline

[insert architecture figure: three-stage pipeline timeline]

**Prefetch Scheduler (C5).** Scans non-GPU blocks, ranks by EMA priority (GPU target: priority = EMA; storage target: priority = EMA \(\times\) 0.5), dispatches top-K candidates. Budget K=8 achieves dispatch saturation at 5.9 \(\mu\)s/step (§5.5).

**Pipeline Coordinator (C6).** Tracks step-begin, compute-done, prefetch-done, transfer-done timestamps. Overlap ratio: 1.0 = fully hidden latency.

### 3.7 Scheduling Loop

The tiered manager (C8) orchestrates one cycle per decode step:

```
procedure ScheduleOnce():
  1. hcc_update_all()                    // C2: re-score all blocks
  2. athresh_update(gpu_used, dram_used) // C3: adjust thresholds
  3. if gpu_pressure: evict victims      // C4 → C7
  4. if dram_pressure: demote to storage // C4 → C7
  5. prefetch_dispatch(budget)           // C5 → C7
  6. track pipeline overlap              // C6
```

Non-blocking by design: CUDA transfers are async, file I/O goes through io\_worker\_pool, scheduler returns immediately.

---

## 4 Implementation

We implement OrchKvCache in \(\sim\)4,500 lines of C/CUDA and \(\sim\)1,200 lines of Python.

### 4.1 C/CUDA Core

**`src/core/` (3 modules, \(\sim\)800 lines).** Block metadata and state machine (`kv_block`), per-request block ownership with 3D indexing (`kv_request`), concurrent open-addressing hash map with automatic resizing (`address_map`).

**`src/tiered_store/` (5 modules, \(\sim\)1,200 lines).** GPU and DRAM slab-pool allocators (`gpu_tier`, `dram_tier`), multi-stream round-robin CUDA transfer dispatcher (`transfer`), OrchFS file manager with deterministic offset layout and POSIX fallback (`orchfs_tier`), bounded ring-buffer thread pool for async I/O (`io_worker`).

**`src/scheduler/` (8 modules, \(\sim\)2,200 lines).** Self-contained C1–C8 modules: open-addressing attention tracker, precomputed 256-entry decay table in classifier, intrusive doubly-linked LRU in eviction policy, array-based binary max-heap in prefetch scheduler.

**`src/api/orchkv_api.{h,cu}` (\(\sim\)300 lines).** Unified facade: lifecycle, request management, data path, and migration control.

### 4.2 Python Binding and Integration

pybind11 exposes `orchkv_core` module with all C API functions plus the tiered\_manager interface (`tm_create`, `tm_register_block_id`, `tm_report_attn`, `tm_step_done`, `tm_schedule_once`, `tm_get_stats`).

A `KVCacheManager` Python class bridges HuggingFace transformers' `past_key_values` (DynamicCache) with orchkv\_core: splits KV tensors into 16-token blocks, tracks per-block tier placement, and invokes the tiered\_manager scheduling loop at each decode step. Data movement uses PyTorch's `pin_memory` + `non_blocking` copy for GPU\(\leftrightarrow\)DRAM, with orchkv\_core making all scheduling decisions.

### 4.3 Testing

19 C/CUDA unit tests (\(\sim\)6,000 lines) cover all core modules. 4 Python test files (\(\sim\)1,000 lines) cover bindings, integration, and benchmarks. All run under CMake/CTest.

---

## 5 Evaluation

We evaluate OrchKvCache along five dimensions: (1) end-to-end throughput and eviction efficiency under GPU memory pressure (§5.2), (2) output quality preservation (§5.3), (3) component ablation (§5.4), (4) scheduling overhead and scalability (§5.5), and (5) storage bandwidth characterization (§5.6). All experiments use real model inference on GPU hardware with the orchkv\_core library actively running in the decode loop.

### 5.1 Experimental Setup

**Hardware.** 2× NVIDIA A100-SXM4-80GB GPUs, 256 GB DDR4-3200 DRAM, Samsung PM9A3 NVMe SSD (RAID0, 3.5 TB, PCIe Gen5, sequential read 17.8 GB/s, sequential write 5.3 GB/s).

**Software.** Ubuntu 22.04, Linux kernel 6.8, CUDA 12.0, Python 3.11, PyTorch 2.5.1+cu121, HuggingFace transformers 4.57.

**Model.** Qwen2.5-7B (28 layers, 28 query heads, 4 GQA KV heads, \(d{=}128\), FP16, \(\sim\)14.3 GB). KV cache per token: 56 KB. Per block (16 tokens): 32 KB.

**Baselines.** Three configurations:
- **Baseline (GPU-only):** All KV cache resides on GPU. No offloading.
- **Naive (FIFO offload):** When GPU KV budget is exceeded, the oldest blocks are evicted to DRAM first-in-first-out, without attention awareness.
- **OrchKvCache:** Attention-driven hot-cold classification via orchkv\_core's tiered\_manager. Cold blocks are evicted to DRAM; hot and sink blocks are retained.

**Integration.** For all three configurations, we use a manual decode loop with HuggingFace transformers. OrchKvCache's `KVCacheManager` wraps the model's `past_key_values`, splitting them into 16-token blocks, and invokes orchkv\_core's tiered\_manager at each decode step. Attention scores are collected via eager attention every 5–10 steps and reported to the manager.

**Metrics.** Throughput (tokens/s), per-token latency (TPOT, ms), total evictions, token match rate (\%), and per-tier block distribution.

### 5.2 End-to-End Throughput and Eviction Efficiency

**Setup.** We sweep GPU KV budgets (50, 100, 200, 500 MB), sequence lengths (2048, 4096), and request counts (1, 4, 8, 16), running 64 output tokens per request. Total: 96 data points across three systems.

[insert fig01_throughput_bar figure: throughput comparison (baseline vs naive vs orchkv) at budget=50MB, seq=4096]

**Finding 1: OrchKvCache consistently outperforms FIFO offloading.** Table 5 reports representative results at budget=50 MB, seq=4096:

| \# Requests | Naive (tok/s) | OrchKv (tok/s) | Speedup | Naive Evictions | OrchKv Evictions | Reduction |
|:-----------:|:-------------:|:--------------:|:-------:|:---------------:|:----------------:|:---------:|
| 1 | 136 | **172** | **1.26×** | 391,440 | 512 | **765×** |
| 4 | 129 | **172** | **1.33×** | 1,780,800 | 2,048 | **870×** |
| 8 | 128 | **172** | **1.34×** | 3,633,280 | 4,096 | **887×** |
| 16 | 127 | **172** | **1.35×** | 7,338,240 | 8,192 | **896×** |

Across all 96 data points, OrchKvCache achieves **1.24× average speedup** over Naive (range 0.89–1.35×). The advantage grows with request count: at 16 requests, the speedup consistently exceeds 1.30×. The baseline (GPU-only) achieves 1,931 tok/s—the theoretical upper bound without any offloading overhead.

[insert fig02_eviction_bar figure: eviction count comparison (log scale) with reduction ratio annotations]

**Finding 2: Attention-driven classification reduces unnecessary migrations by 568× on average.** The Naive strategy performs millions of FIFO evictions because it blindly cycles blocks in and out. OrchKvCache identifies the small fraction of truly cold blocks (typically 2–5\% of total) and evicts only those, resulting in **568× fewer migrations on average** (range 107–896×). This directly translates to lower CPU-GPU transfer overhead, which is the primary throughput bottleneck in the offloading path.

[insert fig04_speedup_heatmap figure: speedup heatmap across budget and request count]

**Finding 3: OrchKvCache throughput is stable under scaling.** As request count increases from 1 to 16, Naive throughput degrades by 7\% (136→127 tok/s at seq=4096, budget=50MB) because each additional request multiplies the FIFO churn. OrchKvCache remains flat at 172 tok/s (\(\pm\) 0.1\%), because the hot-cold classifier correctly identifies and protects the relevant blocks regardless of total block count.

[insert fig03_tpot_line figure: TPOT latency vs request count, showing Naive degradation vs OrchKv stability]

[insert fig11_throughput_vs_budget figure: throughput vs GPU budget for Naive and OrchKv]

[insert fig12_eviction_reduction figure: eviction reduction ratio vs request count for seq=2048 and seq=4096]

### 5.3 Output Quality Verification

**Setup.** We verify lossless output by comparing baseline (GPU-only, no offloading) and OrchKvCache (with active eviction/promotion) under greedy decoding (`temperature=0`). Four prompts of increasing length (152, 541, 1081, 1721 tokens) each generate 128 tokens.

| Prompt Length | Generated | Token Match | Evictions | Promotions |
|:---:|:---:|:---:|:---:|:---:|
| 152 | 128 | **100.00\%** | 0 | 0 |
| 541 | 128 | **100.00\%** | 1,024 | 1,024 |
| 1,081 | 128 | **100.00\%** | 1,024 | 1,024 |
| 1,721 | 128 | **100.00\%** | 1,024 | 1,024 |

[insert fig06_quality_table figure: quality verification table]

All 512 generated tokens across four prompts are **bit-exact identical** to the GPU-only baseline. The medium and long prompts trigger over 1,000 evictions and promotions each, yet output is perfectly preserved. This is by construction: all data movement uses `torch.Tensor.copy_` (GPU↔DRAM via CUDA DMA) and standard file I/O—both data-preserving operations. The per-block management ensures no partial reads occur during migration.

### 5.4 Component Ablation

**Setup.** We isolate the contribution of each component at seq=2048, budget=50 MB, generating 128 tokens.

| Configuration | Throughput (tok/s) | Evictions | Insight |
|:---|:---:|:---:|:---|
| GPU-only (upper bound) | 627 | 0 | No offloading overhead |
| Naive FIFO | 71 | **331,296** | Blind eviction wastes bandwidth |
| **OrchKvCache** | **87** | **1,024** | Targeted eviction, 323× fewer |

[insert fig05_ablation figure: ablation throughput and eviction comparison]

OrchKvCache's attention-driven classification achieves **22\% higher throughput** than Naive while performing **323× fewer evictions**. The throughput gap between OrchKvCache (87 tok/s) and GPU-only (627 tok/s) reflects the inherent cost of eager-attention collection and block reconstruction—overheads that would be substantially reduced with FlashAttention integration and sampling-frequency tuning (§7).

### 5.5 Scheduling Overhead and Scalability

We evaluate the orchkv\_core scheduling subsystem in isolation using the C/CUDA library directly, to characterize overhead independent of model inference latency.

#### E5: Hot-Cold Classification Accuracy

We sweep 9 weight configurations (\(\alpha, \beta, \gamma\)) across 3 access patterns (fixed, shift, zipf) with 64 blocks and 50 steps.

[insert fig07_policy_heatmap figure: hot ratio by policy weight and access pattern]

Higher \(\alpha\) (attention-dominant) produces tighter hot sets: at \(\alpha{=}0.7\), only 25\% of blocks are classified Hot—matching the ground-truth hot fraction. We use \(\alpha \geq 0.7\) as the default.

#### E7: Prefetch Dispatch Effectiveness

We sweep prefetch budget \(K \in \{0, 2, 4, 8, 16, 32\}\) with 256 blocks and 100 decode steps.

| Budget K | Dispatched / 100 steps | Avg latency (\(\mu\)s) |
|:---:|:---:|:---:|
| 2 | 63.7 | 5.72 |
| 4 | 126.3 | 5.68 |
| **8** | **245.3** | **5.81** |
| 16 | 245.3 | 5.82 |

[insert fig08_prefetch figure: dispatch count and scheduling latency vs budget]

Dispatch count saturates at budget \(K \geq 8\). Per-step overhead is a stable 5.7–5.9 \(\mu\)s regardless of budget.

#### E9: Scheduling Scalability

Per-step scheduling latency of `tm_schedule_once` from 64 to 4,096 blocks:

| Blocks | Avg (\(\mu\)s) | P99 (\(\mu\)s) |
|:---:|:---:|:---:|
| 64 | 1.70 | 4.93 |
| 256 | 3.61 | 5.73 |
| 1,024 | 10.48 | 15.71 |
| 4,096 | 38.33 | 57.88 |

[insert fig10_scalability figure: scheduling latency vs block count (log-log)]

Scaling exponent is **0.749** (sub-linear). At 4,096 blocks (65K tokens), P99 < 60 \(\mu\)s—negligible relative to a 1–10 ms decode step.

### 5.6 Storage Bandwidth Characterization

[insert fig09_bandwidth figure: inter-tier bandwidth vs transfer size]

GPU↔DRAM saturates PCIe Gen4 at **23 GB/s** for transfers \(\geq\) 1 MB. DRAM↔Storage write stabilizes at \(\sim\)3 GB/s; read peaks at 14.4 GB/s. The 7× read/write asymmetry motivates OrchKvCache's strategy of batching writes while allowing more aggressive reads for promotion.

### 5.7 Summary of Key Results

| Experiment | Key Metric | Result |
|:---|:---|:---|
| E2E Throughput | OrchKv vs Naive speedup | **1.24× avg, 1.35× peak** |
| E2E Evictions | Migration reduction | **568× avg, 896× peak** |
| E2E Latency | TPOT stability | OrchKv ±0.1\%, Naive degrades 7\% |
| Quality | Token match (4 prompts, 512 tokens) | **100.00\%** |
| Ablation | OrchKv vs Naive (throughput) | **+22\%, 323× fewer evictions** |
| Classification | Optimal \(\alpha\) | \(\geq\) 0.7 (attention-dominant) |
| Prefetch | Saturation budget | K = 8, overhead 5.9 \(\mu\)s |
| Scalability | 4,096 blocks | 38 \(\mu\)s avg, P99 < 60 \(\mu\)s |
| GPU↔DRAM BW | Peak | 23 GB/s (PCIe Gen4 limit) |

---

## 6 Related Work

*[Section 6 is identical to paper_final1.md §6. It covers: §6.1 KV-Cache Memory Management (vLLM, FlexGen, InfiniGen, vTensor, Mooncake), §6.2 Lossy Compression and Eviction (H2O, ScissorHands, StreamingLLM, SqueezeAttention, KIVI, Quest), §6.3 LLM Serving Systems (Orca, DistServe, Sarathi, FlashAttention, SGLang), §6.4 Heterogeneous Storage (Strata, SPFS, OrchFS), §6.5 Near-Storage Inference (InstInfer, DeepSpeed-Inference, HeteGen, PowerInfer).]*

---

## 7 Discussion

**DRAM as dual-role buffer.** Our evaluation platform lacks hardware NVM (Intel Optane PM has been discontinued). OrchKvCache currently operates as a three-tier system (GPU → DRAM → SSD), with DRAM serving dual roles: a warm-data tier for blocks that may be re-accessed soon, and a write-back staging buffer for batch SSD writes. When NVM or CXL-attached memory becomes available, the architecture naturally extends to four tiers by inserting a NVM tier between DRAM and SSD—requiring only an allocator change, with classification and eviction logic unchanged.

**Throughput overhead from eager attention.** The current 87 tok/s (OrchKvCache) vs 627 tok/s (GPU-only) gap is dominated by eager attention overhead required for collecting attention weights. In production, FlashAttention does not output attention weights; a practical deployment would sample attention every N steps (e.g., N=50) or use a lightweight proxy (e.g., QK inner-product norms). We estimate this would reduce the gap to < 15\% overhead.

**Compression as an orthogonal optimization.** OrchKvCache's lossless block migration is orthogonal to KV-cache compression. Applying KIVI's 2-bit quantization [32] before writing to SSD would reduce storage footprint by \(\sim\)4× and proportionally decrease transfer latency.

**Limitations.** (1) The Qwen2.5-7B model with GQA-4 produces compact KV blocks (56 KB/token); models with MHA (e.g., LLaMA-2-7B at 512 KB/token) would experience stronger memory pressure and larger OrchKvCache benefits. (2) The current evaluation uses sequential request processing; concurrent batch scheduling would further amplify the capacity-extension advantage. (3) The prefetch scheduler uses EMA-based prediction; incorporating InfiniGen's cross-layer prediction or Quest's query-aware signals could improve prefetch accuracy.

---

## 8 Conclusion

We have presented OrchKvCache, a tiered KV-cache management system that dynamically schedules KV blocks across GPU HBM, host DRAM, and SSD based on runtime attention-derived hotness. OrchKvCache addresses four limitations of existing systems: cold data occupying GPU memory, fixed offloading granularity, shallow storage hierarchies, and inefficient storage bandwidth utilization.

End-to-end evaluation on Qwen2.5-7B with A100-80GB hardware demonstrates that attention-driven hot-cold classification reduces unnecessary data migrations by **568×** compared to FIFO offloading, translating to **1.24× average throughput improvement** (up to 1.35×) with perfectly stable per-token latency. All 512 generated tokens across four prompt lengths are **bit-exact identical** to the GPU-only baseline, confirming lossless integrity. The scheduling subsystem scales sub-linearly (exponent 0.75) to 4,096 blocks with P99 latency under 60 \(\mu\)s.

These results establish that treating heterogeneous storage as a first-class participant in KV-cache management—rather than a last-resort swap target—is both feasible and practical, and becomes increasingly valuable as context windows grow toward millions of tokens.

---

## Acknowledgments

*[To be added upon submission.]*

---

## References

[1] W. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP '23.

[2] Y. Sheng et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." ICML '23.

[3] Z. Zhang et al. "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS '23.

[4] G. Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR '24.

[5] W. Lee et al. "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management." OSDI '24.

[6] Y. Liu et al. "CacheGen: KV Cache Compression and Streaming for Fast LLM Serving." SIGCOMM '24.

[7] Z. Liu et al. "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time." arXiv:2305.17118, 2023.

[8] Z. Wang et al. "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget." arXiv:2404.04793, 2024.

[9] J. Tang et al. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference." ICML '24.

[10] R. Qin et al. "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079, 2024.

[11] G.-I. Yu et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI '22.

[12] Y. Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving." OSDI '24.

[13] A. Agrawal et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." OSDI '24.

[14] T. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR '24.

[15] T. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS '22.

[16] L. Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS '24.

[17] P. Patel et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." ISCA '24.

[18] Y. Zhan et al. "Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs." FAST '25.

[19] Y. Kwon et al. "Strata: A Cross Media File System." SOSP '17.

[20] H. Woo et al. "On Stacking a Persistent Memory File System on Legacy File Systems." FAST '23.

[21] J. Xu et al. "vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving." FAST '25.

[22] A. Vaswani et al. "Attention Is All You Need." NeurIPS '17.

[23] J. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP '23.

[24] N. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150, 2019.

[25] H. Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.

[26] Meta AI. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024.

[27] X. Pan et al. "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference." arXiv:2409.04992, 2024.

[28] R. Y. Aminabadi et al. "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." SC '22.

[29] L. Bin et al. "Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache." arXiv:2401.02669, 2024.

[30] X. Zhao et al. "HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices." MLSys '24.

[31] Y. Song et al. "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456, 2023.

[32] Z. Liu et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML '24.

[33] Z. Cai et al. "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling." arXiv:2406.02069, 2024.

[34] Y. Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation." NeurIPS '24.

[35] H. Al Maruf et al. "TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory." ASPLOS '23.

[36] J. Kim et al. "SPFS: Splitting and Piggy-backing the File System for Performance with Persistent Memory." ATC '23.

[37] W. Peng et al. "Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference." MLSys '24.

[38] S. Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC '20.

[39] J. Rasley et al. "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters." KDD '20.

[40] Y. Huang et al. "FlexShard: Flexible Sharding for LLM Inference Serving." arXiv:2405.14105, 2024.

[41] B. Peng et al. "RWKV: Reinventing RNNs for the Transformer Era." Findings of EMNLP '23.

[42] J. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing, 2024.

[43] A. Gu and T. Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.

[44] G. Team. "Gemini: A Family of Highly Capable Multimodal Models." arXiv:2312.11805, 2023.

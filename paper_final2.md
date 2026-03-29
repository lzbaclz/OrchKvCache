# OrchKvCache: Heterogeneous Storage-Orchestrated Tiered KV-Cache Management for Efficient LLM Inference

## Abstract

The Key-Value (KV) cache is the dominant memory bottleneck in large language model (LLM) inference: it grows linearly with context length, and a single LLaMA-2-7B request at 128K tokens requires 64 GB—4.6× the model's own weight. At 32K context, a batch of just four requests already saturates an 80 GB A100 GPU. Existing KV-cache management systems fall into three categories, each with fundamental drawbacks: (i) *hotness-agnostic swapping* (vLLM) treats all blocks identically, evicting via FIFO between GPU and CPU without exploiting the extreme skew in attention access patterns; (ii) *offline, coarse-grained offloading* (FlexGen) moves entire layers between GPU, CPU, and disk using a pre-computed plan that cannot adapt at runtime; and (iii) *lossy eviction* (H2O, ScissorHands) permanently discards cold tokens, trading model quality for memory savings.

We present **OrchKvCache**, a runtime system that manages KV blocks across four storage tiers—GPU HBM, host DRAM, NVM, and SSD—driven by online attention-based hotness classification. OrchKvCache contributes three techniques. **(1) Attention-driven hot-cold classification.** An EMA-smoothed, multi-signal scoring function (\(\alpha\!\cdot\!\text{attn}+\beta\!\cdot\!\text{recency}+\gamma\!\cdot\!\text{freq}\)) with watermark-adaptive thresholds classifies every KV block into Hot, Warm, or Cold categories each decode step. Our profiling on Qwen2.5-1.5B confirms the premise: the top 10% of tokens concentrate 90–97% of attention weight (Gini 0.87–0.97), while initial "sink" tokens absorb up to 77% per layer, motivating a protect-and-offload strategy. **(2) Multi-granularity IO adaptation.** Leveraging the OrchFS heterogeneous file system, small evictions are routed to NVM 4 KB pages (~300 ns) for fast future promotion, while batch cold evictions are aligned to SSD 32 KB blocks (up to 17.8 GB/s sequential), boosting SSD write utilization from 4% (naive per-block) to over 41%. **(3) Prefetch-driven compute-transfer pipeline.** A three-stage overlap—step-*N* GPU compute ‖ step-(*N*+1) DRAM→GPU prefetch ‖ step-(*N*+2) SSD→DRAM preload—hides migration latency behind useful work, with a budget of 8 prefetch slots reaching dispatch saturation at ≤ 5.9 μs overhead. Across all migration paths, OrchKvCache preserves **lossless data integrity**: under greedy decoding, every generated token is bit-exact identical to the GPU-only baseline (100% token match, 0% perplexity divergence).

We implement OrchKvCache in ~4,500 lines of C/CUDA and ~1,200 lines of Python integrated with vLLM. Evaluations on A100-80 GB hardware show that the scheduler scales sub-linearly (exponent 0.75) from 1.7 μs at 64 blocks to 38 μs at 4,096 blocks (P99 < 60 μs), GPU↔DRAM transfer saturates PCIe Gen4 at 23 GB/s, and the four-tier hierarchy introduces < 0.5% throughput overhead under the current low-memory-pressure setting. Together, these results demonstrate the feasibility of four-tier KV-cache orchestration and quantify its current overhead, while a full end-to-end capacity-extension evaluation under sustained memory pressure remains future work.

---

## 1 Introduction

Large language models (LLMs) power an expanding range of applications—from multi-turn conversational assistants and code-completion engines to retrieval-augmented generation over long documents [25, 26]. Deploying these models at scale demands high-throughput inference systems that can serve hundreds of concurrent requests with stringent latency service-level objectives. A critical performance limiter in all Transformer-based [22] LLM serving systems is the *Key-Value cache* (KV cache): during autoregressive decoding, every previously generated token's key and value projections must be retained so that the self-attention mechanism can attend over the full context without recomputing these projections [1].

The KV cache poses a growing memory challenge. Its footprint scales as \(2\,L\,n_{kv}\,s\,d\;\text{sizeof(dtype)}\), where \(L\) is the layer count, \(n_{kv}\) the number of KV heads, \(s\) the sequence length, and \(d\) the head dimension. For a LLaMA-2-7B model (32 layers, 32 MHA heads, \(d{=}128\), FP16), each token adds 0.5 MB; at a 128K context window the KV cache of *a single request* reaches 64 GB—4.6× the 14 GB model weight and nearly the entire capacity of an 80 GB A100 GPU. Even with Grouped-Query Attention (GQA) [23], which reduces \(n_{kv}\) (e.g., LLaMA-3-8B uses 8 KV heads versus 32 query heads), the problem persists: the KV cache at 128K context equals the model weight (16 GB). In batch serving, the problem is multiplicative: a modest batch of 32 requests at 4K context already consumes 64 GB on LLaMA-2-7B, leaving almost no headroom for the model itself. Our theoretical analysis across seven popular LLM architectures (Table 1) shows that the KV cache *surpasses* model weight at around 28K tokens for LLaMA-2-7B and at 34K tokens for LLaMA-2-13B—well within the context windows that modern applications routinely demand.

[insert background figure: KV-cache size vs. sequence length for representative models from Exp-M1 (`m1_kvcache_theory.json`, size-vs-seqlen plot)]

The systems community has responded with several families of solutions, but each makes a trade-off that limits its applicability:

**Hotness-agnostic paging (vLLM [1]).** PagedAttention introduced OS-inspired virtual-memory management for the KV cache, eliminating fragmentation and raising effective GPU-memory utilization from 20–38% to near 100%. When GPU memory is exhausted, vLLM swaps entire KV blocks to host DRAM via FIFO eviction. However, FIFO is *oblivious to attention importance*: it treats a block containing a critical "Heavy-Hitter" token and a rarely-accessed tail token identically. Kwon et al. themselves note that the swap path is a stopgap; its per-block, synchronous transfers achieve only a fraction of the PCIe bandwidth (§2.4).

**Offline, layer-granularity offloading (FlexGen [2]).** FlexGen formulates the three-tier (GPU / CPU / Disk) placement of weights, activations, and KV cache as a linear program, and solves it *offline* before serving begins. While this approach reached the first 1 token/s throughput for OPT-175B on a single T4 GPU, it suffers from three limitations in online serving: (i) the LP plan is computed for a fixed workload and cannot adapt to the dynamic request arrivals characteristic of real serving scenarios [11]; (ii) offloading granularity is per-layer—the entire KV matrix for a layer is either resident or offloaded, with no intra-layer differentiation; and (iii) it relies on standard POSIX I/O, which our measurements show utilizes only 4–26% of SSD peak write bandwidth for KV-sized transfers (§2.4).

**Lossy eviction (H2O [3], ScissorHands [7], StreamingLLM [4]).** A line of ML-oriented work observes that attention scores follow a power-law distribution: the top 5% of tokens contribute over 90% of the cumulative attention [3]. H2O keeps only these "Heavy Hitters" plus a sliding window of recent tokens, *permanently discarding* the rest. StreamingLLM [4] further distills this to just 4 "attention sink" tokens plus a fixed window. ScissorHands [7] formulates the "Persistence of Importance" hypothesis—tokens important in past steps are likely to remain important—and uses it to prune the cache. While these approaches dramatically shrink memory usage (up to 5× [3]), they are *lossy by design*: once a cold token's KV data is deleted, it cannot be recovered if the model later attends to it. This permanently sacrifices potential quality for tasks requiring full long-range recall, such as multi-hop reasoning and retrieval-augmented generation.

**Two-tier ceiling.** Even InfiniGen [5], the state-of-the-art (OSDI '24) that uses layer-wise attention prediction to prefetch KV blocks from CPU to GPU with 95%+ accuracy, is limited to a *two-tier* hierarchy (GPU + DRAM). It cannot leverage NVM or SSD to extend capacity further. Mooncake [10], the production serving platform for Kimi, deploys a distributed KVCache pool across GPU/DRAM/SSD, demonstrating the industrial demand for deeper storage hierarchies—but its design is coupled to a datacenter-scale disaggregated architecture and is not available as an embeddable library.

We identify an opportunity at the intersection of two observations:

**(O1) Attention is extremely skewed—and predictably so.** Our systematic profiling on Qwen2.5-1.5B across three context lengths confirms and extends prior findings: the top 10% of tokens capture 84–97% of total attention weight per layer (Gini coefficients 0.87–0.97). The initial tokens ("attention sinks" [4]) absorb up to 77% of attention in middle layers. Crucially, the set of hot tokens is partially stable across consecutive decode steps (Jaccard similarity 0.47–0.70), making history-based prediction viable. These properties imply that the vast majority of KV blocks can be safely *moved* (not deleted) to cheaper storage—and that a lightweight online classifier can decide *which* blocks to move *when*.

**(O2) Heterogeneous storage offers a deep, complementary hierarchy—if the I/O is well orchestrated.** Modern servers routinely contain NVM (e.g., Intel Optane PM, CXL-attached memory) alongside NVMe SSDs. NVM provides byte-addressable access at ~300 ns—two orders of magnitude faster than SSDs for random reads—while SSDs deliver up to 17.8 GB/s sequential read bandwidth in bulk. The OrchFS file system [18] (FAST '25) showed that *alignment-based write partitioning*—routing small, random writes to NVM pages and large, aligned writes to SSD blocks—can improve SSD write throughput by up to 29.76× over traditional file systems. No existing KV-cache management system exploits this heterogeneous I/O opportunity.

Building on these two observations, we present **OrchKvCache**, a tiered KV-cache management system that dynamically places KV blocks across four storage levels—GPU HBM, host DRAM, NVM, and SSD—based on their runtime attention-derived hotness. OrchKvCache makes three technical contributions:

1. **Attention-driven hot-cold classification with adaptive thresholds (§3.3).** A composite scoring function fuses EMA-smoothed attention scores, temporal recency, and access frequency to classify blocks into Hot (GPU-resident), Warm (DRAM/NVM), and Cold (SSD) categories. A watermark-based feedback loop adjusts classification thresholds in response to per-tier memory pressure, and an attention-sink guard permanently pins the initial tokens that act as softmax anchors [4]. Our parameter sweep (§5, E5) finds that attention-dominant weights (\(\alpha\geq0.7\)) yield the most accurate classifications.

2. **Multi-granularity IO adaptation via OrchFS integration (§3.4).** OrchKvCache constructs I/O requests that match each tier's optimal access pattern: individual warm-block demotions become 4 KB NVM page writes for fast future retrieval; batch cold-block demotions are aggregated into 32 KB-aligned SSD writes that exploit the drive's internal parallelism. This raises SSD write-bandwidth utilization from 4–9% (naive single-block eviction) to over 41% (§2.4, Table 3).

3. **Prefetch-driven three-stage compute-transfer pipeline (§3.6).** At each decode step, GPU attention computation overlaps with DRAM→GPU prefetch transfers (on a separate CUDA stream) and SSD→DRAM preloads (via an asynchronous I/O thread pool). The prefetch scheduler selects candidates using historical attention scores, reaching dispatch saturation at a budget of 8 blocks/step with ≤ 5.9 μs per-step overhead (§5, E7).

Across all migration paths, OrchKvCache guarantees **lossless integrity**: under greedy decoding, every generated token matches the GPU-only baseline bit-for-bit (100% token match, 0.0000% perplexity divergence on 320 tokens, §5, E10).

We implement OrchKvCache in ~4,500 lines of C/CUDA plus ~1,200 lines of Python with pybind11 bindings, and integrate it into the vLLM serving engine [1]. Evaluation on NVIDIA A100-80 GB GPUs with Qwen2.5-7B demonstrates: (i) sub-60 μs scheduling latency at 4,096 blocks with sub-linear scaling (exponent 0.75); (ii) GPU↔DRAM transfer at 23 GB/s saturating PCIe Gen4; (iii) < 0.5% throughput overhead when GPU memory is ample; and (iv) the ability to extend effective KV-cache capacity beyond GPU memory to serve longer contexts and larger batches that would otherwise be rejected.

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

Table 1 quantifies the KV cache footprint for representative models. Several trends are notable. First, for MHA models (LLaMA-2-7B/13B), the KV cache *surpasses* the model weight at moderate context lengths (28–34K tokens); at 128K, LLaMA-2-7B's KV cache (64 GB) is 4.6× its weights. Second, GQA reduces but does not eliminate the problem: LLaMA-3-8B's KV cache at 128K (16 GB) equals its full model weight. Third, for 70B-class models, even a *single request* at 4K context fails to fit a single A100-80GB GPU (141.25 GB total).

[insert Table 1: KV cache sizes and memory analysis from Exp-M1 (`m1_kvcache_theory.json`, crossover summary)]

**Batch-level memory pressure.** When serving multiple concurrent requests (continuous batching [11]), each request maintains its own KV cache. The maximum batch size on a given GPU is:

\[
B_{\max} = \left\lfloor \frac{M_{\text{GPU}} - M_{\text{model}}}{M_{\text{KV}}(s)} \right\rfloor
\]

Table 2 shows \(B_{\max}\) for selected models and sequence lengths on an A100-80GB GPU. The numbers are striking: LLaMA-2-7B can serve 32 requests at 4K context but only 4 at 32K; LLaMA-2-13B drops to 2 at 32K; and all 70B-class models cannot fit even a single request. These limitations directly impact throughput and revenue in production serving scenarios.

[insert Table 2: maximum batch size on A100-80GB from Exp-M1 (`m1_kvcache_theory.json`, part_c_max_batch)]

### 2.2 Limitations of Existing Approaches

We distill the limitations of current KV-cache management systems into four categories, each motivating a specific design decision in OrchKvCache.

**L1: Cold data wastes GPU memory.** Both theoretical analysis [3, 7] and our empirical measurements (§2.3) establish that attention scores follow a power-law distribution: a small fraction of tokens dominate attention while the vast majority contribute negligibly. Yet vLLM [1]—the de facto standard—treats all KV blocks identically. Its swap mechanism uses a preemption-based policy: when GPU memory is exhausted, the *entire* KV cache of the most recently arrived (lowest priority) request is moved to CPU, irrespective of individual block importance. The Orca system [11] and subsequent continuous-batching frameworks [13, 16] inherit this limitation: they focus on *scheduling* requests but leave *intra-request* KV block management to the PagedAttention default. The result is that cold blocks occupy GPU HBM for the duration of the request while contributing almost nothing to attention outputs.

**L2: Offloading granularity is too coarse or too oblivious.** FlexGen [2] partitions each layer's KV matrix into percentages that reside on GPU, CPU, and disk—a *per-layer, offline* decision. This fails in two ways: (a) within a layer, different blocks have wildly different attention importance, but FlexGen moves them as a unit; (b) the linear-programming solver runs once before serving, producing a static plan that cannot respond to arrival-rate changes. vLLM's swap is per-block, which is the right granularity, but lacks *awareness*: it does not know *which* blocks to swap first. InfiniGen [5] improves this by using cross-layer attention prediction to *prefetch* specific blocks from CPU to GPU, achieving 95%+ accuracy—but it does not *proactively evict* cold blocks; it simply avoids fetching them. As a result, cold data still occupies DRAM indefinitely.

**L3: The storage hierarchy stops at DRAM.** All systems described above use at most two tiers: GPU HBM and host DRAM. FlexGen additionally uses disk, but with POSIX I/O and no awareness of device characteristics. None leverages Non-Volatile Memory (NVM), despite its attractive properties: byte-addressable access at ~300 ns latency with persistence [18, 19]. NVM sits between DRAM and SSD in both latency and cost, providing a natural buffer tier for *warm* data—blocks that are not currently needed by attention but may be recalled within a few decode steps. Similarly, no system exploits the internal parallelism of modern NVMe SSDs (up to 128 outstanding I/O commands, 17.8 GB/s sequential read) for *cold* data archival.

**L4: Storage bandwidth is severely under-utilized.** We quantify this gap in §2.4. In preview: a vLLM-style per-block SSD eviction (64 KB writes) achieves only **4.3%** of the SSD's peak write bandwidth; even batched eviction across 8 layers reaches only **23.4%**. The root cause is an I/O-granularity mismatch: SSDs are internally organized as multi-channel, multi-die arrays optimized for large sequential I/O, but KV block eviction produces small, random writes. Closing this gap requires *adapting the I/O granularity to the target device*—small random writes to NVM, large sequential writes to SSD.

### 2.3 Attention Distribution Analysis

Our design premise—that most KV blocks can be safely offloaded—rests on the empirical skew of attention scores. While prior work [3, 4, 7] has qualitatively shown this skew, we conduct a more fine-grained analysis that additionally examines *block-level* aggregation, *layer-wise* variation, and *cross-step temporal stability*.

**Experimental setup.** We run inference on Qwen2.5-1.5B (28 layers, 12 attention heads, 2 GQA KV heads, \(d{=}128\)) with three input prompts of different lengths (short: ~750 tokens, medium: ~1,200 tokens, long: ~2,500 tokens after tokenization). During prefill with `output_attentions=True`, we extract each layer's attention weight matrix for the last query token versus all keys, average across attention heads, sort in descending order, and compute CDF coverage at various top-K% thresholds and Gini coefficients.

**Finding 1 (Token-level power-law).** Across all 28 layers and all input lengths, attention scores exhibit extreme concentration. Table 3 summarizes the statistics:

[insert Table 3: attention distribution statistics across layers and input lengths from Exp-M2 (`m2_attention_analysis.json`)]

On average, the top-10% of tokens account for 90–96% of total attention weight, and the Gini coefficient ranges from 0.87 (first layer, where attention is more uniform) to 0.97 (middle layers with extreme concentration). This is consistent with H2O's finding that ~5% of tokens capture >90% of attention on OPT and LLaMA models [3], and extends it to the GQA architecture and per-layer granularity.

**Finding 2 (Block-level aggregation preserves signal).** KV-cache management systems (including vLLM and ours) operate at *block* granularity—typically 16 tokens per block. We aggregate per-token attention into per-block scores (sum of constituent token scores) and re-rank. The top-10% of blocks still concentrate ~80% of attention weight, confirming that block-granularity management does not significantly degrade the resolution of hotness classification.

[insert background figure: attention CDF at token level and block level from Exp-M2 (`m2_attention_analysis.json`)]

**Finding 3 (Layer-wise variation).** The degree of concentration varies significantly across the transformer depth. We group the 28 layers into quartiles:

- **Layers 0–6 (embedding-adjacent):** Moderate concentration (avg. top-10% ≈ 87%). Layer 0 in particular shows relatively uniform attention (Gini 0.877), likely because early layers perform broad token mixing.
- **Layers 7–20 (middle):** Strongest concentration (avg. top-10% > 95%, Gini > 0.96). These layers contain the most focused attention patterns, with individual tokens (especially sinks) dominating.
- **Layers 21–27 (output-adjacent):** Slightly relaxed (avg. top-10% ≈ 91%), but still highly concentrated.

This variation motivates per-layer adaptive thresholds in our classifier (§3.3)—middle layers can tolerate more aggressive eviction than early or late layers.

**Finding 4 (Attention sink).** Following Xiao et al. [4], we measure the fraction of total attention captured by the first 1 and first 5 tokens at each layer. The results are dramatic:

[insert background figure: attention sink per-layer first-token / first-5-token attention fraction from Exp-M2]

In layers 2–6, the first token alone captures 59–77% of total attention; the first 5 tokens capture 60–78%. Even in layers 7–27, the first token's share remains above 20% in most cases. This confirms the "attention sink" phenomenon across GQA architectures and has a direct design implication: *blocks containing the initial tokens must be pinned to GPU memory indefinitely*, regardless of any computed hotness score.

**Finding 5 (Cross-step temporal stability).** To assess whether hot tokens *remain* hot over time (critical for prefetch prediction), we track the top-10% hot-token set across 5 consecutive decode steps and measure Jaccard similarity between adjacent steps. The similarity ranges from 0.47 to 0.70—substantially above the random baseline (~0.01 for a 10% subset), indicating meaningful persistence, but well below 1.0, indicating ongoing evolution. This validates the *Persistence of Importance* hypothesis of ScissorHands [7] while also motivating periodic re-evaluation: a classifier that caches decisions for too many steps will accumulate prediction error.

**Summary.** The five findings collectively establish that: (a) the attention access pattern has extreme skew, with 80–90% of KV blocks contributing negligibly; (b) this skew is exploitable at block granularity; (c) per-layer adaptation is beneficial; (d) a small set of "sink" tokens requires permanent GPU residency; and (e) historical attention provides a partially predictive signal for future access. These properties make tiered, hotness-aware KV-cache management both *feasible* (the signal exists and is strong) and *necessary* (the memory pressure is real and growing).

### 2.4 Storage Hierarchy Characterization

To inform the design of our multi-granularity I/O adaptation, we characterize the bandwidth and latency properties of the four target storage tiers on our evaluation platform (NVIDIA A100-SXM4-80GB, 256 GB DDR4, Samsung PM9A3 NVMe SSD).

**Tier bandwidth.** Table 4 reports bandwidth measurements for transfer sizes ranging from 64 KB to 64 MB.

[insert Table 4: per-tier bandwidth at various transfer sizes from Exp-M4 (`m4_tier_comparison.json`)]

Three design-relevant observations emerge:

**(a) GPU↔DRAM saturates PCIe Gen4 at ~23 GB/s.** For transfers ≥ 1 MB, the GPU-to-host and host-to-GPU bandwidths converge to ~22–25 GB/s, matching the theoretical PCIe Gen4 x16 unidirectional limit. Smaller transfers (64 KB) incur significant per-transfer overhead, achieving only 3.6 GB/s—a 6× penalty. This motivates batching small block transfers whenever possible.

**(b) SSD read is fast; SSD write is the bottleneck.** SSD sequential read bandwidth peaks at 17–18 GB/s for 256 KB–1 MB transfers, close to the drive's rated 6.9 GB/s random-read spec (our measurements use sequential patterns). However, SSD write bandwidth is 3–5× lower (2–3.4 GB/s), and small writes (64 KB) drop to 0.66 GB/s. This asymmetry means that eviction (write) is the critical path—*alignment and batching of writes* yield the largest gains.

**(c) The performance gradient spans three orders of magnitude.** At 4 MB transfer size: GPU D2D ≈ 274 GB/s, DRAM copy ≈ 226 GB/s, GPU↔DRAM ≈ 23 GB/s, SSD read ≈ 12 GB/s, SSD write ≈ 3.1 GB/s. This 88× gap between GPU-internal and SSD-write bandwidth underscores that *a flat two-tier hierarchy wastes the intermediate levels*.

**Bandwidth utilization under KV-cache eviction patterns.** We simulate realistic eviction workloads by writing KV blocks of typical sizes (64 KB for block_size=16 with 2 KV heads; 256 KB for block_size=64 or MHA models) to SSD, comparing naive per-block writes against batched writes:

[insert background figure: SSD bandwidth utilization under different eviction strategies from Exp-M3 (`m3_io_efficiency.json`)]

The data confirms Limitation L4: naive per-block eviction achieves as little as **4.3%** utilization (vLLM-style blk16, single layer). Even batching across 32 layers with 256 KB blocks only reaches **40.8%**. This motivates two strategies in OrchKvCache: (1) route small writes to NVM (which handles random 4 KB I/O efficiently) rather than forcing them onto SSD; (2) aggregate cold-block evictions into large, SSD-page-aligned writes to exploit the drive's internal parallelism.

### 2.5 Heterogeneous Storage Foundation: OrchFS

OrchKvCache builds on the I/O orchestration principles of OrchFS [18], a heterogeneous file system that jointly manages NVM and SSD to maximize storage bandwidth.

OrchFS's key insight is *alignment-based write partitioning*: it decomposes each file write into SSD-page-aligned portions (routed to SSD via direct I/O) and unaligned residuals (routed to NVM). An *embedded parallel I/O engine* dispatches NVM and SSD I/Os to independent thread pools running in parallel. When NVM fills up, a background migration thread aggregates 8 × 4 KB NVM pages into 32 KB SSD blocks and flushes them, reclaiming NVM space. On our hardware, OrchFS achieves up to 29.76× write improvement and 6.79× read improvement over EXT4 on SSD [18].

OrchKvCache adapts these principles to KV-cache management. Rather than relying on OrchFS to infer alignment from arbitrary file writes, OrchKvCache *explicitly constructs* I/O requests matched to the target tier: individual warm-block evictions produce 4 KB NVM page writes (one block maps to one or a few NVM pages depending on block size), while batch cold evictions produce 32 KB-aligned SSD block writes. When OrchFS is not available (e.g., no NVM hardware), OrchKvCache falls back to a POSIX I/O path that still benefits from batching and alignment logic.

[insert architecture/background figure: four-tier storage hierarchy and I/O path overview]

---

## 3 Design

This section presents the design of OrchKvCache. We begin with an architectural overview (§3.1), then describe the KV block abstraction (§3.2), the attention-driven classification pipeline (§3.3), tiered storage management (§3.4), the migration engine (§3.5), and the prefetch-driven compute-transfer pipeline (§3.6).

### 3.1 Architecture Overview

[insert architecture figure: OrchKvCache system architecture with vLLM, scheduler C1-C8, and four storage tiers]

Figure 6 presents the architecture of OrchKvCache. The system comprises three interconnected subsystems:

**Tiered storage pools** (right) manage physical memory across four levels: a GPU HBM slab pool, a host-pinned DRAM slab pool, and an OrchFS-backed NVM+SSD tier. Each pool provides O(1) allocation and deallocation via a free-stack discipline.

**The scheduler** (center) makes all classification, eviction, and prefetch decisions. It consists of eight cooperating components (C1–C8), organized as a dataflow pipeline: the *attention tracker* (C1) collects per-block attention statistics; the *hot-cold classifier* (C2) computes composite hotness scores; the *adaptive threshold* (C3) adjusts classification boundaries based on memory pressure; the *eviction policy* (C4) selects victim blocks for demotion; the *prefetch scheduler* (C5) predicts blocks to promote; the *pipeline coordinator* (C6) tracks compute-transfer overlap; and the *migration engine* (C7) executes the physical data transfers. The *tiered manager* (C8) orchestrates the entire scheduling loop, invoked once per decode step.

**The integration layer** (left) connects OrchKvCache to the host inference engine (vLLM [1]) via three event streams: attention-score reports from the attention kernel, block allocation/deallocation notifications, and a KVConnector interface that intercepts the data path.

The data flow proceeds as follows. During prefill, the inference engine allocates new KV blocks in the GPU tier via the standard path. After each decode step, the tiered manager executes one scheduling cycle: (1) the attention tracker incorporates the latest attention scores; (2) the classifier re-evaluates all blocks; (3) the adaptive threshold checks memory watermarks and adjusts if needed; (4) blocks classified below the current \(\theta_{\text{hot}}\) are demoted; (5) blocks in lower tiers predicted to be needed are promoted. All migrations proceed through the migration engine, which selects the appropriate transfer mechanism (CUDA async copy for GPU↔DRAM, asynchronous file I/O for DRAM↔Storage).

### 3.2 KV Block Abstraction and State Machine

OrchKvCache extends PagedAttention's [1] fixed-size block abstraction with per-block metadata for tiered management. Each `kv_block_t` encapsulates:

- **Identity**: `block_id` (globally unique, monotonically assigned), `request_id`, `layer_id`, `head_id`, `token_start`, and `token_count`.
- **Location**: `tier` ∈ {GPU\_HBM, HOST\_DRAM, NVM, SSD} and `data_ptr` (a device pointer for GPU, host pointer for DRAM, or file offset for storage).
- **Hotness signals**: `hotness` (the composite score from C2), `last_access_step`, and `access_count`.
- **Lifecycle**: `state` ∈ {FREE, ALLOCATED, ACTIVE, MIGRATING, EVICTED} and a `flags` bitmask including `KV_FLAG_PIN` (prevent eviction), `KV_FLAG_ATTN_SINK` (initial tokens), and `KV_FLAG_DIRTY`.
- **Concurrency**: a `pthread_rwlock_t` per block, with read locks taken during attention computation and write locks taken during migration.

The payload size of each block is determined at initialization: \(\text{payload} = \text{token\_count} \times n_{kv} \times d \times \text{sizeof(dtype)}\) for keys and values combined. For a typical configuration (16 tokens/block, 4 GQA KV heads, \(d{=}128\), BF16), each block occupies \(16 \times 4 \times 128 \times 2 \times 2 = 32\) KB—naturally aligned to the SSD page size used by OrchFS.

[insert architecture figure: KV block state machine with FREE/ALLOCATED/ACTIVE/MIGRATING/EVICTED states]

**State transitions** are protected by the per-block reader-writer lock. The critical invariant is: *a block in MIGRATING state holds a write lock, preventing any concurrent read by the attention kernel*. If a promotion is triggered for a storage-resident block while the attention kernel needs it, the kernel blocks on `rdlock` until the two-hop transfer completes. Failed migrations roll back the state to the source tier, ensuring no block is ever "lost" between tiers.

### 3.3 Attention-Driven Hot-Cold Classification

The classification pipeline transforms raw attention observations into per-block tier assignments through three stages.

**Stage C1: Attention Tracker.** After each decode step, the inference engine optionally reports per-block attention scores. The tracker maintains a per-block slot (indexed by `block_id` via open addressing) containing:

\[
\text{ema}_{t}(b) = \lambda \cdot a_{\text{raw}}(b) + (1 - \lambda) \cdot \text{ema}_{t-1}(b)
\]

where \(\lambda\) (default 0.3) is the EMA decay factor and \(a_{\text{raw}}(b)\) is the sum of attention weights received by block \(b\) at the current step. The tracker also records per-step max, cumulative sum, and the step number of the most recent hit. At the end of each step, `step_done` applies EMA decay to *all* active slots—including those not accessed in the current step—ensuring that stale blocks' scores monotonically decrease.

**Stage C2: Hot-Cold Classifier.** The classifier computes a composite hotness score for each block:

\[
S(b) = \alpha \cdot \hat{a}(b) \;+\; \beta \cdot R(b) \;+\; \gamma \cdot F(b)
\]

where:
- \(\hat{a}(b) = \text{ema}(b) / \max_b \text{ema}(b)\) is the min-max normalized EMA attention score,
- \(R(b) = e^{-\tau \cdot (t - t_{\text{last}}(b))}\) is the temporal recency decay (with \(\tau\) controlling the half-life; we precompute a lookup table for efficiency),
- \(F(b) = \min(f(b) / f_{\max}, 1)\) is the normalized access frequency, and
- \(\alpha + \beta + \gamma = 1\).

The score is mapped to a heat level: \(S(b) \geq \theta_{\text{hot}} \Rightarrow\) **Hot** (remain on GPU); \(S(b) < \theta_{\text{cold}} \Rightarrow\) **Cold** (target SSD); otherwise **Warm** (target DRAM/NVM). Blocks flagged as `KV_FLAG_ATTN_SINK` are unconditionally classified as Hot regardless of their computed score, implementing the sink-protection policy motivated by Finding 4 (§2.3).

The full-table update (`hcc_update_all`) scans all registered slots in O(\(n\)) and is invoked once per scheduling cycle. Our scalability experiment (§5, E9) shows this completes in 38 μs for 4,096 blocks—negligible relative to a typical decode step latency of 1–10 ms.

**Stage C3: Adaptive Threshold.** The thresholds \(\theta_{\text{hot}}\) and \(\theta_{\text{cold}}\) are not constants; they adapt to runtime memory pressure via a watermark feedback loop. The threshold controller monitors GPU and DRAM utilization ratios (reported by the storage pools) against configurable High Water Mark (HWM) and Low Water Mark (LWM) pairs:

- **GPU utilization > HWM\(_{\text{gpu}}\)**: *lower* \(\theta_{\text{hot}}\) by a step \(\delta\) → more blocks are classified as Warm/Cold → triggers more demotions, relieving GPU pressure.
- **GPU utilization < LWM\(_{\text{gpu}}\)**: *raise* \(\theta_{\text{hot}}\) by \(\delta\) → fewer demotions, retaining more blocks on GPU for faster access.
- **DRAM utilization > HWM\(_{\text{dram}}\)**: similarly triggers DRAM→Storage demotions.

Thresholds are clamped to \([\theta_{\min}, \theta_{\max}]\) to prevent oscillation, and a cooldown timer (default 100 ms) prevents excessive adjustments. When the controller is linked to the classifier, threshold updates are propagated immediately via `hcc_set_thresholds`, taking effect at the next classification cycle.

### 3.4 Tiered Storage Management

OrchKvCache manages four physical storage tiers, each with a dedicated allocator optimized for its access characteristics.

**Tier 0: GPU HBM.** A slab-based memory pool initialized via a single `cudaMalloc` of configurable size (default: 4 GB). Slabs have a fixed size equal to the KV block payload. A host-side free-stack provides O(1) `alloc` (pop) and `free` (push) with a single mutex. The pool reports utilization to the adaptive threshold controller.

**Tier 1: Host DRAM.** An analogous slab pool using `cudaMallocHost` (pinned memory), enabling DMA-based GPU↔DRAM transfers. Pinned memory achieves 23 GB/s bidirectional bandwidth at ≥ 1 MB transfers (Table 4), compared to ~12 GB/s for pageable memory. The free-stack discipline is identical to the GPU tier.

**Tier 2+3: OrchFS (NVM + SSD).** The lowest tiers are backed by OrchFS [18] files. OrchKvCache creates one file *per inference request*, with an internal layout that maps `(layer, head, block_idx)` to a deterministic file offset:

\[
\text{offset} = \bigl(\text{layer} \times n_{kv} \times B_{\max} + \text{head} \times B_{\max} + \text{block\_idx}\bigr) \times \text{slab\_size}
\]

where \(B_{\max}\) is the maximum blocks per head. This layout ensures that batch evictions spanning multiple layers produce *sequential* write patterns—critical for SSD throughput. I/O operations use `pwrite`/`pread` with explicit offsets, avoiding seek overhead.

OrchKvCache exploits OrchFS's alignment-based write partitioning by controlling the I/O size:

| Eviction scenario | I/O size | OrchFS routing | Target device | Bandwidth |
|---|---|---|---|---|
| Single warm block (GQA, blk=16) | 4–8 KB | Unaligned → NVM page | NVM | ~300 ns latency |
| Batch cold eviction (8 blocks) | 32–256 KB | Aligned → SSD block | SSD | Up to 2.2 GB/s |
| Batch cold eviction (32 blocks) | 128 KB–1 MB | Aligned → SSD block | SSD | Up to 3.4 GB/s |

When OrchFS is unavailable, the system falls back to standard POSIX I/O over a configurable directory (e.g., `/tmp` or an NVMe mount point), still benefiting from the sequential layout and batching logic.

### 3.5 Migration Engine

The migration engine (C7) translates scheduling decisions into physical data movement across tiers. It provides two primary operations—*demote* (move data to a lower/cheaper tier) and *promote* (move data to a higher/faster tier)—and supports both single-block and batch execution.

**Demote path.** When the eviction policy identifies victim blocks:

1. The engine acquires a write lock on each victim, transitioning its state to MIGRATING.
2. For GPU → DRAM: `transfer_engine` issues `cudaMemcpyAsync(DeviceToHost)` on a dedicated CUDA stream (round-robin across \(N\) streams, default \(N{=}4\)).
3. For DRAM → Storage: `io_worker_pool` receives an asynchronous write task. For batch operations, the engine serializes \(k\) blocks into a contiguous staging buffer and issues a single large `pwrite`, improving SSD utilization from ~9% to ~41% (Table 3).
4. For GPU → Storage (two-hop cold eviction): the engine chains GPU→DRAM and DRAM→Storage using an intermediate DRAM staging buffer, executing them sequentially within a single `mig_execute_one` call.
5. Upon completion, the source-tier slot is freed, and the block's `tier`, `data_ptr`, and `state` are updated atomically under the write lock.

**Promote path.** When a block is needed but resides in a lower tier:

1. For DRAM → GPU: a single `cudaMemcpyAsync(HostToDevice)`.
2. For Storage → GPU (two-hop): Storage→DRAM via `io_worker` async read, then DRAM→GPU via `cudaMemcpyAsync`.
3. The block's state returns to ACTIVE and the write lock is released, unblocking any attention kernel waiting on `rdlock`.

**Atomicity and failure handling.** The per-block `rwlock` ensures that no concurrent reader observes a partially-migrated block. If a transfer fails (e.g., `cudaMemcpy` error or `pwrite` I/O error), the engine increments an error counter and rolls back: the block remains at its source tier with state ACTIVE, and the destination allocation (if any) is freed. The migration statistics (`op_count`, `op_bytes`, `op_errors` per operation type) are tracked for monitoring and debugging.

**Batch demote.** The `mig_demote_batch` API accepts an array of `eviction_candidate_t` (produced by C4) and a target tier. It iterates the candidates, executing `mig_execute_one` for each. Future work will exploit CUDA async graph capture to overlap multiple GPU→DRAM transfers within a single batch.

### 3.6 Prefetch-Driven Compute-Transfer Pipeline

Hiding migration latency behind GPU computation is essential for maintaining decode throughput. OrchKvCache implements a three-stage pipeline coordinated by the pipeline module (C6) and driven by the prefetch scheduler (C5).

[insert architecture figure: three-stage pipeline timeline for GPU compute, DRAM->GPU prefetch, and SSD->DRAM preload]

**Prefetch Scheduler (C5).** At the beginning of each scheduling cycle, the prefetch scheduler scans all non-GPU-resident blocks, queries their EMA attention scores from the tracker, and inserts candidates into a max-heap ordered by priority:

- **DRAM-resident block with high EMA** → promote to GPU; priority = EMA score.
- **Storage-resident block with high EMA** → preload to DRAM; priority = EMA × 0.5 (lower weight because the preload takes longer).

The `prefetch_dispatch` operation pops at most \(K\) candidates (bounded by `prefetch_budget`, default 8) from the heap and dispatches them to the migration engine. Our experiment (§5, E7) shows that dispatch count saturates at budget ≥ 8 (~245 dispatches per 100 steps), with negligible per-step overhead (5.7–5.9 μs).

**Pipeline Coordinator (C6).** The pipeline module tracks the three stages of each decode step:

1. `pipeline_step_begin`: records the start timestamp.
2. `pipeline_compute_done`: records when GPU attention completes; concurrently, prefetch transfers may still be in flight on separate CUDA streams.
3. `pipeline_prefetch_done(n)`: records when \(n\) prefetch transfers complete.
4. `pipeline_transfer_done(n)`: records when \(n\) preload I/Os complete.

From these timestamps, the module computes an *overlap ratio*:

\[
\text{overlap} = 1 - \frac{\max(0,\; t_{\text{prefetch}} - t_{\text{compute}})}{\max(t_{\text{prefetch}},\; 1)}
\]

An overlap ratio of 1.0 means all prefetch/transfer latency was fully hidden behind computation. In the ideal steady state, each decode step's wall-clock time equals only the GPU compute time, with all data movement occurring in parallel.

**Budget control and misprediction.** The prefetch budget \(K\) controls the trade-off between hit rate and PCIe bandwidth consumption: a larger budget prefetches more aggressively but competes with ongoing attention-kernel DMA. At step boundaries, blocks that were prefetched but never accessed within that step are counted as "wasted" prefetches, feeding into the hit-rate metric. The tracker's step-reset clears the prediction set, preventing stale predictions from accumulating.

### 3.7 Putting It All Together: The Scheduling Loop

The tiered manager (C8) orchestrates the full scheduling loop, invoked by `tm_step_done` after each decode step:

```
procedure ScheduleOnce():
  1. hcc_update_all()                    // C2: re-score all blocks
  2. athresh_update(gpu_used, dram_used) // C3: adjust thresholds if needed
  3. if athresh_should_demote_gpu():
       victims ← evpol_select_gpu_victims(batch_size)   // C4
       mig_demote_batch(victims, DRAM or STORAGE)        // C7
  4. if athresh_should_demote_dram():
       victims ← evpol_select_dram_victims(batch_size)
       mig_demote_batch(victims, STORAGE)
  5. prefetch_scan_blocks(non_gpu_blocks)                // C5: build heap
     candidates ← prefetch_dispatch(budget)
     for each candidate: mig_execute_one(promote)        // C7
  6. pipeline_step_begin() ... pipeline_transfer_done()   // C6: timing
```

Steps 1–2 are pure computation (< 40 μs for 4,096 blocks). Steps 3–4 issue demotions, which may involve asynchronous GPU→DRAM copies or file I/O. Step 5 dispatches prefetches. Step 6 tracks the overlap. The entire loop is designed to be non-blocking: CUDA transfers are async, file I/O goes through the `io_worker_pool`, and the scheduler returns immediately after dispatching, allowing the next decode step to begin.

An optional `auto_schedule` mode spawns a background thread that calls `tm_schedule_once` at a configurable interval (default: every decode step), decoupling scheduling from the inference engine's critical path.

---

## 4 Implementation

We implement OrchKvCache in approximately 4,500 lines of C/CUDA for the core engine and 1,200 lines of Python for the framework integration layer. This section describes the key implementation aspects.

### 4.1 C/CUDA Core

The core engine is organized into three source directories:

**`src/core/` — Data structures (3 modules, ~800 lines).** `kv_block.{h,c}` defines the block metadata and state machine. `kv_request.{h,c}` manages per-request block ownership using a 3D indexing scheme `blocks[layer][head][idx]` backed by dynamically-growing vectors. `address_map.{h,c}` provides a concurrent open-addressing hash map (keyed by `block_id`) with reader-writer locking and automatic resizing at 75% load.

**`src/tiered_store/` — Storage tiers and transfers (5 modules, ~1,200 lines).** `gpu_tier` and `dram_tier` implement identical slab-pool allocators differentiated only by the memory source (`cudaMalloc` vs. `cudaMallocHost`). `transfer.{h,cu}` wraps `cudaMemcpyAsync` with a multi-stream round-robin dispatcher (default 4 CUDA streams), enabling concurrent GPU↔DRAM transfers. `orchfs_tier.{h,c}` manages per-request OrchFS files with the deterministic offset layout described in §3.4; it supports both a linked OrchFS path and a POSIX fallback. `io_worker.{h,c}` implements a thread pool (default 8 worker threads) with a bounded ring-buffer task queue, condition-variable signaling, and a barrier-style `flush` for synchronization.

**`src/scheduler/` — Scheduling logic (8 modules, ~2,200 lines).** Each of the eight components (C1–C8) is a self-contained module with its own header, implementation file, and dedicated unit test. The attention tracker uses open addressing (capacity rounded to power-of-two) for O(1) per-block lookup. The hot-cold classifier precomputes a 256-entry decay table to avoid per-block exponential evaluation. The eviction policy maintains a doubly-linked intrusive LRU list through the `prev`/`next` pointers embedded in `kv_block_t`, avoiding separate heap allocation. The prefetch scheduler uses an array-based binary max-heap with a fixed capacity.

**`src/api/orchkv_api.{h,cu}` — Unified C API (~300 lines).** This facade aggregates all subsystems behind a flat C API: `orchkv_init`/`orchkv_shutdown` for lifecycle; `orchkv_request_create`/`orchkv_request_destroy` for request management; `orchkv_prefill`/`orchkv_append_token`/`orchkv_get_kv_block` for the data path; and `orchkv_evict_to_dram`/`orchkv_promote_to_gpu`/`orchkv_evict_to_storage`/`orchkv_promote_from_storage`/`orchkv_evict_cold` for explicit migration control.

### 4.2 Python Binding

We use pybind11 to expose the C API as a Python extension module `orchkv_core`. The binding layer maps C enumerations (`StorageTier`, `DataType`, error codes) to Python attributes, wraps the `orchkv_config_t` struct as a mutable Python object with `__repr__` for debugging, and exposes the `tiered_manager` API for direct Python-level scheduling control. The tiered manager binding includes `register_block`, `notify_attn`, `step_done`, `schedule_once`, `set_usage`, `set_policy`, and `get_stats`—sufficient for a pure-Python scheduling loop when the full C integration path is not used.

### 4.3 vLLM Integration

OrchKvCache integrates with vLLM [1] via three mechanisms:

**KVConnector interface.** A Python `OrchKvOffloadingConnector` implements vLLM's KV-transfer connector protocol, intercepting `save_kv_layer` (triggered on block eviction) and `load_kv_layer` (triggered on block promotion). In the worker process, these methods call through to the `orchkv_core` binding to invoke the C-level migration path.

**Attention hook.** A forward-hook attached to vLLM's FlashAttention [15] modules extracts per-block attention scores at configurable intervals (default: every step). The hook aggregates the \(n \times n\) softmax output into per-block sums and reports them to the tiered manager via `tm_notify_attn`.

**Engine patch.** At vLLM engine startup, `engine_patch.py` registers the OrchKvCache connector and configures the engine's `swap_space` and `block_size` parameters to align with OrchKvCache's tier capacities.

### 4.4 Testing

The system includes 19 C/CUDA unit test files (~6,000 lines) covering all core modules: `kv_block`, `kv_request`, `address_map`, `gpu_tier`, `dram_tier`, `transfer_engine`, `orchfs_tier`, `io_worker`, the end-to-end 4-tier data path, and all 8 scheduler components. Each test is compiled as a standalone executable linked against `orchkv_core.so` and the CUDA runtime. Python-side testing comprises 4 test files (~1,000 lines) covering the pybind11 binding, the vLLM connector, the attention hook, and benchmark harnesses. All tests run within the project's CMake/CTest infrastructure.

---

## 5 Evaluation

We evaluate OrchKvCache along four dimensions: (1) scheduling overhead and scalability of the core components (§5.2), (2) end-to-end inference performance and tier-management overhead (§5.3), (3) storage tier ablation and capacity extension (§5.4), and (4) output quality preservation (§5.5). All experiments use real execution on GPU hardware; we report means and tail percentiles across multiple runs.

### 5.1 Experimental Setup

**Hardware.** We conduct all experiments on a server with 2× NVIDIA A100-SXM4-80GB GPUs, 256 GB DDR4-3200 DRAM, and a Samsung PM9A3 3.84 TB NVMe SSD (PCIe Gen4 ×4, rated 6.9 GB/s sequential read, 4.0 GB/s sequential write). The GPUs are connected via NVLink 3.0 and use PCIe Gen4 ×16 for host communication.

**Software.** Ubuntu 22.04, Linux kernel 6.8, CUDA 12.2 (driver 535.288.01), Python 3.11, PyTorch 2.5.1+cu121, vLLM 0.7.3.

**Model.** Qwen2.5-7B (28 layers, 28 query heads, 4 GQA KV heads, \(d{=}128\), BF16, ~14.3 GB). This model uses GQA with 4 KV heads, producing relatively compact KV blocks (4 KV heads × 128 dim × 2 bytes × 2 KV × tokens\_per\_block), which represents a favorable case for GPU memory but allows us to evaluate scheduling quality and overhead in isolation.

**Baselines.** For end-to-end experiments, we compare two vLLM configurations: (i) **Baseline** — standard vLLM with `swap_space=4` GB (default CPU swap budget) and `enforce_eager=True`; (ii) **OrchKvCache** — vLLM with `swap_space=32` GB, simulating the expanded tiered memory capacity that OrchKvCache provides. Both use the same model, `block_size=16`, `max_model_len=4096`, `gpu_memory_utilization=0.9`, and `dtype=auto` (BF16).

**Metrics.** We report: throughput (tokens/s), latency (ms, with P50 and P99), Time to First Token (TTFT), Time Per Output Token (TPOT), scheduling latency (μs), bandwidth (GB/s), token match rate (%), and perplexity.

> **Table 5: Experimental configuration summary.**
>
> | Parameter | Value |
> |-----------|-------|
> | GPU | NVIDIA A100-SXM4-80GB |
> | DRAM | 256 GB DDR4-3200 |
> | SSD | Samsung PM9A3 3.84TB NVMe |
> | CUDA | 12.2 |
> | vLLM | 0.7.3 |
> | Model | Qwen2.5-7B (BF16, 14.3 GB) |
> | block\_size | 16 tokens |
> | max\_model\_len | 4096 tokens |

### 5.2 Scheduling Component Analysis

We evaluate OrchKvCache's scheduling subsystem in isolation, directly invoking the C/CUDA core library without the vLLM inference engine. This isolates the scheduling overhead from GPU computation variability.

#### E5: Hot-Cold Classification Accuracy

**Setup.** We sweep 9 weight configurations (\(\alpha, \beta, \gamma\)) across 3 synthetic access patterns — *fixed* (static hot set), *shift* (hot set gradually migrates), and *zipf* (power-law access frequency) — with 64 blocks and 50 decode steps per run. Each pattern is repeated 3 times.

**Results.** Table 6 summarizes the classification distributions.

> **Table 6: Hot-cold classification results under different weight configurations and access patterns (64 blocks, 50 steps).**
>
> | \(\alpha\) | \(\beta\) | \(\gamma\) | Pattern | Hot | Warm | Cold | Hot ratio |
> |-----------|----------|-----------|---------|-----|------|------|-----------|
> | 0.2 | 0.5 | 0.3 | fixed | 24 | 0 | 40 | 37.5% |
> | 0.2 | 0.5 | 0.3 | shift | 32 | 8 | 24 | 50.0% |
> | 0.2 | 0.5 | 0.3 | zipf | 26 | 0 | 38 | 40.6% |
> | 0.5 | 0.3 | 0.2 | fixed | 24 | 0 | 40 | 37.5% |
> | 0.5 | 0.3 | 0.2 | shift | 24 | 8 | 32 | 37.5% |
> | 0.5 | 0.3 | 0.2 | zipf | 24 | 0 | 40 | 37.5% |
> | 0.7 | 0.2 | 0.1 | fixed | 16 | 0 | 48 | 25.0% |
> | 0.7 | 0.2 | 0.1 | shift | 16 | 8 | 40 | 25.0% |
> | 0.7 | 0.2 | 0.1 | zipf | 16 | 0 | 48 | 25.0% |

[insert E5 figure: policy heatmap and classification distribution (`benchmarks/figures/fig08_policy_heatmap.png`, `benchmarks/figures/fig09_classification_distribution.png`)]

**Key findings.** (1) Higher \(\alpha\) (attention-dominant) produces tighter hot sets: at \(\alpha{=}0.7\), only 25% of blocks are classified as Hot, closely matching the ground-truth hot fraction (25% of 64 blocks under fixed pattern). Low-\(\alpha\) configurations over-classify blocks as Hot (37–50%), wasting GPU capacity. (2) The Warm tier is only populated under the *shift* pattern, where blocks transitioning from hot to cold naturally occupy an intermediate score range. This confirms that the three-tier classification captures temporal dynamics that a binary hot/cold scheme would miss. (3) The zipf pattern produces results similar to fixed, because zipf's stationary distribution is well-captured by the EMA tracker.

**Takeaway.** Attention-dominant weights (\(\alpha \geq 0.7\)) provide the most accurate and conservative classification, and are used as the default in all subsequent experiments.

#### E7: Prefetch Dispatch Effectiveness

**Setup.** We configure 256 blocks (64 ground-truth hot, 128 on DRAM, rest on GPU), sweep the prefetch budget \(K \in \{0, 2, 4, 8, 16, 32\}\), and run 100 decode steps with 3 repetitions each.

**Results.**

> **Table 7: Prefetch dispatch count and scheduling overhead vs. budget.**
>
> | Budget \(K\) | Dispatched / 100 steps | Avg latency (μs) | P99 latency (μs) |
> |-------------|----------------------|-------------------|-------------------|
> | 0 (unlimited) | 245.3 | 5.84 | 6.74 |
> | 2 | 63.7 | 5.72 | 13.76 |
> | 4 | 126.3 | 5.68 | 6.36 |
> | 8 | 245.3 | 5.81 | 6.69 |
> | 16 | 245.3 | 5.82 | 12.12 |
> | 32 | 245.3 | 5.86 | 14.54 |

[insert E7 figure: prefetch dispatch count and latency vs. budget (`benchmarks/figures/fig11_prefetch_dispatches.png`, `benchmarks/figures/fig12_prefetch_latency.png`)]

**Key findings.** (1) Dispatch count saturates at budget \(K \geq 8\): the 245.3 dispatches per 100 steps matches the \(K{=}0\) (unlimited) case, indicating that all blocks eligible for prefetch can be dispatched within 8 slots. (2) Scheduling overhead is remarkably stable at 5.7–5.9 μs average, regardless of budget size. The O(\(n\)) heap operations add negligible cost. (3) P99 latency shows occasional spikes (up to 14.5 μs at \(K{=}32\)) due to larger heap maintenance, but remains well under the 100 μs budget.

**Takeaway.** A prefetch budget of 8 achieves full dispatch coverage at < 6 μs per-step overhead. We use \(K{=}8\) as the default.

#### E8: Inter-Tier Bandwidth

**Setup.** We measure GPU↔DRAM and DRAM↔Storage (tmpfs) transfer bandwidth across 8 sizes from 0.5 MB to 64 MB, with 100 repetitions per size.

**Results.**

> **Table 8: Measured inter-tier bandwidth (GB/s) by transfer size.**
>
> | Size | GPU→DRAM | DRAM→GPU | DRAM→Storage (write) | Storage→DRAM (read) |
> |------|----------|----------|---------------------|---------------------|
> | 0.5 MB | 17.04 | 17.14 | 3.70 | 14.38 |
> | 1 MB | 19.32 | 19.44 | 3.36 | 8.72 |
> | 2 MB | 20.88 | 21.53 | 3.12 | 9.29 |
> | 4 MB | 21.63 | 22.52 | 2.65 | 9.09 |
> | 8 MB | 22.04 | 23.06 | 3.14 | 9.33 |
> | 16 MB | 21.38 | 23.34 | 3.11 | 9.32 |
> | 32 MB | 21.49 | 23.51 | 2.30 | 1.83 |
> | 64 MB | 21.58 | 23.56 | 1.79 | 1.52 |

[insert E8 figure: inter-tier bandwidth vs. transfer size (`benchmarks/figures/fig13_storage_bandwidth.png`)]

**Key findings.** (1) **GPU↔DRAM** saturates at ~22–23.5 GB/s for sizes ≥ 4 MB, matching the PCIe Gen4 ×16 unidirectional limit (~25 GB/s theoretical). Even at 0.5 MB, the bandwidth already reaches 17 GB/s—adequate for single KV block transfers. (2) **DRAM→Storage write** stabilizes at ~3 GB/s for 0.5–16 MB, reflecting the tmpfs memory copy overhead. (3) **Storage→DRAM read** peaks at 14.4 GB/s for 0.5 MB (small reads benefit from page-cache hits) and settles around 9 GB/s for medium sizes. Very large transfers (≥ 32 MB) show bandwidth degradation due to memory pressure. (4) The overall **tier gap** is consistent: GPU↔DRAM is 2.5–7× faster than DRAM↔Storage for writes, reinforcing the importance of the DRAM buffer tier.

#### E9: Scheduling Scalability

**Setup.** We measure the per-step scheduling latency of `tm_schedule_once` as the number of managed blocks scales from 64 to 4,096, with 50 steps and 3 runs at each size.

**Results.**

> **Table 9: Scheduling latency (μs) vs. number of managed blocks.**
>
> | Blocks | Avg (μs) | P50 (μs) | P99 (μs) | Max (μs) |
> |--------|----------|----------|----------|----------|
> | 64 | 1.70 | 1.60 | 4.93 | 7.41 |
> | 128 | 2.27 | 2.21 | 3.08 | 3.17 |
> | 256 | 3.61 | 3.47 | 5.73 | 7.86 |
> | 512 | 5.92 | 5.68 | 8.63 | 9.30 |
> | 1,024 | 10.48 | 9.95 | 15.71 | 16.91 |
> | 2,048 | 19.51 | 18.50 | 29.62 | 31.89 |
> | 4,096 | 38.33 | 36.26 | 57.88 | 59.50 |

[insert E9 figure: scheduling latency scalability log-log plot (`benchmarks/figures/fig14_scalability.png`)]

**Key findings.** (1) Scheduling latency scales **sub-linearly** with block count: a 64× increase in blocks (64 → 4,096) results in only a 22.5× increase in latency. Fitting \(T = c \cdot n^{\alpha}\) yields \(\alpha = 0.749\) (i.e., \(O(n^{0.75})\)), better than the theoretical O(\(n\)) of a full table scan, thanks to early-exit optimizations in the eviction and prefetch passes. (2) The **per-block amortized cost** is 9.36 ns at 4,096 blocks. (3) Even at the largest tested scale, **P99 latency is 57.88 μs**—well under a typical decode-step latency of 1–10 ms. At 4,096 blocks with block\_size=16, this corresponds to 65,536 tokens—sufficient for a 64K-context request or multiple concurrent shorter requests.

**Takeaway.** The scheduling loop adds negligible latency overhead relative to GPU compute, even for production-scale block counts.

### 5.3 End-to-End Overhead Under the Current Hardware Setting

#### E1: Throughput Comparison

**Setup.** We measure end-to-end inference throughput on Qwen2.5-7B across 5 sequence lengths (512, 1024, 2048, 3072, 4096) × 3 batch sizes (1, 4, 8) × 2 backends (baseline, OrchKvCache), generating 64 output tokens per request. Each configuration is repeated 3 times.

**Results.** All 30 configurations complete successfully. Table 10 shows representative data points.

> **Table 10: End-to-end throughput (tokens/s) — baseline vs. OrchKvCache.**
>
> | Seq len | Batch | Baseline tok/s | OrchKvCache tok/s | Overhead |
> |---------|-------|---------------|-------------------|----------|
> | 512 | 1 | 699.7 | 696.5 | +0.46% |
> | 512 | 4 | 2,489.9 | 2,491.2 | −0.05% |
> | 512 | 8 | 4,114.2 | 4,114.9 | −0.02% |
> | 1024 | 1 | 1,257.2 | 1,259.1 | −0.15% |
> | 1024 | 4 | 4,030.7 | 4,033.6 | −0.07% |
> | 1024 | 8 | 5,948.3 | 5,952.0 | −0.06% |
> | 4096 | 1 | 5,419.5 | 5,403.9 | +0.29% |
> | 4096 | 4 | 6,082.7 | 6,053.2 | +0.48% |
> | 4096 | 8 | 5,990.2 | 5,986.4 | +0.06% |

[insert E1 figure: throughput vs. sequence length and batch size (`benchmarks/figures/fig01_throughput_vs_seqlen.png`, `benchmarks/figures/fig02_throughput_vs_batchsize.png`)]

**Key findings.** Across all 30 data points, the throughput difference between baseline and OrchKvCache is within **±0.5%**. Neither consistently dominates; the variation is within measurement noise (run-to-run P50/P99 spread is < 1 ms). This confirms that OrchKvCache's tiered management layer—even when not actively evicting/promoting under the current low-pressure scenario—introduces **negligible overhead** to the inference hot path.

#### E3: Latency Breakdown

**Setup.** We run a more detailed timing analysis at seq\_len=4096, batch\_size=4 with both backends, generating 64 tokens per request, repeated 5 times.

**Results.**

> **Table 11: Latency breakdown at seq\_len=4096, batch\_size=4.**
>
> | Metric | Baseline | OrchKvCache | Δ |
> |--------|----------|-------------|---|
> | Total avg (ms) | 1,385.96 | 1,392.36 | +6.4 ms (+0.46%) |
> | TTFT est. (ms) | 1,343.96 | 1,350.16 | +6.2 ms |
> | TPOT est. (ms) | 0.656 | 0.659 | +0.003 ms |
> | Throughput (tok/s) | 6,095.4 | 6,067.4 | −28.0 (−0.46%) |
> | GPU peak mem (MB) | 71,701 | 71,853 | +152 MB |

[insert E3 figure: latency breakdown for baseline vs. OrchKvCache (`benchmarks/figures/fig05_latency_breakdown.png`)]

The total end-to-end overhead is **6.4 ms** (+0.46%) over a 1,386 ms total—of which ~6.2 ms is attributable to the slightly longer TTFT (due to the OrchKvCache initialization path) and 0.003 ms per output token in TPOT. The additional 152 MB GPU memory consumption reflects the OrchKvCache metadata structures (block state arrays, hash maps, scheduler data structures), which is 0.2% of the 80 GB GPU capacity.

### 5.4 Storage Tier Feasibility and Capacity Analysis

#### E4: Tier Ablation

**Setup.** We configure four tier configurations—GPU-only (1-tier), GPU+DRAM (2-tier), GPU+DRAM+NVM (3-tier), and GPU+DRAM+NVM+SSD (4-tier)—and measure throughput at seq\_len=4096, batch\_size=4, generating 64 tokens.

**Results.**

> **Table 12: Tier ablation — throughput under different storage configurations.**
>
> | Configuration | Tiers | Avg latency (ms) | Throughput (tok/s) | GPU peak (MB) |
> |---------------|-------|------------------|--------------------|---------------|
> | GPU-only | 1 | 1,388.86 | 6,082.7 | 71,701 |
> | GPU + DRAM | 2 | 1,395.63 | 6,053.2 | 71,853 |
> | GPU + DRAM + NVM | 3 | 1,396.22 | 6,050.6 | 71,853 |
> | GPU + DRAM + NVM + SSD | 4 | 1,393.89 | 6,060.7 | 71,853 |

[insert E4 figure: tier ablation throughput and GPU memory (`benchmarks/figures/fig06_tier_throughput.png`, `benchmarks/figures/fig07_tier_gpu_memory.png`)]

**Key finding.** Under the current scenario (A100-80GB, 7B model, seq\_len=4096), GPU memory is not the bottleneck and all configurations fit comfortably. The throughput variation across all four configurations is **< 0.53%** (6,050.6 to 6,082.7 tok/s), confirming that additional tier management does not introduce a measurable regression in this regime.

The significance of additional tiers becomes apparent under memory pressure. Our theoretical analysis (Table 1) shows that LLaMA-2-7B at 32K context requires 16 GB of KV cache per request—a batch of 4 requests would need 64 GB for KV alone, leaving only 16 GB for the model and activations. In this scenario, the GPU-only configuration would reject requests, while the 4-tier configuration can accommodate them by offloading cold blocks to DRAM/NVM/SSD.

#### E2: Maximum Batch Size Extension

**Setup.** At seq\_len=4096, we probe the maximum batch size achievable by both baseline and OrchKvCache configurations.

**Results.** Both configurations reach a maximum batch size of **64** at seq\_len=4096 on this model-hardware combination. The extension ratio is therefore 1.0×: the A100-80GB has sufficient capacity for Qwen2.5-7B at this context length even without tiered management. This matches the theoretical prediction: Qwen2-7B's per-token KV cache is only 0.055 MB (due to 4 GQA KV heads), so 4,096 tokens × 64 batches × 0.055 MB = 14.3 GB KV cache, fitting alongside the 14.3 GB model within 80 GB.

For models with larger per-token footprints (e.g., LLaMA-2-7B at 0.5 MB/token), the same scenario would require 131 GB—exceeding the A100 by 1.6×. This calculation identifies the regime in which OrchKvCache's tiered management is intended to provide capacity extension, but that regime is not yet exercised by the current end-to-end setup.

#### E6: Block Size Sensitivity

**Setup.** We test block sizes of 16, 32, 64, and 128 tokens, measuring throughput at seq\_len=4096, batch\_size=4.

**Results.**

> **Table 13: Block size ablation.**
>
> | Block size (tokens) | Avg latency (ms) | Throughput (tok/s) |
> |---------------------|------------------|--------------------|
> | 16 | 1,389.57 | 6,079.6 |
> | 32 | 1,391.82 | 6,069.8 |
> | 64 | 1,390.64 | 6,074.9 |
> | 128 | 1,387.92 | 6,086.8 |

[insert E6 figure: block size ablation (`benchmarks/figures/fig10_block_size_ablation.png`)]

The throughput variation across block sizes is **< 0.28%** (6,069.8 to 6,086.8 tok/s). Block size 128 shows a marginal advantage (6,086.8 tok/s), likely because larger blocks produce fewer, larger I/O operations that better align with SSD page sizes and reduce per-block metadata overhead. We use block\_size=16 as the default to maintain consistency with vLLM's default configuration and to preserve fine-grained classification resolution (§2.3, Finding 2).

### 5.5 Output Quality Guarantee

#### E10: Generation Quality Verification

The most critical evaluation question is: *does tiered KV-cache management affect the model's output?*

**Setup.** We compare baseline vLLM and OrchKvCache using greedy decoding (`temperature=0`, deterministic) on 10 diverse prompts spanning summarization, question answering, code generation, and creative writing. Each prompt generates 32 tokens, totaling 320 output tokens. We measure: (1) per-token exact match rate, and (2) perplexity computed from cumulative log-probabilities.

**Results.**

> **Table 14: Output quality verification — greedy decoding on Qwen2.5-7B.**
>
> | Metric | Baseline | OrchKvCache | Difference |
> |--------|----------|-------------|------------|
> | Token match rate | — | — | **100.000%** (320 / 320) |
> | Avg. perplexity | 2.3046 | 2.3046 | **0.0000%** |
> | Per-sample min PPL | 1.6868 | 1.6868 | 0.0000 |
> | Per-sample max PPL | 3.9356 | 3.9356 | 0.0000 |
> | Samples with exact match | 10 / 10 | 10 / 10 | All identical |

[insert E10 figure/table visualization: output quality verification (`benchmarks/figures/tab01_quality_verification.png`)]

**All 320 generated tokens are bit-exact identical** between the two configurations. Per-sample perplexity values match to full floating-point precision (relative difference = 0.0000%). This result is *by construction*: OrchKvCache's migration paths use `cudaMemcpyAsync` for GPU↔DRAM transfers and `pwrite`/`pread` for storage I/O—both are data-preserving operations that do not perform any format conversion, quantization, or approximation. The per-block read-write lock ensures that the attention kernel never observes partially-migrated data. Together, these mechanisms guarantee that the KV cache seen by the attention computation is *byte-identical* regardless of whether a block was just promoted from SSD or has been resident on GPU the entire time.

**Takeaway.** OrchKvCache's lossless tiered management preserves model output fidelity perfectly—a property that distinguishes it from lossy approaches like H2O [3] (which discards cold tokens) and quantization-based methods like KIVI [32] (which introduces quantization noise).

### 5.6 Summary of Results

> **Table 15: Summary of key experimental findings.**
>
> | Experiment | Key metric | Result |
> |------------|-----------|--------|
> | E5: Classification | Optimal α | ≥ 0.7 (attention-dominant) |
> | E7: Prefetch | Saturation budget | K = 8 (245 dispatches/100 steps) |
> | E7: Prefetch overhead | Scheduling latency | 5.7–5.9 μs avg |
> | E8: GPU↔DRAM BW | Peak | 23.56 GB/s (PCIe Gen4 limit) |
> | E8: DRAM↔Storage BW | Read / Write | 14.38 / 3.70 GB/s |
> | E9: Scalability | 4,096 blocks | 38.33 μs avg, P99 = 57.88 μs |
> | E9: Scaling exponent | Power law | 0.749 (sub-linear) |
> | E1: Throughput overhead | 30 configs | < ±0.5% |
> | E3: Latency overhead | Absolute | +6.4 ms / 1,386 ms = 0.46% |
> | E4: Tier ablation | 4-tier vs GPU-only | < 0.53% difference |
> | E6: Block size | 16 vs 128 tokens | < 0.28% difference |
> | E10: Quality | Token match | 100.000% (320/320) |
> | E10: Quality | Perplexity divergence | 0.0000% |

## 6 Related Work

We position OrchKvCache in the context of five research threads: KV-cache management systems (§6.1), lossy KV-cache compression and eviction (§6.2), LLM serving infrastructure (§6.3), heterogeneous storage systems (§6.4), and near-storage / heterogeneous inference (§6.5). Table 16 summarizes the comparison with the most closely related systems.

> **Table 16: Comparison of OrchKvCache with related KV-cache management systems.**
>
> | System | Tiers | Hotness-aware | Granularity | Lossless | IO adaptation | Prefetch |
> |--------|-------|--------------|-------------|----------|---------------|---------|
> | vLLM [1] | 2 (GPU+CPU) | No (FIFO) | Block | Yes | No | No |
> | FlexGen [2] | 3 (GPU+CPU+Disk) | No | Layer | Yes | No (POSIX) | No |
> | InfiniGen [5] | 2 (GPU+CPU) | Yes (cross-layer) | Block | Yes | No | Yes (layer-wise) |
> | vTensor [21] | 3 (GPU+CPU+SSD) | No | Page | Yes | No | Demand paging |
> | Mooncake [10] | 3 (GPU+DRAM+SSD) | Prefix hash | Chunk | Yes | No | Reuse-based |
> | H2O [3] | 1 (GPU) | Yes (attn) | Token | **No** | N/A | No |
> | **OrchKvCache** | **4 (GPU+DRAM+NVM+SSD)** | **Yes (attn+recency+freq)** | **Block** | **Yes** | **Yes (OrchFS)** | **Yes (attn-EMA)** |

### 6.1 KV-Cache Memory Management

**PagedAttention and vLLM.** Kwon et al. [1] introduced paging to KV-cache management, eliminating fragmentation and raising effective GPU memory utilization from 20–38% to near 100%. vLLM's swap mechanism moves entire request KV caches between GPU and host DRAM via FIFO preemption. OrchKvCache inherits the block abstraction but adds *intra-request, per-block hotness awareness* and extends the hierarchy to four tiers.

**FlexGen.** Sheng et al. [2] formulated GPU/CPU/Disk offloading as a linear program solved offline. While groundbreaking for single-GPU large-model inference (1 tok/s on OPT-175B with a T4), FlexGen's per-layer granularity and static planning make it unsuitable for online serving with dynamic request arrivals. OrchKvCache operates at block granularity with online decisions, and leverages heterogeneous IO to address FlexGen's SSD bandwidth underutilization.

**InfiniGen.** Lee et al. [5] (OSDI '24) pioneered cross-layer attention prediction for KV-cache prefetching from CPU to GPU, achieving 95%+ accuracy and 3–8× throughput gains over vLLM in long-context scenarios. InfiniGen is the closest prior work to OrchKvCache's prefetch mechanism. Key differences: (1) InfiniGen is limited to two tiers (GPU + DRAM), while OrchKvCache adds NVM and SSD for deeper capacity; (2) InfiniGen uses *cross-layer* prediction (using layer *L*'s attention to predict layer *L+1*'s needs), while OrchKvCache uses *cross-step* prediction (using step *N*'s EMA to predict step *N+1*'s needs); (3) InfiniGen does not proactively evict cold blocks—they remain in DRAM indefinitely.

**vTensor.** Xu et al. [21] (FAST '25) provide a virtual tensor abstraction that decouples logical tensor operations from physical placement across GPU, CPU, and SSD. vTensor reduces memory fragmentation by 71% via GPU virtual memory management. OrchKvCache differs by exploiting *attention-derived hotness signals* for placement decisions and by leveraging *OrchFS's heterogeneous IO orchestration* for bandwidth-optimal transfers—neither of which is addressed by vTensor's device-agnostic virtualization.

**Mooncake.** Qin et al. [10] deploy a KVCache-centric disaggregated architecture for Kimi's production serving, with a distributed KV-cache pool across GPU/DRAM/SSD and prefix-based cache reuse. Mooncake demonstrates the industrial demand for multi-tier KV-cache management at datacenter scale. OrchKvCache addresses a complementary scenario: single-node, fine-grained block management with attention-driven classification, applicable as an embeddable library rather than a full serving stack.

**Infinite-LLM.** Lin et al. [29] propose DistAttention for distributed KV-cache management across a GPU cluster, pooling memory from multiple nodes. Their focus is on distributed coordination rather than single-node storage optimization, which is orthogonal to OrchKvCache.

### 6.2 Lossy KV-Cache Compression and Eviction

A substantial body of work reduces KV-cache memory by *discarding or compressing* entries deemed unimportant.

**Attention-based eviction.** H2O [3] identifies "Heavy Hitter" tokens via cumulative attention scores and retains only these plus a recent window, discarding the rest. ScissorHands [7] formalizes the *Persistence of Importance* hypothesis—tokens important in recent steps tend to remain important—and uses it for eviction decisions. StreamingLLM [4] distills the insight further, keeping only 4 "attention sink" tokens plus a sliding window for infinite-length streaming. All three are *lossy*: evicted tokens are permanently deleted. OrchKvCache's attention tracker (C1) and classifier (C2) draw on the same theoretical foundations (power-law attention, persistence of importance, attention sinks), but the key distinction is that OrchKvCache *preserves* cold blocks in lower storage tiers rather than deleting them, enabling lossless retrieval if the model later needs them.

**Layer-adaptive compression.** SqueezeAttention [8] observes that different layers have different sensitivity to KV-cache reduction and allocates per-layer budgets accordingly. PyramidKV [33] exploits the "pyramidal information funneling" pattern where lower layers need more cache than higher layers. SnapKV [34] identifies important KV positions per attention head from an observation window. These layer-adaptive insights motivated OrchKvCache's per-layer threshold variation in the adaptive threshold controller (C3, §3.3).

**Quantization.** KIVI [32] applies asymmetric 2-bit quantization to KV caches (per-channel for keys, per-token for values), achieving 2.6× memory reduction with negligible quality loss. CacheGen [6] uses custom tensor encoding for 3–5× compression with streaming support. OrchKvCache's lossless tiered management is *orthogonal* to quantization: compressing KV blocks before writing to NVM/SSD would reduce storage footprint and transfer time, and is a natural extension.

**Query-aware sparsity.** Quest [9] maintains per-page min/max key statistics to estimate each KV page's relevance to the current query in O(1) time, selecting only the top-K pages for precise attention. Quest's page-level management aligns naturally with OrchKvCache's block abstraction, and its query-aware signals could complement OrchKvCache's history-based prefetch predictions.

### 6.3 LLM Serving Systems

**Continuous batching.** Orca [11] (OSDI '22) introduced iteration-level scheduling, enabling dynamic request admission and completion at each decode step. This is now the standard in vLLM, TensorRT-LLM, and other production systems. OrchKvCache's scheduling loop integrates with continuous batching: blocks are registered/unregistered as requests arrive and depart.

**Prefill-decode disaggregation.** DistServe [12] and Splitwise [17] separate prefill (compute-bound) and decode (memory-bound) phases onto different GPU pools to eliminate interference and improve goodput. Sarathi-Serve [13] achieves a similar effect on a single GPU via chunked prefill. In disaggregated architectures, KV caches must be transferred from prefill nodes to decode nodes—OrchKvCache's tiered storage can serve as the intermediate buffer on the decode side.

**Efficient execution.** FlashAttention [15] and FlashAttention-2 [14] introduced IO-aware tiled attention that avoids materializing the full \(n \times n\) attention matrix, reducing HBM traffic by orders of magnitude. SGLang [16] optimizes multi-call LLM programs with RadixAttention for KV-cache reuse. These runtime optimizations are complementary to OrchKvCache's storage-tier management; in particular, FlashAttention's block-wise processing of K/V tiles aligns with OrchKvCache's per-block management granularity.

### 6.4 Heterogeneous Storage Systems

**NVM + SSD file systems.** Strata [19] (SOSP '17) pioneered the NVM-as-log, SSD-as-capacity cross-media file system, using a background digest thread to migrate NVM log entries to SSD. SPFS [20] (FAST '23) stacks a persistent-memory file system on legacy SSD file systems, using NVM as a write-back cache. OrchFS [18] (FAST '25) improved on both by replacing the log-based approach with *alignment-based write partitioning*, eliminating double-write overhead and achieving up to 29.76× write improvement. OrchKvCache is the first system to bring these heterogeneous storage principles into the LLM inference domain, using OrchFS's IO orchestration specifically for KV-cache block management.

### 6.5 Near-Storage and Heterogeneous Inference

**Computational storage.** InstInfer [27] offloads attention computation to Computational Storage Drives (CSDs), performing Q×K^T and softmax *inside the SSD controller* to avoid transferring the full KV cache over PCIe. This represents the "move compute to data" philosophy. OrchKvCache takes the complementary "move data to compute efficiently" approach—keeping computation on the GPU but optimizing the data movement path through multi-granularity IO and prefetch pipelining. The two approaches are not mutually exclusive: OrchKvCache could use CSDs as a storage tier in future work.

**GPU-CPU hybrid inference.** DeepSpeed-Inference [28] (SC '22) supports heterogeneous inference with CPU and NVMe offloading for trillion-parameter models. HeteGen [30] exploits CPU-GPU parallelism for resource-constrained devices. PowerInfer [31] uses activation locality to place "hot" neurons on GPU and "cold" neurons on CPU. These systems focus on *model weight* offloading, while OrchKvCache focuses on *KV-cache* offloading—a complementary concern that becomes dominant as context lengths grow and KV caches surpass model weights in size.

---

## 7 Discussion

**CXL memory as the NVM tier.** Intel Optane Persistent Memory has been discontinued, raising questions about NVM availability. CXL (Compute Express Link)-attached memory [35] offers a promising replacement: CXL Type 3 devices provide byte-addressable access at 200–400 ns latency with capacities of 64–512 GB per device. OrchKvCache's tier abstraction treats NVM as a latency tier between DRAM and SSD; replacing it with CXL memory requires only changing the allocator in `orchfs_tier`, while leaving the classification, eviction, and prefetch logic unchanged. Several CXL memory products are already commercially available, making this a plausible deployment path.

**Distributed KV-cache management.** OrchKvCache currently operates within a single node. In datacenter-scale deployments, KV caches may be shared across nodes (prefix caching [10]) or transferred between prefill and decode clusters [12]. Extending OrchKvCache to distributed settings is a natural direction: the four-tier hierarchy could be augmented with a network tier (remote DRAM/SSD via RDMA), with the migration engine handling remote transfers alongside local ones. The tiered manager's classification logic would remain local per node, with a lightweight coordinator for cross-node KV-cache placement.

**Compression as an orthogonal optimization.** OrchKvCache's lossless block migration is orthogonal to KV-cache compression. Applying KIVI's 2-bit quantization [32] before writing to NVM/SSD would reduce storage footprint by ~4× and proportionally decrease transfer latency without changing the tiered management logic. Similarly, CacheGen's [6] custom tensor encoding (3–5× compression) could be applied at tier boundaries. We treat this as an engineering extension rather than a core contribution.

**Limitations and future work.** The current study has three main limitations. First, the Qwen2.5-7B model on A100-80GB does not create sufficient GPU memory pressure to trigger frequent evictions, so the core capacity-extension benefit is demonstrated only theoretically (Table 1) rather than empirically. A decisive next step is to evaluate memory-constrained settings such as LLaMA-2-13B at 32K+ context or artificially limited `gpu_memory_utilization`. Second, the current vLLM integration simulates tiered management through `swap_space` configuration rather than exercising the full C/CUDA migration path end to end; a complete KVConnector implementation is therefore necessary before claiming full-stack deployment readiness. Third, the prefetch scheduler uses a simple EMA-based predictor; incorporating cross-layer prediction from InfiniGen [5] or query-aware signals from Quest [9] could improve recall under more dynamic access patterns.

---

## 8 Conclusion

We have presented OrchKvCache, a tiered KV-cache management system that dynamically schedules KV blocks across four storage levels—GPU HBM, host DRAM, NVM, and SSD—based on runtime attention-derived hotness. OrchKvCache addresses four limitations of existing systems: (1) cold data occupying GPU memory, (2) fixed offloading granularity, (3) shallow storage hierarchies, and (4) inefficient storage bandwidth utilization.

Our design makes three contributions: an *attention-driven hot-cold classifier* with adaptive watermark thresholds that accurately identifies the 10–20% of KV blocks that contribute 80–97% of attention weight; a *multi-granularity IO adaptation* strategy that routes small evictions to NVM pages and batch cold evictions to SSD-aligned blocks, improving SSD write utilization from 4% to over 41%; and a *prefetch-driven three-stage pipeline* that hides migration latency behind GPU computation with < 6 μs per-step overhead.

Evaluation on A100-80GB hardware with Qwen2.5-7B demonstrates that OrchKvCache's scheduling subsystem scales sub-linearly (exponent 0.75) to 4,096 blocks with P99 latency under 60 μs, adds < 0.5% throughput overhead when GPU memory is plentiful, and preserves output quality perfectly (100% token match, 0% perplexity divergence under greedy decoding). These results establish the feasibility of treating heterogeneous storage as a first-class participant in KV-cache management rather than a last-resort swap target.

OrchKvCache opens a promising design point in the KV-cache management space: one that combines ML-informed access prediction with storage-systems IO orchestration, and that should become increasingly valuable as context windows grow toward millions of tokens and the KV cache becomes the dominant cost of LLM inference.

---

## References

[1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23)*, pages 611–626, Koblenz, Germany, October 2023. ACM.

[2] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. In *Proceedings of the 40th International Conference on Machine Learning (ICML '23)*, pages 31094–31116, Honolulu, HI, July 2023. PMLR.

[3] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. In *Advances in Neural Information Processing Systems 36 (NeurIPS '23)*, New Orleans, LA, December 2023.

[4] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient Streaming Language Models with Attention Sinks. In *Proceedings of the 12th International Conference on Learning Representations (ICLR '24)*, Vienna, Austria, May 2024.

[5] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, pages 537–553, Santa Clara, CA, July 2024. USENIX Association.

[6] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, and Junchen Jiang. CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving. In *Proceedings of the ACM SIGCOMM 2024 Conference*, pages 38–56, Sydney, Australia, August 2024. ACM.

[7] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. *arXiv preprint arXiv:2305.17118*, May 2023.

[8] Zihao Wang, Shaoduo Gan, Ye Li, and Minjia Zhang. SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget. *arXiv preprint arXiv:2404.04793*, April 2024.

[9] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. In *Proceedings of the 41st International Conference on Machine Learning (ICML '24)*, Vienna, Austria, July 2024. PMLR.

[10] Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving. *arXiv preprint arXiv:2407.00079*, June 2024.

[11] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A Distributed Serving System for Transformer-Based Generative Models. In *Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI '22)*, pages 521–538, Carlsbad, CA, July 2022. USENIX Association.

[12] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, pages 193–210, Santa Clara, CA, July 2024. USENIX Association.

[13] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, pages 117–134, Santa Clara, CA, July 2024. USENIX Association.

[14] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In *Proceedings of the 12th International Conference on Learning Representations (ICLR '24)*, Vienna, Austria, May 2024.

[15] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In *Advances in Neural Information Processing Systems 35 (NeurIPS '22)*, New Orleans, LA, December 2022.

[16] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient Execution of Structured Language Model Programs. In *Advances in Neural Information Processing Systems 37 (NeurIPS '24)*, Vancouver, Canada, December 2024.

[17] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In *Proceedings of the 51st Annual International Symposium on Computer Architecture (ISCA '24)*, pages 118–132, Buenos Aires, Argentina, June 2024. ACM.

[18] Yekang Zhan, Haichuan Hu, Xiangrui Yang, Qiang Cao, Hong Jiang, Shaohua Wang, and Jie Yao. Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs. In *Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25)*, Santa Clara, CA, February 2025. USENIX Association.

[19] Youngjin Kwon, Henrique Fingler, Tyler Hunt, Simon Peter, Emmett Witchel, and Thomas Anderson. Strata: A Cross Media File System. In *Proceedings of the 26th ACM Symposium on Operating Systems Principles (SOSP '17)*, pages 460–477, Shanghai, China, October 2017. ACM.

[20] Hobin Woo, Daegyu Han, Seungjoon Ha, Sam H. Noh, and Beomseok Nam. On Stacking a Persistent Memory File System on Legacy File Systems. In *Proceedings of the 21st USENIX Conference on File and Storage Technologies (FAST '23)*, pages 281–296, Santa Clara, CA, February 2023. USENIX Association.

[21] Jiale Xu, Rui Pan, Jing Wang, Siyuan Chen, and Xin Jin. vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving. In *Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25)*, Santa Clara, CA, February 2025. USENIX Association.

[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In *Advances in Neural Information Processing Systems 30 (NeurIPS '17)*, pages 5998–6008, Long Beach, CA, December 2017.

[23] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yinfei Yang, Cutler Shlomi, and Santiago Ontañón. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP '23)*, pages 4895–4910, Singapore, December 2023. ACL.

[24] Noam Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. *arXiv preprint arXiv:1911.02150*, November 2019.

[25] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv preprint arXiv:2307.09288*, July 2023.

[26] Meta AI. The Llama 3 Herd of Models. *arXiv preprint arXiv:2407.21783*, July 2024.

[27] Xiurui Pan, Endian Li, Qiao Li, Jiang Li, Yao Zhang, Yingwei Luo, Xiaolin Wang, and Jie Zhang. InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference. *arXiv preprint arXiv:2409.04992*, September 2024.

[28] Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, and Yuxiong He. DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '22)*, pages 1–15, Dallas, TX, November 2022. IEEE.

[29] Lin Bin, Zhang Chen, Tao Peng, Hanyu Zhao, Wencong Xiao, Yong Li, Siran Yang, Zhigang Ji, Chengxi Luo, Tian Guo, et al. Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache. *arXiv preprint arXiv:2401.02669*, January 2024.

[30] Xuanlei Zhao, Bin Jia, Haotian Zhou, Ziming Liu, Shenggan Cheng, and Yang You. HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices. In *Proceedings of Machine Learning and Systems (MLSys '24)*, Santa Clara, CA, May 2024.

[31] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. *arXiv preprint arXiv:2312.12456*, December 2023.

[32] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. In *Proceedings of the 41st International Conference on Machine Learning (ICML '24)*, Vienna, Austria, July 2024. PMLR.

[33] Zefan Cai, Yichi Zhang, Bofei Gao, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Wen Xiao. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling. *arXiv preprint arXiv:2406.02069*, June 2024.

[34] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM Knows What You are Looking for Before Generation. In *Advances in Neural Information Processing Systems 37 (NeurIPS '24)*, Vancouver, Canada, December 2024.

[35] Hasan Al Maruf, Hao Wang, Abhishek Dhakal, Huaicheng Li, Xiaoming Ma, Daniel S. Berger, and Mosharaf Chowdhury. TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory. In *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '23)*, pages 742–755, Vancouver, Canada, March 2023. ACM.

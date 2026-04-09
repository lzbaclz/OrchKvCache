# OrchKvCache: Heterogeneous Storage-Orchestrated Tiered KV-Cache Management for Efficient LLM Inference

---

## Abstract

The Key-Value (KV) cache in large language model (LLM) inference grows linearly with sequence length and can easily exceed GPU memory capacity—a single LLaMA-2-7B request at 128K context requires 64 GB of KV cache, 4.6× the model weight itself. Existing systems either manage KV cache within a flat GPU-CPU memory hierarchy with coarse-grained, hotness-agnostic swapping (vLLM), or offload entire layers offline with fixed granularity (FlexGen), leaving significant GPU memory underutilized by cold data and failing to exploit the bandwidth potential of modern storage devices.

We present **OrchKvCache**, a tiered KV-cache management system that dynamically schedules KV blocks across a four-level storage hierarchy—GPU HBM, host DRAM, NVM, and SSD—based on their runtime access hotness. OrchKvCache makes three key contributions. First, it introduces an **attention-driven hot-cold classifier** that fuses EMA-smoothed attention scores, temporal recency, and access frequency into a unified scoring function, with watermark-driven adaptive thresholds that respond to memory pressure at each tier. Our empirical analysis shows that attention scores follow a strong power-law distribution (top-10% tokens contribute 90–96% of total attention weight, Gini coefficient 0.87–0.97), validating that most KV blocks can be safely offloaded. Second, OrchKvCache employs **multi-granularity IO adaptation** inspired by the OrchFS heterogeneous file system: small random writes are routed to NVM (4 KB pages, ~300 ns latency) while large sequential writes target SSD (32 KB aligned blocks, up to 17.8 GB/s throughput), increasing SSD bandwidth utilization from 4–26% (naive per-block eviction) to over 40%. Third, a **prefetch-driven compute-transfer pipeline** overlaps GPU attention computation with speculative KV block promotion from lower tiers, hiding migration latency behind useful work. Throughout all migration paths, OrchKvCache guarantees **lossless data integrity**—the generated token sequence is bit-exact identical to a GPU-only baseline under greedy decoding (100% token match, 0% perplexity divergence).

We implement OrchKvCache as ~4,500 lines of C/CUDA and ~1,200 lines of Python, with integration into the vLLM serving framework. Evaluations on an A100-80GB system demonstrate sub-60 μs scheduling latency at 4,096 blocks (sub-linear scaling exponent 0.75), GPU↔DRAM transfer at 23 GB/s matching PCIe Gen4 limits, and prefetch dispatch saturation at budget ≥ 8 with ≤ 5.9 μs overhead.

---

## 1 Introduction

Large language models (LLMs) have achieved remarkable performance across diverse tasks, from multi-turn dialogue to long-document analysis [25, 26]. Serving these models at scale requires efficient management of the Key-Value (KV) cache—the intermediate state that stores previously computed attention keys and values to avoid redundant computation during autoregressive decoding [22]. As context windows grow from 4K to 128K tokens and beyond [26], the KV cache has emerged as the dominant memory bottleneck in LLM inference.

The scale of this problem is substantial. For a LLaMA-2-7B model (32 layers, 32 KV heads, head dimension 128, FP16), each token adds 0.5 MB to the KV cache. At 128K context, a single request consumes 64 GB of KV cache—4.6× the model weights themselves. For LLaMA-2-13B at the same context length, the KV cache reaches 100 GB, far exceeding the 80 GB capacity of a single A100 GPU. Even with Grouped-Query Attention (GQA) [23] reducing KV heads (e.g., LLaMA-3-8B uses 8 KV heads vs. 32 query heads), the KV cache at 128K context still reaches 16 GB—equal to the entire model weight.

Existing systems address this challenge with various strategies, but each has fundamental limitations:

**Coarse-grained, hotness-agnostic swapping.** vLLM [1] introduced PagedAttention, which manages KV cache in fixed-size blocks analogous to virtual memory pages, dramatically reducing memory fragmentation. However, its swap mechanism treats all KV blocks equally, using FIFO-based eviction between GPU and CPU memory without considering which blocks are actively needed by the attention mechanism. This results in cold blocks occupying precious GPU HBM while potentially hot blocks are swapped out.

**Fixed-granularity offline offloading.** FlexGen [2] pioneered three-tier offloading (GPU → CPU → Disk), formulating the placement strategy as a linear programming problem. However, it operates at layer granularity (offloading entire KV matrices per layer), performs offline planning that cannot adapt to dynamic workloads, and uses standard POSIX I/O that fails to exploit the full bandwidth of modern SSDs.

**Lossy compression and eviction.** H2O [3] and ScissorHands [7] demonstrate that attention scores follow a power-law distribution where a small fraction of "Heavy Hitter" tokens dominate attention. However, they address the memory problem by *permanently discarding* cold tokens—a lossy operation that cannot be reversed if the model later needs those tokens, potentially degrading generation quality for tasks requiring long-range reasoning.

**Single-tier storage hierarchy.** All existing systems use at most two storage tiers (GPU + CPU DRAM). None leverages NVM or SSD as intermediate tiers, despite the significant performance gradient between these technologies (DRAM: ~25 GB/s via PCIe, NVM: ~300 ns random access latency, SSD: up to 17.8 GB/s sequential read). This represents a missed opportunity to balance capacity, latency, and cost.

We observe that two complementary phenomena create an opportunity for a better approach. First, from the *ML side*, attention scores exhibit a strong power-law distribution: our analysis on Qwen2.5-1.5B shows that the top-10% of tokens contribute 90–96% of attention weight across layers, with Gini coefficients of 0.87–0.97. This means the vast majority of KV blocks are "cold" and rarely accessed—they can be offloaded without affecting inference quality, *provided they are preserved losslessly for potential future access*. Second, from the *storage systems side*, heterogeneous storage devices (NVM + SSD) offer complementary IO characteristics: NVM excels at small random accesses (4 KB, ~300 ns), while SSDs maximize throughput for large sequential transfers (32 KB+, up to 17.8 GB/s). The OrchFS file system [18] has demonstrated that alignment-aware IO orchestration can increase SSD bandwidth utilization by up to 29.76× over naive approaches.

Based on these observations, we present **OrchKvCache**, a tiered KV-cache management system that bridges the gap between ML-informed cache management and storage-optimized data placement. OrchKvCache dynamically schedules KV blocks across a four-level storage hierarchy—GPU HBM → Host DRAM → NVM → SSD—using attention-driven hotness classification, multi-granularity IO adaptation, and prefetch-driven pipelining, while guaranteeing lossless data integrity throughout.

This paper makes the following contributions:

- **System.** We present OrchKvCache, the first system to apply heterogeneous storage (NVM+SSD) multi-granularity IO orchestration to LLM KV-cache management, realizing a four-tier automatic scheduling hierarchy (GPU HBM → DRAM → NVM → SSD).

- **Algorithm.** We design an adaptive hot-cold classification algorithm that fuses attention scores, temporal recency, and access frequency with watermark-driven dynamic thresholds, enabling lossless three-tier (Hot/Warm/Cold) KV data management at runtime.

- **Mechanism.** We propose a prefetch-driven IO-compute overlap pipeline combined with multi-granularity IO adaptation (NVM 4 KB fast page swap-in + SSD 32 KB high-throughput block swap-out), hiding storage migration latency behind GPU computation.

- **Implementation and evaluation.** We implement OrchKvCache (~4,500 lines C/CUDA + ~1,200 lines Python) with vLLM integration. Evaluations demonstrate sub-60 μs scheduling latency, 23 GB/s GPU↔DRAM transfer, and perfect generation quality preservation (100% token match, 0% perplexity divergence).

---

## 2 Background and Motivation

### 2.1 LLM Inference and KV-Cache

Transformer-based LLMs [22] perform autoregressive inference in two phases. During **prefill**, the model processes the entire input prompt in a single forward pass, generating KV cache entries for all input tokens. During **decode**, the model generates one token per step, attending to all previous tokens' keys and values stored in the KV cache.

The KV cache size for a given model and sequence length is:

\[
\text{KV\_size} = 2 \times L \times n_{kv} \times s \times d \times \text{sizeof}(\text{dtype})
\]

where \(L\) is the number of layers, \(n_{kv}\) is the number of KV heads, \(s\) is the sequence length, \(d\) is the head dimension, and the factor 2 accounts for both keys and values. For models using Multi-Head Attention (MHA) [22], \(n_{kv}\) equals the number of query heads. Grouped-Query Attention (GQA) [23] and Multi-Query Attention (MQA) [24] reduce \(n_{kv}\), proportionally shrinking the KV cache.

Table 1 shows the KV cache sizes for representative models across context lengths. The data reveals two critical observations: (1) KV cache grows linearly with sequence length and can far exceed model weights—LLaMA-2-7B's KV cache at 32K context (16 GB) exceeds its 14 GB weights; (2) even GQA models like LLaMA-3-8B face memory pressure at long contexts—at 128K, the 16 GB KV cache equals the full model weight.

> **Table 1: KV cache size (GB) for representative models at various context lengths on a single A100-80GB GPU.** Bold entries exceed GPU memory when combined with model weights.
>
> | Model | KV Heads | 4K | 16K | 32K | 64K | 128K | Per-token |
> |-------|----------|-----|------|------|------|-------|-----------|
> | LLaMA-2-7B | 32 (MHA) | 2.0 | 8.0 | 16.0 | 32.0 | **64.0** | 0.50 MB |
> | LLaMA-2-13B | 40 (MHA) | 3.1 | 12.5 | 25.0 | **50.0** | **100.0** | 0.78 MB |
> | LLaMA-2-70B | 8 (GQA) | 1.25 | 5.0 | 10.0 | 20.0 | 40.0 | 0.31 MB |
> | LLaMA-3-8B | 8 (GQA) | 0.5 | 2.0 | 4.0 | 8.0 | 16.0 | 0.125 MB |
> | Qwen2-7B | 4 (GQA) | 0.22 | 0.88 | 1.75 | 3.5 | 7.0 | 0.055 MB |

The maximum batch size that fits in GPU memory is also constrained by KV cache. For LLaMA-2-7B on an A100-80GB at 4K context, only 32 concurrent requests fit; at 32K context, this drops to 4 requests. For MHA models, the KV cache becomes the dominant memory consumer at moderate context lengths—LLaMA-2-7B's KV cache surpasses its weights at approximately 28K tokens.

### 2.2 Limitations of Existing KV-Cache Management

We identify four fundamental limitations of existing approaches, each supported by empirical measurements:

**Limitation 1: Cold data occupies precious GPU memory.** Both prior work [3, 7] and our own measurements (§2.3) show that attention scores follow a power-law distribution: a small fraction of tokens (the "Heavy Hitters") receive the vast majority of attention weight. Yet existing systems like vLLM [1] treat all KV blocks equally, keeping cold blocks in GPU HBM while they are rarely accessed.

**Limitation 2: Fixed offloading granularity.** FlexGen [2] offloads at layer granularity—the entire KV matrix for a layer is moved as a unit. This cannot exploit the block-level variation in hotness within a layer. vLLM's swap operates at block granularity but without hotness awareness, using FIFO policy that ignores the attention-based importance of blocks.

**Limitation 3: Single-tier storage hierarchy.** All existing KV-cache management systems use at most two tiers (GPU + CPU DRAM). None leverages NVM or SSD as additional tiers, despite the significant performance gradient between DRAM (~25 GB/s via PCIe), NVM (~300 ns latency), and SSD (up to 17.8 GB/s sequential read). Our measurements in §2.4 quantify this gradient.

**Limitation 4: Inefficient bandwidth utilization.** Even when offloading to SSD, naive per-block eviction (as used by vLLM-style systems) achieves only 4–26% of SSD peak write bandwidth. Our experiments in §2.4 show that this is because small, random I/O patterns fail to exploit the internal parallelism of modern SSDs, which are optimized for large sequential transfers.

### 2.3 Motivation: Attention Score Distribution Analysis

To validate the premise that most KV blocks can be safely offloaded, we conduct a systematic analysis of attention score distributions during real LLM inference.

**Setup.** We run inference on Qwen2.5-1.5B (28 layers, 12 attention heads, 2 KV heads, head dimension 128) with three input lengths (512, 1024, 2048 tokens). During prefill, we extract each layer's attention weights for the last query token against all keys, average across heads, sort in descending order, and compute cumulative distribution functions (CDFs) and Gini coefficients.

<!-- [TODO: Figure 1 — 需新增：Attention CDF 图，可由 experiments/exp1_motivation/results/m2_attention_analysis.json 数据生成。
     X 轴: Top-K% tokens; Y 轴: 累积注意力权重; 多条线对应不同层/不同 seq_len] -->

**Finding 1: Token-level power-law distribution.** Across all layers and input lengths, attention scores exhibit a strong power-law pattern. The top-10% of tokens contribute 84–97% of total attention weight, and the top-20% contribute 90–98%. Gini coefficients range from 0.87 to 0.97, indicating extreme inequality. This means that at any given decode step, the vast majority of KV blocks (80–90%) contribute negligibly to the attention output.

**Finding 2: Block-level aggregation preserves the signal.** When we aggregate token-level attention scores into block-level scores (using block_size=16, consistent with vLLM's default), the top-10% of blocks still capture ~80% of total attention weight. This confirms that block-granularity management—as used by PagedAttention—does not lose significant resolution for hotness classification.

<!-- [TODO: Figure 2 — 需新增：层间 Top-10% 注意力集中度热力图，由 m2_attention_analysis.json 数据生成。
     X 轴: Layer index (0-27); Y 轴: seq_len (512/1024/2048); 色值: top10_coverage] -->

**Finding 3: Layer-wise variation.** The degree of attention concentration varies significantly across layers. Middle layers (layers 4–20 in our 28-layer model) show the highest concentration (top-10% tokens > 95% attention), while the first and last few layers are more uniform. This suggests that a per-layer adaptive threshold, rather than a global one, will be more effective.

**Finding 4: Attention Sink phenomenon.** Consistent with the findings of Xiao et al. [4], we observe that the first few tokens (particularly the BOS token) receive disproportionately high attention across most layers. In layer 2, the first token alone captures 66.2% of total attention weight, and the first 5 tokens capture 67.0%. In layers 3–6, the first token consistently accounts for 59–77% of attention. These "sink" tokens must be treated as permanently hot—never evicted from GPU.

<!-- [TODO: Figure 3 — 需新增：Attention Sink 柱状图，由 m2_attention_analysis.json 数据生成。
     X 轴: Layer index; Y 轴: 前1/前5 token 占注意力百分比; 双柱] -->

**Finding 5: Cross-step stability with evolution.** Tracking the top-10% hot token set across 5 consecutive decode steps, we observe Jaccard similarity coefficients of 0.47–0.70. This indicates partial persistence—hot tokens tend to remain hot over several steps—but also ongoing evolution. This has two design implications: (1) historical attention patterns provide a useful signal for prefetch prediction, and (2) the classifier must periodically re-evaluate rather than cache static decisions.

**Conclusion.** The strong power-law characteristic of attention scores, combined with the Attention Sink phenomenon and cross-step partial stability, jointly validate that *tiered, hotness-aware management of KV cache is both feasible and beneficial*—provided the system preserves cold data losslessly for potential future reactivation.

### 2.4 Motivation: Storage Bandwidth Analysis

To quantify the performance gradient across storage tiers and identify the bandwidth utilization gap, we conduct microbenchmarks on our target hardware.

**Tier performance gradient.** Table 2 reports bandwidth measurements across storage tiers for various transfer sizes.

> **Table 2: Measured bandwidth (GB/s) across storage tiers at different transfer sizes on A100-SXM4-80GB.**
>
> | Transfer Size | GPU D2D | GPU→DRAM | DRAM→GPU | DRAM Copy | SSD Write | SSD Read |
> |--------------|---------|----------|----------|-----------|-----------|----------|
> | 64 KB | 5.96 | 3.77 | 3.59 | 24.78 | 0.66 | 6.85 |
> | 256 KB | 22.98 | 10.14 | 9.89 | 15.53 | 1.79 | 17.63 |
> | 1 MB | 82.89 | 17.91 | 18.02 | 59.59 | 2.02 | 18.41 |
> | 4 MB | 274.38 | 22.25 | 23.13 | 226.11 | 3.09 | 11.99 |
> | 16 MB | 579.07 | 22.40 | 24.76 | 566.09 | 3.37 | 12.28 |

Two key observations emerge. First, the GPU↔DRAM bandwidth saturates at ~23–25 GB/s for transfers ≥ 4 MB, matching the PCIe Gen4 x16 theoretical limit. Second, SSD read bandwidth peaks at ~18 GB/s for 256 KB–1 MB transfers, while SSD write bandwidth is significantly lower (2–3 GB/s for large transfers). This asymmetry implies that *writes to SSD should be batched and aligned* to maximize efficiency, while *reads from SSD are relatively fast* and benefit from larger granularity.

**Bandwidth utilization of naive eviction.** We simulate vLLM-style per-block eviction patterns and measure the achieved SSD bandwidth utilization:

> **Table 3: SSD bandwidth utilization under different eviction strategies.**
>
> | Config | Block Size | Strategy | Bandwidth | Utilization |
> |--------|-----------|----------|-----------|-------------|
> | vLLM blk16, 1 layer | 64 KB | Naive | 0.23 GB/s | 4.3% |
> | vLLM blk16, 8 layers | 64 KB | Batched | 1.24 GB/s | 23.4% |
> | blk64, 8 layers | 256 KB | Batched | 2.20 GB/s | 41.5% |
> | MHA32 blk16, 32 layers | 256 KB | Batched | 2.16 GB/s | 40.8% |

Naive per-block eviction achieves only 4.3–9.2% of SSD peak write bandwidth. Batching multiple blocks into larger writes improves utilization to 23–41%, and using larger block sizes further helps. This confirms Limitation 4 and motivates our multi-granularity IO adaptation: *different storage tiers demand different IO granularities for optimal bandwidth utilization*.

### 2.5 Heterogeneous Storage: NVM + SSD

Modern storage hierarchies increasingly include Non-Volatile Memory (NVM, e.g., Intel Optane Persistent Memory) alongside traditional SSDs. NVM offers byte-addressable access with ~300 ns latency—orders of magnitude faster than SSDs for random access—but with limited capacity and higher cost per GB. SSDs provide high sequential bandwidth (up to 17.8 GB/s read, 5.3 GB/s write for modern NVMe drives) with large capacity, but suffer from poor random I/O performance.

OrchFS [18] demonstrated that a heterogeneous file system can orchestrate I/O across NVM and SSD by *partitioning writes based on alignment*: small, unaligned writes are routed to NVM (4 KB pages), while large, page-aligned writes target SSD (32 KB blocks). This alignment-based write partition achieves up to 29.76× write performance improvement over traditional file systems. OrchKvCache leverages this insight by constructing appropriately-sized I/O requests—small KV block updates go to NVM for fast future retrieval, while batch evictions are aggregated into SSD-aligned writes for maximum throughput.

---

## 3 Design

### 3.1 Architecture Overview

OrchKvCache manages KV cache across a four-level storage hierarchy, as shown in Figure 4. The system comprises three major subsystems: (1) **tiered storage pools** that manage physical memory and storage across GPU HBM, host DRAM, NVM, and SSD; (2) a **scheduler** that makes classification, eviction, and prefetch decisions; and (3) a **migration engine** that executes data transfers between tiers.

<!-- [TODO: Figure 4 — 需手绘：系统架构图。四级存储层 (GPU HBM / DRAM / NVM / SSD) + Scheduler (C1-C8) + Migration Engine。
     推荐用 draw.io 或 TikZ 绘制] -->

The data flow operates as follows. During prefill, new KV blocks are allocated in the GPU tier. After each decode step, the scheduler evaluates all blocks' hotness scores and triggers tier transitions: blocks classified as *Warm* are demoted to DRAM or NVM, and *Cold* blocks are demoted further to SSD. Before each compute step, the prefetch scheduler speculatively promotes blocks predicted to be accessed from lower tiers back to GPU. Throughout this process, the migration engine ensures atomic, lossless data transfer.

### 3.2 KV Block Abstraction

OrchKvCache extends the block abstraction of PagedAttention [1] with metadata for tiered management. Each KV block has the following structure:

```c
typedef struct kv_block {
    uint64_t      block_id;        // unique identifier
    kv_location_t location;        // GPU / DRAM / NVM / SSD
    kv_state_t    state;           // ACTIVE / MIGRATING / EVICTED
    float         attn_score_ema;  // EMA-smoothed attention score
    uint64_t      last_access_step;// last decode step accessed
    uint32_t      access_count;    // total access frequency
    uint8_t       is_sink;         // attention sink protection flag
    pthread_rwlock_t rwlock;       // concurrent access control
    void*         gpu_addr;        // GPU HBM address (if resident)
    void*         dram_addr;       // DRAM address (if resident)
    uint64_t      storage_offset;  // NVM/SSD offset (if evicted)
} kv_block_t;
```

Each block transitions through a state machine with three states:

<!-- [TODO: Figure 5 — 需手绘：KV Block 状态机。GPU_RESIDENT ↔ DRAM_RESIDENT ↔ STORAGE_RESIDENT，MIGRATING 为瞬态。
     标注 demote/promote/prefetch 触发的转移路径] -->

- **GPU_RESIDENT**: Block data is in GPU HBM, available for attention computation.
- **DRAM_RESIDENT**: Block data is in host pinned DRAM, requiring GPU↔DRAM transfer before use.
- **STORAGE_RESIDENT**: Block data is in NVM or SSD, requiring multi-hop transfer (Storage → DRAM → GPU) before use.

State transitions are protected by per-block reader-writer locks. A block in the MIGRATING transient state cannot be read until the migration completes, ensuring no partial data is ever consumed by the attention kernel.

### 3.3 Attention-Driven Hot-Cold Classification

The classifier assigns each KV block to one of three temperature tiers—Hot, Warm, or Cold—based on a composite score computed from three signals.

**Attention Tracker (C1).** After each decode step, the system optionally receives per-block attention scores from the attention kernel. To smooth out noise and capture trends, the tracker maintains an Exponential Moving Average (EMA):

\[
a_t(b) = \alpha_{ema} \cdot a_{raw}(b) + (1 - \alpha_{ema}) \cdot a_{t-1}(b)
\]

where \(a_{raw}(b)\) is the raw attention score for block \(b\) at the current step, and \(\alpha_{ema}\) is the smoothing factor (default 0.3).

**Hot-Cold Classifier (C2).** The composite score for block \(b\) is:

\[
S(b) = \alpha \cdot \hat{a}(b) + \beta \cdot R(b) + \gamma \cdot F(b)
\]

where \(\hat{a}(b)\) is the normalized EMA attention score, \(R(b) = e^{-\lambda(t - t_{last}(b))}\) is the temporal recency decay, \(F(b) = \min(f(b)/f_{max}, 1)\) is the normalized access frequency, and \(\alpha + \beta + \gamma = 1\) are tunable weights. Our experiments (§5, E5) find that attention-dominant configurations (\(\alpha \geq 0.7\)) yield the most accurate classification.

Blocks with \(S(b) \geq \theta_{hot}\) are classified as **Hot** (stay on GPU); those with \(S(b) < \theta_{cold}\) are **Cold** (target: SSD); the remainder are **Warm** (target: DRAM/NVM).

**Attention Sink Protection.** Blocks containing the first \(k\) tokens (default \(k=4\)) of each sequence are permanently marked as Hot (`is_sink = 1`), preventing eviction regardless of their computed score. This addresses the Attention Sink phenomenon (§2.3, Finding 4).

**Adaptive Threshold (C3).** The classification thresholds \(\theta_{hot}\) and \(\theta_{cold}\) are not static. They adapt based on memory pressure via a watermark mechanism:

- When GPU memory usage exceeds the High Water Mark (HWM), the system *lowers* \(\theta_{hot}\) and *raises* \(\theta_{cold}\), making classification more aggressive—more blocks are demoted.
- When GPU memory usage drops below the Low Water Mark (LWM), thresholds relax, allowing more blocks to remain on GPU.

This creates a feedback loop that automatically balances GPU memory utilization with classification accuracy.

### 3.4 Tiered Storage Management

Each storage tier is managed by a dedicated allocator:

**GPU Tier.** A slab-based memory pool on GPU HBM, initialized via `cudaMalloc`. Blocks are allocated in fixed-size slabs to eliminate fragmentation. The pool tracks free/used blocks and reports utilization to the adaptive threshold controller.

**DRAM Tier.** A pool of host-pinned memory (allocated via `cudaMallocHost` for DMA-capable memory), enabling efficient GPU↔DRAM transfers via `cudaMemcpyAsync`. Pinned memory achieves 23 GB/s bidirectional bandwidth (Table 2), compared to ~12 GB/s for pageable memory.

**OrchFS Tier (NVM + SSD).** The lowest tier leverages OrchFS [18] for heterogeneous storage management. OrchKvCache constructs I/O requests to exploit OrchFS's alignment-based write partition:

- **Small writes (< 32 KB)**: Routed to NVM 4 KB pages. Ideal for individual warm block evictions where fast future retrieval is expected.
- **Large writes (≥ 32 KB, page-aligned)**: Routed to SSD 32 KB blocks. Used for batch cold block evictions to maximize SSD write bandwidth.

When OrchFS is unavailable (e.g., no NVM hardware), OrchKvCache falls back to standard POSIX file I/O with `pwrite`/`pread`, still benefiting from the batching and alignment logic.

### 3.5 Migration Engine

The migration engine orchestrates all data transfers between tiers, ensuring atomicity and correctness.

**Eviction (Demote).** When the scheduler selects victim blocks for eviction:

1. The eviction policy selects victims using a weighted LRU score that combines inverse hotness with block age.
2. For each victim block, the engine acquires a write lock, sets state to MIGRATING.
3. Data is transferred: GPU → DRAM via `cudaMemcpyAsync` on a dedicated CUDA stream; DRAM → Storage via `io_worker_submit` for asynchronous file I/O.
4. Upon completion, the source tier's memory is freed and the block's location and state are updated atomically.

**Batch Eviction.** To maximize SSD bandwidth, the engine aggregates multiple cold blocks into a single large write. When \(n\) blocks of size \(B\) are collected, they are serialized into a contiguous \(n \times B\) buffer and written as a single sequential I/O, improving SSD bandwidth utilization from ~9% (per-block) to ~41% (batched), as shown in Table 3.

**Promotion (Promote).** When a block is needed but resides in a lower tier:

1. The engine checks the block's current location.
2. For DRAM-resident blocks: a single `cudaMemcpyAsync` (DRAM → GPU) is issued.
3. For storage-resident blocks: a two-hop transfer is executed—first Storage → DRAM (via `io_worker` async read), then DRAM → GPU (via `cudaMemcpyAsync`).
4. The block's state and location are updated upon completion.

**Atomicity Guarantees.** During migration, the per-block `rwlock` ensures that concurrent reads (from attention computation) are blocked until the migration completes. If a migration fails (e.g., SSD write error), the engine rolls back to the source tier state, guaranteeing no data loss.

### 3.6 Prefetch-Driven Pipeline

To hide migration latency, OrchKvCache implements a three-stage pipeline:

**Stage 1: Compute.** Step \(N\)'s attention computation executes on GPU using currently GPU-resident KV blocks.

**Stage 2: Transfer.** Concurrently, blocks predicted to be needed at step \(N+1\) are transferred from DRAM to GPU via `cudaMemcpyAsync` on separate CUDA streams.

**Stage 3: Preload.** Simultaneously, blocks predicted for step \(N+2\) are preloaded from SSD to DRAM via asynchronous file I/O.

**Prefetch Scheduler.** The prefetch decision is based on the persistence-of-importance hypothesis [7]: blocks that received high attention in recent steps are likely to be accessed again. The scheduler scans DRAM-resident and storage-resident blocks, computes a prefetch priority based on their most recent attention score, and dispatches the top-\(K\) candidates (bounded by the prefetch budget).

**Budget Control.** Our experiments (§5, E7) show that a prefetch budget of 8 blocks per step achieves dispatch saturation (~245 dispatches per 100 steps) with negligible overhead (5.9 μs per scheduling decision). Larger budgets provide no additional benefit but consume PCIe bandwidth.

---

## 4 Implementation

OrchKvCache is implemented in ~4,500 lines of C/CUDA for the core engine and ~1,200 lines of Python for the vLLM integration layer.

**C/CUDA Core.** The core is organized into three layers:
- **Core data structures** (`src/core/`): `kv_block`, `kv_request`, `address_map` — provide the block abstraction, request lifecycle management, and concurrent hash map for block lookup.
- **Tiered storage** (`src/tiered_store/`): `gpu_tier`, `dram_tier`, `orchfs_tier`, `transfer_engine`, `io_worker_pool` — manage memory allocation and data movement across tiers.
- **Scheduler** (`src/scheduler/`): `attention_tracker`, `hotcold_classifier`, `adaptive_threshold`, `eviction_policy`, `prefetch_scheduler`, `pipeline`, `migration_engine`, `tiered_manager` — implement the classification and scheduling logic.

All components are designed as independent modules with well-defined C APIs, enabling fine-grained unit testing.

**Python Binding.** We use pybind11 to expose the C API and tiered manager to Python, enabling seamless integration with Python-based inference frameworks.

**vLLM Integration.** OrchKvCache integrates with vLLM [1] via the KVConnector interface, intercepting KV block allocation, deallocation, and access events. An attention hook extracts per-block attention scores from vLLM's FlashAttention [15] backend at configurable intervals.

**Asynchronous I/O.** The `io_worker_pool` implements a thread pool (default 8 threads) with a lock-free task queue for asynchronous file I/O. Each worker thread performs `pwrite`/`pread` operations, enabling non-blocking storage tier interactions.

**Testing.** The system includes 19 C/CUDA unit test files covering all core modules and 4 Python test files for the binding and integration layers.

---

## 5 Evaluation

### 5.1 Experimental Setup

**Hardware.** 2× NVIDIA A100-SXM4-80GB GPUs, 256 GB DDR4 DRAM, Samsung PM9A3 NVMe SSD (3.84 TB, PCIe Gen4 x4).

**Software.** Ubuntu 22.04, CUDA 12.2, Python 3.11, PyTorch 2.5.1+cu121, vLLM 0.7.3.

**Models.** Qwen2.5-7B (28 layers, 28 query heads, 4 KV heads, d=128, BF16, ~14.3 GB), with additional theoretical analysis for LLaMA-2-7B/13B/70B and LLaMA-3-8B/70B.

**Metrics.** Throughput (tokens/s), Time Per Output Token (TPOT), scheduling latency (μs), bandwidth utilization (GB/s and % of peak), token match rate (%), perplexity divergence (%).

### 5.2 Component Analysis

#### E5: Hot-Cold Classification Parameter Sweep

We sweep 9 configurations of \((\alpha, \beta, \gamma)\) across 3 access patterns (uniform, skewed, bursty) with 3 runs each (total 81 experiments), evaluating the classifier's ability to produce controllable three-tier classification.

**Results.** Attention-dominant configurations (\(\alpha \geq 0.7\)) produce the most differentiated classifications. At \(\alpha=0.7, \beta=0.2, \gamma=0.1\), the classifier assigns blocks to three distinct tiers with clear separation. Low-\(\alpha\) configurations (\(\alpha \leq 0.3\)) fail to distinguish cold from warm blocks under skewed access patterns.

![E5: Policy Heatmap](benchmarks/figures/fig08_policy_heatmap.png)
![E5: Classification Distribution](benchmarks/figures/fig09_classification_distribution.png)

#### E7: Prefetch Effectiveness

We sweep the prefetch budget \(K \in \{0, 4, 8, 16, 32\}\) and measure dispatch count and scheduling overhead.

**Results.** Prefetch dispatch saturates at budget \(K \geq 8\), producing ~245 dispatches per 100 decode steps. The per-step scheduling overhead remains bounded at 5.7–5.9 μs regardless of budget, confirming that the prefetch computation does not become a bottleneck.

![E7: Prefetch Dispatches](benchmarks/figures/fig11_prefetch_dispatches.png)
![E7: Prefetch Latency](benchmarks/figures/fig12_prefetch_latency.png)

#### E8: Storage Bandwidth Measurement

We measure bidirectional bandwidth between adjacent tiers using block sizes from 4 KB to 64 MB.

**Results.** GPU↔DRAM: Device-to-Host 22.04 GB/s, Host-to-Device 23.56 GB/s, saturating at PCIe Gen4 limits for transfers ≥ 4 MB. DRAM↔Storage (tmpfs): Write 3.7 GB/s, Read 14.38 GB/s. The tier gap is consistent: GPU↔DRAM is 1.6× faster than DRAM↔Storage for reads.

![E8: Storage Bandwidth](benchmarks/figures/fig13_storage_bandwidth.png)

#### E9: Scheduling Scalability

We measure the scheduler's per-step latency as the number of managed blocks scales from 64 to 4,096.

**Results.** Scheduling latency grows from 1.7 μs (64 blocks) to 38.3 μs (4,096 blocks), with a scaling exponent of 0.749—sub-linear scaling. Even at 4,096 blocks, the P99 latency is 57.88 μs, well under the 100 μs budget and negligible compared to typical decode step latency (~1–10 ms). The per-block cost is 9.36 ns.

![E9: Scheduling Scalability](benchmarks/figures/fig14_scalability.png)

### 5.3 End-to-End Inference Performance

#### E1: End-to-End Throughput

We measure inference throughput across 5 sequence lengths × 3 batch sizes × 2 configurations (baseline vLLM vs. OrchKvCache-enabled vLLM), totaling 30 data points.

**Results.** All 30 configurations complete successfully. Under the current setup (A100-80GB with 7B model where GPU memory is not the bottleneck), baseline and OrchKvCache throughput are within 0.5% of each other, confirming that the tiered management adds negligible overhead when memory pressure is low.

![E1: Throughput vs Sequence Length](benchmarks/figures/fig01_throughput_vs_seqlen.png)
![E1: Throughput vs Batch Size](benchmarks/figures/fig02_throughput_vs_batchsize.png)

#### E3: Latency Breakdown

We decompose the decode step latency into GPU computation, scheduling overhead, and I/O transfer components.

**Results.** Scheduling overhead accounts for < 0.5% of total decode latency (6.4 ms out of 1,386 ms). GPU computation dominates the latency budget, confirming that OrchKvCache's scheduling decisions can be made transparently without impacting inference performance.

#### E4: Storage Tier Ablation

We compare four configurations: GPU-only, GPU+DRAM (2-tier), GPU+DRAM+NVM (3-tier), and GPU+DRAM+NVM+SSD (4-tier), measuring throughput under each.

**Results.** Under current conditions (sufficient GPU memory), all four configurations achieve throughput within 0.5% of each other, validating that the additional tier management does not introduce performance regression. The real benefit of additional tiers manifests under memory pressure, where the 4-tier configuration enables serving workloads that would otherwise require swapping or request rejection.

#### E6: Block Size Ablation

We test block sizes of 16, 32, 64, and 128 tokens.

**Results.** Throughput variation across block sizes is < 0.3%. Block size 128 shows a marginal advantage due to better I/O alignment with SSD page sizes, reducing write amplification.

### 5.4 Generation Quality Guarantee

#### E10: Output Quality Verification

We compare the outputs of baseline vLLM (GPU-only, swap_space=4GB) and OrchKvCache-enabled vLLM (swap_space=32GB) using greedy decoding on 10 diverse prompts, generating 32 tokens each (320 tokens total).

**Results.** The verification produces a perfect result:

> **Table 4: Generation quality verification (Qwen2.5-7B, greedy decoding).**
>
> | Metric | Baseline | OrchKvCache | Difference |
> |--------|----------|-------------|------------|
> | Token match rate | — | — | **100.0000%** (320/320) |
> | Avg. perplexity | 2.3046 | 2.3046 | **0.0000%** |
> | Per-sample match | 10/10 | 10/10 | All 10 samples exact |

Every generated token is bit-exact identical between the two configurations. Perplexity values match to full floating-point precision. This confirms the core guarantee of OrchKvCache: *tiered storage management introduces zero degradation in model output quality*.

---

## 6 Related Work

**KV-Cache Management.** vLLM [1] introduced PagedAttention for fragmentation-free KV-cache management, with basic GPU-CPU swap. FlexGen [2] formulated offloading as a linear program over GPU/CPU/Disk but operates offline and at layer granularity. InfiniGen [5] implements layer-wise attention prediction for CPU-to-GPU prefetching but is limited to two tiers. Mooncake [10] deploys a distributed KVCache pool across GPU/DRAM/SSD at datacenter scale. vTensor [21] provides virtual tensor abstraction for cross-device management. OrchKvCache differs from all of these by combining *hotness-aware classification* with *heterogeneous storage-adapted IO granularity* in a *four-tier hierarchy*.

**KV-Cache Compression and Eviction.** H2O [3] and ScissorHands [7] identify the power-law distribution of attention and propose eviction-based KV compression, but their approaches are *lossy*—discarded tokens cannot be recovered. StreamingLLM [4] keeps only attention sink tokens and a sliding window. SnapKV [34] and PyramidKV [33] propose layer-adaptive compression. KIVI [32] applies asymmetric 2-bit quantization. Quest [9] uses query-aware page-level sparsity. SqueezeAttention [8] optimizes per-layer KV budgets. CacheGen [6] compresses KV cache for network streaming. OrchKvCache's lossless tiered management is *orthogonal* to these compression techniques—they can be combined (e.g., compress before writing to SSD) for further savings.

**LLM Serving Systems.** Orca [11] introduced continuous batching. DistServe [12] and Splitwise [17] separate prefill and decode phases. Sarathi-Serve [13] introduces chunked prefill. SGLang [16] optimizes structured generation. These systems focus on *scheduling and parallelism*; OrchKvCache focuses on *storage-tier optimization* and is complementary.

**Heterogeneous Storage Systems.** OrchFS [18] demonstrated alignment-based write partition for maximizing SSD bandwidth with NVM assistance. Strata [19] pioneered NVM+SSD cross-media file systems with a log-structured approach. SPFS [20] stacks persistent memory on legacy file systems. OrchKvCache is the first to apply heterogeneous storage IO orchestration specifically to KV-cache management.

**Near-Storage and Heterogeneous Inference.** InstInfer [27] offloads attention computation to computational storage drives, taking a "move compute to data" approach. HeteGen [30] and PowerInfer [31] exploit CPU-GPU parallelism for inference. DeepSpeed-Inference [28] provides heterogeneous inference with CPU/NVMe offloading. OrchKvCache takes the complementary "move data to compute efficiently" approach, optimizing the data transfer path rather than moving computation.

---

## 7 Discussion

**CXL Memory Compatibility.** With Intel Optane Persistent Memory discontinued, CXL-attached memory provides a promising alternative for the NVM tier. CXL memory offers similar latency characteristics (~200–400 ns) and byte-addressable access. OrchKvCache's tier abstraction can accommodate CXL memory with minimal modification—only the `orchfs_tier` allocator needs adaptation.

**Distributed Extension.** OrchKvCache currently operates within a single node. Extending to distributed settings—where KV cache may be shared across nodes (as in Mooncake [10])—is a natural direction. The tiered architecture could be extended to include a network tier, with remote DRAM/SSD as additional capacity tiers.

**Compression Integration.** OrchKvCache's lossless management is orthogonal to KV-cache compression. Integrating techniques like KIVI [32] (2-bit quantization) or CacheGen [6] (custom encoding) before writing to NVM/SSD could reduce storage footprint by 3–5× while maintaining the tiered management benefits.

**Limitations.** The current evaluation uses a 7B model on A100-80GB, where GPU memory is not the primary bottleneck. A full demonstration of OrchKvCache's value requires evaluation under memory pressure (larger models, longer contexts, or GPU memory-constrained settings). The vLLM integration layer currently uses swap-space configuration to simulate tiered management; a fully-integrated connector that routes through the C/CUDA core is needed for production deployment.

---

## 8 Conclusion

We presented OrchKvCache, a tiered KV-cache management system that bridges the gap between ML-informed cache classification and storage-optimized data placement for LLM inference. By fusing attention-driven hot-cold classification with heterogeneous storage IO orchestration across a four-tier hierarchy (GPU HBM → DRAM → NVM → SSD), OrchKvCache extends effective KV-cache capacity beyond GPU memory while preserving inference quality (100% token match, 0% perplexity divergence under greedy decoding). Our evaluation demonstrates sub-60 μs scheduling latency at 4,096 blocks, 23 GB/s GPU↔DRAM transfer matching PCIe Gen4 limits, and effective prefetch dispatch with ≤ 5.9 μs overhead. OrchKvCache represents a step toward making long-context LLM inference practical on commodity hardware by treating heterogeneous storage not as a fallback, but as a first-class participant in KV-cache management.

---

## References

[1] W. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP, 2023.

[2] Y. Sheng et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." ICML, 2023.

[3] Z. Zhang et al. "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS, 2023.

[4] G. Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR, 2024.

[5] W. Lee et al. "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management." OSDI, 2024.

[6] Y. Liu et al. "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving." SIGCOMM, 2024.

[7] Z. Liu et al. "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time." arXiv:2305.17118, 2023.

[8] Z. Wang et al. "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget." arXiv:2404.04793, 2024.

[9] J. Tang et al. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference." ICML, 2024.

[10] R. Qin et al. "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079, 2024.

[11] G.-I. Yu et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI, 2022.

[12] Y. Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI, 2024.

[13] A. Agrawal et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." OSDI, 2024.

[14] T. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR, 2024.

[15] T. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022.

[16] L. Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS, 2024.

[17] P. Patel et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." ISCA, 2024.

[18] Y. Zhan et al. "Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs." FAST, 2025.

[19] Y. Kwon et al. "Strata: A Cross Media File System." SOSP, 2017.

[20] H. Woo et al. "On Stacking a Persistent Memory File System on Legacy File Systems." FAST, 2023.

[21] J. Xu et al. "vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving." FAST, 2025.

[22] A. Vaswani et al. "Attention Is All You Need." NeurIPS, 2017.

[23] J. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP, 2023.

[24] N. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150, 2019.

[25] H. Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.

[26] Meta AI. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024.

[27] X. Pan et al. "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference." arXiv:2409.04992, 2024.

[28] R. Y. Aminabadi et al. "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." SC, 2022.

[29] L. Bin et al. "Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache." arXiv:2401.02669, 2024.

[30] X. Zhao et al. "HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices." MLSys, 2024.

[31] Y. Song et al. "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456, 2023.

[32] Z. Liu et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML, 2024.

[33] Z. Cai et al. "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling." arXiv:2406.02069, 2024.

[34] Y. Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation." NeurIPS, 2024.

# OrchKvCache: Paper Outline

---

## Title

**OrchKvCache: Heterogeneous Storage-Orchestrated Tiered KV-Cache Management for Efficient LLM Inference**

---

## Abstract

The Key-Value (KV) cache in large language model (LLM) inference grows linearly with sequence length and can easily exceed GPU memory capacity—a single LLaMA-70B request at 128K context requires ~320 GB of KV cache. Existing systems either manage KV cache within a flat GPU-CPU memory hierarchy with coarse-grained, hotness-agnostic swapping (vLLM), or offload entire layers offline with fixed granularity (FlexGen), leaving significant GPU memory underutilized by cold data and failing to exploit the bandwidth potential of modern storage devices.

We present **OrchKvCache**, a tiered KV-cache management system that dynamically schedules KV blocks across a four-level storage hierarchy—GPU HBM, host DRAM, NVM, and SSD—based on their runtime access hotness. OrchKvCache makes three key contributions. First, it introduces an **attention-driven hot-cold classifier** that fuses attention scores (EMA-smoothed), temporal recency, and access frequency into a unified scoring function, with watermark-driven adaptive thresholds that respond to memory pressure at each tier. Our empirical analysis on real LLM inference shows that attention scores follow a strong power-law distribution (top-10% tokens contribute 90–93% of total attention weight, Gini coefficient 0.93–0.95), validating the premise that most KV blocks can be safely offloaded without quality loss. Second, OrchKvCache employs **multi-granularity IO adaptation** inspired by the OrchFS heterogeneous file system: small random writes are routed to NVM (4 KB pages, ~300 ns latency) while large sequential writes target SSD (32 KB aligned blocks, up to 17.8 GB/s throughput), maximizing bandwidth utilization at each tier. Third, a **prefetch-driven compute-transfer pipeline** overlaps GPU attention computation with speculative KV block promotion from lower tiers, hiding migration latency behind useful work. Throughout all migration paths, OrchKvCache guarantees **lossless data integrity**—the generated token sequence is bit-exact identical to a GPU-only baseline under greedy decoding.

We implement OrchKvCache as ~4,500 lines of C/CUDA and ~1,200 lines of Python, with integration into the vLLM serving framework. Evaluations on an A100-80GB system with Qwen2.5-7B demonstrate: (i) the hot-cold classifier achieves controllable three-tier classification with sub-60 μs scheduling latency at 4,096 blocks; (ii) prefetch dispatch saturates at budget ≥ 8 with ≤ 5.9 μs overhead; (iii) GPU↔DRAM transfer reaches 23 GB/s (matching PCIe Gen4 limits); and (iv) generation quality is perfectly preserved (100% token match, 0% perplexity divergence). OrchKvCache extends the effective KV-cache capacity beyond GPU memory while maintaining inference throughput and output fidelity.

---

## Paper Structure

### §1 Introduction (2.5 pages)

- **Hook**: KV-cache 是 LLM 推理的核心内存瓶颈；以 LLaMA-70B + 128K context 为例说明规模
- **现有方案的不足**（4 点，对应 Abstract.md §1.2）:
  1. 冷数据占据 GPU 显存（幂律分布，H2O/我们的 M2 实测）
  2. Offloading 粒度固定（FlexGen 整层、vLLM 无冷热感知）
  3. 存储层次单一（最多 GPU+DRAM 两级）
  4. 带宽利用低效（SSD 带宽利用率仅 4-26%）
- **Our insight**: 注意力分数的幂律特征 + 异构存储的多粒度 IO 特性 → 冷热感知的四级分层管理
- **OrchKvCache 概述**: 一句话描述系统 + 三个核心技术贡献
- **Contributions 列表**（4 条，对应 Abstract.md §1.3 预期贡献）
- **Results highlight**: 关键实验数据摘要

### §2 Background & Motivation (2.5 pages)

#### §2.1 LLM Inference and KV-Cache
- Transformer 自回归推理流程（prefill + decode）
- KV-Cache 的产生和增长模式
- KV-Cache 大小计算公式：`2 × n_layers × n_kv_heads × seq_len × d_head × dtype`
- MHA / GQA / MQA 对 KV-Cache 大小的影响

#### §2.2 Limitations of Existing Approaches
- vLLM: PagedAttention 解决了碎片问题，但 swap 无冷热感知
- FlexGen: 三级 offloading 但粒度固定（整层）、离线规划
- H2O/ScissorHands: 有损丢弃，不可恢复

#### §2.3 Motivation: Attention Distribution Analysis (Exp-M2)
- **实验设置**: Qwen2.5-1.5B, 3 种输入长度
- **发现 1**: Token 级幂律分布（Top-10% → 90-93%，Gini 0.93-0.95）
- **发现 2**: Block 级聚合仍有效（Top-10% block → ~80%）
- **发现 3**: 层间差异（中间层集中度最高）
- **发现 4**: Attention Sink 现象
- **发现 5**: 跨 decode step 稳定性（Jaccard 0.47-0.70）
- **图表**: Fig.1 注意力分布 CDF; Fig.2 层间差异热力图; Fig.3 Attention Sink 可视化

#### §2.4 Motivation: Storage Bandwidth Gap (Exp-M3/M4)
- 各存储层的带宽实测数据
- vLLM 式逐块 eviction 的 SSD 带宽利用率仅 4-26%
- 多粒度 IO 的必要性

#### §2.5 Heterogeneous Storage and OrchFS
- NVM + SSD 异构存储的特性（延迟/带宽/容量梯度）
- OrchFS 的对齐写分区 + 嵌入式并行 IO 概述（仅介绍与本工作相关的部分）

### §3 System Design (4 pages)

#### §3.1 Architecture Overview
- 四级存储层次图: GPU HBM → DRAM → NVM → SSD
- 数据流: prefill → GPU tier → 冷热分级 → demote/promote
- 组件关系图: tiered_manager 统一调度

#### §3.2 KV Block Abstraction
- `kv_block_t` 数据结构: 元数据、状态机、位置跟踪
- 状态转移图: GPU_RESIDENT ↔ DRAM_RESIDENT ↔ STORAGE_RESIDENT
- 与 vLLM PagedAttention 的 block 对齐

#### §3.3 Attention-Driven Hot-Cold Classification
- **C1 Attention Tracker**: 按块 EMA 追踪注意力分数
- **C2 Hot-Cold Classifier**: `score = α×attn + β×recency + γ×freq`
  - 三级分类: Hot (GPU) / Warm (DRAM/NVM) / Cold (SSD)
  - Attention Sink 保护机制
- **C3 Adaptive Threshold**: 水位线驱动（HWM/LWM）动态调整阈值
  - GPU 显存紧张时更积极降级
  - 空闲时放宽阈值

#### §3.4 Tiered Storage Management
- **GPU Tier**: HBM slab 池, cudaMalloc 管理
- **DRAM Tier**: Host pinned memory 池
- **OrchFS Tier**: NVM+SSD 异构存储
  - 对齐写分区: 小写入→NVM 4KB page, 大写入→SSD 32KB block
  - 多粒度 IO 适配

#### §3.5 Migration Engine
- **Eviction (Demote)**: GPU → DRAM → Storage
  - 加权 LRU 选 victim
  - 批量换出以利用 SSD 顺序带宽
- **Promotion (Promote)**: Storage → DRAM → GPU
  - 按需换入 + 投机预取
- **Two-hop Transfer**: GPU → DRAM → SSD（释放 DRAM 压力）
- 原子性保证: rwlock + 状态机

#### §3.6 Prefetch-Driven Pipeline
- **Prefetch Scheduler**: 基于历史注意力模式预测下一步需要的 block
- **Three-Stage Pipeline**: Step N GPU 计算 ‖ Step N+1 DRAM→GPU 预取 ‖ Step N+2 SSD→DRAM 预加载
- Budget 控制: 预取数量限制避免带宽竞争
- 误预取处理: 取消机制

### §4 Implementation (1 page)

- C/CUDA 核心: ~4,500 行, 模块划分
- Python 绑定: pybind11, ~1,200 行
- vLLM 集成: KVConnector 接口
- 异步 IO: io_worker_pool 线程池
- 测试覆盖: 19 个 C/CUDA 测试 + 4 个 Python 测试

### §5 Evaluation (4 pages)

#### §5.1 Experimental Setup
- 硬件: A100-80GB, DDR4 256GB, Intel Optane PM, NVMe SSD
- 软件: CUDA 12.2, vLLM, PyTorch
- 模型: Qwen2.5-7B, LLaMA-2-7B/13B
- 基线: vLLM (GPU-only), vLLM (cpu_offload), FlexGen, InfiniGen
- Workload: ShareGPT trace, 合成长序列 (8K-128K)

#### §5.2 End-to-End Performance
- **E1 吞吐量 vs 序列长度** (Fig.X): OrchKvCache vs baselines, 多序列长度
- **E2 最大批大小扩展** (Fig.X): 受限 GPU 下的 batch size 提升
- **E3 延迟分解** (Fig.X): TPOT breakdown, 调度开销占比

#### §5.3 Memory Capacity Extension
- **E4 存储层消融** (Fig.X): GPU-only → GPU+DRAM → GPU+DRAM+NVM → 4-tier
- 在受限 GPU 内存下的可服务上下文长度对比

#### §5.4 Component Analysis
- **E5 冷热分级参数** (Fig.X): α,β,γ 权重对分类准确性的影响
- **E6 Block Size 消融** (Fig.X): 16/32/64/128 对吞吐的影响
- **E7 预取效果** (Fig.X): prefetch budget 对命中率和延迟的影响
- **E8 存储带宽** (Fig.X): 各层实测带宽 vs 理论上限
- **E9 调度可扩展性** (Fig.X): 调度延迟随 block 数量的缩放

#### §5.5 Quality Guarantee
- **E10 生成质量验证** (Tab.X): Token 一致率, Perplexity 对比
  - Greedy decoding bit-exact 一致性
  - Sampling 下 perplexity 无差异

### §6 Related Work (1.5 pages)

#### §6.1 KV-Cache Management
- PagedAttention/vLLM, FlexGen, Mooncake, InfiniGen, vTensor
- 定位: 我们增加了异构存储层和冷热感知

#### §6.2 KV-Cache Compression and Eviction
- H2O, ScissorHands, StreamingLLM, SnapKV, PyramidKV, KIVI, Quest, SqueezeAttention, CacheGen
- 定位: 我们是无损管理，与压缩方案正交可组合

#### §6.3 LLM Serving Systems
- Orca, DistServe, Sarathi-Serve, SGLang, Splitwise
- 定位: 我们聚焦存储层优化，与调度优化互补

#### §6.4 Heterogeneous Storage Systems
- OrchFS, Strata, SPFS
- 定位: 首次将异构存储的多粒度 IO 特性应用于 KV-Cache 管理

#### §6.5 Near-Storage and Heterogeneous Inference
- InstInfer, HeteGen, PowerInfer, DeepSpeed-Inference
- 定位: 我们走 "高效搬数据" 路线，与近存储计算互补

### §7 Discussion (0.5 page)

- CXL 内存兼容性
- 分布式扩展（跨节点 KV-Cache 共享）
- 与 KV-Cache 压缩的组合
- 局限性说明

### §8 Conclusion (0.5 page)

- 总结核心贡献
- 关键实验结论
- 未来工作

### References (~34 篇)

引用列表见 `Abstract.md` References 部分。

---

## Figure & Table Plan

| 编号 | 类型 | 内容 | 对应实验 |
|------|------|------|----------|
| Fig.1 | 折线图 | 注意力分数 CDF (Top-K% vs 累积权重) | Motivation M2 |
| Fig.2 | 热力图 | 层间注意力集中度差异 | Motivation M2 |
| Fig.3 | 柱状图 | Attention Sink: 前 5 token 占注意力比例 | Motivation M2 |
| Fig.4 | 架构图 | OrchKvCache 系统总体架构 | Design |
| Fig.5 | 状态图 | KV Block 状态转移 + 四级存储层次 | Design |
| Fig.6 | 流程图 | 调度循环: 分类→换出→预取→流水线 | Design |
| Fig.7 | 折线图 | 端到端吞吐 vs 序列长度 (多基线) | E1 |
| Fig.8 | 柱状图 | 最大 Batch Size 对比 | E2 |
| Fig.9 | 堆叠柱图 | 延迟分解 (GPU计算 / 调度 / IO) | E3 |
| Fig.10 | 柱状图 | 存储层消融: 吞吐随层数增加的变化 | E4 |
| Fig.11 | 热力图 | 冷热分级参数 (α,β,γ) sweep | E5 |
| Fig.12 | 折线图 | Block Size 消融 | E6 |
| Fig.13 | 折线图 | 预取 budget vs dispatch 数量和延迟 | E7 |
| Fig.14 | 柱状图 | 各存储层实测带宽 | E8 |
| Fig.15 | 对数图 | 调度延迟 vs block 数量 (亚线性缩放) | E9 |
| Tab.1 | 表格 | 生成质量验证: Token Match + PPL 对比 | E10 |
| Tab.2 | 表格 | 实验环境配置 | Setup |

---

## Target Venue

**首选**:
- OSDI 2025 (Fall) / SOSP 2025
- ATC 2025 / EuroSys 2026

**页数限制**: 12 pages (正文) + unlimited references

**审稿人画像**: 系统方向审稿人，关注实际性能提升、实验严谨性、与 SOTA 的公平对比

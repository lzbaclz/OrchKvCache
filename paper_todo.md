# OrchKvCache 论文阅读清单

> 按重要性分为三级：⭐⭐⭐ 必精读（逐段细读）、⭐⭐ 必读（通读全文）、⭐ 推荐读（重点读设计和实验）
>
> 标签说明：`开山` = 开创性工作　`竞品` = 直接对比系统　`理论` = 提供理论/观察基础　`技术` = 可复用技术　`参考` = 实验/写作参考

---

## 一、KV-Cache 管理（核心主线，必须全部精读）

这是你项目所在的赛道，每一篇都直接影响你的论文定位和实验设计。

---

### 1. vLLM / PagedAttention ⭐⭐⭐ `开山` `竞品`

**Efficient Memory Management for Large Language Model Serving with PagedAttention**
Woosuk Kwon et al., SOSP 2023

**为什么必须读**：
这是 KV-Cache 管理领域的开山之作。在此之前，所有推理框架都用连续内存存放 KV-Cache，导致严重的显存碎片和浪费。vLLM 提出了 PagedAttention——把 KV-Cache 按固定大小的 block 分页管理（类比操作系统的虚拟内存分页），解决了显存碎片问题，将有效显存利用率从 ~20% 提升至 >90%。

**对你的项目的价值**：
- vLLM 是你的**核心集成目标和基线系统**。你的 OrchKvCache 需要替换 vLLM 的 `BlockSpaceManager`，因此必须深入理解它的 block 分配、引用计数、swap 机制
- vLLM 已有基本的 GPU↔CPU swap，但**粒度固定、无冷热感知、无 NVM/SSD 层**——这正是你的改进点
- 论文的实验设计（throughput vs latency、ShareGPT trace、不同模型规模）是你实验设计的模板

**重点关注**：§3 PagedAttention 设计、§4 Block 管理器、§5.2 Swap 机制、§6 评估方法

---

### 2. FlexGen ⭐⭐⭐ `竞品`

**FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**
Ying Sheng et al., ICML 2023

**为什么必须读**：
这是你**最直接的竞品**。FlexGen 首次系统性地研究了 KV-Cache 在 GPU/CPU/Disk 三级存储之间的 offloading，并用线性规划求解最优调度策略。它证明了通过合理的 offloading，单张 GPU 就能运行 OPT-175B 这样的超大模型。

**对你的项目的价值**：
- FlexGen 的三级 offloading（GPU→CPU→Disk）与你的四级（GPU→DRAM→NVM→SSD）直接对标
- FlexGen 的核心缺陷正是你的改进空间：**粒度固定（整层 offload）、无冷热感知（按层而非按 token 调度）、无 NVM 层、POSIX IO 导致 SSD 带宽利用率低**
- 它的线性规划调度器思路可以参考，但你的方案需要更动态（在线调度 vs FlexGen 的离线规划）

**重点关注**：§3 Offloading 策略空间、§4 线性规划调度、§5.2 IO 调度、§6 实验设置（你的实验需要复现其配置作为基线）

---

### 3. H2O ⭐⭐⭐ `开山` `理论`

**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
Zhenyu Zhang et al., NeurIPS 2023

**为什么必须读**：
这是 KV-Cache 冷热分级的**理论基石**。H2O 首次系统性地揭示了 LLM 注意力分数的**幂律分布**特征：少量 "Heavy Hitter" token 持续获得高注意力分数，绝大多数 token 的注意力分数趋近于零。基于此，H2O 只保留 Heavy Hitter 的 KV-Cache，丢弃其余部分。

**对你的项目的价值**：
- H2O 的核心发现（注意力的幂律分布）是你冷热分级算法的**理论依据**——你在论文里必须引用它来证明"冷热分化确实存在"
- H2O 的做法是**有损的**（直接丢弃冷 token），而你的做法是**无损的**（冷数据下刷到 NVM/SSD 而非丢弃）——这是你相对 H2O 的关键差异化
- H2O 的 Heavy Hitter 识别算法可以作为你冷热分级器的一个对比基线

**重点关注**：§3.1 Heavy Hitter 现象分析（你需要复现类似图表作为 Motivation）、§3.2 动态保留策略、§4.2 不同模型/任务上的 Heavy Hitter 比例

---

### 4. Attention Sink ⭐⭐⭐ `理论`

**Efficient Streaming Language Models with Attention Sinks**
Guangxuan Xiao et al., ICLR 2024

**为什么必须读**：
这篇论文发现了一个关键现象：LLM 中**最开头的几个 token（通常是 BOS token）会持续获得极高的注意力分数**，无论当前生成到多远的位置。这些 token 被称为 "Attention Sink"。即使这些 token 在语义上并不重要，删除它们的 KV-Cache 会导致模型输出严重劣化。

**对你的项目的价值**：
- 你的冷热分级器必须特殊处理 Attention Sink：**将初始 token 标记为永久 Hot，禁止被换出**
- 这个现象解释了为什么纯 LRU 策略在 KV-Cache 管理上不够好（Attention Sink 虽"老"但"热"）
- Streaming LLM 的 "Window Attention + Sink" 思路与你的 "Sliding Window 强制 Hot + Sink 永久 Hot" 设计直接对应

**重点关注**：§2 Attention Sink 现象分析（复现其可视化图）、§3 StreamingLLM 设计、Figure 1~3（注意力模式可视化）

---

### 5. InfiniGen ⭐⭐⭐ `竞品`

**InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
Lee et al., OSDI 2024

**为什么必须读**：
InfiniGen 是 2024 年 OSDI 上发表的 KV-Cache offloading 系统，代表了该方向的**最新 SOTA**。它的核心思路是：在 decode 阶段，用 GPU 上的 prefill 前几层的计算结果来**预测**后续层需要哪些 KV-Cache block，然后从 CPU 内存中**预取**这些 block 到 GPU，与 GPU 计算重叠。

**对你的项目的价值**：
- InfiniGen 是你最强的竞品之一，你必须在实验中与之对比
- InfiniGen 的预取思路与你的预取调度引擎类似，但它**只有 GPU-CPU 两级，没有 NVM/SSD 层**
- 你的优势点：（1）更多存储层次提供更大容量，（2）OrchFS 的异构 IO 提供更高带宽，（3）多粒度管理更灵活
- 它的预取准确率分析方法和实验设计值得参考

**重点关注**：§3 预取策略设计、§4 系统实现（GPU-CPU 数据传输细节）、§5 评估方法和基线选择

---

### 6. CacheGen ⭐⭐ `竞品` `技术`

**CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**
Yuhan Liu et al., SIGCOMM 2024

**为什么必须读**：
CacheGen 从**压缩**角度切入 KV-Cache 管理。它发现 KV-Cache 具有层间和层内的冗余，通过自定义的编码方案可以将 KV-Cache 压缩 3~5×，从而减少存储和传输开销。

**对你的项目的价值**：
- CacheGen 的压缩技术可以与你的分层调度**正交组合**——在换出到 NVM/SSD 前先压缩，减少 IO 量
- 它是你实验中的一个对比基线
- 但 CacheGen 侧重网络传输场景（分布式），你侧重本地异构存储——赛道不同，威胁有限
- 它的压缩比数据可以帮助你估算压缩模块的潜在收益

**重点关注**：§3 KV-Cache 特征分析（压缩性）、§4 编码方案、§6.3 压缩比和精度影响

---

### 7. ScissorHands ⭐⭐ `理论` `技术`

**ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time**
Zichang Liu et al., NeurIPS 2023

**为什么必须读**：
ScissorHands 提出了 "Persistence of Importance" 假说——如果一个 token 在历史 decode step 中持续获得高注意力分数，那它在未来 step 中大概率仍然重要。这个假说为基于历史注意力模式做**预测**提供了理论基础。

**对你的项目的价值**：
- 这个假说直接支撑你的**冷热分级**和**预取调度**的设计：用历史注意力分数预测未来访问模式
- ScissorHands 的 "重要性持续性" 分析是你论文 Motivation 部分的重要引用
- 它的实验方法（跨 decode step 追踪注意力分数变化）可以指导你的 Exp-M2

**重点关注**：§3 Persistence of Importance 假说及实证、§4 基于历史的 token 淘汰策略、Figure 2~3（跨 step 的注意力稳定性分析）

---

### 8. SqueezeAttention ⭐⭐ `竞品` `技术`

**SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget**
Zihao Wang et al., 2024

**为什么必须读**：
SqueezeAttention 提出了一个非常重要的观察：**不同层的 KV-Cache 重要性差异很大**。它按层分配不同的 KV-Cache 预算（某些层保留更多 token，某些层积极淘汰），实现了层级自适应的 KV-Cache 管理。

**对你的项目的价值**：
- 直接启发你的**层自适应冷热阈值**设计——不同层可以有不同的 θ_hot 和 θ_cold
- 它的层重要性分析方法可以集成到你的分级器中
- 证明了 "一刀切" 的全局策略不如层级自适应策略

**重点关注**：§3 层间重要性差异分析、§4 层级预算分配算法

---

### 9. Quest ⭐⭐ `技术`

**Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference**
Jiaming Tang et al., ICML 2024

**为什么必须读**：
Quest 发现注意力稀疏性是**query-dependent**的——同一个 KV-Cache token 在不同 query 下的重要性完全不同。它提出按 KV block 维护的 min/max 统计量来快速判断哪些 block 需要参与 attention 计算。

**对你的项目的价值**：
- Quest 的 "按 block 管理 + 基于统计量快速筛选" 与你的 KV Block 管理粒度天然对齐
- 它的 page-level min/max 统计量可以用来辅助你的冷热判定——不需要每次都计算完整 attention
- Quest 的实验中对比了不同 block 大小（与你的 E6 粒度敏感性实验相关）

**重点关注**：§3 Query-aware 稀疏性分析、§4 Page-level 统计量设计、§5 不同 page 大小的实验

---

### 10. Mooncake ⭐⭐ `参考`

**Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving**
Ruoyu Qin et al., ATC 2025 (或 arXiv 2024)

**为什么必须读**：
Mooncake 是月之暗面（Kimi）的 KV-Cache 管理系统，将 KV-Cache 作为一等公民进行分布式管理。它展示了工业界如何处理海量 KV-Cache 的存储和调度问题。

**对你的项目的价值**：
- 了解工业界的需求和痛点，帮助你在论文中论证问题的实际意义
- Mooncake 的 KV-Cache 跨节点传输/复用思路可以延伸到你的分布式版本（future work）
- 它的 prefill-decode 分离架构是当前趋势，你的系统需要兼容

**重点关注**：§2 KVCache-centric 设计动机、§3 整体架构、§5 评估规模和指标

---

## 二、LLM 推理系统（理解大背景，明确你的位置）

这些论文构成了 LLM 推理的系统生态，你的工作需要在这个生态中找到自己的位置。

---

### 11. Orca ⭐⭐⭐ `开山`

**Orca: A Distributed Serving System for Transformer-Based Generative Models**
Gyeong-In Yu et al., OSDI 2022

**为什么必须读**：
Orca 是 LLM serving 系统的**开山之作**，提出了**连续批处理（Continuous Batching / Iteration-Level Scheduling）**。在此之前，一个 batch 必须等所有请求都生成完才能接新请求；Orca 允许每个 decode step 独立调度，请求可以随到随走。

**对你的项目的价值**：
- 连续批处理是当前所有 LLM serving 系统（包括 vLLM）的基础，理解它才能理解 KV-Cache 管理为什么复杂——不同请求的序列长度不同，KV-Cache 的增长和释放是动态的
- 你的生命周期管理器必须兼容连续批处理的调度模型
- 论文的性能建模方法（prefill vs decode 的计算量分析）可以帮助你理解 IO 开销的相对大小

**重点关注**：§3 Iteration-Level Scheduling、§4 Selective Batching

---

### 12. DistServe ⭐⭐ `参考`

**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**
Yinmin Zhong et al., OSDI 2024

**为什么必须读**：
DistServe 将 Prefill 和 Decode 阶段**分离到不同的 GPU** 上执行，因为两者的计算特征截然不同（Prefill 是 compute-bound，Decode 是 memory-bound）。这是当前 LLM serving 的重要趋势。

**对你的项目的价值**：
- 在 Prefill-Decode 分离的架构下，KV-Cache 需要从 Prefill GPU 传输到 Decode GPU，这个传输过程可以利用你的分层存储作为中间缓冲
- 理解 Prefill-Decode 分离有助于你思考 OrchKvCache 在分布式场景中的扩展
- 它的 goodput 指标定义和实验方法值得参考

**重点关注**：§3 Prefill-Decode 分离动机、§4 KV-Cache 传输机制、§6 评估指标

---

### 13. Sarathi-Serve ⭐⭐ `参考`

**Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**
Amey Agrawal et al., OSDI 2024

**为什么必须读**：
Sarathi-Serve 提出了 Chunked Prefill——将 Prefill 切分为多个小块，与 Decode 请求混合在同一个 batch 中执行。这解决了长 prefill 阻塞 decode 导致延迟尖峰的问题。

**对你的项目的价值**：
- Chunked Prefill 改变了 KV-Cache 的生成模式（不是一次性全生成，而是分块增量生成），你的系统需要兼容这种模式
- 它的 throughput-latency tradeoff 分析方法可以用于你的实验设计

**重点关注**：§3 Chunked Prefill 设计、§4.2 对 KV-Cache 分配的影响

---

### 14. FlashAttention-2 ⭐⭐ `开山` `技术`

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
Tri Dao, ICLR 2024 (FlashAttention-1: NeurIPS 2022)

**为什么必须读**：
FlashAttention 彻底改变了 attention 的 GPU 实现方式——通过 tiling 和 kernel fusion，避免将完整的 attention 矩阵写入 HBM，将 attention 从 memory-bound 变为 compute-bound。它是目前几乎所有推理框架的标配。

**对你的项目的价值**：
- FlashAttention 的 tiling 思路与你的 KV Block 分块管理有天然对应——FlashAttention 本身就是 block-wise 处理 KV
- 当 KV-Cache 不在 GPU 上（被换出到 DRAM/NVM/SSD）时，你需要理解 FlashAttention 如何被修改以支持部分 KV 在 GPU、部分不在的情况
- 理解 GPU HBM 带宽瓶颈有助于你设计合理的换入时机

**重点关注**：§2 IO-aware attention 算法、§3 Tiling 和 Work Partitioning、Appendix（实现细节）

---

## 三、存储系统（你的技术底座）

这些论文提供了异构存储管理的关键技术，是你系统设计的技术来源。

---

### 15. OrchFS ⭐⭐⭐ `技术`（核心依赖）

**Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs**
Yekang Zhan et al., FAST 2025

**为什么必须读**：
这是你项目的**核心技术底座**。你对 OrchFS 的理解深度直接决定了你系统设计的质量。

**必须深入理解的核心机制**：
1. **对齐写分区（Alignment-based Write Partition）**：按 SSD 页对齐拆分写请求→你需要让 KV Block 对齐 32KB
2. **异构数据布局（Heterogeneous-unit Data Layout）**：4KB NVM Page + 32KB SSD Block→你的多粒度管理基础
3. **STRATA_NODE**：同一逻辑块内 NVM+SSD 共存→支持部分换出
4. **嵌入式并行 IO 引擎**：独立 NVM/SSD 线程池→你的异步 IO 基础
5. **NVM→SSD 迁移**：LRU 触发、合并 8×4KB 为 32KB→你的冷数据下刷机制

**重点关注**：全文精读，特别是 §4 异构数据布局、§5 对齐写分区、§6 并行 IO 引擎、§7 迁移机制

---

### 16. Strata ⭐⭐ `开山` `参考`

**Strata: A Cross Media File System**
Youngjin Kwon et al., SOSP 2017

**为什么必须读**：
Strata 是 NVM+SSD 混合文件系统的**开山之作**。它将 NVM 作为日志层（接收所有写入），SSD 作为持久化层（后台迁移），开创了 "NVM 做 write buffer + SSD 做 capacity tier" 的架构模式。OrchFS 在 Strata 的基础上做了大幅改进。

**对你的项目的价值**：
- 理解 Strata 有助于你理解 OrchFS 的设计动机和改进点
- Strata 的 "日志→持久化" 两级架构与你的 "温数据→冷数据" 存储层次在思路上对应
- 作为 Related Work 中存储系统部分的重要引用

**重点关注**：§3 Cross-media 架构设计、§4 日志层和持久化层的协调、§6 迁移策略

---

### 17. SPFS ⭐⭐ `参考`

**SPFS: Splitting and Piggy-backing the File System for Performance with Persistent Memory**
Jongseok Kim et al., ATC 2023

**为什么必须读**：
SPFS 是另一个 PM+SSD 混合文件系统，将元数据和热数据放在 PM，冷数据放在 SSD。与 OrchFS 同属 NVM+SSD 混合文件系统赛道。

**对你的项目的价值**：
- 了解 PM+SSD 混合文件系统的设计空间，有助于你在 Related Work 中全面覆盖
- SPFS 的冷热数据分离方法可以与你的方案对比

**重点关注**：§3 数据分离策略、§5 与 Strata 的对比

---

### 18. vTensor ⭐⭐ `竞品` `参考`

**vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving**
Jiale Xu et al., FAST 2025

**为什么必须读**：
vTensor 是 FAST 2025 上发表的（与 OrchFS 同期），专门针对 LLM serving 中的张量（包括 KV-Cache）管理。它提出了虚拟张量的概念，实现了灵活的张量放置和迁移。

**对你的项目的价值**：
- 这是与你工作**最接近的并发工作之一**，必须仔细读并明确差异化
- 你的优势点可能在于：OrchFS 提供的底层存储优化（对齐写分区、并行 IO）是 vTensor 没有的
- 了解 FAST 2025 审稿人对这类工作的期望

**重点关注**：全文精读，特别是虚拟张量管理机制、与 vLLM 的集成方式、实验设计

---

### 19. InstInfer ⭐⭐ `竞品` `技术`

**InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference**
Xiurui Pan et al., 2024

**为什么必须读**：
InstInfer 将 KV-Cache 直接放在**计算型 SSD（Computational Storage）** 内部，在存储设备端直接执行部分 attention 计算，减少 Host-SSD 数据传输。这是一个极端的 "数据不动计算动" 的思路。

**对你的项目的价值**：
- 代表了 KV-Cache offloading 的另一个方向（近存储计算），帮助你理解设计空间的边界
- 它的 SSD 带宽利用率分析数据可以参考
- 在 Related Work 中需要讨论与之的对比

**重点关注**：§3 In-Storage Attention 设计、§4 SSD 内部实现、§6 与 FlexGen 的对比

---

## 四、注意力机制与稀疏性（冷热分级的理论支撑）

这些论文从 ML 角度分析注意力的稀疏性，为你的冷热分级提供理论依据。

---

### 20. Attention Is All You Need ⭐⭐⭐ `开山`

**Attention Is All You Need**
Ashish Vaswani et al., NeurIPS 2017

**为什么必须读**：
Transformer 架构的原始论文。KV-Cache 问题的根源就在于 Transformer 的 self-attention 机制——每次 decode 都需要访问所有历史 token 的 K、V 向量。

**对你的项目的价值**：
- 理解 KV-Cache 为什么存在：它是 self-attention 计算复杂度从 O(n²) 中 "缓存计算结果" 的产物
- 论文 §3 的 attention 公式是你整个系统设计的出发点
- 虽然很基础，但审稿人期望你在 Background 中正确引用和描述

**重点关注**：§3.2 Scaled Dot-Product Attention、§3.2.2 Multi-Head Attention

---

### 21. GQA (Grouped Query Attention) ⭐⭐ `技术`

**GQA: Training Generalized Multi-Query Attention Models from Multi-Head Checkpoints**
Joshua Ainslie et al., EMNLP 2023

**为什么必须读**：
GQA 是当前主流 LLM（LLaMA-2 70B、Mistral、LLaMA-3 等）采用的注意力变体。它让多个 query head 共享同一组 KV head，大幅减少了 KV-Cache 的大小（如 64 个 query head 共享 8 个 KV head，KV-Cache 缩小 8×）。

**对你的项目的价值**：
- GQA 改变了 KV-Cache 的数据布局——你的 `kv_block_meta_t` 中 head_id 需要按 KV head group 管理而非逐 query head
- 不同模型的 GQA ratio 不同，你的系统需要灵活适配
- GQA 减小了 KV-Cache 大小，但对超长上下文（128K+）仍然不够——这正是你系统的目标场景

**重点关注**：§2 MQA vs GQA vs MHA 对比、§3 GQA 的实现方式

---

### 22. Efficient Memory Management Techniques (综合) ⭐

**A Survey on Efficient Inference for Large Language Models**
Zixuan Zhou et al., 2024 (arXiv survey)

**为什么推荐读**：
这是一篇全面的 LLM 高效推理综述，涵盖了量化、剪枝、蒸馏、KV-Cache 管理、推测解码等方方面面。

**对你的项目的价值**：
- 一站式了解所有相关方向，避免遗漏重要相关工作
- 帮助你在 Related Work 部分覆盖全面
- 快速定位你可能没注意到的新论文

**重点关注**：KV-Cache 管理章节、Memory Offloading 章节

---

## 五、补充阅读（特定技术点深入）

这些论文针对你系统中特定的技术点，按需阅读。

---

### 23. CUDA Unified Memory / HMM ⭐ `技术`

**Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking** (或 NVIDIA 官方文档)

**为什么推荐读**：
理解 CUDA Unified Memory 和 GPU↔Host 内存传输的底层机制，这直接影响你的 GPU↔DRAM 传输设计。

**对你的项目的价值**：
- cudaMemcpyAsync 的实际行为、CUDA Stream 的并发模型
- PCIe/NVLink 带宽限制和延迟特征
- 影响你的 IO-计算重叠流水线设计

---

### 24. CXL Memory Expansion ⭐ `参考`

**TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory**
Hasan Al Maruf et al., ASPLOS 2023

**Pond: CXL-Based Memory Pooling Systems for Cloud Platforms**
Huaicheng Li et al., ASPLOS 2023

**为什么推荐读**：
CXL 是 NVM 的潜在替代品（Intel Optane PM 已停产）。CXL 内存扩展提供了类似 NVM 的 "比 DRAM 慢但比 SSD 快" 的存储层次。

**对你的项目的价值**：
- 如果拿不到 NVM 硬件，CXL 内存是重要的 Plan B
- CXL 的延迟特征（~200-400ns）与 NVM 接近，你的设计可能可以直接迁移
- 在论文的 Discussion 部分可以讨论 CXL 兼容性

---

### 25. FastGen / Mixed-Precision KV-Cache ⭐ `技术`

**Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs**
Suyu Ge et al., ICLR 2024

**为什么推荐读**：
FastGen 提出按注意力头的特征对不同头采用不同的 KV-Cache 策略（有的头保留完整、有的头只保留 sink+recent），并支持混合精度 KV-Cache。

**对你的项目的价值**：
- 头间差异化管理的思路可以融入你的冷热分级（不同头的冷热阈值不同）
- 混合精度思路可以融入你的压缩模块

---

### 26. RetrievalAttention ⭐ `技术`

**RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval**
Di Liu et al., 2024

**为什么推荐读**：
RetrievalAttention 用向量检索（ANNS）的方式从 CPU 内存中高效检索需要的 KV-Cache，而不是将所有 KV-Cache 都搬到 GPU。

**对你的项目的价值**：
- 它的 "按需检索而非全量加载" 思路与你的预取调度相关
- 如果你的预取不够准确，可以考虑用向量检索作为 fallback
- 了解另一种减少 GPU↔Host 传输量的方法

---

## 六、阅读优先级与建议顺序

### 第一周（最紧急，建立核心认知）

按此顺序阅读，每篇花 3~4 小时精读：

```
Day 1-2:  #1  vLLM/PagedAttention  — 理解 KV-Cache 管理的基本框架
Day 2-3:  #2  FlexGen             — 理解 offloading 的设计空间和局限
Day 3-4:  #3  H2O                 — 理解冷热分化的理论基础
Day 4-5:  #4  Attention Sink      — 理解特殊 token 的处理
Day 5:    #15 OrchFS              — 深入理解你的技术底座（已读过，再精读一遍）
```

### 第二周（建立竞争格局认知）

```
Day 1-2:  #5  InfiniGen           — 理解 OSDI'24 SOTA 的预取思路
Day 2-3:  #18 vTensor             — 理解最接近的并发工作
Day 3:    #11 Orca                — 理解连续批处理基础
Day 4:    #7  ScissorHands        — 理解重要性持续性假说
Day 5:    #8  SqueezeAttention    — 理解层级差异化
```

### 第三周（扩展视野，完善 Related Work）

```
Day 1:    #6  CacheGen            — 压缩方向
Day 2:    #9  Quest               — Query-aware 稀疏性
Day 3:    #10 Mooncake            — 工业界实践
Day 4:    #12 DistServe + #13 Sarathi — Serving 系统趋势
Day 5:    #14 FlashAttention-2    — Attention 实现
```

### 按需阅读（遇到具体问题时查）

```
#16 Strata, #17 SPFS              — 写 Related Work 存储部分时
#19 InstInfer                      — 写 Related Work 时对比
#20 Attention Is All You Need      — 写 Background 时引用
#21 GQA                           — 实现 GQA 模型支持时
#22 Survey                        — 检查是否遗漏相关工作
#23~26                            — 实现特定模块时按需查阅
```

---

## 七、阅读笔记模板

对每篇论文，建议按以下模板记录笔记（存入 `docs/paper_notes/` 目录）：

```markdown
# [论文标题] - 阅读笔记

## 基本信息
- 会议/年份：
- 作者：
- 一句话总结：

## 核心问题
这篇论文要解决什么问题？

## 核心方法
它是怎么解决的？（3~5 句话）

## 关键发现/数据
- 发现 1：...
- 发现 2：...
- 关键数据点：...

## 与 OrchKvCache 的关系
- 可借鉴：...
- 差异化：...
- 需要对比的点：...

## 实验设计参考
- 使用了哪些 benchmark？
- 使用了哪些指标？
- 有哪些实验配置值得参考？

## 审稿人可能的关注点
如果审稿人读过这篇论文，他可能会问我们什么？
```

---

## 八、关键论文关系图谱

```
                         ┌──────────────────┐
                         │ Attention Is All  │
                         │  You Need (2017)  │
                         │   [Transformer]   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼──────────────┐
                    │             │              │
              ┌─────┴─────┐ ┌────┴────┐  ┌──────┴──────┐
              │ Flash     │ │  GQA    │  │  Orca       │
              │ Attention │ │ (2023)  │  │  (2022)     │
              │ (2022)    │ └─────────┘  │ [Continuous │
              └───────────┘              │  Batching]  │
                                         └──────┬──────┘
                                                │
          ┌────────────────────┬────────────────┤
          │                    │                │
    ┌─────┴─────┐     ┌───────┴──────┐  ┌──────┴──────┐
    │  vLLM     │     │  DistServe   │  │  Sarathi    │
    │  (2023)   │     │  (2024)      │  │  (2024)     │
    │ [Paged   │     └──────────────┘  └─────────────┘
    │  KV-Cache]│
    └─────┬─────┘
          │
    ┌─────┼──────────────┬──────────────┐
    │     │              │              │
┌───┴───┐ │        ┌─────┴─────┐  ┌────┴────┐
│FlexGen│ │        │ InfiniGen │  │Mooncake │
│(2023) │ │        │ (2024)    │  │(2024)   │
│[3-tier│ │        │[Prefetch] │  │[KVCache │
│offload│ │        └───────────┘  │ Centric]│
└───────┘ │                       └─────────┘
          │
  ┌───────┼──────────────────────┐
  │       │                      │
┌─┴──┐ ┌─┴────────┐  ┌──────────┴────────┐
│H2O │ │Attention │  │ ScissorHands      │
│2023│ │Sink 2024 │  │ (2023)            │
│    │ └──────────┘  │[Persistence of    │
│    │               │ Importance]       │
└─┬──┘               └──────────┬────────┘
  │                              │
  │    ┌─────────────────────────┤
  │    │                         │
  │  ┌─┴──────────┐  ┌──────────┴───────┐
  │  │SqueezeAttn │  │ Quest            │
  │  │(2024)      │  │ (2024)           │
  │  │[Layer-wise]│  │[Query-aware]     │
  │  └────────────┘  └──────────────────┘
  │
  └──────────────────┐
                     │
        ┌────────────┴────────────────────┐
        │        OrchKvCache (Ours)       │
        │  冷热分级 + 多粒度 + 异构存储    │
        └────────────┬────────────────────┘
                     │
              ┌──────┴──────┐
              │   OrchFS    │
              │  (FAST'25)  │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │   Strata    │
              │  (SOSP'17)  │
              └─────────────┘
```

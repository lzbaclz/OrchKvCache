# OrchKvCache 论文详细笔记

> 每篇论文包含：摘要翻译、Introduction 介绍、解决的问题、系统设计、实验内容、结论
> 共覆盖 21 篇核心论文，按 paper_todo.md 中的编号排列

---
---

# 第一部分：KV-Cache 管理（核心赛道）

---

## Paper #1: vLLM / PagedAttention

**Efficient Memory Management for Large Language Model Serving with PagedAttention**
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
**SOSP 2023**

### 摘要翻译

大语言模型（LLM）的高吞吐量服务需要对大量请求进行批处理。然而，现有系统由于对 KV-Cache 内存管理效率低下而受到限制。KV-Cache 内存管理的低效主要表现在：（1）大量的内存碎片化和过度预留导致的内存浪费，（2）无法利用内存共享机制。为解决这些问题，本文提出了 **PagedAttention**——一种受操作系统虚拟内存与分页思想启发的注意力算法，它允许 KV-Cache 存储在非连续的内存空间中。在此基础上构建的 **vLLM** 系统实现了：（1）KV-Cache 内存的近零浪费；（2）灵活的 KV-Cache 共享（在同一 prompt 的并行采样之间）。实验表明，vLLM 在不影响模型精度的前提下，将 LLM 服务的吞吐量提升了 **2~4 倍**。

### Introduction 介绍

LLM 的推理过程分为 prefill（处理 prompt 生成所有 KV 缓存）和 decode（逐 token 生成，每步需查询历史 KV）两个阶段。现有系统（如 FasterTransformer、Orca）为每个请求**预先分配**一块连续的显存来存储 KV-Cache，但这种方式存在三大问题：
1. **内部碎片**：预分配的显存按最大序列长度分配，但实际生成长度远小于最大长度，大量空间被浪费
2. **外部碎片**：不同请求的 KV-Cache 大小不同，请求结束后释放的内存块难以被新请求完整利用
3. **无法共享**：并行采样（beam search、parallel sampling）中，多个序列共享相同的 prompt KV-Cache，但连续内存方案无法实现这种共享

论文指出，在现有系统中，**实际仅有 20.4%~38.2% 的 GPU 显存被有效利用**。

### 解决的问题

如何高效管理 LLM 推理过程中动态变化的 KV-Cache 显存，消除碎片化、减少过度预留、支持 KV 共享，从而最大化批处理吞吐量。

### 系统设计

**核心思想：将操作系统的虚拟内存分页机制引入 KV-Cache 管理。**

1. **PagedAttention 算法**：
   - 将每个请求的 KV-Cache 拆分为固定大小的 **KV Block**（如 16 个 token 一个 block）
   - Block 不需要在显存中连续存放，通过 **Block Table**（类比页表）记录逻辑 block 到物理 block 的映射
   - Attention 计算时按 block 粒度进行，每个 block 独立寻址

2. **Block 管理器（BlockSpaceManager）**：
   - 维护全局的物理 block 空闲池
   - 请求到来时按需分配 block（而非预先分配最大长度）
   - 请求结束时立即回收 block
   - 支持 **Copy-on-Write (CoW)**：并行采样的序列共享公共前缀的 KV block，分叉时才拷贝

3. **Swap 机制**：
   - 当 GPU 显存不足时，可以将低优先级请求的 KV block **swap 到 CPU 内存**
   - 待 GPU 有空间时再 swap 回来
   - 按 block 粒度 swap，比整个请求级别更灵活

4. **调度策略**：
   - First-Come-First-Served (FCFS) 为基础
   - 当显存不足时，preempt（抢占）最后到达的请求，swap 其 KV-Cache 到 CPU

### 实验内容

- **模型**：OPT-13B, OPT-175B, LLaMA-13B
- **基线**：FasterTransformer, Orca (学术模拟)
- **工作负载**：ShareGPT 真实对话 trace（输入/输出长度分布不均）、合成固定长度请求
- **核心指标**：吞吐量（requests/s, tokens/s）、延迟（TTFT, TPOT）
- **关键实验**：
  - **E1 吞吐量对比**：vLLM 在 ShareGPT 上比 FasterTransformer 提升 **14~24×**，比 Orca 提升 **2.2~4.3×**
  - **E2 并行采样**：Beam search 场景下共享 KV block 减少 55% 的内存使用
  - **E3 ChatBot 模拟**：长时间运行的多轮对话场景下，vLLM 保持稳定高吞吐

### 结论

PagedAttention 通过将 KV-Cache 分页管理，几乎消除了显存浪费（有效利用率接近 100%），大幅提升了 LLM serving 的吞吐量。vLLM 成为后续几乎所有 KV-Cache 管理工作的基线系统。

---

## Paper #2: FlexGen

**FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**
Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang
**ICML 2023**

### 摘要翻译

大语言模型（LLM）的高计算和内存需求使得在资源有限的硬件上部署变得困难。本文提出了 **FlexGen**，一个用于在单个 GPU 上运行大型语言模型的高吞吐量生成引擎。FlexGen 将 LLM 推理的吞吐量优化问题形式化为一个 **图遍历问题**，通过搜索 GPU、CPU 内存和磁盘三级存储层次上的最优张量放置策略，实现高效的 offloading。FlexGen 还集成了**分组量化**和**稀疏注意力**等压缩技术来进一步减少内存需求。在 OPT-175B 模型上，FlexGen 实现了比现有 offloading 系统高出数个数量级的吞吐量，在单张 NVIDIA T4 GPU 上达到了 **1 token/s** 的生成速度。

### Introduction 介绍

论文指出了 LLM 推理面临的关键矛盾：大模型（如 OPT-175B 需要 325GB 存储权重）远超单张 GPU 的显存容量。现有的解决方案有两种思路：
1. **模型并行**：用多张 GPU 分担，但成本高
2. **Offloading**：将部分数据放在 CPU 内存或磁盘上，按需加载

现有 offloading 方案（如 DeepSpeed-Inference、Hugging Face Accelerate）的问题是**IO 调度效率低**——每层的权重、KV-Cache、激活值的放置策略是手动或启发式的，未充分利用 CPU 内存和磁盘的带宽。FlexGen 将此建模为优化问题并系统性求解。

### 解决的问题

如何在单张消费级 GPU（如 T4 16GB）上高效运行超大 LLM（如 175B 参数），通过 GPU/CPU/Disk 三级 offloading 最大化吞吐量。

### 系统设计

1. **Offloading 策略空间建模**：
   - 将 LLM 的每层计算抽象为一个图节点，每个节点涉及三类张量：权重（W）、激活值（A）、KV-Cache（C）
   - 每类张量可以放在 GPU、CPU 或 Disk 上
   - 定义放置策略为：每层每类张量在三级存储上的分配比例

2. **线性规划求解**：
   - 目标函数：最大化吞吐量 = 最小化每 token 的总执行时间
   - 约束：GPU 显存容量、CPU 内存容量、磁盘空间
   - 决策变量：每类张量在各存储层的比例
   - 离线求解后，按固定策略执行

3. **IO 调度优化**：
   - **Overlap**：将当前层的 GPU 计算与下一层的 IO（从 CPU/Disk 加载权重/KV-Cache）重叠
   - **按层调度**：每层的权重只加载一次，处理完整个 batch 所有请求后再换出

4. **压缩技术集成**：
   - **分组量化**：权重和 KV-Cache 压缩为 4-bit，显著减少存储和传输量
   - **稀疏注意力**：结合 token 级稀疏来减少 KV-Cache 大小

### 实验内容

- **模型**：OPT-6.7B, OPT-30B, OPT-175B
- **硬件**：单张 NVIDIA T4 (16GB) / A6000 (48GB)
- **基线**：DeepSpeed Zero-Inference, Hugging Face Accelerate, Petals (分布式)
- **核心指标**：吞吐量 (tokens/s)、端到端延迟
- **关键实验**：
  - **E1 OPT-175B on T4**：FlexGen 达到 1 token/s，比 DeepSpeed 快 **69×**，比 Accelerate 快 **40×**
  - **E2 Offloading 策略搜索**：线性规划找到的策略显著优于手工策略
  - **E3 压缩效果**：4-bit 量化进一步提升 2× 吞吐，精度损失 <1%
  - **E4 延迟分解**：IO 时间占 ~70%，GPU 计算占 ~30%

### 结论

FlexGen 通过系统性的 offloading 策略优化，将单 GPU 上大模型推理的吞吐量提升了 1~2 个数量级。但其**核心局限**在于：（1）策略是离线计算的，不能动态适应负载变化；（2）以整层为粒度 offload，无法按 token 级别做冷热区分；（3）仅使用 POSIX IO，未针对存储设备特性做优化。

---

## Paper #3: H2O (Heavy-Hitter Oracle)

**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen
**NeurIPS 2023**

### 摘要翻译

大语言模型推理中，KV-Cache 随序列长度线性增长，严重限制了批处理大小和上下文长度。本文首先揭示了 Attention 的一个关键特征：**少量 token（称为 Heavy Hitter, H2）持续贡献了绝大部分的注意力分数，而大多数 token 几乎不被关注**。基于此发现，本文提出了 **H2O（Heavy-Hitter Oracle）**——一种 KV-Cache 驱逐策略，仅保留 Heavy Hitter token 和最近的 token 的 KV-Cache，其余全部丢弃。H2O 可将 KV-Cache 缩减至原来的 **20%**，同时在多种下游任务上保持与完整 KV-Cache 相近的性能。

### Introduction 介绍

论文首先量化了 KV-Cache 的内存瓶颈：以 OPT-30B 为例，处理一个 2048 token 的请求需要约 1.8GB 的 KV-Cache（仅次于模型权重）。在长上下文场景下，KV-Cache 甚至超过模型权重成为内存的主要消耗者。

论文的核心观察是：**并非所有 token 的 KV-Cache 都同等重要**。通过可视化多个模型在多个任务上的注意力分数矩阵，论文发现：
1. 注意力分数呈现**幂律分布**：Top-5% 的 token 贡献了 >90% 的累积注意力
2. **Heavy Hitter 的位置是动态的**：不同 decode step 关注的 token 子集不同，但 Heavy Hitter 具有一定的跨步稳定性
3. **初始 token（Attention Sink）几乎总是 Heavy Hitter**：无论在哪个 decode step

### 解决的问题

如何在推理时（test-time）动态缩减 KV-Cache 大小，在大幅节省显存的同时保持生成质量。

### 系统设计

1. **Heavy Hitter 识别**：
   - 在每个 decode step，计算当前 query 与所有 key 的注意力分数
   - 将注意力分数累积到每个 token 位置的**重要性计数器**上
   - 重要性最高的 Top-K 个 token 被标记为 Heavy Hitter

2. **KV-Cache 驱逐策略**：
   - 保留两部分 KV-Cache：**Heavy Hitter token（全局重要）** + **Recent token（最近 W 个 token）**
   - 其余 token 的 KV-Cache 直接丢弃（有损操作）
   - 每层独立维护 Heavy Hitter 集合

3. **动态更新**：
   - 每个 decode step 重新评估 Heavy Hitter 集合
   - 新进入 Heavy Hitter 的 token 被保留
   - 落出 Heavy Hitter 的 token 被驱逐（不可恢复）

4. **与 attention 计算集成**：
   - 在 softmax 之后利用注意力分数更新 Heavy Hitter 集合
   - 额外开销极小（仅需一次 top-k 操作）

### 实验内容

- **模型**：OPT-6.7B, OPT-13B, OPT-30B, LLaMA-7B, LLaMA-13B
- **基线**：Full KV-Cache, Local (sliding window), Random eviction, Learned sparse attention (Reformer, LongFormer)
- **评估任务**：COPA, PIQA, StoryCloze, Winogrande, OpenBookQA, HellaSwag (zero-shot)；WikiText 困惑度
- **核心指标**：任务准确率、困惑度（Perplexity）
- **关键实验**：
  - **E1 压缩比 vs 质量**：保留 20% KV-Cache 时，准确率仅下降 1~3%
  - **E2 Heavy Hitter 比例分析**：~5% 的 token 贡献了 >90% 的注意力
  - **E3 跨层分析**：不同层的 Heavy Hitter 集合不完全重叠，需要按层独立维护
  - **E4 吞吐量提升**：KV-Cache 缩减后，batch size 可增大 2~5×

### 结论

H2O 揭示了 LLM 注意力的幂律分布特性，证明了 KV-Cache 中大量数据是"冷数据"可被安全移除。但其局限性在于**有损**——丢弃的 token 无法恢复，可能在某些需要回溯的任务中导致质量损失。OrchKvCache 的改进思路是：将冷数据下刷到低层存储（无损），需要时再换入。

---

## Paper #4: StreamingLLM / Attention Sink

**Efficient Streaming Language Models with Attention Sinks**
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
**ICLR 2024**

### 摘要翻译

将大语言模型部署在流式应用中（如多轮对话、实时文本处理）需要处理无限长度的输入。然而，现有 LLM 受限于注意力窗口大小和 KV-Cache 的线性增长。本文发现了一个关键现象——**Attention Sink**：自回归 LLM 对序列**最开头的几个 token 分配了异常高的注意力分数**，即使这些 token 在语义上并不重要。丢弃这些初始 token 的 KV-Cache 会导致模型输出严重崩溃。基于此，本文提出了 **StreamingLLM**：仅保留 Attention Sink token（开头几个 token）和最近的滑动窗口内 token 的 KV-Cache，即可实现稳定的无限长度流式推理。StreamingLLM 无需任何模型微调，可直接应用于现有预训练模型。

### Introduction 介绍

论文首先界定了 LLM 流式推理的三种策略及其问题：
1. **Dense Attention**：保留所有 KV-Cache，计算量和内存随长度线性增长，不可持续
2. **Window Attention**：只保留最近 W 个 token 的 KV-Cache，但当序列长度超过预训练长度后**立即崩溃**
3. **Sliding Window + Re-computation**：每次重新计算窗口内的 KV-Cache，计算量巨大

Window Attention 崩溃的根本原因是什么？论文发现：**是因为丢弃了 Attention Sink token 的 KV-Cache**。

核心观察：
- 在几乎所有层和所有头中，序列**第 1 个 token**（通常是 BOS 或句首标记）都获得了极高的注意力分数
- 这些 token 之所以获得高注意力，**并非因为其语义重要**，而是因为 softmax 的数学特性——模型需要一个"垃圾桶"来存放多余的注意力概率质量
- 丢弃这些 sink token 会破坏 softmax 的数值稳定性，导致输出崩溃

### 解决的问题

如何在有限的 KV-Cache 预算下实现 LLM 的稳定无限长度流式推理。

### 系统设计

1. **StreamingLLM 框架**：
   - 保留两部分 KV-Cache：
     - **Attention Sink token**：序列最开头的 S 个 token（通常 S=4 即可）
     - **滑动窗口 token**：最近的 W 个 token
   - 总 KV-Cache 大小固定为 S + W，不随序列长度增长
   - 超出窗口的中间 token 被丢弃

2. **位置编码处理**：
   - 使用 RoPE 等相对位置编码时，需要重新调整保留 token 的位置 ID
   - Sink token 保持位置 0~S-1，窗口 token 从 S 开始重新编号
   - 确保位置编码的连续性

3. **Sink Token 数量选择**：
   - 实验表明，保留 4 个 Attention Sink token 即可在所有测试模型上保持稳定
   - 更多 Sink token 带来的提升边际递减

### 实验内容

- **模型**：LLaMA-2-7B/13B/70B, Falcon-7B/40B, MPT-7B/30B, Pythia 系列
- **评估**：在长度为 4M token 的文本上计算困惑度（Perplexity）
- **基线**：Dense Attention, Window Attention, 重计算方案
- **关键实验**：
  - **E1 崩溃分析**：Window Attention 在序列超过预训练长度时困惑度暴增（>10^3），而 StreamingLLM 保持稳定（~与 Dense 相当）
  - **E2 Sink 数量消融**：1 个 Sink 就能防止崩溃，4 个 Sink 达到最优
  - **E3 可视化**：热力图清晰展示第 1 个 token 在各层各头上获得的高注意力
  - **E4 推理速度**：StreamingLLM 比重计算方案快 **22×**

### 结论

Attention Sink 是 LLM 注意力机制的一个根本性特征，而非个别模型的偶然现象。StreamingLLM 提供了一种零成本的无限长度推理方案。对 OrchKvCache 的启示：**Attention Sink token 必须永久标记为 Hot，禁止被换出到低层存储**。

---

## Paper #5: InfiniGen

**InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
Lee et al.
**OSDI 2024**

### 摘要翻译

大语言模型推理中，将 KV-Cache offload 到 CPU 内存可以扩展上下文长度，但在 decode 阶段将 KV-Cache 从 CPU 换入 GPU 的**传输延迟**成为新的瓶颈。本文提出 **InfiniGen**，一个基于预取的 KV-Cache 管理系统。核心思想是：利用 **当前层的 attention 计算结果来预测下一层需要哪些 KV-Cache block**，然后在 GPU 计算当前层 attention 的同时，从 CPU 内存预取下一层的 KV-Cache，实现 **IO-计算重叠**。InfiniGen 的预取准确率达到 **95%+**，在长上下文推理中比 vLLM 提升 **3~8×** 吞吐量。

### Introduction 介绍

论文分析了 KV-Cache offloading 面临的核心挑战：
1. **带宽瓶颈**：PCIe 带宽（~32GB/s）远低于 GPU HBM 带宽（~2TB/s），从 CPU 加载 KV-Cache 的延迟远大于 GPU 本地访问
2. **全量加载不可行**：将所有 KV-Cache 从 CPU 加载到 GPU 的延迟太高
3. **稀疏访问模式**：实际上每个 decode step 只有少量 KV-Cache block 对 attention 输出有显著贡献

InfiniGen 的关键洞察：LLM 的注意力模式在**相邻层之间具有相关性**——如果当前层 L 对某些 token 给予了高注意力，那么层 L+1 大概率也会关注类似的 token。因此可以利用层 L 的 attention 分数来预测层 L+1 需要的 KV-Cache。

### 解决的问题

如何在 KV-Cache offload 到 CPU 的场景下，通过精准预取消除 CPU→GPU 的传输延迟，使长上下文推理的吞吐量接近全部 KV 在 GPU 上的水平。

### 系统设计

1. **跨层预测机制**：
   - 在 GPU 计算层 L 的 attention 时，分析 attention 分数分布
   - 识别出注意力集中的 KV block（即 "重要 block"）
   - 基于层间相关性，预测层 L+1 也可能需要这些 block（加上一定的扩展窗口）

2. **预取流水线**：
   - **Stage 1**（GPU 计算）：层 L 的 attention 计算
   - **Stage 2**（并行 IO）：根据层 L 的预测结果，从 CPU 预取层 L+1 的 KV-Cache 到 GPU
   - Stage 1 和 Stage 2 在不同 CUDA Stream 上并行执行
   - 当层 L 计算完成、开始层 L+1 计算时，所需的 KV-Cache 已经到位

3. **部分 KV-Cache 保留**：
   - Attention Sink token 和最近的 token 始终保留在 GPU 上（不需要预取）
   - 只有中间位置的 "历史 token" 的 KV-Cache 存放在 CPU，按需预取

4. **预取粒度控制**：
   - 预取以 KV page 为单位（类比 vLLM 的 block）
   - 通过阈值控制预取数量：只预取注意力分数超过阈值的 page
   - 阈值自适应调整以平衡命中率和带宽消耗

### 实验内容

- **模型**：LLaMA-2-7B, LLaMA-2-13B, Yi-6B, Yi-34B
- **硬件**：NVIDIA A100 80GB GPU, 大容量 CPU DRAM
- **基线**：vLLM (swap), FlexGen, Full GPU (上界), Sparse Attention 方案
- **工作负载**：LongBench, 合成长序列 (8K~128K tokens)
- **关键实验**：
  - **E1 预取准确率**：95%+ 的 decode step 中，所有需要的 KV block 都被正确预取
  - **E2 吞吐量对比**：在 64K 序列长度下，比 vLLM 提升 **3~8×**，接近全 GPU 水平的 85~95%
  - **E3 延迟分析**：预取将 CPU→GPU 传输延迟从关键路径上移除，IO 开销 <10%
  - **E4 层间相关性**：相邻层的 top-K 注意力 token 重叠度 >80%，证实了跨层预测的可行性

### 结论

InfiniGen 证明了基于层间注意力模式的预取可以几乎完全消除 KV-Cache offloading 的传输延迟。对 OrchKvCache 的启示：（1）预取策略非常有效，你应该实现类似的预取机制；（2）InfiniGen 只有 GPU-CPU 两级，你可以通过增加 NVM/SSD 层提供更大容量；（3）InfiniGen 的预取准确率数据是你的重要参考基准。

---
---

# 第二部分：KV-Cache 优化扩展

---

## Paper #6: CacheGen

**CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**
Yuhan Liu et al.
**SIGCOMM 2024**

### 摘要翻译

大语言模型推理中，KV-Cache 的存储和传输是主要的性能瓶颈之一。本文提出 **CacheGen**，一个 KV-Cache 压缩和流式传输系统。CacheGen 发现 KV-Cache 张量在层间和层内存在显著的冗余模式，利用自定义的编码方案可将 KV-Cache 压缩 **3~5×**。CacheGen 还设计了流式传输协议，在分布式 serving 场景中减少了 KV-Cache 在节点间传输的网络开销。在实际负载下，CacheGen 将首 token 延迟（TTFT）降低了 **3.7×**，同时生成质量损失可忽略。

### Introduction 介绍

在多种 LLM serving 场景中，KV-Cache 需要被存储或传输：
1. **Prefill-Decode 分离**：Prefill 节点生成 KV-Cache 后需要传输到 Decode 节点
2. **KV-Cache 复用**：相同 prompt 的 KV-Cache 可以被缓存并复用，避免重复 prefill
3. **长上下文 offloading**：KV-Cache 需要在 GPU 和 CPU/Disk 之间传输

这些场景中，KV-Cache 的传输量很大（如 LLaMA-13B 处理 4K token 需传输约 1.6GB），网络或 IO 带宽成为瓶颈。CacheGen 的核心思路是：在传输/存储前压缩 KV-Cache，在使用时解压。

### 解决的问题

如何高效压缩 KV-Cache 以减少存储和传输开销，同时保持生成质量。

### 系统设计

1. **KV-Cache 特征分析**：
   - 层间特征：不同层的 KV-Cache 分布差异大，但**同层不同请求的 KV-Cache 分布相似**
   - 层内特征：KV-Cache 的通道维度存在显著的**低秩结构**
   - Key 和 Value 的分布特征不同：Key 更稀疏，Value 更稠密

2. **自适应编码方案**：
   - 对每层 KV-Cache 独立编码
   - 使用**量化 + Delta 编码**：先对通道维度做 PCA 提取主成分，对残差量化
   - 量化位宽按层自适应调整（根据该层的敏感度）
   - Key 和 Value 使用不同的编码策略

3. **流式传输协议**：
   - 按层顺序流式发送压缩后的 KV-Cache
   - 接收端边解压边使用，不等所有层传输完成
   - 支持部分传输重试和带宽自适应

4. **质量控制**：
   - 设定目标精度约束（如困惑度增加 <1%）
   - 在离线阶段校准每层的量化位宽
   - 运行时使用固定的编码方案，无需在线调整

### 实验内容

- **模型**：LLaMA-2-7B, LLaMA-2-13B, Falcon-7B
- **场景**：KV-Cache 跨节点传输（prefill-decode 分离）、KV-Cache 本地存储复用
- **基线**：未压缩传输、通用压缩（zstd, gzip）、量化方案（INT8, FP8）
- **关键实验**：
  - **E1 压缩比**：CacheGen 达到 3~5× 压缩（优于通用压缩的 1.5~2×）
  - **E2 质量影响**：困惑度增加 <0.5%，下游任务准确率差异 <1%
  - **E3 TTFT 加速**：在 10Gbps 网络下，TTFT 降低 3.7×
  - **E4 编解码开销**：压缩/解压延迟相比传输延迟可忽略

### 结论

KV-Cache 具有良好的可压缩性，CacheGen 的自适应编码方案在高压缩比下保持了生成质量。对 OrchKvCache 的启示：压缩可以与你的分层调度正交组合——在 KV-Cache 从 DRAM 换出到 NVM/SSD 时先压缩，换入时解压，进一步减少 IO 量。

---

## Paper #7: ScissorHands

**ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time**
Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava
**NeurIPS 2023**

### 摘要翻译

本文提出了 **"重要性持续性假说"（Persistence of Importance Hypothesis）**：如果一个 token 在过去的 decode step 中被认为是重要的（获得高注意力分数），那么它在未来的 decode step 中大概率仍然是重要的。基于此假说，本文提出 **ScissorHands**——一种推理时的 KV-Cache 压缩方法，利用历史注意力分数识别关键 token，只保留这些 token 的 KV-Cache。ScissorHands 可将 KV-Cache 压缩至原来的 **20%**，在多种模型和任务上的质量损失可忽略。

### Introduction 介绍

论文观察到了一个重要的现象：在连续的 decode step 中，被注意力机制关注的 token 集合具有**高度的稳定性**。具体地说，如果一个 token 在 step t 时获得了 top-K 注意力分数，那么它在 step t+1、t+2、...中有 80%+ 的概率仍然在 top-K 中。这种稳定性被称为 "Persistence of Importance"。

这个发现有两个重要推论：
1. 可以用过去的注意力分数**预测**未来哪些 token 会被关注→支持预取决策
2. 可以安全地丢弃过去持续不被关注的 token→支持 KV-Cache 缩减

### 解决的问题

如何利用注意力分数的时间稳定性，在推理时动态精简 KV-Cache。

### 系统设计

1. **重要性持续性验证**：
   - 在多种模型（GPT-2, OPT, LLaMA）上，追踪每个 token 在连续 decode step 中的注意力排名
   - 度量指标：**Jaccard 相似度**——连续两步的 top-K 集合的交集与并集之比
   - 实验结果：Jaccard 相似度 >0.8（非常高的稳定性）

2. **基于历史的 Token 选择**：
   - 维护每个 token 的**重要性分数**（历史注意力分数的 EMA）
   - 每隔 R 个 decode step，重新评估 token 重要性并更新保留集合
   - 保留策略："Pivotal Token（关键 token）" = 最近 R 步中至少有一次进入 top-K 的 token

3. **与标准 attention 的集成**：
   - 在标准 attention 计算前，先用保留的 KV-Cache 子集计算
   - 对未保留的 token，其注意力贡献被近似为零
   - 开销仅为一次排序操作

### 实验内容

- **模型**：OPT-6.7B/13B/66B, LLaMA-7B/13B/30B
- **基线**：Full KV-Cache, Random eviction, Local (sliding window), H2O
- **评估**：WikiText-2 困惑度、PG-19 长文本困惑度、下游零样本任务
- **关键实验**：
  - **E1 持续性验证**：Jaccard 相似度跨模型 >0.8，跨层 >0.75
  - **E2 压缩比 vs 质量**：保留 20% 时困惑度增加 <0.3；保留 10% 时增加 <1.0
  - **E3 与 H2O 对比**：ScissorHands 在极低保留率（<15%）下优于 H2O

### 结论

重要性持续性假说得到了广泛验证，这为基于历史注意力分数的预测提供了坚实的理论基础。对 OrchKvCache 的核心启示：（1）你的冷热分级器可以利用历史注意力分数的 EMA 来预测未来热度——这是有理论支撑的；（2）重要性持续性也支撑了你的预取调度的可行性。

---

## Paper #8: SqueezeAttention

**SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget**
Zihao Wang et al., 2024

### 摘要翻译

现有 KV-Cache 管理方案对所有 Transformer 层施加**统一的**缓存预算（如每层保留相同数量的 token），但不同层的注意力模式差异显著。本文提出 **SqueezeAttention**，通过分析每层的注意力稀疏度，为每层分配**最优的 KV-Cache 预算**——重要性高的层多保留，不重要的层激进缩减。在总预算相同的约束下，SqueezeAttention 比统一分配策略显著提升了生成质量。

### Introduction 介绍

论文通过大量实验发现了一个被忽视的现象：**不同 Transformer 层对 KV-Cache 缩减的敏感度差异极大**。
- **底层（如 layer 0~10）**：注意力模式高度局部化（主要关注相邻 token），KV-Cache 可以激进缩减
- **中间层（如 layer 10~25）**：注意力模式混合，部分头全局、部分头局部
- **高层（如 layer 25+）**：注意力模式更全局化，对 KV-Cache 缩减最敏感

如果统一将所有层的 KV-Cache 缩减为 20%，底层浪费了预算（本来可以更激进），高层预算不足（导致质量下降）。

### 解决的问题

如何在固定的总 KV-Cache 预算下，按层级分配最优的缓存预算，最大化生成质量。

### 系统设计

1. **层重要性评估**：
   - 对每层独立计算 "注意力熵"（attention entropy）：熵高 = 注意力分散 = 需要更多 KV-Cache
   - 计算每层的 "KV-Cache 敏感度"：对该层 KV-Cache 缩减后困惑度增加的梯度
   - 离线 profiling：在校准数据集上运行一次，获取各层敏感度

2. **预算分配**：
   - 给定总预算 B，按层敏感度比例分配
   - 敏感度高的层分配更多 token，敏感度低的层分配更少
   - 形式化为约束优化问题求解

3. **与驱逐策略组合**：
   - SqueezeAttention 是一个预算分配框架，可以与任意 token 驱逐策略（H2O, ScissorHands 等）组合
   - 每层按分配的预算独立执行驱逐

### 实验内容

- **模型**：LLaMA-2-7B/13B, Mistral-7B
- **基线**：统一预算 + H2O, 统一预算 + ScissorHands, 统一预算 + Local
- **关键实验**：
  - **E1 层敏感度差异**：不同层的敏感度相差最多 **10×**
  - **E2 质量提升**：相同总预算下，分层分配比统一分配降低困惑度 **5~15%**
  - **E3 与 H2O 组合**：SqueezeAttention + H2O 在 20% 预算下达到接近 Full KV 的质量

### 结论

"一刀切"的 KV-Cache 管理策略是次优的。对 OrchKvCache 的启示：你的冷热分级器应该**按层独立维护阈值**——底层可以更激进地将 KV-Cache 换出，高层应该尽可能保留在 GPU 上。

---

## Paper #9: Quest

**Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference**
Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han
**ICML 2024**

### 摘要翻译

长上下文 LLM 推理中，KV-Cache 的 attention 计算复杂度与序列长度成线性关系，成为延迟瓶颈。现有的稀疏注意力方法（如 H2O）使用**静态的**重要性评估，但实际上 token 的重要性是**依赖于当前 query 的**——同一个 KV token 对不同 query 可能完全不同地重要或不重要。本文提出 **Quest**，通过维护每个 KV **page 的统计量**（每个 channel 的 min/max 值），在 attention 计算前快速估算每个 page 与当前 query 的相关性，仅对相关 page 执行精确 attention 计算。Quest 在 128K 上下文中实现了 **2.23× 的自注意力加速**，质量损失可忽略。

### Introduction 介绍

Quest 指出了现有 KV-Cache 管理方法的一个核心局限：**query-agnostic（不考虑当前 query）**。H2O、ScissorHands 等方法基于历史注意力分数来判断 token 的重要性，但这个判断对所有未来的 query 都是相同的。然而实际中，每个 query 关注的 token 子集是不同的。

Quest 的关键观察：如果能在 attention 计算之前（O(1) 时间内）估算每个 KV page 与当前 query 的最大可能注意力分数，就可以快速过滤掉不相关的 page，只对少量 "可能重要" 的 page 做精确计算。

### 解决的问题

如何在保持生成质量的前提下，利用 query-aware 的稀疏性加速长上下文 attention 计算。

### 系统设计

1. **Page 组织**：
   - 将 KV-Cache 按 page 组织（每个 page 包含 P 个连续 token 的 KV 数据）
   - page 大小 P 通常为 16~64

2. **Page 统计量维护**：
   - 对每个 page，维护 Key 向量每个 channel 的 **min 值和 max 值**
   - 统计量在 token 追加时增量更新，开销极小

3. **Query-Aware 选择**：
   - 对当前 query Q，利用 page 的 min/max 统计量快速计算一个注意力上界
   - 上界计算：`upper_bound = sum_over_channels(max(Q_ch * Key_min_ch, Q_ch * Key_max_ch))`
   - 只选择上界超过阈值的 page 参与精确 attention 计算

4. **自适应 page 选择**：
   - 阈值根据 top-K page 的上界自适应调整
   - 保证至少选择 K 个 page（防止全部过滤）
   - K 值根据精度要求配置

### 实验内容

- **模型**：LLaMA-2-7B/13B, Mistral-7B-128K
- **基线**：Full Attention, H2O, StreamingLLM, Local Attention
- **工作负载**：RULER (合成长序列), LongBench, Needle-in-a-Haystack
- **关键实验**：
  - **E1 加速比**：128K 上下文中自注意力加速 2.23×
  - **E2 质量保持**：RULER 上准确率几乎无损，优于 H2O 和 StreamingLLM
  - **E3 Page 大小影响**：page=32 在速度和质量间取得最佳平衡
  - **E4 选择准确率**：Quest 的 page 选择与 Oracle 的重叠度 >90%

### 结论

Query-aware 的 KV-Cache 选择显著优于 query-agnostic 方法。对 OrchKvCache 的启示：（1）你的 KV Block 粒度管理与 Quest 的 page 粒度天然对应；（2）Quest 的 min/max 统计量可以用作你预取调度的辅助信号——快速估算哪些 block 可能被当前 query 关注，优先预取这些 block。

---

## Paper #10: Mooncake

**Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving**
Ruoyu Qin et al., 2024 (月之暗面/Kimi)

### 摘要翻译

本文提出 **Mooncake**，一个以 KV-Cache 为中心的分离式 LLM 推理架构，部署于月之暗面的 Kimi 长上下文模型服务中。Mooncake 将 prefill 和 decode 阶段分离到不同的节点集群上，并设计了一个分布式 KV-Cache 存储池，由 CPU DRAM、GPU 显存和 SSD 共同构成。Mooncake 通过 **KV-Cache 复用**（相同 prefix 的多个请求共享 KV-Cache）和**基于预测的调度**（预估 decode 长度来优化 KV-Cache 放置），大幅提升了长上下文服务的吞吐和资源利用率。

### Introduction 介绍

Mooncake 的设计动机源于 Kimi（月之暗面的商业 LLM 产品）面临的真实挑战：
1. 超长上下文（128K~1M token）导致单个请求的 KV-Cache 非常大
2. 很多请求共享相同的系统 prompt 或文档 prefix，KV-Cache 存在大量复用机会
3. Prefill 和 Decode 对硬件的需求截然不同（compute vs memory），混合部署效率低
4. KV-Cache 的调度直接决定了系统吞吐和延迟

### 解决的问题

如何在大规模商业部署中高效管理海量 KV-Cache，最大化 GPU 利用率和服务吞吐量。

### 系统设计

1. **Prefill-Decode 分离**：
   - Prefill 节点使用计算密集型 GPU，decode 节点使用内存密集型 GPU
   - Prefill 完成后将 KV-Cache 传输到分布式 KV-Cache 池

2. **分布式 KV-Cache 池**：
   - 三层存储：GPU 显存（最快）→ CPU DRAM（中间）→ SSD（最大容量）
   - KV-Cache 以 chunk 为单位管理（chunk = 一组连续 token 的 KV 数据）
   - 基于访问热度自动在三层之间迁移

3. **KV-Cache 复用**：
   - 基于 prefix hash 识别可复用的 KV-Cache
   - 相同 prefix 的请求直接使用已有的 KV-Cache，避免重复 prefill
   - prefix 缓存使用 LRU 管理

4. **预测式调度**：
   - 预测每个请求的生成长度，据此估算 KV-Cache 的总需求量
   - 提前将可能需要的 KV-Cache 迁移到合适的存储层

### 实验内容

- **规模**：数千张 GPU 的生产集群
- **模型**：Kimi 系列长上下文模型（具体架构未完全公开）
- **评估指标**：请求吞吐（req/s）、TTFT、TPOT、GPU 利用率、KV-Cache 命中率
- **关键实验**：
  - **E1 KV-Cache 复用率**：在真实流量下，prefix 命中率约 60~70%
  - **E2 分离架构收益**：相比混合部署，prefill 吞吐提升约 2×
  - **E3 三层存储效果**：SSD 层使可缓存的 KV-Cache 量增加 10×+

### 结论

KV-Cache 是 LLM serving 的核心资源，以 KV-Cache 为中心的设计可以显著提升系统效率。对 OrchKvCache 的启示：Mooncake 验证了多层存储管理 KV-Cache 的工业界需求和可行性；你的 OrchFS 后端可以被视为一种更高效的存储池实现（利用异构 IO 优化）。

---
---

# 第三部分：LLM 推理系统基础设施

---

## Paper #11: Orca

**Orca: A Distributed Serving System for Transformer-Based Generative Models**
Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun
**OSDI 2022**

### 摘要翻译

基于 Transformer 的生成模型的 serving 需要处理自回归解码的迭代特性。现有系统使用**请求级调度**——一个 batch 中所有请求必须全部生成完毕才能接受新请求，导致严重的资源浪费和延迟增加。本文提出 **Orca**，引入**迭代级调度（Iteration-Level Scheduling）**——在每个 decode step（迭代）结束后，已完成的请求立即退出 batch，新请求立即加入。这种细粒度调度使得 GPU 始终被充分利用。Orca 的吞吐量比现有系统提升了 **36.9×**。

### Introduction 介绍

论文指出了请求级调度的根本低效之处：在一个 batch 中，不同请求的生成长度差异很大（如一些请求生成 10 个 token 就结束了，另一些需要 512 个 token）。在请求级调度下，短请求必须等待最长的请求完成，GPU 在短请求完成后的等待期间处于空闲状态。

**关键洞察**：自回归推理的每个 decode step 是独立的——每步只需要上一步的输出和 KV-Cache。因此可以在每一步重新组合 batch，而不必等所有请求完成。

### 解决的问题

如何在 LLM serving 中实现细粒度的动态批处理，消除因请求长度差异导致的 GPU 空闲。

### 系统设计

1. **迭代级调度**：
   - 每个 decode step 后检查：是否有请求生成了 EOS token → 退出 batch
   - 检查等待队列：是否有新请求可以加入当前 batch → 加入 batch
   - 每步的 batch 组成是动态变化的

2. **Selective Batching**：
   - Prefill 和 Decode 的计算模式不同（prefill 是 GEMM，decode 是 GEMV），将两者分开执行
   - 新加入 batch 的请求先执行 prefill，再与其他请求一起执行 decode

3. **KV-Cache 管理**：
   - 为每个请求独立维护 KV-Cache
   - KV-Cache 大小随生成长度动态增长
   - 请求退出时立即释放 KV-Cache

### 实验内容

- **模型**：GPT-3 (175B, 模拟), GPT-J (6B, 真实运行)
- **基线**：FasterTransformer (请求级调度), Megatron-LM
- **工作负载**：合成请求（不同生成长度分布）
- **关键实验**：
  - **E1 吞吐量**：Orca 在 GPT-3 上比 FasterTransformer 提升 **36.9×**
  - **E2 延迟分布**：P99 延迟降低显著，因为短请求不再被长请求阻塞
  - **E3 GPU 利用率**：持续保持 >90% 的 GPU 计算利用率

### 结论

迭代级调度是 LLM serving 的基础能力，已被 vLLM、TensorRT-LLM 等后续系统广泛采纳。对 OrchKvCache 的启示：你的 KV-Cache 管理必须兼容动态 batch——请求随到随走，KV-Cache 的分配和释放是高度动态的。

---

## Paper #12: DistServe

**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**
Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
**OSDI 2024**

### 摘要翻译

现有 LLM serving 系统将 prefill 和 decode 混合在同一个 GPU 上执行，但两者的计算特征截然不同：prefill 是 compute-bound（大矩阵乘），decode 是 memory-bound（小矩阵乘、大量 KV-Cache 访问）。这种混合部署导致了干扰——长 prefill 会延迟共处的 decode 请求，反之亦然。本文提出 **DistServe**，将 prefill 和 decode **分离到不同的 GPU 集群**上执行，并设计了 **KV-Cache 传输机制** 将 prefill 产生的 KV-Cache 高效传输到 decode 节点。在延迟 SLA 约束下，DistServe 的有效吞吐（goodput）比混合部署提升 **2~4×**。

### Introduction 介绍

论文分析了 prefill-decode 混合部署的问题：
1. **Prefill 干扰 Decode**：一个长 prompt 的 prefill 可能需要数百毫秒，期间同 GPU 上的 decode 请求被阻塞
2. **资源浪费**：Prefill 需要大量计算但少量 KV-Cache 访问，Decode 反之——用同一种 GPU 服务两者效率都不高
3. **SLA 难以保证**：由于相互干扰，TTFT 和 TPOT 都难以稳定控制

### 解决的问题

如何通过分离 prefill 和 decode 消除相互干扰，在满足延迟 SLA 的前提下最大化有效吞吐。

### 系统设计

1. **分离架构**：
   - **Prefill 集群**：专门执行 prefill，选择计算密集型 GPU（如 A100）
   - **Decode 集群**：专门执行 decode，可使用内存密集型 GPU
   - 两个集群通过高速网络连接

2. **KV-Cache 传输**：
   - Prefill 完成后，将 KV-Cache 从 Prefill GPU 传输到 Decode GPU
   - 传输通过 NCCL/RDMA 实现，带宽 ~100Gbps
   - 按层流水线传输：传完第 0 层 KV-Cache 后，Decode 端可以开始执行第 0 层 decode，同时继续接收后续层

3. **Goodput 优化调度**：
   - 定义 goodput = 满足 SLA 的请求吞吐
   - 根据 TTFT SLA 和 TPOT SLA 分别配置 Prefill 和 Decode 集群的 GPU 数量
   - 动态分配请求到两个集群

### 实验内容

- **模型**：LLaMA-2-7B/13B/70B, Yi-34B
- **基线**：vLLM (混合部署), Sarathi (混合+chunked prefill)
- **关键实验**：
  - **E1 Goodput**：在 TTFT<200ms + TPOT<50ms 的 SLA 下，goodput 提升 2~4×
  - **E2 尾延迟**：P99 TPOT 从 ~200ms 降至 ~50ms
  - **E3 KV 传输开销**：传输延迟约 10~30ms，可被 prefill 时间覆盖

### 结论

Prefill-Decode 分离是 LLM serving 的重要趋势。对 OrchKvCache 的启示：在分离架构下，KV-Cache 需要跨节点传输后存储在 decode 端——你的分层存储可以作为 decode 端的 KV-Cache 存储后端，NVM/SSD 层提供大容量缓存。

---

## Paper #13: Sarathi-Serve

**Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**
Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee
**OSDI 2024**

### 摘要翻译

LLM 推理中，throughput（吞吐）和 latency（延迟）之间存在根本性的权衡。增大批处理大小提升吞吐，但长 prefill 会阻塞 decode 导致 decode 延迟飙升。本文提出 **Sarathi-Serve**，引入 **Chunked Prefill** 技术——将长 prompt 的 prefill 切分为多个小块，每个块与 decode 请求混合在同一个 micro-batch 中执行。这样既利用了 prefill 的计算密度提升 GPU 利用率，又避免了长 prefill 阻塞 decode。Sarathi-Serve 在保持低 decode 延迟的同时，吞吐量接近理论上界。

### Introduction 介绍

论文分析了 vLLM 和 Orca 的调度局限：
1. **Orca/vLLM 的问题**：虽然实现了迭代级调度，但在一个 batch 中混合 prefill 和 decode 仍然存在问题——prefill 的大 GEMM 和 decode 的小 GEMV 混在一起，GPU kernel 效率低
2. **纯分离的问题**：完全将 prefill 和 decode 分开（如 DistServe），需要多套 GPU，成本高

Chunked Prefill 是一个折中方案：将 prefill 切成小块（如每块 512 token），每块的计算量与若干 decode 请求相当，可以高效混合执行。

### 解决的问题

如何在单 GPU 上同时服务 prefill 和 decode，在不增加硬件成本的前提下同时优化吞吐和延迟。

### 系统设计

1. **Chunked Prefill**：
   - 将一个 prompt（如 4096 token）切分为多个 chunk（如 8 个 512-token chunk）
   - 每个 chunk 作为一个 "小 prefill" 与 decode 请求放在同一个 micro-batch 中
   - GPU 先执行 chunk prefill，再执行 decode，一个迭代周期内完成

2. **Chunk 大小选择**：
   - Chunk 过大→阻塞 decode 延迟；Chunk 过小→prefill 效率低
   - 论文分析了最优 chunk size 与 batch size、模型大小的关系
   - 典型选择：256~1024 token/chunk

3. **KV-Cache 增量管理**：
   - Chunked Prefill 使得 KV-Cache 是分批生成的（每个 chunk 生成一部分）
   - 需要在 chunk 之间保存和拼接中间 KV-Cache

4. **调度策略**：
   - 优先调度 decode 请求（保证低延迟）
   - 用 prefill chunk 填充剩余的 GPU 计算能力
   - 动态调整 chunk 数量以适应当前 GPU 负载

### 实验内容

- **模型**：LLaMA-2-7B/13B/70B, Yi-34B
- **基线**：vLLM, Orca, DistServe
- **关键实验**：
  - **E1 吞吐-延迟权衡**：Sarathi-Serve 在 TPOT < 50ms 约束下的吞吐比 vLLM 高 2×
  - **E2 Chunk 大小影响**：512-token chunk 在大多数配置下最优
  - **E3 vs DistServe**：单 GPU 性能接近 DistServe 的双 GPU 配置

### 结论

Chunked Prefill 是一种优雅的 throughput-latency 权衡方案。对 OrchKvCache 的启示：（1）Chunked Prefill 下 KV-Cache 是增量生成的，你的分配器需要支持动态增长；（2）Chunk 间的间隙可以用来执行 KV-Cache 迁移操作（在 prefill chunk 计算时同步执行换出/换入）。

---

## Paper #14: FlashAttention-2

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
Tri Dao
**ICLR 2024**（FlashAttention-1: NeurIPS 2022）

### 摘要翻译

Attention 是 Transformer 的核心运算，但标准实现需要 O(n²) 的显存来存储注意力矩阵，且受 GPU 内存带宽限制。**FlashAttention** 提出了一种 IO-aware 的精确 attention 算法：通过 tiling（分块计算）和 kernel fusion（算子融合），在 GPU SRAM（L1/L2 cache）上完成 attention 计算，避免将完整的 n×n 注意力矩阵写入 HBM。**FlashAttention-2** 在此基础上优化了并行性和工作分配，在 A100 GPU 上达到理论峰值 FLOPS 的 **72%**，比 FlashAttention-1 快 **2×**。

### Introduction 介绍

标准 attention 的 GPU 实现流程：
1. 计算 QK^T → n×n 矩阵 → 写入 HBM
2. 对 QK^T 做 softmax → 读/写 HBM
3. 计算 softmax(QK^T) × V → 读 HBM

每一步都需要读写 O(n²) 的数据到 GPU HBM，而 HBM 带宽（~2TB/s）远低于 GPU 计算能力（~312 TFLOPS on A100）。Attention 因此是 **memory-bound** 的。

FlashAttention 的核心思想：**永远不将完整的 n×n 矩阵写入 HBM**。通过 tiling 将 Q、K、V 切成小块，每次在 GPU SRAM（~20MB，带宽 ~19TB/s）上计算一个 tile 的 attention，使用 online softmax 技术将 tile 结果增量合并。

### 解决的问题

如何在不牺牲精确性的前提下，将 attention 计算从 memory-bound 转变为 compute-bound，最大化 GPU 利用率。

### 系统设计

1. **Tiling 策略**（FlashAttention-1）：
   - 将 Q 按行切块（block_size_q 行），K/V 按行切块（block_size_kv 行）
   - 外层循环遍历 K/V 的 block，内层循环遍历 Q 的 block
   - 每次加载一个 Q-block 和 K/V-block 到 SRAM，计算部分 attention

2. **Online Softmax**：
   - 由于 softmax 需要全局信息（max 和 sum），不能直接分块计算
   - 使用 online softmax 技巧：维护运行中的 max 和 sum，每个 tile 计算后更新
   - 数学上严格等价于标准 softmax

3. **FlashAttention-2 改进**：
   - **减少非 matmul 计算量**：将 rescaling 操作延迟到最后一步
   - **改进并行策略**：在 sequence 维度上并行（而非 batch/head 维度），更适合长序列
   - **更好的 warp 分配**：将 Q 和 K/V 的加载分配给不同的 warp group，减少共享内存冲突

4. **反向传播优化**：
   - 前向传播保存 softmax 的统计量（logsumexp），反向传播时重新计算 attention 矩阵
   - 用计算换存储，峰值显存从 O(n²) 降至 O(n)

### 实验内容

- **硬件**：A100 80GB GPU
- **基线**：PyTorch 标准 attention, xFormers, Triton attention
- **关键实验**：
  - **E1 Kernel 性能**：FlashAttention-2 达到 A100 峰值 FLOPS 的 72%（vs FA1 的 ~50%）
  - **E2 端到端训练加速**：GPT-2 训练速度提升 2×
  - **E3 长序列扩展**：支持 64K+ 序列长度而不 OOM

### 结论

FlashAttention 彻底改变了 attention 的实现方式，成为所有主流推理框架的标配。对 OrchKvCache 的启示：（1）FlashAttention 本身就是 block-wise 处理 KV，与你的 KV Block 管理天然对应；（2）在 KV-Cache 部分换出的场景下，你需要修改 FlashAttention kernel 以支持 "部分 KV 在 GPU、部分需要从 Host 加载" 的模式；（3）理解 GPU HBM 带宽瓶颈有助于你设计合理的换入时机和粒度。

---
---

# 第四部分：存储系统

---

## Paper #15: OrchFS

**Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs**
Yekang Zhan, Haichuan Hu, Xiangrui Yang, Qiang Cao, Hong Jiang, Shaohua Wang, Jie Yao
**FAST 2025**

### 摘要翻译

现代 SSD 的容量和带宽持续增长，但现有 SSD 文件系统将用户请求转换为内存页对齐的同构块 IO，未能充分利用 SSD 的高写带宽。本文通过实验分析发现了三个导致写入低效的根本原因：（1）SSD 页对齐开销，（2）页缓存开销，（3）IO 并发不足。为此，本文提出 **OrchFS**——一个基于对齐写分区的异构 IO 编排文件系统。OrchFS 利用少量 NVM（非易失性内存）来最大化 SSD 性能，将文件写入转换为 SSD 页对齐的 SSD-IO 和未对齐的 NVM-IO，通过各自的最优数据路径以多线程方式执行。实验表明，OrchFS 在写和读性能上分别比现有方案提升最高 **29.76×** 和 **6.79×**。

### Introduction 介绍

论文首先量化了现有文件系统在高带宽 SSD 上的表现不佳：即使是最新的高带宽 NVMe SSD（理论带宽 ~7GB/s），传统文件系统（EXT4, F2FS）和混合文件系统（Strata, SPFS）都只能利用其 20~50% 的带宽。

三个根本原因：
1. **SSD 页对齐开销**：SSD 的内部写入以 4KB 页为粒度，非页对齐的写入需要额外的 Read-Modify-Write
2. **页缓存开销**：传统文件系统通过页缓存中转所有 IO，引入了额外的内存拷贝
3. **IO 并发不足**：文件系统通常使用单线程提交 IO，无法充分利用 SSD 的内部并行性（多通道、多 die）

### 解决的问题

如何重新设计文件系统的请求-IO 转换过程，充分利用高带宽 SSD 的写性能。

### 系统设计

1. **异构数据布局（Heterogeneous-unit Data Layout）**：
   - 两种存储单元：4KB NVM Page + 32KB SSD Block
   - 每个 32KB 逻辑块由 8 个 4KB 槽位组成
   - 三种节点类型：
     - **SSD_BLOCK**：整块 32KB 在 SSD 上
     - **VIR_LEAF_NODE**：8 个 4KB 页都在 NVM 上
     - **STRATA_NODE**：混合——部分槽位在 NVM，部分在 SSD

2. **对齐写分区（Alignment-based Write Partition）**：
   - 分析每次写请求覆盖的 4KB 页：
     - 若覆盖 ≥6 个连续对齐页（STRATA_THRESHOLD=6）→ 写入 SSD
     - 否则 → 写入 NVM
   - 头部/尾部不对齐的部分始终写入 NVM

3. **统一映射结构（Unified Per-file Mapping）**：
   - `offset_info_t` / `virtual_node_t` 统一描述每个逻辑块在 NVM 和 SSD 上的位置
   - 通过 `nvm_page_id[8]` 和 `ssd_dev_addr` 记录混合布局

4. **嵌入式并行 IO 引擎**：
   - NVM IO 线程池（默认 5 线程）和 SSD IO 线程池（默认 32 线程）独立运行
   - 写请求被拆分后分别提交到两个线程池并行执行
   - SSD IO 可以进一步按 split_size 切分以提升并发

5. **NVM → SSD 迁移**：
   - 当 NVM 使用量超过阈值时，触发后台迁移
   - LRU 选出最久未使用的 NVM 页 → 合并 8 个 4KB 页为 32KB 块 → 写入 SSD
   - 调用 `change_virnd_to_ssdblk()` 更新索引

### 实验内容

- **硬件**：Intel Optane PM 128GB (NVM) + Samsung PM9A3 NVMe SSD
- **基线**：EXT4, F2FS (纯 SSD), NOVA, OdinFS, ArckFS (纯 NVM), Strata, SPFS, PHFS (混合)
- **Benchmark**：顺序写、随机写、随机读、混合读写、Filebench (varmail, fileserver)
- **关键实验**：
  - **E1 顺序写**：OrchFS 比 EXT4 快 **29.76×**，达到 SSD 理论带宽的 90%+
  - **E2 随机写**：比 F2FS 快 **11.2×**
  - **E3 混合读写**：比 Strata 快 **3.8×**
  - **E4 Filebench**：varmail 场景比 EXT4 快 **5.7×**

### 结论

OrchFS 证明了通过异构 IO 编排，可以几乎完全释放高带宽 SSD 的性能潜力。对 OrchKvCache：这是你的核心技术底座——对齐写分区、多粒度布局、并行 IO 引擎、迁移机制都可以直接服务于 KV-Cache 管理。

---

## Paper #16: Strata

**Strata: A Cross Media File System**
Youngjin Kwon, Henrique Fingler, Tyler Hunt, Simon Peter, Emmett Witchel, Thomas Anderson
**SOSP 2017**

### 摘要翻译

NVM（非易失性内存）的出现为文件系统设计带来了新的可能。本文提出 **Strata**——一个跨介质文件系统，联合管理 NVM 和 SSD（或 HDD）。Strata 的核心思想是将 NVM 用作**操作日志层**：所有写入首先追加到 NVM 上的日志中（快速、小粒度），然后由后台 "Digest" 线程将日志内容合并到 SSD 上的持久化存储中（批量、大粒度）。这种分层架构兼顾了 NVM 的低延迟写入和 SSD 的大容量持久化。

### Introduction 介绍

论文提出的背景：NVM（如 Intel Optane PM）提供了接近 DRAM 的延迟和持久性，但容量有限且价格高于 SSD。如何同时利用 NVM 的低延迟和 SSD 的大容量是核心挑战。

Strata 的关键洞察：**写入路径和读取路径对存储介质的需求不同**。
- 写入需要低延迟 → 用 NVM
- 持久化存储需要大容量 → 用 SSD
- 通过后台 Digest 将 NVM 日志中的数据迁移到 SSD

### 解决的问题

如何设计一个文件系统，高效利用 NVM+SSD 两种存储介质的各自优势。

### 系统设计

1. **日志层（NVM）**：
   - 所有文件操作（写入、创建、删除）首先追加到 NVM 上的日志中
   - 日志操作是同步的，写入返回后即持久化
   - 粒度：每次写入操作一条日志记录

2. **Digest（消化/合并）**：
   - 后台线程将 NVM 日志的内容合并到 SSD 上的 "shared area"
   - Digest 是批量操作，将多个小写合并为大块 SSD 写入
   - Digest 完成后释放 NVM 日志空间

3. **LibFS + KernelFS 分层**：
   - **LibFS**：用户态库，处理文件操作和 NVM 日志
   - **KernelFS**：内核模块，管理 SSD 和执行 Digest
   - 这种 User-Kernel 分层架构被 OrchFS 继承

4. **读取路径**：
   - 先查 NVM 日志（最新数据），再查 SSD（持久化数据）
   - 日志中的数据覆盖 SSD 中的旧数据

### 实验内容

- **硬件**：模拟 NVM (Intel DCPMM 尚未上市时) + Samsung SSD
- **基线**：EXT4, XFS, NOVA (纯 NVM)
- **关键实验**：
  - **E1 写延迟**：NVM 日志写入延迟 ~1μs（vs EXT4 on SSD ~10μs）
  - **E2 读取性能**：热数据在 NVM 中读取快，冷数据从 SSD 读取
  - **E3 Digest 开销**：后台 Digest 对前台性能影响 <5%

### 结论

Strata 开创了 "NVM 做快速日志 + SSD 做大容量存储" 的架构模式。OrchFS 在此基础上做了重大改进——从 "日志模式" 改为 "对齐写分区模式"，避免了 Digest 的双写开销。对 OrchKvCache：理解 Strata 有助于你理解 OrchFS 的设计演进脉络。

---

## Paper #17: SPFS

**SPFS: Splitting and Piggy-backing the File System for Performance with Persistent Memory**
Jongseok Kim et al.
**ATC 2023**

### 摘要翻译

持久内存（PM）为传统 SSD 文件系统提供了加速机会。本文提出 **SPFS**，一种将 PM 作为 SSD 文件系统"加速层"的方案。SPFS 的核心思想是**分离（Splitting）**——将文件系统的元数据和热数据放在 PM 上，冷数据放在 SSD 上——以及**捎带（Piggy-backing）**——利用 PM 的低延迟将前台写入先缓冲在 PM，后台异步刷入 SSD。SPFS 作为现有 SSD 文件系统（如 EXT4）的透明加速层，无需修改应用。

### Introduction 介绍

与 Strata 的全新文件系统设计不同，SPFS 采用了更实用的方案：不替换现有文件系统，而是在 EXT4/XFS 之上叠加一个 PM 加速层。这降低了部署门槛。

### 解决的问题

如何以最小的改动利用 PM 加速现有 SSD 文件系统。

### 系统设计

1. **元数据分离**：inode、目录等元数据存放在 PM 上（低延迟访问）
2. **写缓冲**：写请求先缓冲到 PM，立即返回给应用（低延迟响应）
3. **异步刷入**：后台线程将 PM 缓冲的数据批量写入 SSD 的底层文件系统
4. **读加速**：热数据缓存在 PM，命中时直接从 PM 读取

### 实验内容

- **基线**：EXT4 on SSD, NOVA on PM
- **关键结果**：比 EXT4 提升 2~5× 写性能，元数据操作提升 10×+

### 结论

PM 作为 SSD 加速层的方案在工程上更可行。对 OrchKvCache：SPFS 的 "冷热分离 + PM 缓冲" 思路与你的 DRAM-NVM-SSD 分层类似，可在 Related Work 中对比讨论。

---

## Paper #18: vTensor

**vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving**
Jiale Xu et al.
**FAST 2025**

### 摘要翻译

LLM serving 中，KV-Cache 等张量的管理面临显存碎片、跨设备迁移低效、缺乏灵活性等问题。本文提出 **vTensor**——一个虚拟张量管理系统，对 LLM serving 中的各种张量（KV-Cache、模型权重、激活值）提供统一的虚拟化抽象。vTensor 将物理内存管理与逻辑张量操作解耦，支持 GPU、CPU DRAM 和 SSD 之间的透明张量放置和迁移。vTensor 集成到 vLLM 中，在长上下文场景下将吞吐量提升了 **1.86×**，最大可服务的序列长度扩大了 **3.1×**。

### Introduction 介绍

论文指出了现有 LLM serving 系统在张量管理上的局限：
1. vLLM 的 PagedAttention 只管 KV-Cache，不管模型权重和激活值
2. 张量的物理放置与逻辑操作耦合——一旦张量在 GPU 上分配，就难以透明地迁移到 CPU 或 SSD
3. 不同请求的张量生命周期不同，导致显存碎片

vTensor 的核心思想：借鉴操作系统的虚拟内存，**让上层应用只看到虚拟张量地址，物理放置和迁移由 vTensor 层自动管理**。

### 解决的问题

如何为 LLM serving 提供统一、灵活的张量虚拟化管理，支持跨设备透明迁移。

### 系统设计

1. **虚拟张量抽象**：
   - 每个张量有一个虚拟地址和一组物理页的映射（类似页表）
   - 物理页可以在 GPU、CPU DRAM 或 SSD 上
   - 上层推理引擎使用虚拟地址访问张量

2. **按需分页（Demand Paging）**：
   - 张量页在首次访问时才分配物理内存
   - 不在 GPU 上的页触发 "page fault"，自动从 CPU/SSD 换入

3. **预取和换出策略**：
   - 基于访问模式预测的预取
   - 基于 LRU/优先级的换出

4. **与 vLLM 集成**：
   - 替换 vLLM 的 BlockManager
   - 利用 CUDA Unified Memory 或自定义的 page fault handler

### 实验内容

- **模型**：LLaMA-2-7B/13B/70B
- **基线**：vLLM (原始), FlexGen
- **关键实验**：
  - **E1 吞吐量**：长序列（32K~128K）下吞吐提升 1.86×
  - **E2 最大序列长度**：可服务的最大序列长度扩大 3.1×
  - **E3 碎片率**：物理内存碎片率降低 60%+

### 结论

虚拟张量管理提供了更灵活的内存管理能力。对 OrchKvCache 的关键差异化：vTensor 侧重**通用虚拟化**，对底层存储设备无感知；OrchKvCache 侧重**利用 OrchFS 的异构 IO 特性**，针对 KV-Cache 的冷热特征做精细化的多粒度管理——这是你相对 vTensor 的核心优势点。

---

## Paper #19: InstInfer

**InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference**
Xiurui Pan et al., 2024

### 摘要翻译

长上下文 LLM 推理中，将 KV-Cache offload 到 CPU/SSD 后，Host 与存储设备之间的数据传输成为瓶颈。本文提出 **InstInfer**——一种将 attention 计算直接 offload 到 **计算型 SSD（Computational Storage Drive, CSD）** 内部的方案。InstInfer 在 SSD 控制器内部执行部分 attention 计算（Q×K^T 和 softmax），只将计算结果（而非完整 KV-Cache）传输回 Host，大幅减少了 Host-SSD 数据传输量。

### Introduction 介绍

论文分析了传统 KV-Cache offloading 的带宽瓶颈：
- 以 LLaMA-70B + 128K context 为例，每个 decode step 需要从 SSD 读取 ~5GB 的 KV-Cache
- 即使用最新的 PCIe 5.0 SSD（~14GB/s），也需要 ~350ms/step——完全不可接受
- 问题的根源：传统方案是 "数据搬到计算"，但搬的数据量太大

InstInfer 的思路：**计算搬到数据**——在 SSD 内部完成 attention 计算，只传输结果向量。

### 解决的问题

如何通过近存储计算消除 KV-Cache offloading 的带宽瓶颈。

### 系统设计

1. **In-Storage Attention**：
   - KV-Cache 直接存储在 SSD 的 NAND Flash 上
   - SSD 控制器内置的 FPGA/ASIC 执行 Q×K^T 矩阵乘和 softmax
   - 计算结果（attention weights 或 softmax(QK^T)×V 的输出）通过 PCIe 传回 Host

2. **数据传输量分析**：
   - 传统方案：传输 KV-Cache → O(seq_len × d_model) 数据量
   - InstInfer：传输 Query + 返回结果 → O(d_model) 数据量
   - 减少了约 **seq_len 倍**的传输量

3. **Host-SSD 协同**：
   - Host GPU 执行 MLP、LayerNorm 等计算
   - SSD 执行 attention 计算
   - 通过 PCIe 交换中间结果

### 实验内容

- **硬件**：FPGA 模拟的计算型 SSD
- **模型**：LLaMA-2-7B/13B, OPT-6.7B
- **基线**：FlexGen, vLLM (CPU offloading)
- **关键实验**：
  - **E1 延迟**：每 decode step 延迟从 350ms 降至 ~10ms
  - **E2 吞吐量**：比 FlexGen 提升 **10~20×**
  - **E3 成本效率**：相比多 GPU 方案，成本降低 5~10×

### 结论

近存储计算为超长上下文推理提供了一条新路径。对 OrchKvCache：InstInfer 代表了 "计算搬到数据" 的极端方案，你的方案是 "数据搬到计算（但更高效地搬）"。两者是互补而非竞争的关系——在 Related Work 中需要讨论对比。

---
---

# 第五部分：基础理论

---

## Paper #20: Attention Is All You Need (Transformer)

**Attention Is All You Need**
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**NeurIPS 2017**

### 摘要翻译

主流的序列转录模型基于包含编码器和解码器的复杂循环或卷积神经网络。性能最优的模型还通过注意力机制连接编码器和解码器。本文提出了一种新的简洁网络架构——**Transformer**，完全基于注意力机制，摒弃了循环和卷积。在两个机器翻译任务上的实验表明，该模型在质量上更优越，同时具有更高的并行性和更短的训练时间。在 WMT 2014 英德翻译上达到 **28.4 BLEU**（提升 2 BLEU），在英法翻译上达到 **41.8 BLEU**（单模型新 SOTA）。

### Introduction 介绍

论文指出了 RNN（循环神经网络）的根本性局限：
1. **顺序依赖**：RNN 必须按时间步顺序计算，无法并行化
2. **长程依赖**：尽管 LSTM/GRU 有所改善，但仍然难以捕获很长距离的依赖关系
3. **计算效率低**：训练时间长

Transformer 的核心创新：**完全用 attention 机制替代循环连接**，实现全局信息交互和完全并行化。

### 解决的问题

如何构建一个高效、可并行化的序列到序列模型，同时具有强大的长程依赖建模能力。

### 系统设计

1. **Scaled Dot-Product Attention**：
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ```
   - Q (Query)、K (Key)、V (Value) 均为输入序列的线性变换
   - softmax(QK^T) 产生注意力权重矩阵（n×n）
   - 加权求和 V 得到输出
   - **这就是 KV-Cache 问题的源头**：decode 时，K 和 V 需要包含所有历史 token 的信息

2. **Multi-Head Attention (MHA)**：
   - 将 Q、K、V 分成 h 个头，每个头独立计算 attention
   - 多头并行计算后拼接
   - 允许模型同时关注不同位置、不同表示子空间的信息

3. **自回归解码**：
   - 解码器在生成第 t 个 token 时，只能关注位置 1~t（因果 mask）
   - 为避免重复计算位置 1~t-1 的 K、V，将其缓存 → **这就是 KV-Cache**

4. **其他组件**：位置编码（正弦/余弦）、残差连接、Layer Norm、FFN

### 实验内容

- **任务**：WMT 2014 英德/英法机器翻译
- **关键结果**：BLEU 分数 SOTA，训练时间比 RNN 快 10×+

### 结论

Transformer 架构通过 self-attention 机制实现了高质量的序列建模。KV-Cache 是 Transformer 自回归推理的必然产物——它是计算效率和内存效率之间的权衡。你的整个项目就是在优化这个权衡的内存端。

---

## Paper #21: GQA (Grouped Query Attention)

**GQA: Training Generalized Multi-Query Attention Models from Multi-Head Checkpoints**
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yinfei Yang, Cutler Shlomi, Santiago Ontañón
**EMNLP 2023**

### 摘要翻译

Multi-Head Attention (MHA) 为每个注意力头独立维护 K 和 V 向量，导致 KV-Cache 随头数线性增长。**Multi-Query Attention (MQA)** 让所有 query 头共享同一组 K、V，大幅减少 KV-Cache，但可能损失模型质量。本文提出 **Grouped Query Attention (GQA)**——将 query 头分为 G 组，每组共享一组 K、V。GQA 在 MHA 和 MQA 之间取得了灵活的平衡：KV-Cache 缩减为 MHA 的 1/G，质量接近 MHA。此外，本文展示了通过 **uptraining（从 MHA checkpoint 微调转换为 GQA）** 的方法，无需从零训练 GQA 模型。

### Introduction 介绍

论文分析了 MHA 在推理时的内存瓶颈：
- MHA 中每个头独立有 K、V → KV-Cache 大小 = 2 × n_heads × seq_len × d_head
- LLaMA-70B 有 64 个头 → KV-Cache 非常大
- MQA 将 n_heads 变为 1 → KV-Cache 缩减 64×，但质量下降明显

GQA 的折中：将 64 个 query 头分为 8 组，每组 8 个 query 头共享 1 组 KV → KV-Cache 缩减 8×，质量损失极小。

### 解决的问题

如何在减少 KV-Cache 大小和保持模型质量之间找到最优平衡。

### 系统设计

1. **GQA 定义**：
   - 标准 MHA：n_heads 个 Q 头，n_heads 个 KV 头（1:1 对应）
   - MQA：n_heads 个 Q 头，1 个 KV 头（n_heads:1）
   - GQA-G：n_heads 个 Q 头，G 个 KV 头（n_heads/G : 1）
   - 每组 Q 头共享同一个 KV 头的 K、V 向量

2. **Uptraining**：
   - 从已有的 MHA 模型 checkpoint 开始
   - 将原来 n_heads 个 KV 头按组平均合并为 G 个（取组内均值）
   - 用少量数据（原训练数据的 ~5%）微调
   - 实验表明 uptraining 后的 GQA 接近从零训练的质量

3. **KV-Cache 影响**：
   - MHA: KV-Cache = 2 × n_heads × seq_len × d_head
   - GQA-8: KV-Cache = 2 × 8 × seq_len × d_head（减少 n_heads/8 倍）
   - LLaMA-2-70B 使用 GQA-8：64 个 Q 头，8 个 KV 头

### 实验内容

- **模型**：T5-Large, T5-XXL (从 MHA uptraining 到 GQA)
- **基线**：MHA (原始), MQA, 从零训练的 GQA
- **关键实验**：
  - **E1 质量**：GQA-8 在摘要和翻译任务上与 MHA 质量差距 <0.5%
  - **E2 推理速度**：GQA-8 的 decode 速度比 MHA 快 ~2×（因为 KV-Cache 访问量减少）
  - **E3 vs MQA**：GQA-8 的质量显著优于 MQA（特别是在大模型上）

### 结论

GQA 是当前主流大模型的标配（LLaMA-2-70B、LLaMA-3、Mistral 等均采用）。对 OrchKvCache 的启示：（1）你的 KV Block 需要按 **KV head group** 管理，而非按 query head——同一组的 query head 共享同一份 KV-Cache；（2）GQA 虽然减小了 KV-Cache，但对于超长上下文（128K+）仍然不够——这验证了你的多级存储方案的必要性；（3）不同模型的 GQA 分组数不同，你的系统需要灵活适配。

---
---

# 附录：论文核心信息速查表

| # | 论文 | 会议 | 核心贡献 | 与 OrchKvCache 的关系 |
|---|------|------|---------|---------------------|
| 1 | vLLM | SOSP'23 | KV-Cache 分页管理 | 基线系统，集成目标 |
| 2 | FlexGen | ICML'23 | GPU/CPU/Disk 三级 offloading | 最直接竞品 |
| 3 | H2O | NeurIPS'23 | 注意力幂律分布，Heavy Hitter | 冷热分级理论基础 |
| 4 | Attention Sink | ICLR'24 | 初始 token 永久高注意力 | Sink token 必须永久 Hot |
| 5 | InfiniGen | OSDI'24 | 层间预测 + 预取 | 预取策略竞品 |
| 6 | CacheGen | SIGCOMM'24 | KV-Cache 压缩 3~5× | 压缩可正交组合 |
| 7 | ScissorHands | NeurIPS'23 | 重要性持续性假说 | 冷热预测理论支撑 |
| 8 | SqueezeAttention | 2024 | 层级差异化 KV 预算 | 按层自适应阈值 |
| 9 | Quest | ICML'24 | Query-aware 稀疏选择 | 预取信号辅助 |
| 10 | Mooncake | 2024 | 工业级 KV-Cache 管理 | 验证多层存储需求 |
| 11 | Orca | OSDI'22 | 连续批处理 | 动态 batch 兼容 |
| 12 | DistServe | OSDI'24 | Prefill-Decode 分离 | KV 传输后的存储 |
| 13 | Sarathi | OSDI'24 | Chunked Prefill | KV 增量生成 |
| 14 | FlashAttention-2 | ICLR'24 | IO-aware 分块 attention | KV Block 粒度对齐 |
| 15 | OrchFS | FAST'25 | 异构 IO 编排文件系统 | 核心技术底座 |
| 16 | Strata | SOSP'17 | NVM+SSD 跨介质文件系统 | OrchFS 前驱工作 |
| 17 | SPFS | ATC'23 | PM 加速 SSD 文件系统 | Related Work 对比 |
| 18 | vTensor | FAST'25 | 虚拟张量管理 | 最接近的并发工作 |
| 19 | InstInfer | 2024 | SSD 内部 attention 计算 | 近存储计算对比 |
| 20 | Transformer | NeurIPS'17 | Self-attention 架构 | KV-Cache 问题根源 |
| 21 | GQA | EMNLP'23 | 分组查询注意力 | KV 头组管理适配 |

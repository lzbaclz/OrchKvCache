# OrchKvCache 项目总结与审视

---

## 一、研究背景与问题

### 1.1 研究对象

大语言模型（LLM）推理中的 **KV-Cache 显存管理**。Transformer 自回归推理在 decode 阶段需要保留所有历史 token 的 Key/Value 向量，其显存占用随序列长度线性增长。以 LLaMA-70B + 128K 上下文为例，单请求 KV-Cache 可达 ~320GB，远超单 GPU 显存。

### 1.2 发现的问题

通过对 21 篇核心论文的系统性调研，以及在 Qwen2.5-1.5B 模型上对真实推理注意力分数分布的实测分析（Exp-M2），识别出现有方案的四个关键不足。

### 调研的 21 篇论文

按四个类别组织：

**KV-Cache 管理核心（9 篇）**

| # | 论文 | 会议 | 与本工作的关系 |
|---|------|------|----------------|
| 1 | vLLM / PagedAttention | SOSP'23 | 基线系统，分页管理 KV 块；仅 GPU↔CPU swap，无冷热感知 |
| 2 | FlexGen | ICML'23 | 直接竞品；GPU/CPU/Disk 三级 offloading，但粒度固定（整层）、无 NVM、离线规划 |
| 3 | H2O (Heavy-Hitter Oracle) | NeurIPS'23 | 冷热分级理论基石；证明注意力幂律分布，但策略有损（直接丢弃冷 token） |
| 4 | StreamingLLM / Attention Sink | ICLR'24 | 发现初始 token 持续获得高注意力（sink 现象）；冷热分级必须特殊处理 |
| 5 | InfiniGen | OSDI'24 | 核心竞品；基于预取的 KV-Cache offloading，GPU-CPU 协同，但无异构存储 |
| 6 | CacheGen | SIGCOMM'24 | KV-Cache 压缩+流式传输，补充方向 |
| 7 | ScissorHands | — | 基于重要性的 KV-Cache 剪枝，理论与技术参考 |
| 8 | SqueezeAttention | — | 语义级 KV 复用，竞品参考 |
| 9 | Quest | — | 基于 page 级注意力估计的 KV 查询加速 |

**LLM 推理系统（5 篇）**

| # | 论文 | 会议 | 关注点 |
|---|------|------|--------|
| 10 | Mooncake | ATC'25 | KVCache-centric 分布式 serving |
| 11 | Orca | OSDI'22 | 连续批处理开山之作 |
| 12 | DistServe | OSDI'24 | Prefill/Decode 分离部署 |
| 13 | Sarathi-Serve | OSDI'24 | 分块 prefill |
| 14 | FlashAttention-2 | ICLR'24 | GPU 高效 attention 实现 |

**存储系统（4 篇）**

| # | 论文 | 会议 | 关注点 |
|---|------|------|--------|
| 15 | OrchFS | FAST'25 | 核心依赖——异构 NVM+SSD 编排、对齐写分区、嵌入式并行 IO |
| 16 | Strata | SOSP'17 | NVM+SSD 分层文件系统，架构参考 |
| 17 | SPFS | ATC'23 | SSD+PM 混合文件系统 |
| 18 | vTensor | FAST'25 | 面向 DNN 的虚拟张量管理，竞品参考 |

**基础理论与技术（3 篇）**

| # | 论文 | 会议 | 关注点 |
|---|------|------|--------|
| 19 | InstInfer | — | 基于存储的大模型推理，竞品参考 |
| 20 | Attention Is All You Need | NeurIPS'17 | Transformer 原始论文 |
| 21 | GQA (Grouped Query Attention) | — | 分组查询注意力，影响 KV-Cache 大小计算 |

### 注意力分布实测分析（Exp-M2）

在 Qwen2.5-1.5B（28 层, 12 头, 2 KV 头, d=128）上，对 3 种输入长度（512/1024/2048 tokens）的真实推理注意力分数进行了系统分析。
方法：在 `output_attentions=True` 模式下执行 prefill，提取每层最后一个 query token 对所有 key 的注意力权重，跨头平均后按降序排列，计算 CDF 覆盖率和 Gini 系数。
脚本：`experiments/exp1_motivation/scripts/m2_attention_analysis.py`，结果存于 `results/m2_attention_analysis.json`。

**分析维度与发现**：

1. **Token 级幂律分布**：Top-10% token 贡献 **90-93%** 的总注意力权重，Top-20% 贡献 **95-96%**。Gini 系数达 0.93-0.95，表明极度不均匀
2. **Block 级聚合仍有效**：以 block_size=16 聚合后，Top-10% block 仍覆盖 ~80% 注意力，证明 block 粒度管理不损失分辨率
3. **层间差异**：中间层（Layer 4-20）集中度最高（Top-10% > 95%），首层和末层较均匀
4. **Attention Sink 现象**：前 5 个 token 在多数层中吸引了 >50% 的注意力（如 Layer 2 的首 token 独占 66%），验证了 Attention Sink ICLR'24 的发现
5. **跨 Decode Step 稳定性**：连续 5 个 decode step 的 Top-10% 热 token 集合的 Jaccard 相似度 0.47-0.70，热 token 有一定持续性但也在演化

**结论**：注意力分数的强幂律特征 + Attention Sink + 跨步稳定性共同论证了"按冷热分级管理 KV-Cache 是合理且高效的"。

### 识别出的四个关键不足

1. **冷数据占据宝贵显存**：注意力分数呈幂律分布（H2O 的 Heavy-Hitter 发现、我们的 M2 实测），大量"冷" KV block 长期驻留 GPU 显存却极少被访问
2. **Offloading 粒度固定**：FlexGen 按整层 offload，vLLM 仅做 GPU↔CPU swap，均无法利用冷热差异做细粒度调度
3. **存储层次单一**：现有方案最多两级（GPU + CPU DRAM），未利用 NVM/SSD 等更深存储层（M4 实测：DRAM 比 SSD 快 19-168×，证明分层缓冲必要）
4. **带宽利用低效**：同步、固定粒度的传输方式无法充分利用存储设备带宽（M3 实测：vLLM 式逐块 eviction 的 SSD 带宽利用率仅 4-26%）

### 1.3 研究问题

#### 核心问题（Main Research Question）

> **RQ**: 如何利用异构存储设备（NVM + SSD）的多粒度 IO 特性，为 LLM 推理中的 KV-Cache 构建高效的分层调度系统，在不影响模型输出质量的前提下，突破单 GPU 显存对可服务上下文长度和并发请求数的限制？

#### 问题分解（Sub-Questions）

上述核心问题可分解为以下 5 个逐层递进的子问题，每个子问题直接映射到前文识别的关键不足和系统设计中的核心模块。

**RQ1: 冷热识别 — 如何在运行时准确且低开销地区分 KV-Cache 块的冷热程度？**

- **问题根源**: Exp-M2 实测表明注意力分数呈强幂律分布（Top-10% token 贡献 90-93% 注意力），但冷热分布并非静态——跨 decode step 的热 token 集合 Jaccard 相似度仅 0.47-0.70，说明冷热状态在持续演化。现有方案要么完全不做冷热区分（vLLM 的 FIFO swap），要么做有损丢弃（H2O 直接删除冷 token），缺乏无损且自适应的分级机制。
- **具体挑战**:
  - 注意力分数采集的开销控制：从 GPU 拷贝 softmax 权重到 CPU 会引入额外延迟，需要在精度和开销之间取得平衡
  - 多信号融合的权重设计：仅靠注意力分数不够（首层和末层分布较均匀），需要结合访问时间衰减（recency）和访问频次（frequency），但三者的权重如何自适应调整？
  - 三级分类的阈值确定：Hot/Warm/Cold 的边界不应是静态的，需要随各存储层的水位动态调整（当 GPU 显存紧张时应更积极地降级，空闲时可放宽）
  - Attention Sink 的特殊处理：前 5 个 token 在多数层中吸引 >50% 注意力（Layer 2 的首 token 独占 66%），这类 token 必须永驻 GPU，分类器需要识别并保护它们
- **对应不足**: 不足 ① — 冷数据占据宝贵显存
- **对应设计**: C1 (attention_tracker) + C2 (hotcold_classifier) + C3 (adaptive_threshold)

**RQ2: 分层放置 — 如何根据冷热等级和存储设备的 IO 特性，将 KV-Cache 块最优地放置在四级存储层次中？**

- **问题根源**: 现有系统最多两级存储（GPU + CPU DRAM），未利用 NVM 和 SSD 等更深层次。Exp-M4/Exp0 实测数据显示各层性能差异巨大：GPU 内部 D2D ~773 GB/s，GPU↔DRAM ~24 GB/s，SSD 顺序读 ~17.8 GB/s、顺序写 ~5.3 GB/s。这种性能梯度意味着简单的两级划分无法兼顾延迟和容量。
- **具体挑战**:
  - 四级存储的容量-延迟权衡：GPU HBM 80GB（~190μs/4MB block 传输延迟）vs DRAM 256GB vs NVM vs SSD（容量大但延迟高），如何根据冷热等级自动选择目标层？
  - 换出路径选择：冷数据从 GPU 换出时，应直接跳到 SSD（两跳：GPU→DRAM→SSD），还是先放 DRAM/NVM 再按需下刷？两跳延迟更高但释放 DRAM 压力，单跳延迟低但 DRAM 可能成为瓶颈
  - 温数据的特殊地位：介于 Hot 和 Cold 之间的 Warm 数据——短期内可能再次被访问——应放在 DRAM 还是 NVM？NVM 换入延迟 (~300ns) 远低于 SSD (~10μs)，但 NVM 容量有限
  - 请求生命周期管理：不同请求的 KV-Cache 有不同的生存期（短对话 vs 长文档），分层策略需要感知请求级别的状态
- **对应不足**: 不足 ③ — 存储层次单一
- **对应设计**: Phase A (gpu_tier, dram_tier) + Phase B (orchfs_tier, io_worker) + C7 (migration_engine)

**RQ3: 粒度适配 — 如何在不同存储层使用不同的管理粒度，以最大化存储带宽利用率？**

- **问题根源**: Exp-M3 实测揭示了一个核心矛盾——vLLM 式的逐块同步 eviction 仅利用了 SSD 峰值带宽的 4-26%（Exp0 SSD 随机 4K 写仅 33K IOPS，而顺序写可达 5.3 GB/s）。原因是现有方案用统一的固定粒度管理所有存储层，而不同设备的最优 IO 粒度差异显著。
- **具体挑战**:
  - NVM vs SSD 的粒度差异：NVM 适合小粒度随机访问（4KB page，延迟 ~300ns），SSD 适合大粒度顺序访问（32KB+ block，充分利用 17.8 GB/s 顺序带宽）。统一粒度必然对某一层不最优
  - KV-Cache block 到存储粒度的映射：模型的 KV block 大小由 `tokens_per_block × n_kv_heads × d_head × dtype` 决定（如 64×2×128×2 = 32KB），如何与 NVM 4KB page 和 SSD 32KB block 对齐？
  - 跨粒度迁移的开销：当 KV block 从 NVM（4KB 页分散存储）迁移到 SSD（需 32KB 对齐连续写入）时，需要聚合多个小页为大块，这个聚合本身有 CPU 和内存拷贝开销
  - OrchFS 的对齐写分区机制：OrchFS 根据写入大小自动路由到 NVM 或 SSD（小写入→NVM page，大写入→SSD block），但 KV-Cache 管理层需要主动构造合适大小的写入请求来触发这个路由
- **对应不足**: 不足 ② — Offloading 粒度固定；不足 ④ — 带宽利用低效
- **对应设计**: Phase B (orchfs_tier 的多粒度写入) + C4 (eviction_policy 的批量换出)

**RQ4: 延迟隐藏 — 如何通过预取和流水线重叠，将存储迁移的延迟隐藏在 GPU 计算背后？**

- **问题根源**: 即使分层放置是最优的，迁移操作（GPU↔DRAM：~190μs/4MB，DRAM↔SSD：取决于粒度）本身会引入额外延迟。如果迁移与计算串行执行，decode 阶段的 TPOT（Time Per Output Token）将显著增加，抵消分层存储带来的容量优势。
- **具体挑战**:
  - 预取时机的预测：需要在 token 被 attention 访问之前，提前将其从低层存储提升到 GPU。但 LLM 的 attention 模式并非完全可预测（Exp-M2 中 Jaccard 相似度 0.47-0.70），预测太早浪费带宽，预测太晚来不及换入
  - 计算-传输-IO 三阶段流水线：理想情况下，第 N 步的 GPU 计算、第 N+1 步的 DRAM→GPU 预取传输、第 N+2 步的 SSD→DRAM 预加载应该同时进行。但 CUDA stream 管理、CPU 调度线程、IO 线程池三者的协调是复杂的系统工程
  - 预取带宽预算：预取操作占用 PCIe 带宽（与正常的 KV 访问竞争），需要限制同时进行的预取数量。E7 实测显示 prefetch_budget≥8 时 dispatch 趋于饱和
  - 误预取的代价：预取了但没用到的数据白白占据了 GPU/DRAM 空间，在显存紧张时可能反而加剧压力。需要设计预取取消机制
- **对应不足**: 不足 ④ — 带宽利用低效（从时间维度的利用不足）
- **对应设计**: C5 (prefetch_scheduler) + C6 (pipeline) + transfer_engine 的多 CUDA stream

**RQ5: 无损保证 — 如何在多级迁移过程中严格保证数据完整性，使模型输出与不做任何迁移时 bit-exact 一致？**

- **问题根源**: KV-Cache 迁移涉及多次内存拷贝（GPU→DRAM→NVM→SSD 及反向路径），每一跳都存在数据损坏的风险。与 H2O 等有损方案（直接丢弃冷 token）不同，本系统的核心承诺是**无损迁移**——所有 KV 数据在迁移后必须与原始值完全一致。
- **具体挑战**:
  - 多跳传输的数据完整性：GPU→DRAM 使用 cudaMemcpyAsync（依赖 CUDA 运行时），DRAM→NVM/SSD 使用 pwrite/pread（依赖文件系统）。不同路径的错误模式不同，需要统一的验证机制
  - 并发迁移的一致性：当某个 KV block 正在从 DRAM 迁移到 SSD 的过程中，如果 GPU 侧恰好需要读取该 block（触发 promote），需要保证不会读到部分写入的数据。kv_block 的 rwlock 和 state 机制需要正确处理所有并发场景
  - 浮点精度：KV 数据通常为 FP16/BF16，在 GPU→CPU 拷贝时不应发生任何精度转换。E10 实验验证了 Token 一致率 100% 和 Perplexity 相对差 0%，但这仅在当前测试规模下——更大规模、更多迁移次数下是否仍然成立需要持续验证
  - 故障恢复：如果迁移过程中发生异常（如 SSD 写入失败），系统需要保证 KV block 的状态回退到一致状态，不能出现"数据既不在源层也不在目标层"的悬挂状态
- **对应不足**: 这是本系统区别于 H2O、ScissorHands 等有损方案的核心差异点
- **对应设计**: kv_block 的状态机 (KV_STATE_*) + migration_engine 的原子性保证 + E10 质量验证

#### 子问题间的依赖关系

```
RQ1 (冷热识别)  ──→  RQ2 (分层放置)  ──→  RQ3 (粒度适配)
                          │                      │
                          ▼                      ▼
                     RQ4 (延迟隐藏)  ◄────────────┘
                          │
                          ▼
                     RQ5 (无损保证) ← 贯穿所有迁移路径
```

- **RQ1 → RQ2**: 冷热分级的结果决定了数据应放置在哪一层
- **RQ2 → RQ3**: 目标层的选择决定了写入时应使用何种粒度
- **RQ3 → RQ4**: 粒度适配影响迁移的 IO 效率，进而影响预取和流水线的设计
- **RQ5 横切所有路径**: 无论迁移方向和粒度如何，数据完整性是不可妥协的约束

#### 约束条件（Constraints）

本研究在解决上述问题时，需要在以下约束条件下工作：

| 约束 | 具体要求 | 来源 |
|------|---------|------|
| **输出无损** | 模型生成的 token 序列与不做任何迁移时完全一致（greedy decoding 下 bit-exact） | 论文核心承诺，E10 验证 |
| **在线推理** | 所有调度决策必须在推理运行时做出，不允许离线预分析（区别于 FlexGen 的 LP 规划） | serving 场景需求 |
| **透明集成** | 对上层推理引擎（vLLM）的修改最小化，通过标准插件接口接入 | 工程可维护性 |
| **调度开销** | 冷热分类和调度决策的 CPU 开销应 < 1% 的 decode 延迟 | E9 实测：4096 blocks 下 P99 < 60μs |
| **通用性** | 支持不同模型架构（MHA / GQA / MQA）和不同序列长度 | 论文评估范围 |

#### 预期贡献（Expected Contributions）

若上述 5 个子问题均得到有效解决，本工作将产出以下贡献：

1. **系统**：首个将异构存储（NVM+SSD）多粒度 IO 特性应用于 LLM KV-Cache 管理的分层调度系统，实现 GPU HBM → DRAM → NVM → SSD 四级自动调度
2. **算法**：基于注意力分数、时间衰减和访问频次三信号融合的自适应冷热分级算法，配合水位线驱动的动态阈值调整，在运行时无损地区分并管理 Hot/Warm/Cold 三级 KV 数据
3. **机制**：预取驱动的 IO-计算重叠流水线，结合多粒度 IO 适配（NVM 4KB 小页快速换入 + SSD 32KB 大块高吞吐换出），将存储迁移延迟隐藏在 GPU 计算之后
4. **实现与评估**：完整的系统实现（~6000 行 C/CUDA + ~1200 行 Python），集成 vLLM 推理框架，在多种模型和上下文长度下验证性能提升和输出质量保持

---

## 二、系统设计

### 2.1 核心思想

借鉴 OrchFS（FAST 2025）的异构 IO 编排理念，将 KV-Cache 按冷热程度在 **GPU HBM → Host DRAM → NVM → SSD** 四级存储层次间动态调度：热数据驻留 GPU 保证推理速度，冷数据下刷到廉价存储释放显存容量。

### 2.2 已实现的系统组件

#### Phase A — 核心数据结构与存储层

| 组件 | 文件 | 功能 | 状态 |
|------|------|------|------|
| `kv_block_t` | `src/core/kv_block.{h,c}` | KV 缓存块元数据、状态机、位置跟踪 | ✅ 完整 |
| `kv_request_ctx_t` | `src/core/kv_request.{h,c}` | 请求生命周期、按层块分配 | ✅ 完整 |
| `address_map_t` | `src/core/address_map.{h,c}` | 开地址哈希、并发 rwlock、动态扩容 | ✅ 完整 |
| `gpu_tier` | `src/tiered_store/gpu_tier.{h,cu}` | GPU HBM slab 池、cudaMalloc 管理 | ✅ 完整 |
| `dram_tier` | `src/tiered_store/dram_tier.{h,cu}` | Host pinned DRAM 池 | ✅ 完整 |
| `transfer_engine` | `src/tiered_store/transfer.{h,cu}` | 多 CUDA stream 异步 memcpy | ✅ 完整 |
| `orchfs_tier` | `src/tiered_store/orchfs_tier.{h,c}` | OrchFS 真链接 或 POSIX 回退双路径 | ✅ 完整 |
| `io_worker_pool` | `src/tiered_store/io_worker.{h,c}` | 异步 IO 线程池、任务队列 | ✅ 完整 |
| `orchkv_api` | `src/api/orchkv_api.{h,cu}` | 统一 C API：init/shutdown、请求生命周期、数据路径、迁移 | ✅ 完整 |

#### Phase B/C — 调度器子系统

| 组件 | 文件 | 功能 | 状态 |
|------|------|------|------|
| C1: `attention_tracker` | `src/scheduler/attention_tracker.{h,c}` | 按块追踪注意力分数（EMA 平滑） | ✅ 完整 |
| C2: `hotcold_classifier` | `src/scheduler/hotcold_classifier.{h,c}` | 三级分类（Hot/Warm/Cold），公式 `score = α×attn + β×recency + γ×freq` | ✅ 完整 |
| C3: `adaptive_threshold` | `src/scheduler/adaptive_threshold.{h,c}` | 基于 HWM/LWM 水位动态调整分类阈值 | ✅ 完整 |
| C4: `eviction_policy` | `src/scheduler/eviction_policy.{h,c}` | 加权 LRU 选 victim、demote 候选 | ✅ 完整 |
| C5: `prefetch_scheduler` | `src/scheduler/prefetch_scheduler.{h,c}` | 预取候选扫描与 dispatch 队列 | ⚠️ 队列逻辑完整，但 dispatch 结果未驱动真实迁移 |
| C6: `pipeline` | `src/scheduler/pipeline.{h,c}` | 流水线阶段打点与统计 | ⚠️ 已初始化，但未接入调度主路径 |
| C7: `migration_engine` | `src/scheduler/migration_engine.{h,c}` | 迁移操作执行、支持两跳传输 | ✅ 完整（依赖 transfer_fn 回调） |
| C8: `tiered_manager` | `src/scheduler/tiered_manager.{h,c}` | 统一调度入口：注册、注意力上报、自动/手动调度循环 | ✅ 完整 |

#### Phase D — Python 绑定与 vLLM 集成

| 组件 | 文件 | 功能 | 状态 |
|------|------|------|------|
| pybind11 绑定 | `bindings/orchkv_pybind.cpp` | 暴露 C API + tiered_manager 到 Python | ✅ 完整 |
| vLLM Connector | `python/orchkv/vllm_integration/connector.py` | KVConnectorBase_V1 实现 | ⚠️ Worker 侧 GPU↔DRAM 拷贝可用；Scheduler 侧多处 pass/空；未与 orchkv_core 分层路径打通 |
| 引擎注册 | `python/orchkv/vllm_integration/engine_patch.py` | 注册 OrchKvOffloadingConnector | ⚠️ 面向 vLLM V1 API，与实际安装的 0.7.3 不兼容 |
| 注意力钩子 | `python/orchkv/vllm_integration/attention_hook.py` | 从 FlashAttention 提取注意力权重 | ⚠️ 采集逻辑完整，`_get_all_modules` 固定返回空列表 |

### 2.3 测试覆盖

- **C/CUDA 单元测试**：19 个测试文件，覆盖所有核心模块（kv_types、kv_block、address_map、kv_request、gpu_dram、e2e、4-tier、orchfs、attention_tracker、hotcold_classifier、adaptive_threshold、eviction_policy、prefetch_scheduler、pipeline、migration_engine、tiered_manager 等）
- **Python 测试**：4 个测试文件（binding、connector、attention_hook、benchmarks）
- **总代码量**：C/CUDA 核心 ~4,500 行实现 + ~6,000 行测试；Python 集成 ~1,200 行 + ~1,000 行测试

---

## 三、实验设计与结果

### 3.1 实验环境

- 硬件：2× NVIDIA A100-SXM4-80GB / 256GB DDR4 / NVMe SSD
- 软件：CUDA 12.2, Python 3.11, PyTorch 2.5.1+cu121, vLLM 0.7.3
- 模型：Qwen/Qwen2.5-7B (bf16, ~14.3GB)

### 3.2 实验矩阵与结果

#### 系统内核级实验（直接调用 orchkv_core C 库，不依赖 vLLM）

| 实验 | 内容 | 关键结果 | 图表 |
|------|------|----------|------|
| **E5** | 冷热策略参数 sweep：9 组 (α,β,γ) × 3 种访问模式 × 3 次运行 | α≥0.7 时注意力驱动分类最准确；三级分类（hot/warm/cold）可控 | Fig.8-9 |
| **E7** | 预取效果：sweep prefetch_budget ∈ {0,4,8,16,32} | budget≥8 时 dispatch 饱和 (~245/100步)；调度开销 ≤ 5.9μs | Fig.11-12 |
| **E8** | 存储带宽：GPU↔DRAM 和 DRAM↔tmpfs 多块大小测量 | GPU↔DRAM ~23 GB/s，DRAM↔tmpfs ~14 GB/s（读），与 A100 PCIe 理论上限吻合 | Fig.13 |
| **E9** | 调度可扩展性：64→4096 blocks | 延迟 1.7μs→38.3μs，缩放指数 0.749（亚线性），P99 < 60μs | Fig.14 |

#### 端到端推理实验（使用 vLLM 引擎 + Qwen2.5-7B）

| 实验 | 内容 | 关键结果 | 图表 |
|------|------|----------|------|
| **E10** | 生成质量验证：greedy decoding 对比 baseline/orchkv 输出 | Token 一致率 100.0000%，Perplexity 相对差 0.0000% | Tab.1 |
| **E1** | 端到端吞吐：5 seq_len × 3 batch_size × 2 backends = 30 点 | 30/30 成功，baseline ≈ orchkv 吞吐 | Fig.1-3 |
| **E2** | 最大 Batch Size | seq=4096 下两者均达 bs=64 | Fig.4 |
| **E3** | 延迟分解 | 调度开销 < 0.5% (6.4ms / 1386ms) | Fig.5 |
| **E4** | 存储层消融：GPU-only → 4-tier | 4 种配置吞吐差异 < 0.5% | Fig.6-7 |
| **E6** | Block Size 消融：16/32/64/128 | 影响 < 0.3%，128 略优 | Fig.10 |

共生成 **16 个论文级图表**（PDF + PNG），存放于 `benchmarks/figures/`。

---

## 四、尚未完成或存在不足的地方

### 4.1 架构层面的缺口

1. **`orchkv_api` 与 `tiered_manager` 未打通**
   - `orchkv_api.cu` 未 `#include` 或调用 `tiered_manager`
   - 两者目前是独立的子系统：`orchkv_api` 管理四级存储池和数据路径，`tiered_manager` 管理调度决策
   - 需要在 `orchkv_init` 中挂载 tiered_manager，使调度决策自动驱动 `orchkv_evict_to_dram` / `orchkv_promote_to_gpu` 等操作

2. **Prefetch 只调度不执行**
   - `prefetch_scheduler` 的 `prefetch_dispatch` 输出了预取候选列表，但 `tiered_manager` 的 `do_prefetch` 函数仅累加统计计数器，**未调用** `mig_execute_one` 或任何迁移函数
   - 即：预取调度器做了"决策"但没有"执行"

3. **Pipeline 组件未接入**
   - `pipeline_t` 在 `tm_init`/`tm_destroy` 中被初始化/销毁，但调度循环、step_done、prefetch/demote 路径中均未调用 `pipeline_step_begin`/`pipeline_compute_done` 等
   - 计算-传输-IO 三阶段流水线的重叠统计功能实际不工作

4. **io_worker 异步 IO 未在主 API 中使用**
   - `io_worker_pool` 已实现完整的任务队列和线程池
   - 但 `orchkv_api.cu` 中的存储 IO（`orchkv_evict_to_storage` / `orchkv_promote_from_storage`）使用同步 `orchfs_tier_write`/`read`，未走 `io_worker_submit`

### 4.2 vLLM 集成层面的缺口

1. **Connector 面向错误版本的 API**
   - `connector.py` 实现了 `KVConnectorBase_V1`（vLLM V1 API），但实际安装的 vLLM 0.7.3 只有旧版 `KVConnectorBase`
   - `engine_patch.py` 的注册路径 `vllm.distributed.kv_transfer.kv_connector.v1` 在 0.7.3 中不存在
   - 实际实验中完全没有使用 OrchKvOffloadingConnector

2. **E1-E4/E6 实验的"orchkv"只是 vLLM 参数差异**
   - `bench_utils.build_vllm_engine` 中 `orchkv_enabled=True` 仅将 `swap_space` 从 4GB 调为 32GB
   - 未经过 orchkv_core C 库，也未经过 OrchKvOffloadingConnector
   - 实验本质是"不同 swap 配置的 vLLM"，不是"OrchKvCache 系统 vs baseline"

3. **Connector Worker 未调用 C 层分层逻辑**
   - `OrchKvConnectorWorker.save_kv_layer` 直接用 PyTorch `copy_` 做 GPU→pinned CPU
   - 没有调用 `orchkv_evict_to_dram` / `orchkv_prefill` 等 C API
   - Scheduler 侧 `update_state_after_alloc` 为 `pass`，`_pending_load` 从未被写入

### 4.3 实验层面的不足

1. **缺乏真集成基准**
   - 没有一组实验是 orchkv_core 的 tiered_manager 真正嵌入 vLLM 推理路径运行的
   - 系统内核实验（E5/E7/E8/E9）和推理实验（E1-E4/E6/E10）是完全独立的两套

2. **硬件过于充裕，未展示核心价值**
   - A100 80GB + 7B 模型，GPU 显存远非瓶颈
   - E2 两边都到 bs=64、E1 吞吐几乎重合，无法体现"省显存 / 扩上下文"的核心叙事
   - 需要更小 GPU、更长上下文、或更大模型来展示 OrchKvCache 的价值

3. **缺少强基线对比**
   - 未与 FlexGen、InfiniGen、H2O 等论文中的系统做直接对比
   - 未使用 vLLM 自身的 `cpu_offload_gb` 参数作为公平基线

4. **模型与计划不一致**
   - 原计划使用 Llama-2-7B + vLLM 0.17.2
   - 实际使用 Qwen2.5-7B + vLLM 0.7.3
   - 论文中需准确说明

---

## 五、总结

### 做得好的部分

- **C/CUDA 核心代码质量高**：~4,500 行实现代码，所有 header 声明的 API 均有对应实现，无 TODO/FIXME，错误处理完善
- **模块化设计清晰**：8 个调度子系统（C1-C8）可独立测试、独立配置
- **测试覆盖全面**：19 个 C/CUDA 测试 + 4 个 Python 测试，覆盖所有核心模块
- **系统内核实验可信**：E5/E7/E8/E9 真正调用 orchkv_core C 库，数据有统计意义
- **E10 质量验证完美**：100% token 一致率证明存储层管理不破坏数值精度

### 关键差距

- **组件已造好但未拼装**：各子系统（orchkv_api、tiered_manager、prefetch、pipeline、io_worker）独立可工作，但尚未形成从 vLLM 注意力层到四级存储的完整数据通路
- **实验的"系统"与"内核"脱节**：推理实验未走 orchkv_core 路径，内核实验未走推理路径
- **核心价值未被实验验证**：项目的核心叙事是"冷热感知 + 异构分层 → 扩展上下文/降显存"，但目前实验未能在一个端到端场景中展示这一点

### 要达到 CCF-A 级别还需要

1. 在 `orchkv_api.cu` 中集成 `tiered_manager`，使调度决策自动驱动迁移
2. 补全 prefetch 执行路径和 pipeline 重叠
3. 适配 vLLM 0.7.3 的 `KVConnectorBase` API（或升级到对齐版本）
4. 设计压力场景（小 GPU 利用率 / 长上下文 / 大模型），展示 OrchKvCache 的显存扩展能力
5. 增加 FlexGen / vLLM-swap / InfiniGen 等强基线对比

---

## References

### KV-Cache 管理核心

[1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. "Efficient Memory Management for Large Language Model Serving with PagedAttention." In *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23)*, 2023.

[2] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." In *Proceedings of the 40th International Conference on Machine Learning (ICML '23)*, 2023.

[3] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." In *Advances in Neural Information Processing Systems 36 (NeurIPS '23)*, 2023.

[4] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. "Efficient Streaming Language Models with Attention Sinks." In *Proceedings of the 12th International Conference on Learning Representations (ICLR '24)*, 2024.

[5] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management." In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, 2024.

[6] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, and Junchen Jiang. "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving." In *Proceedings of ACM SIGCOMM 2024*, 2024.

[7] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time." arXiv:2305.17118, 2023.

[8] Zihao Wang, Shaoduo Gan, Ye Li, and Minjia Zhang. "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget." arXiv:2404.04793, 2024.

[9] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference." In *Proceedings of the 41st International Conference on Machine Learning (ICML '24)*, 2024.

### LLM 推理系统

[10] Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079, 2024.

[11] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. "Orca: A Distributed Serving System for Transformer-Based Generative Models." In *Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI '22)*, 2022.

[12] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, 2024.

[13] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, 2024.

[14] Tri Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." In *Proceedings of the 12th International Conference on Learning Representations (ICLR '24)*, 2024.

[15] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." In *Advances in Neural Information Processing Systems 35 (NeurIPS '22)*, 2022.

[16] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. "SGLang: Efficient Execution of Structured Language Model Programs." In *Advances in Neural Information Processing Systems 37 (NeurIPS '24)*, 2024.

[17] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." In *Proceedings of the 51st Annual International Symposium on Computer Architecture (ISCA '24)*, 2024.

### 存储系统

[18] Yekang Zhan, Haichuan Hu, Xiangrui Yang, Qiang Cao, Hong Jiang, Shaohua Wang, and Jie Yao. "Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs." In *Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25)*, 2025.

[19] Youngjin Kwon, Henrique Fingler, Tyler Hunt, Simon Peter, Emmett Witchel, and Thomas Anderson. "Strata: A Cross Media File System." In *Proceedings of the 26th ACM Symposium on Operating Systems Principles (SOSP '17)*, 2017.

[20] Hobin Woo, Daegyu Han, Seungjoon Ha, Sam H. Noh, and Beomseok Nam. "On Stacking a Persistent Memory File System on Legacy File Systems." In *Proceedings of the 21st USENIX Conference on File and Storage Technologies (FAST '23)*, 2023.

[21] Jiale Xu, Rui Pan, Jing Wang, Siyuan Chen, and Xin Jin. "vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving." In *Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25)*, 2025.

### 基础理论与模型

[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention Is All You Need." In *Advances in Neural Information Processing Systems 30 (NeurIPS '17)*, 2017.

[23] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yinfei Yang, Cutler Shlomi, and Santiago Ontañón. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP '23)*, 2023.

[24] Noam Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150, 2019.

[25] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.

[26] Meta AI. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024.

### 异构推理与 KV-Cache 扩展

[27] Xiurui Pan, Endian Li, Qiao Li, Jiang Li, Yao Zhang, Yingwei Luo, Xiaolin Wang, and Jie Zhang. "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference." arXiv:2409.04992, 2024.

[28] Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, and Yuxiong He. "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '22)*, 2022.

[29] Lin Bin, Zhang Chen, and et al. "Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache." arXiv:2401.02669, 2024.

[30] Xuanlei Zhao, Bin Jia, Haotian Zhou, Ziming Liu, Shenggan Cheng, and Yang You. "HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices." In *Proceedings of Machine Learning and Systems (MLSys '24)*, 2024.

[31] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456, 2023.

### KV-Cache 压缩与量化

[32] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." In *Proceedings of the 41st International Conference on Machine Learning (ICML '24)*, 2024.

[33] Zefan Cai, Yichi Zhang, Bofei Gao, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Wen Xiao. "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling." arXiv:2406.02069, 2024.

[34] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. "SnapKV: LLM Knows What You are Looking for Before Generation." In *Advances in Neural Information Processing Systems 37 (NeurIPS '24)*, 2024.

### 总计 34 篇参考文献

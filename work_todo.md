# OrchKvCache 工作计划：面向 CCF-A 会议投稿的全流程规划

---

## 〇、目标会议选择

### 推荐目标（按优先级排序）

| 会议 | CCF 等级 | 适配度 | 理由 |
|------|---------|--------|------|
| **OSDI** | A | ★★★★★ | 系统顶会，InfiniGen (OSDI'24) 同方向，审稿人理解 KV-Cache offloading |
| **SOSP** | A | ★★★★★ | 系统顶会，vLLM (SOSP'23) 同方向，对存储+ML 系统交叉工作友好 |
| **ATC** | A | ★★★★☆ | USENIX 系统会议，接受面更广，OrchFS 团队（FAST'25）审稿池重叠 |
| **EuroSys** | A | ★★★★☆ | 欧洲系统顶会，近年大量 ML 系统工作 |
| **ASPLOS** | A | ★★★★☆ | 体系结构+系统交叉，对异构存储+GPU 的工作非常友好 |
| **FAST** | A | ★★★☆☆ | 存储顶会，但 OrchFS 已在 FAST'25 发表，同团队再投需显著差异化 |

### 投稿时间线建议

请根据目标会议的最新 CFP 确认 deadline，以下为典型周期：
- **OSDI**：通常 5~6 月截稿，11~12 月开会
- **SOSP**：通常 4~5 月截稿，10~11 月开会（隔年举办，需确认 2026/2027）
- **ATC**：通常 1~2 月截稿，7 月开会
- **EuroSys**：通常 10~11 月截稿，次年 3~4 月开会
- **ASPLOS**：通常有多个 deadline cycle（春/夏/秋），需查最新 CFP

**建议策略**：先瞄准一个中期 deadline（如 ASPLOS 或 EuroSys），留足 4~5 个月开发+实验时间。如果结果优秀，升级投 OSDI/SOSP。

---

## 一、论文调研（第 1~2 周）

### 1.1 必读论文清单

#### 1.1.1 KV-Cache 管理核心论文（必精读）

| # | 论文 | 会议 | 核心贡献 | 与本工作的关系 |
|---|------|------|---------|---------------|
| 1 | **PagedAttention / vLLM** | SOSP'23 | 分页管理 KV-Cache，解决显存碎片 | 基线系统；我们在其上扩展多级存储 |
| 2 | **FlexGen** | ICML'23 | CPU/GPU/Disk 三级 offloading，线性规划调度 | 最直接竞品；粒度固定、无 NVM、无热度感知 |
| 3 | **H2O** | NeurIPS'23 | Heavy-Hitter Oracle，识别关键 token 保留 KV | 冷热分级的理论依据；证明注意力分数的幂律分布 |
| 4 | **InfiniGen** | OSDI'24 | 基于预取的 KV-Cache offloading，GPU-CPU 协同 | 核心竞品；预取思路类似但无异构存储 |
| 5 | **CacheGen** | SIGCOMM'24 | KV-Cache 压缩+流式传输 | 压缩方向的补充工作 |
| 6 | **Attention Sink** | ICLR'24 | 发现初始 token 的 attention sink 现象 | 冷热分级中永久 Hot token 的理论基础 |
| 7 | **DistServe** | OSDI'24 | Prefill/Decode 分离部署 | 系统架构参考，理解 serving 场景需求 |
| 8 | **Mooncake** | ATC'25 | KVCache-centric 分布式 serving | 了解分布式场景下的 KV-Cache 管理 |

- [ ] 精读上述 8 篇论文，重点提取：
  - 各系统的 KV-Cache 管理粒度和数据布局
  - offloading/换入换出的延迟和带宽数据
  - 冷热判定的具体标准和算法
  - 实验中使用的 benchmark 和评估指标

#### 1.1.2 存储系统相关论文

| # | 论文 | 会议 | 关注点 |
|---|------|------|--------|
| 9 | **OrchFS** | FAST'25 | 核心依赖，需深入理解所有细节 |
| 10 | **Strata** | SOSP'17 | NVM+SSD 分层文件系统，对比参考 |
| 11 | **SPFS** | ATC'23 | SSD+PM 混合文件系统 |
| 12 | **CXL-SSD / TPP** | ASPLOS'23 | CXL 内存扩展，异构内存管理 |
| 13 | **Pond** | ASPLOS'23 | CXL 内存池化 |

- [ ] 精读存储系统论文，提取异构存储管理的关键技术

#### 1.1.3 LLM 推理优化相关论文

| # | 论文 | 会议 | 关注点 |
|---|------|------|--------|
| 14 | **FlashAttention / FlashAttention-2** | NeurIPS'22, ICLR'24 | GPU 高效 attention 实现 |
| 15 | **Orca** | OSDI'22 | 连续批处理 |
| 16 | **Sarathi-Serve** | OSDI'24 | 分块 prefill |
| 17 | **SpecInfer / SpecDecode** | 各 | 投机推理对 KV-Cache 的影响 |
| 18 | **vTensor** | FAST'25 | 面向 DNN 的虚拟张量管理 |

- [ ] 泛读推理优化论文，建立完整的 related work 知识体系

#### 1.1.4 最新动态追踪（2025~2026）

- [ ] 搜索 arXiv 关键词：`KV-Cache offloading`, `KV-Cache tiered storage`, `LLM inference NVM`, `heterogeneous memory LLM`
- [ ] 关注 OSDI'25, SOSP'25, ATC'25, EuroSys'26 已录用论文列表中相关工作
- [ ] 确认无高度重叠的并发工作（如有，需差异化 or 加速投稿）

### 1.2 调研产出

- [ ] 撰写 `docs/related_work.md`，按分类整理所有论文
- [ ] 制作对比表格：各系统在"存储层次数 × 管理粒度 × 冷热策略 × 预取机制"四个维度的对比
- [ ] 明确本工作的差异化定位（Positioning Statement）

---

## 二、问题设定与 Motivation（第 2~3 周）

### 2.1 研究问题定义（Research Question）

**核心问题**：

> 如何利用异构存储设备（NVM + SSD）的多粒度 IO 特性，为 LLM 推理中的 KV-Cache 构建一个高效的分层调度系统，在不显著增加推理延迟的前提下大幅扩展可服务的上下文长度和并发请求数？

**子问题拆解**：

1. **Q1 (冷热感知)**：如何高效判定 KV-Cache 中每个 token 块的冷热程度？注意力分数的稀疏性和动态性如何被利用？
2. **Q2 (多粒度匹配)**：如何将 KV-Cache 的冷热特征与异构存储的不同粒度（4KB NVM Page vs 32KB SSD Block）最优匹配？
3. **Q3 (延迟隐藏)**：如何通过预取和流水线技术，使存储层迁移的延迟不成为推理的关键路径？
4. **Q4 (系统效率)**：相比现有 offloading 方案，异构存储编排能在存储带宽利用率上带来多大提升？

### 2.2 Motivation 实验设计（关键！用于论文 §2）

必须用 **真实数据** 证明问题的存在和严重性。

#### Exp-M1: KV-Cache 显存瓶颈量化

- [ ] 测量不同模型、不同序列长度下 KV-Cache 的显存占用
- [ ] 绘制 "序列长度 vs KV-Cache 大小 vs GPU显存容量" 对比图
- [ ] 量化显存限制对 batch size 的约束

```
预期结论：128K tokens 下，单个请求的 KV-Cache 即可占满一块 A100 80GB，
         batch size 被限制为 1，GPU 计算利用率极低。
```

#### Exp-M2: KV-Cache 注意力分数分布分析

- [ ] 在 LLaMA-7B/13B 上运行长上下文 benchmark（如 LongBench）
- [ ] 收集每层每头的注意力分数矩阵
- [ ] 统计分析：
  - 各 token 位置的平均注意力分数分布（证明幂律特征）
  - Top-K% token 占总注意力的比例（如 Top-10% 占 90%+）
  - 不同层/头的注意力模式差异
- [ ] 绘制热力图、CDF 图

```
预期结论：大部分 KV-Cache 的注意力分数非常低，存在明显的冷热分化。
         仅保留 Top-20% 的 KV-Cache 在 GPU 即可保证 >95% 的注意力覆盖。
```

#### Exp-M3: 现有 offloading 方案的存储带宽利用率分析

- [ ] Profile FlexGen / vLLM 的 offloading IO 模式
- [ ] 测量实际 SSD 写带宽 vs 理论峰值带宽
- [ ] 分析原因：粒度不对齐、同步 IO、无并行

```
预期结论：现有方案仅利用了 SSD 峰值带宽的 30~50%，
         主因是未按 SSD 页对齐、固定粒度管理、缺乏 IO 并行。
         OrchFS 的异构 IO 编排可将利用率提升至 85%+。
```

#### Exp-M4: NVM 作为中间层的价值量化

- [ ] 对比 GPU→DRAM→SSD (2 级) vs GPU→DRAM→NVM→SSD (3 级) 的换入延迟
- [ ] 测量 NVM 直接读取 4KB 页的延迟 vs SSD 读取 32KB 块的延迟
- [ ] 量化"温数据"命中 NVM 带来的延迟节省

```
预期结论：NVM 作为中间层可将温数据的换入延迟从 ~10μs (SSD) 降低至 ~300ns (NVM)，
         减少 97% 的换入延迟，有效隐藏 IO 开销。
```

### 2.3 Motivation 产出

- [ ] 4 组 motivation 实验的代码和数据
- [ ] 论文 §2 (Background & Motivation) 的初稿和全部配图
- [ ] 明确的 **insight 列表**（3~4 个 key insights，每个对应一个设计决策）

### 2.4 Insight → Design 映射（论文核心叙事线）

| Insight | 对应的 Motivation 实验 | 设计决策 |
|---------|----------------------|---------|
| **I1**: KV-Cache 存在显著冷热分化，少量 token 贡献绝大部分注意力 | Exp-M2 | 基于注意力分数的三级冷热分类 |
| **I2**: 冷/温数据的粒度需求不同——温数据需快速换入（小粒度），冷数据需高吞吐换出（大粒度） | Exp-M3, M4 | 多粒度管理：4KB NVM Page + 32KB SSD Block |
| **I3**: NVM 可作为 GPU 和 SSD 之间的高速缓冲，大幅降低温数据换入延迟 | Exp-M4 | GPU HBM → DRAM → NVM → SSD 四级存储架构 |
| **I4**: 现有 offloading 未充分利用存储带宽，对齐和并行是关键 | Exp-M3 | 借鉴 OrchFS 的对齐写分区和并行 IO 引擎 |

---

## 三、系统设计与实现（第 3~9 周）

### 3.1 整体设计确认（第 3 周）

- [ ] 细化 README.md 中的架构图，画正式的系统架构图（用于论文 Figure 1）
- [ ] 确定四级存储层的管理策略和参数范围
- [ ] 确定 KV Block 的粒度映射方案（考虑不同模型的 d_head 和 dtype）
- [ ] 确定冷热分级的具体算法和参数（α, β, γ 的取值范围）
- [ ] 确定与 OrchFS 的集成接口和修改范围
- [ ] 撰写论文 §3 (Design Overview) 的初稿

### 3.2 核心模块实现

#### Phase A: 基础框架（第 4~5 周）

- [ ] **A1**: 搭建 CMake 构建系统，配置 OrchFS 链接
- [ ] **A2**: 实现 `kv_types.h` — 公共类型定义
- [ ] **A3**: 实现 `kv_block.h/c` — KV Block 元数据结构和基本操作
- [ ] **A4**: 实现 `kv_request.h/c` — 请求级 KV-Cache 上下文管理
- [ ] **A5**: 实现 `address_map.h/c` — 统一地址映射
- [ ] **A6**: 实现 `gpu_tier.h/c` — GPU HBM 层（CUDA malloc/free/memcpy async）
- [ ] **A7**: 实现 `dram_tier.h/c` — Host DRAM 层（mmap/munmap 或 malloc）
- [ ] **A8**: 基本的 C API (`orchkv_init`, `orchkv_request_create`, `orchkv_prefill`)
- [ ] **A9**: 单元测试：KV Block 创建/销毁、地址映射、GPU↔DRAM 传输

#### Phase B: OrchFS 后端（第 5~7 周）

- [ ] **B1**: 实现 `orchfs_tier.h/c` — OrchFS 后端适配层
  - 封装 OrchFS 的文件操作 (open/write/read/close via `/Or` 路径)
  - 支持 4KB (NVM) 和 32KB (SSD) 两种粒度的写入/读取
- [ ] **B2**: 实现 KV Block → OrchFS 偏移量的映射逻辑
  - 每个请求一个 OrchFS 文件
  - 文件内按 `[layer][head][block_idx]` 三维寻址
- [ ] **B3**: 实现 DRAM → NVM 换出路径
  - 小粒度写入（4KB page），走 NVM IO 线程池
- [ ] **B4**: 实现 DRAM → SSD 换出路径
  - 大粒度写入（32KB block），走 SSD IO 线程池，确保 SSD 页对齐
- [ ] **B5**: 实现 NVM/SSD → DRAM 换入路径
  - NVM 读取：低延迟直接读
  - SSD 读取：多线程并行读
- [ ] **B6**: 集成 OrchFS 的 NVM → SSD 迁移
  - 扩展 `migrate.c` 支持 KV-aware 的迁移决策
  - 迁移优先级：按热度分数而非纯 LRU
- [ ] **B7**: 实现 STRATA_NODE 混合布局支持
  - 同一 KV Block 内部分 slot 在 NVM、部分在 SSD
- [ ] **B8**: 集成测试：端到端 GPU → DRAM → NVM → SSD 数据流

#### Phase C: 冷热分级与调度（第 7~9 周）

- [ ] **C1**: 实现 `attention_tracker.h/c`
  - 异步采集注意力分数（从 GPU 拷贝到 CPU）
  - 按 block 粒度聚合注意力分数
  - 滑动窗口统计
- [ ] **C2**: 实现 `hotcold_classifier.h/c`
  - 热度计算模型（注意力分数 + 时间衰减 + 频次）
  - EMA 更新
  - 三级分类（Hot/Warm/Cold）
- [ ] **C3**: 实现 `adaptive_threshold.h/c`
  - 根据各存储层使用率动态调整阈值
  - 水位线机制（HWM/LWM）
- [ ] **C4**: 实现 `eviction_policy.h/c`
  - 换出策略：热度+LRU 混合
  - 批量换出优化
- [ ] **C5**: 实现 `prefetch_scheduler.h/c`
  - 基于注意力模式预测的预取
  - 预取优先级队列
  - 预取命中率统计
- [ ] **C6**: 实现 `pipeline.h/c`
  - IO-计算重叠流水线
  - CUDA Stream 管理
  - 异步迁移线程
- [ ] **C7**: 实现 `migration.h/c` — 跨层迁移引擎总控
  - 统一管理 Demote（换出）和 Promote（换入）
  - 与 OrchFS 迁移线程协调
- [ ] **C8**: 实现 `tiered_manager.h/c` — 分层存储管理器总控
  - 整合 classifier、migration、prefetch
  - 周期性调度循环
- [ ] **C9**: 冷热分级器的单元测试
- [ ] **C10**: 预取调度器的单元测试
- [ ] **C11**: 端到端集成测试（模拟 decode 循环）

#### Phase D: 推理引擎集成（第 9~10 周）

- [ ] **D1**: Python binding 实现（pybind11 或 ctypes）
- [ ] **D2**: vLLM 集成 — 自定义 BlockManager
  - 替换 vLLM 的 `BlockSpaceManager`，接管 KV-Cache 分配
- [ ] **D3**: vLLM 集成 — 自定义 Attention Backend
  - 在 attention 计算后 hook 注意力分数采集
  - 在 get_kv 时触发换入
- [ ] **D4**: 端到端推理测试
  - 验证生成质量（与原始 vLLM 输出对比，确保 bit-exact 或误差可控）
  - 验证功能正确性

### 3.3 设计文档产出

- [ ] 论文 §4 (Detailed Design) 初稿，包含：
  - §4.1 多粒度 KV Block 布局
  - §4.2 冷热分级算法
  - §4.3 分层存储管理与迁移
  - §4.4 预取与 IO-计算流水线
  - §4.5 与 OrchFS 的集成
- [ ] 设计图：数据流图、状态机图、时序图

---

## 四、实验设计与执行（第 10~14 周）

### 4.1 实验环境准备（第 10 周）

- [ ] 硬件环境确认与配置
  - GPU: NVIDIA A100/H100
  - NVM: Intel Optane PM (DAX mode)
  - SSD: 高带宽 NVMe SSD
  - DRAM: 足够大（≥128GB）
- [ ] 安装并验证 OrchFS 在目标机器上正常工作
- [ ] 安装并验证 vLLM 基线
- [ ] 安装并验证 FlexGen 基线
- [ ] 准备模型权重（LLaMA-2-7B, 13B, 70B；或更新的 LLaMA-3 系列）
- [ ] 准备 benchmark 数据集

### 4.2 Benchmark 详细规划

#### 4.2.1 推理吞吐/延迟评估 Benchmark

| Benchmark | 数据集 | 目的 | 具体配置 |
|-----------|--------|------|---------|
| **ShareGPT** | ShareGPT 真实对话 trace | 模拟真实 serving 场景 | 请求到达率扫描 (0.5~8 req/s) |
| **LongBench** | 长上下文理解任务集 | 评估长上下文场景 | 序列长度 4K/8K/16K/32K |
| **RULER** | 合成长上下文任务 | 精确控制序列长度 | 序列长度 32K/64K/128K |
| **Synthetic Fixed-Length** | 合成定长请求 | 隔离序列长度变量 | input_len=1K/4K/16K/64K, output_len=256 |
| **Multi-turn** | 合成多轮对话 | 评估 KV-Cache 累积场景 | 10 轮，每轮 2K~4K tokens |

#### 4.2.2 微基准测试（Microbenchmark）

| Benchmark | 测量目标 | 具体操作 |
|-----------|---------|---------|
| **Tier Latency** | 各层存储的读/写延迟 | 顺序/随机读写 4KB 和 32KB 块 |
| **Migration BW** | 层间迁移带宽 | GPU→DRAM, DRAM→NVM, DRAM→SSD, NVM→SSD 的迁移吞吐 |
| **Classifier Overhead** | 冷热分级的 CPU 开销 | 测量热度计算和分类决策的延迟 |
| **Prefetch Accuracy** | 预取命中率 | 不同预取策略下的命中率、误预取率 |
| **OrchFS IO** | OrchFS 存储路径性能 | 对比 OrchFS vs 裸 POSIX IO 的带宽 |

#### 4.2.3 模型覆盖

| 模型 | 参数量 | 层数 | 头数 | d_head | KV-Cache/token | 测试目的 |
|------|--------|------|------|--------|---------------|---------|
| LLaMA-2-7B | 7B | 32 | 32 | 128 | 1MB | 基础功能验证 |
| LLaMA-2-13B | 13B | 40 | 40 | 128 | 1.6MB | 中等规模 |
| LLaMA-2-70B | 70B | 80 | 64 | 128 | 5MB | 大模型压力测试 |
| Mistral-7B | 7B | 32 | 32 | 128 | GQA 验证 | GQA (Grouped Query Attention) |
| Yi-34B-200K | 34B | 60 | 56 | 128 | 超长上下文 | 200K context window |

### 4.3 实验矩阵详细设计

#### E1: 端到端推理吞吐对比（主实验，用于论文 Figure 6~7）

```
系统: {OrchKvCache, vLLM, FlexGen, InfiniGen, No-Offloading}
模型: {LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B}
序列长度: {4K, 8K, 16K, 32K, 64K, 128K}
Batch size: {1, 4, 8, 16, 32}
指标: Throughput (tokens/s), TTFT (ms), TPOT (ms)
```

- [ ] 运行全矩阵实验（5×3×6×5 = 450 组配置）
- [ ] 对于 No-Offloading，标记 OOM 的配置
- [ ] 绘制柱状图 + 折线图

#### E2: 最大可服务 Batch Size 对比（用于论文 Figure 8）

```
系统: {OrchKvCache, vLLM, FlexGen}
模型: {LLaMA-2-7B, LLaMA-2-70B}
序列长度: {16K, 32K, 64K, 128K}
约束: TPOT < 100ms (SLA)
指标: Max Batch Size under SLA
```

- [ ] 对每个配置二分搜索最大 batch size
- [ ] 绘制柱状图

#### E3: 延迟分解分析（用于论文 Figure 9）

```
系统: OrchKvCache
模型: LLaMA-2-13B
序列长度: 32K
分解: {GPU Compute, GPU↔DRAM Transfer, DRAM↔NVM Transfer,
       DRAM↔SSD Transfer, Classifier Overhead, Prefetch Overhead}
```

- [ ] 使用 CUDA Event + 高精度计时器分解各阶段延迟
- [ ] 绘制堆叠柱状图（Stacked Bar）

#### E4: 存储层消融实验（用于论文 Figure 10）

```
配置:
  (a) OrchKvCache-Full: GPU + DRAM + NVM + SSD （完整系统）
  (b) OrchKvCache-NoNVM: GPU + DRAM + SSD （去掉 NVM 层）
  (c) OrchKvCache-NVMOnly: GPU + DRAM + NVM （去掉 SSD 层）
  (d) OrchKvCache-2Tier: GPU + SSD （只有两层，类似 FlexGen）
模型: LLaMA-2-13B
序列长度: {16K, 32K, 64K}
```

- [ ] 对比四种配置的吞吐和延迟
- [ ] 量化 NVM 层的价值（预期：降低温数据换入延迟 10~30×）
- [ ] 量化 SSD 层的价值（预期：扩展冷数据容量 10~50×）

#### E5: 冷热分级策略对比（用于论文 Figure 11）

```
策略:
  (a) OrchKvCache-Attn: 基于注意力分数的分级（本文方法）
  (b) Pure-LRU: 纯 LRU 替换
  (c) Pure-Frequency: 纯频率替换
  (d) Random: 随机替换
  (e) Oracle: 离线最优（事后知道所有访问序列的理想策略）
模型: LLaMA-2-7B
序列长度: 32K
```

- [ ] 对比命中率、推理延迟、吞吐
- [ ] 分析注意力感知策略相比 LRU 的优势

#### E6: 多粒度 vs 单粒度管理（用于论文 Figure 12）

```
粒度配置:
  (a) Multi-Granularity: 4KB (NVM) + 32KB (SSD)（本文方法）
  (b) Fixed-4KB: 统一 4KB 管理
  (c) Fixed-32KB: 统一 32KB 管理
  (d) Fixed-64KB: 统一 64KB 管理
模型: LLaMA-2-7B
序列长度: {8K, 32K, 128K}
```

- [ ] 对比存储带宽利用率、换入换出延迟
- [ ] 分析多粒度在不同序列长度下的优势

#### E7: 预取效果评估（用于论文 Figure 13）

```
配置:
  (a) Prefetch-Attn: 基于注意力模式预取（本文方法）
  (b) Prefetch-Spatial: 基于空间局部性预取
  (c) No-Prefetch: 不预取，按需换入
模型: LLaMA-2-7B
序列长度: 32K
```

- [ ] 对比预取命中率、TPOT 延迟
- [ ] 分析预取的 IO 重叠效果

#### E8: 存储带宽利用率对比（用于论文 Figure 14）

```
系统: {OrchKvCache (via OrchFS), FlexGen (via POSIX), Direct-IO}
操作: 纯 KV-Cache 换出（控制变量）
块大小: {4KB, 32KB, 256KB}
```

- [ ] 测量 SSD 实际写带宽 vs 理论峰值
- [ ] 分析 OrchFS 的对齐写分区和并行 IO 带来的带宽提升

#### E9: 可扩展性实验（用于论文 Figure 15）

```
维度:
  (a) 序列长度扩展: {4K, 8K, 16K, 32K, 64K, 128K, 256K}
  (b) Batch size 扩展: {1, 2, 4, 8, 16, 32, 64}
  (c) IO 线程数扩展: NVM {1,2,4,8}, SSD {4,8,16,32,64}
模型: LLaMA-2-7B
```

- [ ] 绘制扩展性曲线
- [ ] 识别系统瓶颈（GPU 计算 or IO 带宽 or CPU 调度）

#### E10: 生成质量验证（用于论文 §5.x）

```
模型: LLaMA-2-7B
数据: LongBench 各子任务
对比: OrchKvCache 输出 vs vLLM 原始输出
指标: 精确匹配率、困惑度差异、任务准确率
```

- [ ] 确认 KV-Cache 迁移不影响生成质量（bit-exact 或误差 < 0.1%）
- [ ] 若使用量化压缩，测量精度损失

### 4.4 实验执行时间表

| 周次 | 实验 | 预计耗时 |
|------|------|---------|
| 第 10 周 | 环境搭建 + 基线系统部署 + 微基准 | 1 周 |
| 第 11 周 | E1 (端到端吞吐) 前半部分 + E10 (质量验证) | 1 周 |
| 第 12 周 | E1 后半部分 + E2 (Max Batch) + E3 (延迟分解) | 1 周 |
| 第 13 周 | E4 (消融) + E5 (冷热策略) + E6 (粒度) | 1 周 |
| 第 14 周 | E7 (预取) + E8 (带宽) + E9 (扩展性) + 补充实验 | 1 周 |

---

## 五、论文撰写（第 11~16 周，与实验并行）

### 5.1 论文结构

```
Title: OrchKvCache: Heterogeneous-IO Orchestrated KV-Cache Management
       for Efficient Long-Context LLM Inference
       (暂定，后续精炼)

Abstract (250 words)

§1 Introduction (1.5 pages)
   - 问题：长上下文 LLM 推理的 KV-Cache 显存瓶颈
   - 现有方案的不足：固定粒度、同构存储、低存储带宽利用率
   - 我们的方法：基于 OrchFS 的异构多粒度 KV-Cache 分层调度
   - 贡献列表（3~4 条）

§2 Background & Motivation (2 pages)
   - §2.1 KV-Cache in LLM Inference
   - §2.2 Heterogeneous Storage Landscape (NVM + SSD)
   - §2.3 Motivation Experiments (Exp-M1 ~ M4)
   - §2.4 Key Insights

§3 Design Overview (1 page)
   - 系统架构图
   - 核心组件概览
   - 设计原则

§4 Detailed Design (4~5 pages)
   - §4.1 Multi-Granularity KV Block Layout
   - §4.2 Attention-Aware Hot-Cold Classification
   - §4.3 Tiered Storage Management & Migration
   - §4.4 Prefetch-Driven IO-Compute Pipeline
   - §4.5 Integration with OrchFS

§5 Implementation (0.5~1 page)
   - 代码量、关键实现细节
   - 与 vLLM 的集成方式

§6 Evaluation (4~5 pages)
   - §6.1 Experimental Setup
   - §6.2 End-to-End Performance (E1, E2)
   - §6.3 Latency Breakdown (E3)
   - §6.4 Ablation Studies (E4, E5, E6, E7)
   - §6.5 Storage Bandwidth Utilization (E8)
   - §6.6 Scalability (E9)
   - §6.7 Generation Quality (E10)

§7 Related Work (1 page)
   - KV-Cache Management for LLM Inference
   - Heterogeneous Storage Systems
   - Memory Tiering and Offloading

§8 Conclusion (0.5 page)

References
```

### 5.2 撰写时间表

| 周次 | 撰写内容 | 备注 |
|------|---------|------|
| 第 11 周 | §1 Introduction 初稿 + §2 Motivation 初稿 | Motivation 实验数据需就绪 |
| 第 12 周 | §3 Design Overview + §4 Detailed Design 初稿 | 配合设计图绘制 |
| 第 13 周 | §5 Implementation + §7 Related Work 初稿 | |
| 第 14 周 | §6 Evaluation 初稿（边出数据边写） | 核心实验数据需就绪 |
| 第 15 周 | 全文统稿、补充实验、精炼图表 | 内部审阅 |
| 第 16 周 | Abstract + Introduction 精修、格式调整、投稿 | 导师/合作者审阅 |

### 5.3 图表清单

| 图/表编号 | 类型 | 内容 | 对应实验 |
|-----------|------|------|---------|
| Figure 1 | 架构图 | OrchKvCache 系统架构 | - |
| Figure 2 | 柱状图 | KV-Cache 显存占用 vs GPU 容量 | Exp-M1 |
| Figure 3 | 热力图 + CDF | 注意力分数分布 | Exp-M2 |
| Figure 4 | 柱状图 | 现有方案 SSD 带宽利用率 | Exp-M3 |
| Figure 5 | 柱状图 | NVM vs SSD 换入延迟对比 | Exp-M4 |
| Figure 6~7 | 折线图/柱状图 | 端到端推理吞吐对比 | E1 |
| Figure 8 | 柱状图 | 最大可服务 Batch Size | E2 |
| Figure 9 | 堆叠柱状图 | 延迟分解 | E3 |
| Figure 10 | 柱状图 | 存储层消融 | E4 |
| Figure 11 | 柱状图 | 冷热分级策略对比 | E5 |
| Figure 12 | 柱状图 | 多粒度 vs 单粒度 | E6 |
| Figure 13 | 柱状图 + 折线图 | 预取效果 | E7 |
| Figure 14 | 柱状图 | 存储带宽利用率 | E8 |
| Figure 15 | 折线图 | 可扩展性 | E9 |
| Table 1 | 表格 | 实验环境配置 | - |
| Table 2 | 表格 | 与相关工作的功能对比 | - |
| Table 3 | 表格 | 生成质量验证 | E10 |

---

## 六、结果分析框架（第 14~15 周）

### 6.1 需要回答的核心问题

对每组实验，需要在论文中明确回答：

| 实验 | 核心问题 | 预期结论 |
|------|---------|---------|
| E1 | OrchKvCache 的吞吐提升有多大？ | 长序列场景下相比 vLLM 提升 2~5×，相比 FlexGen 提升 3~8× |
| E2 | 能服务多少并发请求？ | 相同 SLA 下 batch size 扩大 3~10× |
| E3 | IO 开销占比多少？ | IO 开销占 TPOT 的 <15%（通过流水线和预取隐藏） |
| E4 | NVM 层的贡献有多大？ | 去掉 NVM 后温数据换入延迟增加 10~30× |
| E5 | 注意力感知分级是否优于 LRU？ | 命中率提升 15~30%，TPOT 降低 20~40% |
| E6 | 多粒度管理是否值得？ | 相比固定粒度，带宽利用率提升 30~60% |
| E7 | 预取能隐藏多少延迟？ | 预取命中率 >70%，TPOT 降低 30~50% |
| E8 | OrchFS 是否真正提升了存储带宽利用率？ | 相比 POSIX IO 利用率提升 40~80% |
| E9 | 系统是否能线性扩展？ | 序列长度 4K→128K 吞吐下降 <40%（vs 基线 >80%） |
| E10 | 是否保持生成质量？ | 精确匹配率 100%（无损）或精度损失 <0.1%（量化） |

### 6.2 数据分析方法

- [ ] 对每组实验，计算均值 ± 标准差（至少 3 次重复）
- [ ] 使用箱线图或 error bar 展示变异性
- [ ] 关键对比使用加速比（Speedup）或归一化性能
- [ ] 延迟分析使用百分位数（P50, P90, P99）

### 6.3 意外结果处理策略

| 可能的意外 | 应对方案 |
|-----------|---------|
| NVM 延迟高于预期 | 检查 DAX 配置、NUMA 亲和性、memcpy 实现 |
| 预取命中率低 | 调整预取窗口大小、尝试层级预取策略 |
| OrchFS IO 性能不佳 | Profile IO 路径、调整线程数、检查对齐 |
| 短序列无优势 | 正常，重新定义目标场景为中长序列 |
| 生成质量下降 | 排查迁移过程数据损坏、降低量化精度 |

---

## 七、论文核心贡献提炼（贯穿全程）

### 7.1 预期贡献声明（4 条）

1. **首个利用异构存储（NVM+SSD）多粒度特性进行 LLM KV-Cache 分层管理的系统**，实现 4KB NVM 页细粒度快速换入和 32KB SSD 块高吞吐批量换出的协同。

2. **基于注意力分数的自适应冷热分级算法**，结合时间衰减和频率统计，动态将 KV-Cache 分为 Hot/Warm/Cold 三级，比纯 LRU 策略提升 X% 命中率。

3. **预取驱动的 IO-计算重叠流水线**，利用 OrchFS 的并行 IO 引擎和 CUDA Stream，将存储迁移延迟隐藏在 GPU 计算之后，IO 开销占比 <15%。

4. **完整的系统实现和广泛评估**，集成 vLLM 推理框架，在多种模型和序列长度下证明 OrchKvCache 相比 SOTA 系统在吞吐、延迟和并发能力上的显著提升。

### 7.2 Novelty 自检清单

- [ ] 区别于 FlexGen：不是简单三级 offloading，而是利用 NVM+SSD 异构 IO 特性做多粒度管理
- [ ] 区别于 InfiniGen：不仅做预取，更做基于存储特性的分级放置
- [ ] 区别于 H2O：H2O 做 token 剪枝（有损），我们做无损迁移 + 可选有损压缩
- [ ] 区别于 CacheGen：CacheGen 侧重压缩和网络传输，我们侧重本地异构存储调度
- [ ] OrchFS 的独特价值：不是任意文件系统都能替代，OrchFS 的对齐写分区、STRATA_NODE、并行 IO 是关键使能技术

---

## 八、风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|------|------|------|---------|
| 无 NVM 硬件（Intel Optane 停产） | 中 | 高 | 1. 用 CXL 内存模拟 NVM 2. 用 DRAM 模拟 NVM（限制带宽） 3. 简化为 GPU+DRAM+SSD 三级 |
| OrchFS 适配工作量超预期 | 中 | 中 | 1. 只用 OrchFS 的 IO 引擎部分 2. 实现简化版的异构 IO 层 |
| 实验提升不显著 | 低 | 高 | 1. 增大序列长度/batch size 2. 聚焦 NVM 的延迟优势场景 3. 增加压缩模块 |
| 并发工作抢先发表 | 中 | 高 | 1. 差异化角度（强调 OrchFS 集成） 2. 加速投稿节奏 |
| 论文被拒 | 中 | 中 | 1. 根据审稿意见修改 2. 转投其他 A/B 类会议 |

---

## 九、整体时间线总览

```
Week 1-2:   ████████  论文调研 + Related Work
Week 2-3:   ████████  问题设定 + Motivation 实验
Week 3:     ████      设计确认 + 论文 §3 初稿
Week 4-5:   ████████  Phase A: 基础框架实现
Week 5-7:   ████████████  Phase B: OrchFS 后端实现
Week 7-9:   ████████████  Phase C: 冷热分级与调度实现
Week 9-10:  ████████  Phase D: 推理引擎集成
Week 10:    ████      实验环境准备 + 微基准
Week 11-12: ████████  主实验 (E1~E3) + 论文 §1§2§3§4 初稿
Week 13:    ████████  消融实验 (E4~E6) + §5§7 初稿
Week 14:    ████████  补充实验 (E7~E10) + §6 初稿
Week 15:    ████      全文统稿 + 补充实验 + 内部审阅
Week 16:    ████      精修 + 投稿
```

**总预计工期：16 周（约 4 个月）**

---

## 十、每周检查清单

每周末进行自检：

- [ ] 本周计划的任务是否全部完成？
- [ ] 是否发现新的相关工作需要补充调研？
- [ ] 当前实现是否与论文叙事一致？
- [ ] 实验数据是否符合预期？偏差的原因是什么？
- [ ] 下周的任务是否明确？是否有阻塞依赖？
- [ ] 论文写作进度是否跟上？

---

## 十一、工具与资源

### 代码管理
- Git 仓库：当前 `/home/lzq/codes/orchkv/`
- 分支策略：`main` (稳定) → `dev` (开发) → `feat/*` (功能分支)

### 实验管理
- 实验日志：`results/` 目录，按日期和实验编号组织
- 性能数据：CSV 格式，便于 Python 绘图
- 绘图工具：Matplotlib + Seaborn（风格对齐顶会论文）

### 论文撰写
- LaTeX 模板：USENIX / ACM 格式（根据目标会议选择）
- 图表工具：draw.io / Tikz（架构图）、Matplotlib（实验图）
- 参考文献：BibTeX 管理
- 协作工具：Overleaf（如有合作者）

### 关键联系人
- OrchFS 作者团队（如有疑问可联系 zhanyekang@foxmail.com）

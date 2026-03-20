# OrchKvCache: 基于 OrchFS 异构存储编排的 LLM 推理 KV-Cache 分层调度系统

## 1. 项目概述

OrchKvCache 是一个面向大语言模型（LLM）推理场景的 KV-Cache 分层调度与管理系统。核心思路是将 LLM 推理过程中产生的 KV-Cache 按照**冷热程度**进行分级，结合 OrchFS 提供的**异构多粒度存储能力**（NVM 4KB 页粒度 + SSD 32KB 块粒度），实现 KV-Cache 在 **GPU HBM → Host DRAM → NVM → SSD** 四级存储层次之间的高效调度与迁移。

### 1.1 解决的问题

在长上下文 LLM 推理（如 128K+ tokens）场景下，KV-Cache 的显存占用随序列长度线性增长，成为推理吞吐的主要瓶颈：

- **显存不足**：单个请求的 KV-Cache 可达数十 GB，GPU HBM 容量有限
- **批处理受限**：KV-Cache 占用过多显存，压缩了可同时服务的请求数
- **冷数据浪费**：Attention 机制中，历史 token 的访问频率呈幂律分布，大量"冷"KV-Cache 长期占据宝贵的 GPU 显存却极少被访问
- **换入换出低效**：现有 offloading 方案多采用固定粒度、同步传输，未能充分利用现代存储设备的带宽

### 1.2 核心思想

借鉴 OrchFS 的异构 IO 编排理念：

1. **冷热分离**：通过访问频率追踪和注意力分数感知，将 KV-Cache 划分为热（Hot）、温（Warm）、冷（Cold）三级
2. **分级存储**：热数据驻留 GPU HBM，温数据下放 Host DRAM，冷数据经 OrchFS 的对齐写分区机制高效写入 NVM/SSD
3. **多粒度管理**：借鉴 OrchFS 的异构数据布局（4KB NVM 页 + 32KB SSD 块），对不同冷热级别的 KV-Cache 采用不同粒度管理，小粒度支持快速换入、大粒度支持高吞吐换出
4. **异步并行 IO**：利用 OrchFS 的嵌入式并行 IO 引擎，NVM 和 SSD 的 IO 线程池并行执行，最大化存储带宽利用

---

## 2. 背景与动机

### 2.1 LLM 推理中的 KV-Cache

Transformer 架构的自回归推理分为两个阶段：

- **Prefill 阶段**：处理完整 prompt，生成所有 token 的 K、V 向量并缓存
- **Decode 阶段**：逐 token 生成，每步需访问历史所有 K、V 向量计算注意力

**KV-Cache 内存模型**（单层、单头）：

```
单个 token 的 KV 占用 = 2 × d_head × sizeof(dtype)
单层总 KV 占用 = 2 × n_heads × seq_len × d_head × sizeof(dtype)
整个模型 KV 占用 = n_layers × 2 × n_heads × seq_len × d_head × sizeof(dtype)
```

以 LLaMA-70B（80 层，64 头，d_head=128，FP16）为例：
- 单个 4K token 请求：80 × 2 × 64 × 4096 × 128 × 2B ≈ 10.0 GB
- 单个 128K token 请求：≈ 320 GB（远超单 GPU 显存）

### 2.2 KV-Cache 的访问模式特征

研究表明，LLM 推理中 KV-Cache 的访问模式具有以下特征：

1. **时间局部性**：最近生成的 token 对应的 KV 更可能在后续步骤中获得高注意力分数
2. **幂律分布**：少量 token（attention sink、关键语义锚点）持续获得高注意力，大部分 token 的注意力分数随距离衰减
3. **层间差异**：不同 Transformer 层的注意力模式不同，底层倾向局部注意力，高层倾向全局注意力
4. **头间差异**：同一层内不同注意力头可能关注不同的 token 子集

这些特征为冷热分级提供了理论基础。

### 2.3 OrchFS 的关键能力

OrchFS（FAST 2025）提供了以下可直接复用的能力：

| 能力 | OrchFS 原始用途 | OrchKvCache 中的应用 |
|------|----------------|---------------------|
| **异构数据布局** | 4KB NVM 页 + 32KB SSD 块 | 温数据用 4KB 页快速换入换出，冷数据用 32KB 块批量写入 SSD |
| **对齐写分区** | 按 SSD 页对齐拆分写请求 | KV-Cache 换出时按对齐边界拆分，最大化 SSD 写带宽 |
| **统一映射结构** | `offset_info_t` 统一管理 NVM/SSD 地址 | 统一管理 KV-Cache 块在各存储层的位置 |
| **嵌入式并行 IO** | NVM/SSD 独立线程池 | 换入换出操作利用多线程并行 IO |
| **LRU + 迁移机制** | NVM 页满时迁移至 SSD | 温数据冷却后从 NVM 迁移至 SSD |
| **STRATA_NODE 混合节点** | 同一逻辑块内 NVM+SSD 共存 | 部分换出场景中，同一 KV block 的不同子块可分布在不同存储层 |

---

## 3. 系统架构

### 3.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Inference Engine                        │
│                   (vLLM / TensorRT-LLM / Custom)                   │
├─────────────────────────────────────────────────────────────────────┤
│                     OrchKvCache 调度层                               │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ 冷热分级器 │  │  分层存储管理  │  │ 预取调度引擎  │  │ 生命周期   │ │
│  │ HotCold   │  │  TieredStore  │  │ Prefetcher   │  │ 管理器     │ │
│  │ Classifier│  │  Manager      │  │ Scheduler    │  │ Lifecycle  │ │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│        │               │                 │                │       │
│  ┌─────┴───────────────┴─────────────────┴────────────────┴─────┐ │
│  │                    统一地址映射层 (Unified Address Map)         │ │
│  │            管理 KV block → 物理存储位置的映射关系               │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
├─────────────────────────────┼───────────────────────────────────────┤
│                             │        存储后端                       │
│  ┌──────────┐  ┌────────────┴───┐  ┌──────────────────────────┐   │
│  │ GPU HBM  │  │   Host DRAM    │  │     OrchFS Backend       │   │
│  │ (热数据)  │  │   (温数据)      │  │  ┌──────┐  ┌──────────┐ │   │
│  │          │  │               │  │  │ NVM   │  │   SSD    │ │   │
│  │          │  │               │  │  │(温/冷) │  │  (冷数据) │ │   │
│  └──────────┘  └───────────────┘  │  └──────┘  └──────────┘ │   │
│                                   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

#### 3.2.1 冷热分级器（HotCold Classifier）

负责实时评估每个 KV-Cache block 的"热度"，决定其应驻留的存储层级。

**输入信号**：
- **注意力分数统计**：每次 attention 计算后，累积各 token 位置的注意力分数
- **访问时间戳**：记录每个 KV block 最近一次被访问的 decode step
- **访问频次计数**：滑动窗口内的累积访问次数

**热度计算模型**：

```
Hotness(block_i) = α × AttnScore_avg(block_i)
                 + β × Recency(block_i)
                 + γ × Frequency(block_i)
```

其中：
- `AttnScore_avg`：block 内所有 token 的平均注意力分数（跨头聚合）
- `Recency`：基于最近访问时间的指数衰减函数
- `Frequency`：滑动窗口内的归一化访问频次
- `α, β, γ`：可调权重，默认 α=0.5, β=0.3, γ=0.2

**分级策略**：

| 级别 | 热度阈值 | 存储层 | 特征描述 |
|------|---------|--------|---------|
| Hot  | > θ_hot | GPU HBM | 高注意力分数、最近频繁访问的 token |
| Warm | θ_cold ~ θ_hot | Host DRAM / NVM | 中等访问频率，可能被近期 decode step 再次引用 |
| Cold | < θ_cold | NVM / SSD | 低注意力分数、长时间未被访问的历史 token |

**特殊处理**：
- **Attention Sink**（初始 token）：标记为永久 Hot，始终驻留 GPU HBM
- **Sliding Window**：当前 decode position 前 W 个 token 强制为 Hot
- **层自适应**：每层独立维护热度阈值，底层和高层可采用不同策略

#### 3.2.2 分层存储管理器（Tiered Storage Manager）

管理四级存储层次，协调数据在各层之间的迁移。

**存储层定义**：

```c
typedef enum {
    TIER_GPU_HBM  = 0,   // GPU 显存，最快、容量最小
    TIER_HOST_DRAM = 1,   // 主机 DRAM，次快
    TIER_NVM       = 2,   // NVM（如 Intel Optane PM），持久化、低延迟
    TIER_SSD       = 3,   // SSD，大容量、高带宽
} StorageTier;
```

**各层特性与配置**：

| 存储层 | 典型容量 | 读延迟 | 写带宽 | 管理粒度 | 用途 |
|--------|---------|--------|--------|---------|------|
| GPU HBM | 24~80 GB | ~ns | ~2 TB/s | Token-group (可变) | 活跃推理数据 |
| Host DRAM | 64~512 GB | ~100 ns | ~50 GB/s | Page (4KB) | 温数据缓冲 |
| NVM | 32~512 GB | ~300 ns | ~8 GB/s | Page (4KB) | 温/冷数据暂存 |
| SSD | 1~8 TB | ~10 μs | ~7 GB/s | Block (32KB) | 冷数据持久化 |

**容量管理**：
- 每层设定 **高水位线（HWM）** 和 **低水位线（LWM）**
- 当某层使用量超过 HWM 时，触发向下一层的**换出（Evict）**操作
- 换出持续进行直到使用量降至 LWM 以下
- 当某层数据被推理引擎请求但不在该层时，触发**换入（Fetch）**操作

#### 3.2.3 预取调度引擎（Prefetch Scheduler）

在 decode 阶段，预测下一步可能需要的 KV-Cache block，提前从低层存储换入高层。

**预取策略**：

1. **基于注意力模式的预取**：
   - 分析历史 decode step 的注意力分数分布，预测下一步高概率访问的 token 位置
   - 对于具有稳定注意力模式的层/头，提前将对应 KV block 换入 DRAM/GPU

2. **基于空间局部性的预取**：
   - 当换入某个 KV block 时，预取相邻 block（特别是同一语义段内的 token）

3. **流水线重叠**：
   - 将预取 IO 与 GPU 计算重叠，利用 PCIe/CXL 带宽在 GPU 计算 attention 的同时预取下一层的数据
   - 使用 CUDA Stream 或异步 DMA 实现 GPU↔Host 的重叠传输

**预取队列管理**：

```
Prefetch Priority = Predicted_AttnScore × Tier_Penalty
```

其中 `Tier_Penalty` 反映当前数据所在层与目标层之间的传输延迟，优先预取延迟最大的冷数据。

#### 3.2.4 生命周期管理器（Lifecycle Manager）

管理 KV-Cache 从创建到销毁的完整生命周期。

**状态机**：

```
                    ┌──────────┐
         Prefill    │          │   Decode 持续访问
        ─────────►  │   HOT    │ ◄──────────────────
                    │ (GPU HBM)│
                    └────┬─────┘
                         │ 热度衰减
                         ▼
                    ┌──────────┐
                    │   WARM   │
                    │(DRAM/NVM)│ ◄─── 换入(预取命中)
                    └────┬─────┘
                         │ 持续冷却
                         ▼
                    ┌──────────┐
                    │   COLD   │
                    │(NVM/SSD) │
                    └────┬─────┘
                         │ 请求结束
                         ▼
                    ┌──────────┐
                    │ EVICTED  │  释放所有存储资源
                    └──────────┘
```

**关键操作**：
- **Allocate**：Prefill 阶段在 GPU HBM 上分配 KV-Cache
- **Demote**：热度下降时，将 KV block 从高层迁移至低层
- **Promote**：换入时，将 KV block 从低层提升至高层
- **Evict**：请求完成或超时后，释放该请求的全部 KV-Cache
- **Compact**：周期性整理碎片，合并小块 NVM 页为 SSD 块（复用 OrchFS 迁移机制）

---

## 4. 与 OrchFS 的集成设计

### 4.1 数据布局与粒度映射

将 KV-Cache 的逻辑管理单元映射到 OrchFS 的物理存储单元：

**KV Block 定义**：

```c
#define KV_TOKEN_GROUP_SIZE    64   // 每个 KV block 包含 64 个 token 的 KV 数据

// 单个 KV block 的大小（以 LLaMA-70B 单层单头为例）：
// 64 tokens × 128 d_head × 2 (K+V) × 2 bytes (FP16) = 32KB
// 恰好对应 OrchFS 的一个 SSD Block (ORCH_BLOCK_SIZE = 32KB)
```

**粒度映射关系**：

| KV-Cache 逻辑单元 | 大小 | OrchFS 存储单元 | 适用场景 |
|-------------------|------|----------------|---------|
| KV Sub-block | 4KB (8 tokens) | NVM Page (ORCH_PAGE_SIZE) | 细粒度换入/部分换出 |
| KV Block | 32KB (64 tokens) | SSD Block (ORCH_BLOCK_SIZE) | 批量冷数据写入 SSD |
| KV Super-block | N×32KB | 多个连续 SSD Block | 超长上下文批量换出 |

**设计要点**：
- KV Block 大小设计为 32KB 以对齐 OrchFS 的 SSD Block，最大化 SSD 写入带宽
- KV Sub-block 为 4KB 对齐 OrchFS 的 NVM Page，支持细粒度的 token 级换入换出
- 当 d_head 或 dtype 不同时，通过调整 `KV_TOKEN_GROUP_SIZE` 保持 32KB 对齐

### 4.2 利用 OrchFS STRATA_NODE 实现混合存储

OrchFS 的 STRATA_NODE 允许同一个 32KB 逻辑块内的 8 个 4KB 槽位分别位于 NVM 或 SSD。这为 OrchKvCache 提供了独特的优势：

**场景举例**：一个 KV Block（32KB，64 tokens）中：
- Token 0~7（Slot 0）是 attention sink → 保留在 NVM（快速访问）
- Token 8~55（Slot 1~6）全部冷 → 在 SSD
- Token 56~63（Slot 7）最近被预取 → 在 NVM

OrchFS 的 `virtual_node_t` 结构天然支持这种混合布局：

```c
struct virtual_node_t {
    uint64_t ndtype;                        // STRATA_NODE
    uint64_t ssd_dev_addr;                  // SSD 上的块地址
    int64_t nvm_page_id[VLN_SLOT_SUM];      // 8 个 NVM 页的 ID（-1 表示不在 NVM）
    int64_t buf_meta_id[VLN_SLOT_SUM];      // 对应的 buffer 元数据
};
```

### 4.3 利用 OrchFS 迁移机制

OrchFS 已有 NVM → SSD 的迁移框架（`migrate.h/c`），OrchKvCache 对其进行扩展：

**原始 OrchFS 迁移**：
- 触发条件：`nvm_page_used > mig_threshold`
- 操作：从 LRU 选出最久未使用的 NVM 页 → 合并 8 个 4KB 页为 32KB 块 → 写入 SSD
- 调用：`change_virnd_to_ssdblk()` 更新索引

**OrchKvCache 扩展迁移**：
- **KV-aware 迁移**：迁移决策不仅考虑 LRU，还结合 KV-Cache 的热度分数
- **批量迁移**：当请求结束时，批量将该请求的所有 KV-Cache 标记为 Cold 并触发迁移
- **迁移优先级**：热度最低的 KV block 优先迁移，保证高热度数据尽可能留在 NVM
- **反向迁移（SSD → NVM）**：当预取命中时，将 SSD 上的冷数据换入 NVM

### 4.4 利用 OrchFS 并行 IO 引擎

OrchFS 的 `io_thdpool` 提供了 NVM 和 SSD 的独立线程池：

```c
// OrchFS 配置
#define ORCH_CONFIG_NVMTHD     5    // NVM IO 线程
#define ORCH_CONFIG_SSDTHD     32   // SSD IO 线程
```

OrchKvCache 利用方式：
1. **换出流水线**：GPU → DRAM 使用 CUDA memcpy async，DRAM → NVM/SSD 使用 OrchFS IO 线程池
2. **换入流水线**：SSD/NVM → DRAM 使用 OrchFS IO 线程池，DRAM → GPU 使用 CUDA memcpy async
3. **IO 与计算重叠**：在 GPU 执行 decode step N 的 attention 计算时，同时通过 OrchFS 线程池换入 step N+1 可能需要的 KV block

---

## 5. 详细设计

### 5.1 KV-Cache 元数据结构

```c
// KV Block 元数据
typedef struct kv_block_meta {
    uint64_t request_id;            // 所属推理请求 ID
    uint32_t layer_id;              // Transformer 层编号
    uint32_t head_id;               // 注意力头编号（或 head group）
    uint64_t token_start;           // 起始 token 位置
    uint32_t token_count;           // 包含的 token 数量

    StorageTier current_tier;       // 当前存储层
    float hotness_score;            // 当前热度分数
    uint64_t last_access_step;      // 最近访问的 decode step
    uint32_t access_count;          // 滑动窗口内访问次数

    // OrchFS 映射信息
    union {
        void* gpu_ptr;              // GPU HBM 地址
        void* dram_ptr;             // Host DRAM 地址
        struct {
            int64_t orchfs_offset;  // OrchFS 文件偏移（可定位到 virtual_node_t）
            int8_t slot_bitmap;     // 8 bit，标记哪些 4KB slot 有效
        } persistent;               // NVM/SSD 存储信息
    } location;

    pthread_rwlock_t lock;          // 读写锁，保护并发迁移
    uint8_t flags;                  // PIN（不可迁移）、DIRTY（已修改）等标记
} kv_block_meta_t;
```

```c
// 请求级 KV-Cache 管理
typedef struct kv_request_ctx {
    uint64_t request_id;
    uint32_t seq_len;               // 当前序列长度
    uint32_t n_layers;
    uint32_t n_heads;

    // 每层每头的 KV block 数组
    // kv_blocks[layer][head] → 动态数组 of kv_block_meta_t*
    kv_block_meta_t*** kv_blocks;

    // 统计
    uint64_t total_blocks;
    uint64_t hot_blocks;
    uint64_t warm_blocks;
    uint64_t cold_blocks;

    // 关联的 OrchFS 文件（每个请求对应一个 OrchFS 文件用于持久化冷数据）
    int orchfs_fd;
    char orchfs_path[256];          // e.g., "/Or/kvcache/<request_id>"
} kv_request_ctx_t;
```

### 5.2 冷热分级算法详细设计

#### 5.2.1 注意力分数采集

在每次 attention 计算后，**异步**采集注意力分数统计信息：

```python
# 伪代码：注意力分数采集（嵌入推理引擎内部）
def post_attention_hook(layer_id, head_id, attn_weights, kv_block_map):
    """
    attn_weights: [batch, n_heads, 1, seq_len] — 当前 decode step 的注意力分数
    """
    for block in kv_block_map[layer_id]:
        token_range = block.token_start : block.token_start + block.token_count
        block_attn = attn_weights[:, head_id, :, token_range].mean()
        block.update_hotness(attn_score=block_attn, current_step=global_step)
```

#### 5.2.2 热度更新策略

采用**指数移动平均（EMA）** 更新热度，避免瞬时波动：

```
hotness_new = (1 - λ) × hotness_old + λ × instant_hotness
```

其中 `instant_hotness` 由当前 step 的注意力分数、访问时间、频次加权计算。

#### 5.2.3 自适应阈值调整

阈值 `θ_hot` 和 `θ_cold` 不固定，而是根据当前存储压力动态调整：

```
if GPU_HBM_usage > HWM_gpu:
    θ_hot ↑  (提高 hot 门槛，更多 block 被降级)
if GPU_HBM_usage < LWM_gpu:
    θ_hot ↓  (降低 hot 门槛，允许更多 block 驻留 GPU)
```

### 5.3 数据迁移流水线

#### 5.3.1 换出流程（Demote: Hot → Warm → Cold）

```
Step 1: 冷热分级器标记 block 为 WARM
    │
Step 2: GPU → DRAM 异步拷贝 (cudaMemcpyAsync, non-blocking)
    │   - 在非关键 CUDA stream 上执行
    │   - 拷贝完成后释放 GPU HBM 空间
    │
Step 3: 若 block 进一步冷却 (WARM → COLD)
    │
Step 4: DRAM → OrchFS 写入
    ├── 若 block 大小 ≥ STRATA_THRESHOLD × 4KB:
    │       通过 OrchFS SSD IO 线程池写入 SSD (32KB block 对齐)
    └── 否则:
            通过 OrchFS NVM IO 线程池写入 NVM (4KB page 粒度)
    │
Step 5: 更新 kv_block_meta_t 的 location 和 current_tier
    │
Step 6: 释放 DRAM 空间
```

#### 5.3.2 换入流程（Promote: Cold → Warm → Hot）

```
Step 1: 预取调度器或 attention 计算发现需要的 block 不在 GPU
    │
Step 2: 查询 kv_block_meta_t 确定 block 所在层
    │
Step 3: 从 OrchFS 读取 (NVM/SSD → DRAM)
    ├── NVM 数据: 直接读取 4KB page (低延迟 ~300ns)
    └── SSD 数据: 读取 32KB block (通过 OrchFS SSD 线程池)
    │
Step 4: DRAM → GPU 异步拷贝 (cudaMemcpyAsync)
    │   - 与 GPU 上其他层的 attention 计算重叠
    │
Step 5: 更新 kv_block_meta_t，标记为 HOT
```

#### 5.3.3 OrchFS 内部迁移（NVM → SSD）

当 NVM 空间不足时，利用 OrchFS 的迁移线程执行冷数据下刷：

```
Step 1: migrate_info_t.nvm_page_used > mig_threshold
    │
Step 2: 从 KV-aware LRU 中选出热度最低的 NVM 页
    │   - 优先选择同一 KV block 的所有 8 个 slot
    │   - 确保凑齐 32KB 的完整块
    │
Step 3: 调用 OrchFS do_migrate_operation()
    │   - 读取 8 × 4KB NVM 页
    │   - 合并为 32KB
    │   - 分配 SSD block 并写入
    │
Step 4: change_virnd_to_ssdblk() 更新索引
    │
Step 5: 释放 NVM 页，更新 kv_block_meta_t
```

### 5.4 与推理引擎的集成接口

#### 5.4.1 C API（核心层）

```c
// ============ 初始化与配置 ============

// 初始化 OrchKvCache 系统
// config 包含存储路径、容量限制、线程数等配置
int orchkv_init(const orchkv_config_t* config);

// 关闭系统，释放所有资源
int orchkv_shutdown(void);


// ============ 请求级操作 ============

// 创建一个推理请求的 KV-Cache 上下文
kv_request_ctx_t* orchkv_request_create(
    uint64_t request_id,
    uint32_t n_layers,
    uint32_t n_heads,
    uint32_t d_head,
    DataType dtype
);

// 销毁请求上下文，释放所有关联的 KV-Cache 存储
int orchkv_request_destroy(kv_request_ctx_t* ctx);


// ============ KV Block 操作 ============

// Prefill 阶段：批量分配并写入 KV-Cache（GPU HBM 上）
int orchkv_prefill(
    kv_request_ctx_t* ctx,
    uint32_t layer_id,
    const void* k_data,       // [seq_len, n_heads, d_head]
    const void* v_data,       // [seq_len, n_heads, d_head]
    uint32_t seq_len
);

// Decode 阶段：追加单个 token 的 KV
int orchkv_append_token(
    kv_request_ctx_t* ctx,
    uint32_t layer_id,
    const void* k_token,      // [n_heads, d_head]
    const void* v_token       // [n_heads, d_head]
);

// 获取指定层的 KV-Cache 用于 attention 计算
// 返回 GPU 上的指针（如果数据不在 GPU，会触发换入）
int orchkv_get_kv(
    kv_request_ctx_t* ctx,
    uint32_t layer_id,
    void** k_out,             // 输出 K 的 GPU 指针
    void** v_out,             // 输出 V 的 GPU 指针
    uint32_t* seq_len_out     // 输出当前有效序列长度
);


// ============ 调度控制 ============

// 上报注意力分数，用于冷热分级
int orchkv_report_attention(
    kv_request_ctx_t* ctx,
    uint32_t layer_id,
    const float* attn_scores,  // [n_heads, seq_len]
    uint32_t current_step
);

// 手动触发一轮冷热分级与迁移
int orchkv_trigger_scheduling(kv_request_ctx_t* ctx);

// 查询当前存储状态
int orchkv_get_stats(orchkv_stats_t* stats);
```

#### 5.4.2 Python Binding（供 vLLM 等框架集成）

```python
import orchkvcache

# 初始化
orchkvcache.init(
    gpu_memory_limit="40GB",
    dram_memory_limit="128GB",
    orchfs_nvm_path="/dev/dax0.0",
    orchfs_ssd_path="/dev/nvme1n1",
    nvm_threads=4,
    ssd_threads=16,
)

# 创建请求
ctx = orchkvcache.RequestContext(
    request_id=12345,
    n_layers=80,
    n_heads=64,
    d_head=128,
    dtype="float16",
)

# Prefill
ctx.prefill(layer_id=0, k_data=k_tensor, v_data=v_tensor)

# Decode loop
for step in range(max_new_tokens):
    for layer in range(n_layers):
        k_gpu, v_gpu, seq_len = ctx.get_kv(layer_id=layer)
        attn_output, attn_scores = attention(query, k_gpu, v_gpu)
        ctx.report_attention(layer_id=layer, attn_scores=attn_scores, step=step)
        ctx.append_token(layer_id=layer, k_token=new_k, v_token=new_v)

    # 每 N 步触发一次调度
    if step % scheduling_interval == 0:
        ctx.trigger_scheduling()

# 清理
ctx.destroy()
orchkvcache.shutdown()
```

### 5.5 KV-Cache 压缩（可选优化）

在换出到低层存储前，可对 KV-Cache 进行压缩以减少存储和 IO 开销：

1. **量化压缩**：FP16 → INT8/INT4，在换出时量化，换入时反量化
2. **Token 剪枝**：对极冷 token（注意力分数趋近于零），直接丢弃而非存储
3. **增量编码**：相邻 token 的 KV 向量差异较小时，存储增量

压缩后的数据仍保持 OrchFS 粒度对齐（4KB/32KB），通过在 `kv_block_meta_t` 中记录压缩方式和原始大小。

---

## 6. 项目结构

```
OrchKvCache/
├── README.md                       # 本文档
├── CMakeLists.txt                  # 顶层构建配置
├── config/
│   ├── orchkv_config.h             # 系统配置（存储层容量、阈值、线程数等）
│   └── orchkv_config_template.h    # 配置模板
│
├── src/
│   ├── core/
│   │   ├── kv_block.h/c            # KV Block 元数据与基本操作
│   │   ├── kv_request.h/c          # 请求级 KV-Cache 上下文管理
│   │   ├── address_map.h/c         # 统一地址映射（逻辑 block → 物理位置）
│   │   └── kv_types.h              # 公共类型定义
│   │
│   ├── classifier/
│   │   ├── hotcold_classifier.h/c  # 冷热分级核心逻辑
│   │   ├── attention_tracker.h/c   # 注意力分数采集与统计
│   │   └── adaptive_threshold.h/c  # 自适应阈值调整
│   │
│   ├── tiered_store/
│   │   ├── tiered_manager.h/c      # 分层存储管理器总控
│   │   ├── gpu_tier.h/c            # GPU HBM 层管理（CUDA 内存分配/传输）
│   │   ├── dram_tier.h/c           # Host DRAM 层管理
│   │   ├── orchfs_tier.h/c         # OrchFS 后端（NVM + SSD 统一接口）
│   │   └── migration.h/c           # 跨层迁移引擎（换入/换出/OrchFS迁移）
│   │
│   ├── scheduler/
│   │   ├── prefetch_scheduler.h/c  # 预取调度引擎
│   │   ├── eviction_policy.h/c     # 换出策略（LRU/热度/混合）
│   │   └── pipeline.h/c            # IO-计算重叠流水线管理
│   │
│   ├── compress/                   # [可选] KV-Cache 压缩模块
│   │   ├── quantizer.h/c           # FP16 → INT8/INT4 量化
│   │   └── delta_encoder.h/c       # 增量编码
│   │
│   └── api/
│       ├── orchkv_api.h            # 对外 C API 头文件
│       ├── orchkv_api.c            # C API 实现
│       └── orchkv_stats.h/c        # 统计信息查询
│
├── python/
│   ├── orchkvcache/
│   │   ├── __init__.py
│   │   ├── binding.py              # ctypes/pybind11 绑定
│   │   ├── config.py               # Python 配置接口
│   │   └── request.py              # RequestContext Python 封装
│   └── setup.py
│
├── integration/
│   ├── vllm/                       # vLLM 集成适配
│   │   ├── orchkv_attention_backend.py
│   │   └── orchkv_block_manager.py
│   └── trtllm/                     # TensorRT-LLM 集成适配
│       └── orchkv_plugin.cpp
│
├── test/
│   ├── unit/
│   │   ├── test_classifier.c       # 冷热分级器单元测试
│   │   ├── test_tiered_store.c     # 分层存储单元测试
│   │   ├── test_migration.c        # 迁移流程测试
│   │   └── test_prefetch.c         # 预取调度测试
│   ├── integration/
│   │   ├── test_orchfs_backend.c   # OrchFS 后端集成测试
│   │   └── test_e2e_inference.py   # 端到端推理测试
│   └── benchmark/
│       ├── bench_tier_latency.c    # 各层存储延迟基准测试
│       ├── bench_migration_bw.c    # 迁移带宽基准测试
│       └── bench_inference.py      # 推理吞吐/延迟基准测试
│
├── scripts/
│   ├── setup_orchfs.sh             # OrchFS 环境配置脚本
│   ├── run_benchmark.sh            # 运行基准测试
│   └── plot_results.py             # 绘制性能对比图
│
├── docs/
│   ├── design.md                   # 详细设计文档
│   ├── api_reference.md            # API 参考手册
│   └── figures/                    # 架构图、流程图等
│
└── third_party/
    └── OrchFS -> ../OrchFS         # 符号链接到 OrchFS 源码
```

---

## 7. 构建与依赖

### 7.1 系统依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| GCC / G++ | ≥ 9.0 | C/C++ 编译 |
| CMake | ≥ 3.18 | 构建系统 |
| CUDA Toolkit | ≥ 11.8 | GPU 内存管理、CUDA Stream |
| libpmem (PMDK) | ≥ 1.12 | NVM 持久化内存操作 |
| Python | ≥ 3.8 | Python binding、测试脚本 |
| PyTorch | ≥ 2.0 | Tensor 操作、与推理引擎交互 |
| pybind11 | ≥ 2.11 | C++ → Python 绑定 |

### 7.2 硬件要求

| 硬件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 1× NVIDIA A100 40GB | 1× NVIDIA H100 80GB |
| DRAM | 64 GB | 256 GB |
| NVM | 32 GB Intel Optane PM | 128 GB Intel Optane PM (DAX mode) |
| SSD | 1 TB NVMe SSD | 2× NVMe SSD (RAID0) |
| CPU | 16 cores | 32+ cores（支撑 IO 线程池） |

### 7.3 构建步骤

```bash
# 1. 克隆项目（假设已有）
cd /path/to/orchkv/OrchKvCache

# 2. 配置 OrchFS
cd ../OrchFS
python config_parameter.py /dev/dax0.0 /dev/nvme1n1 4 16 32k
mkdir -p build && cd build && cmake .. && make
cd ../../OrchKvCache

# 3. 构建 OrchKvCache
mkdir -p build && cd build
cmake .. -DORCHFS_ROOT=../../OrchFS \
         -DCUDA_TOOLKIT_ROOT=/usr/local/cuda \
         -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)

# 4. 安装 Python binding
cd ../python
pip install -e .
```

---

## 8. 实验评估计划

### 8.1 评估指标

| 指标 | 描述 | 测量方法 |
|------|------|---------|
| **推理吞吐** | Tokens/秒 | 固定 batch size，测量单位时间生成 token 数 |
| **首 Token 延迟 (TTFT)** | Prefill 延迟 | 从请求到达到第一个 token 生成的时间 |
| **每 Token 延迟 (TPOT)** | Decode 延迟 | 平均每个 token 的生成时间 |
| **最大批处理量** | 同时服务的请求数 | 在延迟 SLA 约束下可容纳的最大 batch size |
| **存储带宽利用率** | SSD/NVM 实际吞吐 vs 理论峰值 | IO 性能计数器 |
| **命中率** | 预取命中/总换入次数 | 内部统计 |
| **迁移开销** | 迁移操作的 CPU 和延迟开销 | Profiling |

### 8.2 对比基线

| 基线系统 | 描述 |
|---------|------|
| **No Offloading** | 所有 KV-Cache 驻留 GPU HBM（受显存限制） |
| **vLLM PagedAttention** | vLLM 的分页 KV-Cache 管理（纯 GPU） |
| **FlexGen** | CPU/GPU/Disk 三级 offloading |
| **InfiniGen** | 基于预取的 KV-Cache offloading |
| **CacheGen** | KV-Cache 压缩+流式传输 |

### 8.3 实验工作负载

| 工作负载 | 模型 | 序列长度 | 描述 |
|---------|------|---------|------|
| Short Context | LLaMA-2-7B | 2K~4K | 短上下文基准 |
| Medium Context | LLaMA-2-13B | 8K~16K | 中等上下文 |
| Long Context | LLaMA-2-70B / Yi-34B | 32K~128K | 长上下文压力测试 |
| Multi-turn Dialog | LLaMA-2-7B | 累积 32K+ | 多轮对话场景 |
| Shared Prefix | 各模型 | 变长 | 共享前缀 + 不同后缀 |

### 8.4 实验配置矩阵

| 实验 | 变量 | 目的 |
|------|------|------|
| E1: 吞吐对比 | 不同系统 × 不同序列长度 | 验证 OrchKvCache 的吞吐优势 |
| E2: 延迟分析 | 分解 TTFT 和 TPOT 中各阶段延迟 | 分析 IO 开销占比 |
| E3: 存储层效果 | 消融 NVM 层 / SSD 层 | 验证各存储层的价值 |
| E4: 冷热分级效果 | 不同分级策略对比 | 验证冷热分级的有效性 |
| E5: 预取效果 | 开启/关闭预取 | 验证预取的延迟隐藏效果 |
| E6: 粒度敏感性 | 不同 KV Block 大小 | 找到最优管理粒度 |
| E7: 扩展性 | 增加 batch size / 序列长度 | 验证系统扩展能力 |

---

## 9. 开发路线图

### Phase 1: 基础框架（2~3 周）

- [ ] 搭建项目结构和构建系统（CMake）
- [ ] 实现 KV Block 元数据结构和基本操作
- [ ] 实现统一地址映射层
- [ ] 实现 GPU HBM 层管理（CUDA 内存分配/释放）
- [ ] 实现 Host DRAM 层管理
- [ ] 实现 GPU ↔ DRAM 异步传输
- [ ] 基本的 C API 和单元测试

### Phase 2: OrchFS 集成（2~3 周）

- [ ] 创建 OrchFS 后端适配层
- [ ] 实现 KV-Cache 到 OrchFS 文件的映射
- [ ] 实现 DRAM → NVM 写入（4KB 粒度）
- [ ] 实现 DRAM → SSD 写入（32KB 粒度）
- [ ] 实现 NVM/SSD → DRAM 读取
- [ ] 集成 OrchFS 的 NVM → SSD 迁移
- [ ] OrchFS 后端集成测试

### Phase 3: 冷热分级与调度（2~3 周）

- [ ] 实现注意力分数采集模块
- [ ] 实现热度计算与 EMA 更新
- [ ] 实现自适应阈值调整
- [ ] 实现换出策略和调度器
- [ ] 实现预取调度引擎
- [ ] 实现 IO-计算重叠流水线
- [ ] 冷热分级与调度的单元测试和集成测试

### Phase 4: 推理引擎集成（1~2 周）

- [ ] Python binding（pybind11）
- [ ] vLLM 集成适配
- [ ] 端到端推理测试

### Phase 5: 优化与评估（2~3 周）

- [ ] KV-Cache 压缩模块（量化/剪枝）
- [ ] 性能 Profiling 和瓶颈分析
- [ ] 基准测试和对比实验
- [ ] 结果分析和论文图表绘制

---

## 10. 参考文献

1. **OrchFS** — Y. Zhan et al., "Rethinking the Request-to-IO Transformation Process of File Systems for Full Utilization of High-Bandwidth SSDs," FAST 2025.
2. **vLLM / PagedAttention** — W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
3. **FlexGen** — Y. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," ICML 2023.
4. **InfiniGen** — L. Lee et al., "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management," OSDI 2024.
5. **CacheGen** — Y. Liu et al., "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving," SIGCOMM 2024.
6. **Attention Sink** — G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024.
7. **H2O** — Z. Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023.

# Work2: Phase A — 基础框架实现

> **前置已完成**: Step 0 (硬件盘点) + Step 1 (Motivation 实验) + Step 2 (存储基线测量)
> **本阶段目标**: 搭建 OrchKvCache 的 C/CUDA 核心框架，实现 GPU HBM ↔ Host DRAM 两层基本功能
> **预计工期**: 2~3 周

---

## 〇、当前状态与关键约束

### 已有实验结论（指导本阶段设计参数）

| 实验 | 关键数字 | 对 Phase A 的影响 |
|------|---------|------------------|
| Exp0: GPU↔DRAM | Pinned 25 GB/s，Pageable 15 GB/s，甜区 4-16 MB | **必须用 pinned memory pool** |
| Exp0: E2E offload | DRAM 路径 ~190 us (4MB)，SSD 路径 ~2.3 ms (4MB) | KV Block 4MB 是合理的批量传输粒度 |
| Exp2-S1: io_uring | QD≥8 时 io_uring 比 psync 快 10-17x (读) / 2-3x (写) | Phase B 用 io_uring，Phase A 先不涉及 |
| Exp2-S2: fsync | fsync-per-block 惩罚 5-19x | batch eviction 是必须的 |
| Exp2-S3: 多线程 IO | 写 4-8 线程饱和，读线性扩展到 32 线程 | IO 线程池配置参考 |
| Exp2-S5: KV IO 模式 | multi-thread batch eviction 达 3.25 GB/s | 验证了 batch + 多线程的路线 |

### 硬件约束

- **无 NVM 硬件**: Phase A 只做 GPU HBM ↔ Host DRAM 两层，NVM/SSD 后端留给 Phase B
- **GPU**: 2× A100-SXM4-80GB，CUDA 12.2
- **DRAM**: 376 GB
- **SSD**: Samsung RAID0 Gen5 NVMe (挂载 /raid)

### 语言与工具链

- **核心代码**: C11 + CUDA（与 OrchFS 保持一致，OrchFS 是纯 C）
- **构建系统**: CMake ≥ 3.18
- **测试框架**: 自写简单测试 + assert（保持轻量，不引入 gtest 等重依赖）
- **后续 Python 绑定**: Phase D 再做（pybind11 或 ctypes）

---

## 一、目录结构规划

Phase A 只创建框架骨架，后续 Phase 逐步填充：

```
OrchKvCache/
├── CMakeLists.txt                  # 顶层 CMake
├── src/
│   ├── CMakeLists.txt              # src 子目录 CMake
│   ├── core/
│   │   ├── kv_types.h              # [A1] 公共类型定义
│   │   ├── kv_block.h              # [A2] KV Block 元数据结构
│   │   ├── kv_block.c              # [A2] KV Block 操作实现
│   │   ├── kv_request.h            # [A3] 请求级 KV-Cache 上下文
│   │   ├── kv_request.c            # [A3] 请求上下文实现
│   │   ├── address_map.h           # [A4] 统一地址映射
│   │   └── address_map.c           # [A4] 地址映射实现
│   ├── tiered_store/
│   │   ├── gpu_tier.h              # [A5] GPU HBM 层接口
│   │   ├── gpu_tier.cu             # [A5] GPU HBM 层 CUDA 实现
│   │   ├── dram_tier.h             # [A6] Host DRAM 层接口
│   │   ├── dram_tier.c             # [A6] Host DRAM 层实现
│   │   ├── tier_common.h           # [A5/A6] 存储层公共接口
│   │   └── transfer.cu             # [A7] GPU↔DRAM 异步传输引擎
│   └── api/
│       ├── orchkv_api.h            # [A8] 对外 C API 头文件
│       └── orchkv_api.c            # [A8] C API 实现（Phase A 仅 init/prefill/get_kv 基础版本）
├── test/
│   ├── CMakeLists.txt
│   ├── test_kv_block.c             # [A9] KV Block 单元测试
│   ├── test_address_map.c          # [A9] 地址映射单元测试
│   ├── test_gpu_dram.cu            # [A9] GPU↔DRAM 传输测试
│   └── test_e2e_basic.cu           # [A9] 基本端到端测试
├── third_party/
│   └── OrchFS -> ../../OrchFS      # 符号链接 (Phase B 才链接编译)
├── experiments/                    # 已有实验数据 (不动)
│   ├── exp0_hardware_baseline/
│   ├── exp1_motivation/
│   └── exp2_storage_baseline/
├── README.md                       # 已有 (不动)
├── work1.md                        # 已有 (不动)
├── work2.md                        # 本文档
└── work_todo.md                    # 已有 (不动)
```

---

## 二、任务分解

### A1: 公共类型定义 (`kv_types.h`)

**目标**: 定义整个项目共享的类型、常量、错误码。

**关键设计决策**:

```
Q: KV_TOKEN_GROUP_SIZE 定多少？
A: 根据 README 设计，默认 64 tokens/block。
   - LLaMA-70B (GQA, n_kv_heads=8): 64×128×2×2B = 32KB → 恰好对齐 ORCH_BLOCK_SIZE
   - LLaMA-7B (n_kv_heads=32): 64×128×2×2B = 1MB (单层) → 太大
   → 需要区分 "per-head block" 和 "per-layer block"
   → Phase A 先按 per-head 粒度管理，每个 KV Block = 单头单层的 N tokens
```

**需要定义的内容**:

- `StorageTier` 枚举 (GPU_HBM, HOST_DRAM, NVM, SSD)
- `DataType` 枚举 (FP16, BF16, FP32, INT8)
- `KVBlockState` 枚举 (ALLOCATED, HOT, WARM, COLD, EVICTED)
- `orchkv_config_t` 配置结构体
- `orchkv_stats_t` 统计信息结构体
- 错误码宏 (ORCHKV_OK, ORCHKV_ERR_OOM, ORCHKV_ERR_INVALID, ...)
- 大小/对齐常量 (KV_BLOCK_ALIGN, KV_PAGE_SIZE, KV_BLOCK_SIZE)
- 工具宏 (ALIGN_UP, ALIGN_DOWN, MIN, MAX, LOG_INFO, LOG_ERR)

**估时**: 0.5 天

---

### A2: KV Block 元数据 (`kv_block.h/c`)

**目标**: 实现 KV Block 的元数据结构和生命周期管理。

KV Block 是整个系统的基本管理单元。每个 block 描述 "某个请求、某一层、某个头组、连续 N 个 token" 的 K 和 V 数据。

**核心结构体**:

```c
typedef struct kv_block {
    /* 标识 */
    uint64_t block_id;              // 全局唯一 ID
    uint64_t request_id;            // 所属请求
    uint16_t layer_id;              // Transformer 层
    uint16_t head_id;               // 注意力头 (或 KV head group)
    uint32_t token_start;           // 起始 token 位置
    uint16_t token_count;           // 实际 token 数 (≤ KV_TOKEN_GROUP_SIZE)

    /* 存储位置 */
    StorageTier tier;               // 当前所在层
    void *data_ptr;                 // 当前数据指针 (GPU/DRAM/NULL)
    uint64_t persistent_offset;     // OrchFS 文件内偏移 (Phase B 用)

    /* 冷热状态 (Phase C 填充，Phase A 先留占位) */
    float hotness;                  // 热度分数
    uint64_t last_access_step;
    uint32_t access_count;

    /* 管理 */
    KVBlockState state;
    uint8_t flags;                  // PIN, DIRTY, PREFETCHING, ...
    pthread_rwlock_t lock;

    /* 链表节点 (用于空闲链表/LRU 链表) */
    struct kv_block *prev;
    struct kv_block *next;
} kv_block_t;
```

**需要实现的函数**:

| 函数 | 说明 |
|------|------|
| `kv_block_init(block, request_id, layer, head, token_start, count)` | 初始化 block 元数据 |
| `kv_block_destroy(block)` | 销毁 block（释放锁，不释放数据） |
| `kv_block_data_size(block, d_head, dtype)` | 计算 block 数据大小 (bytes) |
| `kv_block_set_tier(block, tier, ptr)` | 更新 block 所在层和指针 |
| `kv_block_lock_read(block)` / `unlock_read` | 读锁 |
| `kv_block_lock_write(block)` / `unlock_write` | 写锁 |

**Block ID 分配策略**: 使用 atomic counter，`block_id = atomic_fetch_add(&global_block_counter, 1)`

**估时**: 1 天

---

### A3: 请求级 KV-Cache 上下文 (`kv_request.h/c`)

**目标**: 管理单个推理请求的完整 KV-Cache 状态。

每个推理请求拥有一个 `kv_request_ctx_t`，内部维护按 `[layer][head][block_idx]` 组织的三维 block 数组。

**核心结构体**:

```c
typedef struct kv_request_ctx {
    uint64_t request_id;
    uint32_t n_layers;
    uint32_t n_kv_heads;            // 注意：GQA 模型 n_kv_heads < n_heads
    uint32_t d_head;
    DataType dtype;
    uint32_t tokens_per_block;      // KV_TOKEN_GROUP_SIZE

    /* 当前序列状态 */
    uint32_t seq_len;               // 已有 token 数
    uint32_t n_blocks_per_head;     // 当前每头 block 数

    /* 三维 block 索引: blocks[layer][head] → 动态数组 of kv_block_t* */
    kv_block_t ***blocks;           // [n_layers][n_kv_heads][n_blocks_per_head]

    /* 统计 */
    uint64_t total_blocks;
    uint64_t blocks_on_gpu;
    uint64_t blocks_on_dram;

    /* 生命周期 */
    bool active;
    pthread_mutex_t ctx_lock;
} kv_request_ctx_t;
```

**需要实现的函数**:

| 函数 | 说明 |
|------|------|
| `kv_request_create(request_id, n_layers, n_kv_heads, d_head, dtype)` | 创建请求上下文 |
| `kv_request_destroy(ctx)` | 销毁上下文，释放所有 block |
| `kv_request_get_block(ctx, layer, head, block_idx)` | 获取指定 block |
| `kv_request_alloc_blocks(ctx, layer, n_tokens)` | 为某层分配 n_tokens 对应的 block |
| `kv_request_append_token(ctx, layer)` | 追加 token，可能触发新 block 分配 |
| `kv_request_get_seq_len(ctx)` | 当前序列长度 |

**设计要点**:
- `blocks` 数组在 `create` 时预分配外层 (`n_layers × n_kv_heads`)，内层按需动态扩展
- Prefill 时一次性分配 `ceil(seq_len / tokens_per_block)` 个 block per head per layer
- Decode 时每积累 `tokens_per_block` 个 token 追加一个新 block

**估时**: 1 天

---

### A4: 统一地址映射 (`address_map.h/c`)

**目标**: 维护 `block_id → kv_block_t*` 的全局映射，支持 O(1) 查找。

**实现方案**: 开放地址哈希表（避免引入外部依赖，OrchFS 的 `util/hashtable.c` 可参考）。

**核心接口**:

```c
typedef struct address_map {
    size_t capacity;
    size_t count;
    kv_block_t **buckets;           // 哈希桶
    pthread_rwlock_t map_lock;      // 全局读写锁
} address_map_t;
```

| 函数 | 说明 |
|------|------|
| `address_map_init(map, initial_capacity)` | 初始化 |
| `address_map_destroy(map)` | 销毁 |
| `address_map_insert(map, block)` | 插入 block (key = block_id) |
| `address_map_lookup(map, block_id)` | 查找 block |
| `address_map_remove(map, block_id)` | 删除 block |
| `address_map_count(map)` | 当前 block 总数 |

**扩容策略**: 当 `count > capacity * 0.75` 时 2x 扩容并 rehash。

**估时**: 0.5 天

---

### A5: GPU HBM 层 (`gpu_tier.h/cu`)

**目标**: 管理 GPU 显存上的 KV-Cache 空间分配与释放。

**设计方案**: 预分配一大块 GPU 显存作为 KV-Cache pool，内部用 slab 分配器管理。

**核心结构体**:

```c
typedef struct gpu_tier {
    int device_id;                  // CUDA device
    void *pool_base;                // cudaMalloc 预分配的 pool 起始地址
    size_t pool_size;               // pool 总大小
    size_t used;                    // 已分配大小

    /* Slab 分配器: 按固定 block_data_size 切分 */
    size_t slab_size;               // 单个 slab 大小 (= kv_block 数据大小)
    uint32_t total_slabs;
    uint32_t free_slabs;
    uint32_t *free_stack;           // 空闲 slab index 栈
    uint32_t free_top;              // 栈顶

    /* 容量控制 */
    float hwm_ratio;                // 高水位 (默认 0.9)
    float lwm_ratio;                // 低水位 (默认 0.7)

    pthread_mutex_t alloc_lock;
    cudaStream_t transfer_stream;   // 用于异步传输的 CUDA stream
} gpu_tier_t;
```

**需要实现的函数**:

| 函数 | 说明 |
|------|------|
| `gpu_tier_init(tier, device_id, pool_size_bytes, slab_size)` | 初始化 GPU pool |
| `gpu_tier_destroy(tier)` | 释放 GPU pool |
| `gpu_tier_alloc(tier)` | 分配一个 slab，返回 GPU 指针 |
| `gpu_tier_free(tier, gpu_ptr)` | 释放一个 slab |
| `gpu_tier_usage(tier)` | 返回使用率 |
| `gpu_tier_above_hwm(tier)` | 是否超过高水位 |
| `gpu_tier_below_lwm(tier)` | 是否低于低水位 |

**为什么用 slab 而不是 cudaMalloc/cudaFree**:
- `cudaMalloc` 每次调用有 ~100us 延迟，频繁分配不可接受
- Slab 分配只需 O(1) 的栈 pop 操作
- 所有 slab 大小相同（等于单个 KV Block 的数据大小），无碎片

**估时**: 1.5 天

---

### A6: Host DRAM 层 (`dram_tier.h/c`)

**目标**: 管理主机 DRAM 上的 KV-Cache 空间。与 GPU 层类似采用 slab 分配。

**关键区别**:
- 使用 **pinned memory** (`cudaMallocHost`) 而非普通 `malloc`，确保 GPU↔DRAM 异步传输可用
- 实验数据表明 pinned 比 pageable 快 1.7x

**核心结构体** (与 gpu_tier 类似):

```c
typedef struct dram_tier {
    void *pool_base;                // cudaMallocHost 预分配
    size_t pool_size;
    size_t used;

    size_t slab_size;
    uint32_t total_slabs;
    uint32_t free_slabs;
    uint32_t *free_stack;
    uint32_t free_top;

    float hwm_ratio;
    float lwm_ratio;

    pthread_mutex_t alloc_lock;
} dram_tier_t;
```

**接口与 gpu_tier 对称**: `dram_tier_init`, `dram_tier_destroy`, `dram_tier_alloc`, `dram_tier_free`, `dram_tier_usage`, ...

**估时**: 1 天

---

### A7: GPU↔DRAM 异步传输引擎 (`transfer.cu`)

**目标**: 封装 CUDA 异步内存传输，支持 GPU↔DRAM 的批量传输和回调。

**核心接口**:

```c
typedef void (*transfer_callback_t)(void *user_data, int status);

typedef struct transfer_request {
    void *src;
    void *dst;
    size_t size;
    enum { TRANSFER_D2H, TRANSFER_H2D } direction;
    transfer_callback_t callback;
    void *callback_data;
} transfer_request_t;

typedef struct transfer_engine {
    cudaStream_t *streams;          // 多条 CUDA stream 用于并发传输
    int num_streams;
    int next_stream;                // round-robin 选 stream

    /* 批量传输队列 */
    transfer_request_t *queue;
    int queue_capacity;
    int queue_count;
    pthread_mutex_t queue_lock;
} transfer_engine_t;
```

| 函数 | 说明 |
|------|------|
| `transfer_engine_init(engine, num_streams)` | 初始化传输引擎 |
| `transfer_engine_destroy(engine)` | 销毁 |
| `transfer_async(engine, src, dst, size, direction, callback, data)` | 提交异步传输 |
| `transfer_batch_submit(engine)` | 批量提交队列中的传输 |
| `transfer_sync_stream(engine, stream_idx)` | 同步等待某 stream 完成 |
| `transfer_sync_all(engine)` | 同步等待所有传输完成 |

**设计要点**:
- 使用 2-4 条 CUDA stream，round-robin 分配，避免单 stream 排队
- 传输完成后通过 `cudaStreamAddCallback` 触发回调（更新 block 元数据）
- 从实验数据看，4-16 MB 是最优传输大小，小于 256KB 会浪费 PCIe 带宽

**估时**: 1.5 天

---

### A8: 基础 C API (`orchkv_api.h/c`)

**目标**: 实现 Phase A 范围内的对外接口，让测试程序可以跑通完整的 Prefill → Decode 基本流程。

Phase A 只实现以下 API（对照 README 中完整 API 的子集）:

| API | Phase A 行为 |
|-----|-------------|
| `orchkv_init(config)` | 初始化 GPU pool, DRAM pool, address_map, transfer_engine |
| `orchkv_shutdown()` | 释放所有资源 |
| `orchkv_request_create(...)` | 创建请求上下文 |
| `orchkv_request_destroy(ctx)` | 销毁请求并释放所有 block |
| `orchkv_prefill(ctx, layer, k_data, v_data, seq_len)` | 在 GPU 上分配 block 并写入 KV 数据 |
| `orchkv_append_token(ctx, layer, k_token, v_token)` | 追加 token |
| `orchkv_get_kv(ctx, layer, k_out, v_out, seq_len_out)` | 返回 GPU 上的 KV 指针 |
| `orchkv_get_stats(stats)` | 返回 GPU/DRAM 使用量等统计 |

**Phase A 暂不实现**:
- `orchkv_report_attention()` — Phase C (冷热分级)
- `orchkv_trigger_scheduling()` — Phase C
- GPU→DRAM eviction 策略 — Phase A 只实现手动 evict 接口用于测试

**额外提供测试用内部接口**:

```c
// 手动将 block 从 GPU 移到 DRAM (测试 A5/A6/A7 的正确性)
int orchkv_test_evict_to_dram(kv_request_ctx_t *ctx, uint16_t layer, uint16_t head, uint32_t block_idx);
// 手动将 block 从 DRAM 移回 GPU
int orchkv_test_promote_to_gpu(kv_request_ctx_t *ctx, uint16_t layer, uint16_t head, uint32_t block_idx);
```

**估时**: 1.5 天

---

### A9: 单元测试与集成测试

**目标**: 验证 Phase A 所有模块的正确性。

| 测试文件 | 测试内容 |
|---------|---------|
| `test_kv_block.c` | block 创建/销毁、状态转换、并发锁 |
| `test_address_map.c` | 插入/查找/删除、扩容、并发安全 |
| `test_gpu_dram.cu` | GPU slab 分配/释放、DRAM slab 分配/释放、异步传输正确性和带宽 |
| `test_e2e_basic.cu` | 完整流程: init → create_request → prefill → get_kv → evict_to_dram → promote_to_gpu → 验证数据一致性 → destroy |

**数据一致性验证方法**:
1. 在 GPU 上 prefill 已知数据 (如全 1.0 或递增序列)
2. 通过 `orchkv_get_kv` 读回 GPU 指针，拷贝到 CPU 比对
3. Evict 到 DRAM，再 promote 回 GPU，再次比对
4. 确保 evict+promote 后数据 bit-exact 相同

**性能验证** (嵌入 test_gpu_dram.cu):
- 测量 GPU slab 分配延迟 (预期 <1 us)
- 测量批量传输带宽 (预期接近 Exp0 数据: 22-25 GB/s)
- 测量 100 次 evict+promote 来回的平均延迟

**估时**: 1.5 天

---

## 三、依赖关系与执行顺序

```
A1 (kv_types.h)
 ├──→ A2 (kv_block)
 │     └──→ A3 (kv_request) ──→ A8 (API) ──→ A9 (测试)
 ├──→ A4 (address_map) ─────────────┘            │
 ├──→ A5 (gpu_tier) ──→ A7 (transfer) ───────────┘
 └──→ A6 (dram_tier) ─────────────────────────────┘
```

**推荐执行顺序**:

| Day | 任务 | 产出 |
|-----|------|------|
| Day 1 | A1: kv_types.h + CMakeLists.txt 搭建 | 项目能编译通过（空壳） |
| Day 2 | A2: kv_block.h/c | block 创建/销毁/状态管理 |
| Day 3 | A4: address_map.h/c + A3 开始 | 哈希表 + 请求上下文框架 |
| Day 4 | A3: kv_request.h/c 完成 | 请求创建/block 分配/追加 token |
| Day 5 | A5: gpu_tier.h/cu | GPU slab pool |
| Day 6 | A6: dram_tier.h/c + A7 开始 | DRAM pinned pool + 传输引擎框架 |
| Day 7 | A7: transfer.cu 完成 | 异步传输 + 回调 |
| Day 8 | A8: orchkv_api.h/c | 对外接口串联所有模块 |
| Day 9-10 | A9: 全部测试 | 通过所有测试用例 |

---

## 四、CMake 构建设计

顶层 `CMakeLists.txt` 骨架:

```cmake
cmake_minimum_required(VERSION 3.18)
project(OrchKvCache LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# CUDA
find_package(CUDAToolkit REQUIRED)

# 编译选项
add_compile_options(-O2 -Wall -pthread)

# 核心库
add_library(orchkv STATIC
    src/core/kv_block.c
    src/core/kv_request.c
    src/core/address_map.c
    src/tiered_store/dram_tier.c
    src/tiered_store/gpu_tier.cu
    src/tiered_store/transfer.cu
    src/api/orchkv_api.c
)
target_include_directories(orchkv PUBLIC src)
target_link_libraries(orchkv CUDA::cudart pthread)

# Phase B 时追加:
# target_link_libraries(orchkv OrchFS_LIBFS pmem)

# 测试
enable_testing()
add_executable(test_kv_block test/test_kv_block.c)
target_link_libraries(test_kv_block orchkv)
add_test(NAME test_kv_block COMMAND test_kv_block)

add_executable(test_address_map test/test_address_map.c)
target_link_libraries(test_address_map orchkv)
add_test(NAME test_address_map COMMAND test_address_map)

add_executable(test_gpu_dram test/test_gpu_dram.cu)
target_link_libraries(test_gpu_dram orchkv)
add_test(NAME test_gpu_dram COMMAND test_gpu_dram)

add_executable(test_e2e_basic test/test_e2e_basic.cu)
target_link_libraries(test_e2e_basic orchkv)
add_test(NAME test_e2e_basic COMMAND test_e2e_basic)
```

---

## 五、关键设计决策记录

### 决策 1: KV Block 粒度

**问题**: README 中定义 `KV_TOKEN_GROUP_SIZE = 64`，但不同模型的 per-block 数据大小差异很大。

| 模型 | n_kv_heads | 单头单层 64-token block 大小 |
|------|-----------|---------------------------|
| LLaMA-2-70B (GQA) | 8 | 64×128×2×2B = 32 KB |
| LLaMA-2-7B | 32 | 64×128×2×2B = 32 KB (单头) |
| LLaMA-3-8B (GQA) | 8 | 64×128×2×2B = 32 KB |
| Mistral-7B (GQA) | 8 | 64×128×2×2B = 32 KB |

**决策**: 每个 KV Block = **单个 KV head** 的 **64 个连续 token**，数据大小 = `64 × d_head × 2(K+V) × sizeof(dtype)`。FP16 下恒为 **32 KB**，完美对齐 OrchFS 的 `ORCH_BLOCK_SIZE`。

### 决策 2: GPU 内存分配策略

**方案 A**: 每次 `cudaMalloc`（简单但慢）
**方案 B**: 预分配 pool + slab（快但浪费）

**决策**: 采用 **方案 B**。原因：
- `cudaMalloc` 延迟 ~100 us，一次 decode step 只有 ~0.5-2 ms
- Slab 分配 O(1)，延迟 <1 us
- 浪费问题通过配置 pool 大小上限来控制

**默认配置**: GPU pool = GPU 显存的 50%（40 GB / 80 GB 中的一半），可通过 `orchkv_config_t.gpu_pool_size` 调节。

### 决策 3: DRAM 使用 pinned memory

**依据**: Exp0 数据显示 pinned 比 pageable 快 1.7x，且 `cudaMemcpyAsync` 要求 pinned memory。

**风险**: `cudaMallocHost` 可能因系统锁页内存限制而失败。

**缓解**: 分批 `cudaMallocHost`，如果失败则降级为 `aligned_alloc` + 同步传输，并打印警告。

### 决策 4: C 还是 C++

**决策**: 核心用 **C11**，CUDA 文件用 `.cu`（自动走 nvcc，支持 C++ 语法）。

**理由**:
- OrchFS 是纯 C，保持一致便于后续链接
- C 的 ABI 稳定，Python ctypes 绑定更简单
- CUDA kernel 和 runtime API 需要 C++ 编译器，但 `.cu` 文件内部可以混用 C 接口

---

## 六、Phase A 验收标准

Phase A 完成时，必须达到以下目标：

- [x] `cmake .. && make` 一次通过 (nvlink 的 Skipping incompatible 是已知无害警告)
- [x] `test_kv_types` 通过: 9 项类型/常量/工具验证
- [x] `test_kv_block` 通过：block 创建、销毁、状态转换、并发锁 (8 项 + 10000 block 压力)
- [x] `test_kv_request` 通过: 请求创建/prefill/decode/payload 计算 (5 项)
- [x] `test_address_map` 通过：增删查改、扩容、10 万 block 压力测试 (4 项)
- [x] `test_gpu_dram` 通过：
  - GPU slab 分配/释放、DRAM slab 分配/释放
  - GPU→DRAM→GPU round-trip 数据一致性 (bit-exact, 32KB & 4MB)
  - 传输带宽: 4MB 块 H2D ~25 GB/s, D2H ~23 GB/s (pinned)
- [x] `test_e2e` 通过 (72 个断言，0 失败)：
  - Init/Shutdown 生命周期正确
  - Prefill 2 层 2 头 80 token → GPU 上 K/V 数据 bit-exact 正确
  - Evict → DRAM 数据一致 → Promote → GPU 数据一致
  - Append 10 token (decode) → 跨 block 边界正确
  - Stats 全链路一致 (GPU/DRAM slabs, transfer 计数)
  - Benchmark: LLaMA-7B scale (32 层 8 头)
    - Prefill 128 tok: 13.1 ms (410.8 us/layer)
    - Decode 100 steps: 387 ms (120.96 us/layer/step)
    - Evict: 10.8 us/block, Promote: 11.1 us/block
- [x] 基准测试数据已保存: `experiments/exp2_storage_baseline/results/a89_e2e_benchmark.json`

---

## 七、Phase A 之后 → Phase B 衔接

Phase A 完成后，Phase B (OrchFS 后端) 的入口是:

1. 创建 `src/tiered_store/orchfs_tier.h/c` — 封装 OrchFS 的 `orchfs_open/pwrite/pread/close`
2. 在 `dram_tier` 中增加 DRAM→OrchFS 的 evict 路径
3. 在 `transfer` 中增加 OrchFS→DRAM→GPU 的 promote 路径
4. **前提**: 需要先完成 NVM 模拟 (参见 `experiments/exp2_storage_baseline/docs/dram_as_nvm_setup.md`) 或修改 OrchFS 使其在无 NVM 时也能运行 (仅 SSD 模式)

---

## 八、TODO 清单

```
Phase A 总览 — 全部完成 ✓ (2026-03-21)
  [A1] ✓ 创建 kv_types.h + CMakeLists.txt 项目骨架
  [A2] ✓ 实现 kv_block.h/c (元数据结构 + 生命周期)
  [A3] ✓ 实现 kv_request.h/c (请求级上下文管理)
  [A4] ✓ 实现 address_map.h/c (block_id 全局哈希表)
  [A5] ✓ 实现 gpu_tier.h/cu (GPU slab pool)
  [A6] ✓ 实现 dram_tier.h/cu (DRAM pinned slab pool)
  [A7] ✓ 实现 transfer.cu (GPU↔DRAM 异步传输引擎)
  [A8] ✓ 实现 orchkv_api.h/cu (对外 C API 子集 — 完整 prefill/decode/evict/promote/stats)
  [A9] ✓ 编写并通过全部单元测试和 E2E 测试 (6/6 ctest pass, 72 assertions)
       ✓ E2E 基准测试: LLaMA-7B scale decode loop 数据已收集

实现统计:
  - 源代码: ~800 行 C/CUDA (src/), ~900 行测试 (test/)
  - 测试覆盖: kv_types, kv_block, kv_request, address_map, gpu_tier, dram_tier, transfer, orchkv_api
  - 关键修正: slab 内 K/V 使用固定偏移布局 (避免 append 时数据移位)
  - 关键修正: kv_block.h 的 stdatomic.h 在 __CUDACC__ 下条件编译
```

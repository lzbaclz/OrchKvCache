# Work3: Phase B — OrchFS 后端集成

> **前置已完成**: Phase A (GPU HBM ↔ Host DRAM 两层框架, 6/6 ctest pass)
> **本阶段目标**: 接入 OrchFS，实现 DRAM ↔ NVM/SSD 的第三/四层存储，完成四级存储全链路
> **预计工期**: 2~3 周

---

## 〇、当前状态与关键约束

### Phase A 已有的模块

| 模块 | 文件 | 作用 |
|------|------|------|
| kv_types.h | `src/core/kv_types.h` | 公共类型、错误码、常量 |
| kv_block | `src/core/kv_block.h/c` | Block 元数据、状态机、并发锁 |
| kv_request | `src/core/kv_request.h/c` | 请求级上下文、block 分配 |
| address_map | `src/core/address_map.h/c` | block_id → kv_block_t* 全局哈希表 |
| gpu_tier | `src/tiered_store/gpu_tier.h/cu` | GPU HBM slab pool |
| dram_tier | `src/tiered_store/dram_tier.h/cu` | DRAM pinned slab pool |
| transfer | `src/tiered_store/transfer.h/cu` | GPU↔DRAM 异步传输引擎 |
| orchkv_api | `src/api/orchkv_api.h/cu` | 对外 API (init/prefill/decode/evict/promote/stats) |

### Phase B 需要扩展的接口

Phase A 的 `orchkv_evict_to_dram` / `orchkv_promote_to_gpu` 只支持 GPU ↔ DRAM。Phase B 需要新增：

```
orchkv_evict_to_storage(ctx, layer, head, block_idx)    → DRAM → OrchFS (NVM/SSD)
orchkv_promote_from_storage(ctx, layer, head, block_idx) → OrchFS → DRAM → GPU
orchkv_evict_cold(ctx, layer, head, block_idx)           → GPU → DRAM → OrchFS (两跳)
```

### 硬件约束

- **无 NVM 硬件**: 必须通过 DRAM 模拟 NVM（`memmap=` 内核参数 + ndctl 创建 `/dev/dax*`）
- **需要重启**: DRAM 模拟 NVM 需要修改 GRUB 并重启
- **SSD**: Samsung RAID0 Gen5 NVMe（挂载 /raid），可用 `/dev/nvme1n1` 或 `/dev/nvme2n1`
- **OrchFS 依赖**: libpmem, Boost headers, 需编译 OrchFS 并启动 kfs_main 守护进程

### OrchFS API 概览（Phase B 用到的核心函数）

```c
// 生命周期
void init_libfs(void);            // 连接 KernelFS 守护进程
void close_libfs(void);           // 断开连接

// 文件操作
int   orchfs_open(const char *pathname, int oflag, ...);
int   orchfs_close(int fd);

// 定位读写 (KV Block 核心 IO 路径)
int64_t orchfs_pwrite(int fd, const void *buf, int64_t write_len, int64_t offset);
int64_t orchfs_pread(int fd, void *buf, int64_t read_len, int64_t byte_offset);

// 文件管理
int   orchfs_unlink(const char *pathname);
int   orchfs_mkdir(const char *pathname, uint16_t mode);
```

### OrchFS 关键常量

| 常量 | 值 | 含义 |
|------|----|------|
| ORCH_PAGE_SIZE | 4 KB | NVM 页大小 |
| ORCH_BLOCK_SIZE | 32 KB | SSD 块大小 |
| ORCH_CONFIG_NVMTHD | 5 | NVM IO 线程数 |
| ORCH_CONFIG_SSDTHD | 32 | SSD IO 线程数 |
| VLN_SLOT_SUM | 8 | 一个 virtual_node 内的 NVM slot 数 (32KB / 4KB) |

**关键对齐**: 我们的 KV Block slab = 32KB (FP16, d_head=128, 64 tokens)，恰好等于 ORCH_BLOCK_SIZE，是 ORCH_PAGE_SIZE 的 8 倍。完美对齐。

---

## 一、环境准备（B0）— 前置步骤

Phase B 的所有代码开发依赖以下环境就绪。这不是代码任务，是一次性的运维操作。

### B0.1 DRAM 模拟 NVM

详见 `experiments/exp2_storage_baseline/docs/dram_as_nvm_setup.md`，核心步骤：

```bash
# 1. 修改 GRUB: memmap=32G!340G
sudo vim /etc/default/grub
sudo update-grub && sudo reboot

# 2. 安装工具
sudo apt-get install -y ndctl daxctl libpmem-dev

# 3. 创建 DAX namespace
sudo ndctl create-namespace --mode=devdax --region=region0 --size=32G
ls /dev/dax0.0    # 应存在
```

### B0.2 编译 OrchFS

```bash
cd /home/lzq/codes/orchkv/OrchFS

# 配置设备路径 (DAX + SSD)
python config_parameter.py /dev/dax0.0 /dev/nvme1n1 5 32 32k

# 编译
mkdir -p build && cd build && cmake .. && make -j$(nproc)
# 产物: libOrchFS.so, mkfs, kfs_main, close_kfs
```

### B0.3 初始化 OrchFS 文件系统

```bash
cd /home/lzq/codes/orchkv/OrchFS/build

# 格式化 (⚠️ 会清空 NVM 和 SSD 设备)
sudo ./mkfs

# 启动 KernelFS 守护进程
sudo ./kfs_main &

# 创建挂载点 (如果不存在)
sudo mkdir -p /Or
```

### B0.4 验证 OrchFS 基本功能

编写简单测试程序调用 `init_libfs()` → `orchfs_open()` → `orchfs_pwrite()` → `orchfs_pread()` → 验证数据一致性。

---

## 二、目录结构变更

Phase B 新增/修改的文件：

```
OrchKvCache/
├── CMakeLists.txt                  # 修改: 链接 OrchFS, 新增 orchfs_tier 源文件和测试
├── src/
│   ├── core/
│   │   ├── kv_types.h              # 修改: 新增 OrchFS 相关配置字段
│   │   └── kv_block.h              # 修改: persistent_offset 实际启用
│   ├── tiered_store/
│   │   ├── orchfs_tier.h           # [B1] 新增: OrchFS 后端适配层声明
│   │   ├── orchfs_tier.c           # [B2] 新增: OrchFS 后端实现
│   │   └── io_worker.h/c          # [B3] 新增: 异步 IO 工作线程池
│   └── api/
│       └── orchkv_api.h/cu         # [B5] 修改: 新增 evict_to_storage / promote_from_storage
├── test/
│   ├── test_orchfs_tier.c          # [B6] 新增: OrchFS 后端单元测试
│   └── test_e2e_4tier.cu           # [B7] 新增: 四级存储 E2E 测试
└── third_party/
    └── OrchFS -> ../../OrchFS      # 已有符号链接
```

---

## 三、任务分解

### B0: 环境准备（运维，非代码）

**目标**: DRAM 模拟 NVM 就绪，OrchFS 编译通过并可运行。

**验收**: `ls /dev/dax0.0` 成功；OrchFS `kfs_main` 运行；简单 pwrite/pread 测试通过。

**估时**: 0.5~1 天（含重启时间）

**⚠️ 需要 root 权限和机器重启**

---

### B1: OrchFS 后端适配层 — 声明 (`orchfs_tier.h`)

**目标**: 定义 `orchfs_tier_t` 结构体和 API，封装 OrchFS 的文件操作。

**核心数据结构**:

```c
typedef struct orchfs_tier {
    bool         initialized;
    char         base_dir[256];      /* OrchFS 根目录, e.g. "/Or/kvcache" */
    size_t       slab_size;          /* = KV Block 大小 (32KB) */

    /* IO 统计 */
    uint64_t     total_writes;
    uint64_t     total_reads;
    uint64_t     bytes_written;
    uint64_t     bytes_read;

    pthread_mutex_t lock;
} orchfs_tier_t;

/*
 * 每个请求在 OrchFS 上对应一个文件。
 * 文件内布局:
 *   offset = (layer * n_kv_heads + head) * max_blocks_per_head * slab_size
 *            + block_idx * slab_size
 *
 * 这样保证同一 layer+head 的 blocks 在文件内连续,
 * 利于 OrchFS 的顺序写优化和 SSD 块对齐。
 */
typedef struct orchfs_file_ctx {
    int          fd;                 /* orchfs_open 返回的 fd */
    char         path[256];          /* 文件路径 */
    uint64_t     request_id;
    uint32_t     n_layers;
    uint32_t     n_kv_heads;
    uint32_t     max_blocks_per_head; /* 预估最大 block 数 */
    size_t       slab_size;
} orchfs_file_ctx_t;
```

**API 声明**:

| 函数 | 说明 |
|------|------|
| `orchfs_tier_init(tier, base_dir, slab_size)` | 初始化 OrchFS 后端 (调用 init_libfs) |
| `orchfs_tier_destroy(tier)` | 关闭后端 (调用 close_libfs) |
| `orchfs_file_open(tier, request_id, n_layers, n_heads, max_blocks)` | 为请求创建 OrchFS 文件 |
| `orchfs_file_close(file_ctx)` | 关闭并删除请求文件 |
| `orchfs_tier_write(file_ctx, layer, head, block_idx, data, size)` | 写入 KV Block 到 OrchFS |
| `orchfs_tier_read(file_ctx, layer, head, block_idx, data, size)` | 从 OrchFS 读取 KV Block |
| `orchfs_tier_delete(file_ctx, layer, head, block_idx)` | 标记 block 无效 (逻辑删除) |

**估时**: 0.5 天

---

### B2: OrchFS 后端适配层 — 实现 (`orchfs_tier.c`)

**目标**: 实现 B1 中声明的所有函数。

**关键实现细节**:

1. **文件偏移计算**:
   ```c
   static inline int64_t compute_offset(orchfs_file_ctx_t *fctx,
                                        uint32_t layer, uint32_t head,
                                        uint32_t block_idx)
   {
       uint32_t head_linear = layer * fctx->n_kv_heads + head;
       return (int64_t)head_linear * fctx->max_blocks_per_head * fctx->slab_size
            + (int64_t)block_idx * fctx->slab_size;
   }
   ```

2. **init_libfs 调用**: `orchfs_tier_init` 时调用一次 `init_libfs()`，连接 KernelFS 守护进程。整个进程生命周期只调用一次。

3. **路径命名**: `/Or/kvcache/req_<request_id>` — 每个请求一个文件。

4. **OrchFS 内部分层**: OrchFS 自动将数据放在 NVM 或 SSD 上：
   - 小数据 / 热数据 → NVM (4KB page 粒度)
   - 大数据 / 冷数据 → SSD (32KB block 粒度)
   - 我们的 slab = 32KB = ORCH_BLOCK_SIZE，OrchFS 会直接走 SSD 路径
   - 若需细粒度 NVM 存储（如 sub-block 级），需拆分为 4KB 写入

5. **错误处理**: `orchfs_pwrite/pread` 返回实际字节数；检查 != 预期大小时返回 ORCHKV_ERR_IO。

**估时**: 1 天

---

### B3: 异步 IO 工作线程池 (`io_worker.h/c`)

**目标**: 提供异步接口，避免 `orchfs_pwrite/pread` 阻塞调用方。

**设计思路**:

OrchFS 的 `orchfs_pwrite`/`orchfs_pread` 是同步调用。为了不阻塞 API 调用方（尤其是 GPU 计算线程），我们需要一个工作线程池来异步执行 OrchFS IO：

```c
typedef enum {
    IO_OP_WRITE = 0,
    IO_OP_READ  = 1,
} IOOpType;

typedef struct io_task {
    IOOpType     op;
    orchfs_file_ctx_t *fctx;
    uint32_t     layer;
    uint32_t     head;
    uint32_t     block_idx;
    void        *buf;           /* 读: 输出缓冲; 写: 输入数据 */
    size_t       size;
    void       (*callback)(int status, void *user_data);
    void        *user_data;
} io_task_t;

typedef struct io_worker_pool {
    int              num_workers;
    pthread_t       *threads;
    /* lock-free 任务队列 (或 mutex + condition variable) */
    io_task_t       *queue;
    uint32_t         queue_cap;
    uint32_t         queue_head;
    uint32_t         queue_tail;
    pthread_mutex_t  queue_lock;
    pthread_cond_t   queue_cond;
    bool             shutdown;

    /* 统计 */
    uint64_t         total_submitted;
    uint64_t         total_completed;
} io_worker_pool_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `io_worker_init(pool, num_workers, queue_cap)` | 创建工作线程池 |
| `io_worker_destroy(pool)` | 销毁线程池 (等待所有任务完成) |
| `io_worker_submit(pool, task)` | 提交异步 IO 任务 |
| `io_worker_flush(pool)` | 等待所有排队任务完成 |

**为什么不直接用 OrchFS 的 IO 线程池**: OrchFS 的线程池是内部实现，对外暴露的是同步 API。我们在 OrchFS 之上再加一层异步封装。OrchFS 内部仍然会用自己的 NVM/SSD 线程池执行实际 IO。

**估时**: 1 天

---

### B4: kv_request 扩展 — OrchFS 文件关联

**目标**: 在 `kv_request_ctx_t` 中增加 OrchFS 文件上下文，使每个请求可以将 KV Block 持久化。

**修改内容**:

1. `kv_types.h`: `orchkv_config_t` 新增字段：
   ```c
   /* OrchFS (Phase B) */
   const char  *orchfs_base_dir;       /* e.g. "/Or/kvcache" */
   int          orchfs_io_workers;     /* IO 工作线程数, 默认 4 */
   uint32_t     max_blocks_per_head;   /* 预估最大 block 数, 默认 256 */
   ```

2. `kv_request_ctx_t`: 新增字段：
   ```c
   orchfs_file_ctx_t *orchfs_fctx;    /* OrchFS 文件句柄 (Phase B) */
   uint64_t           blocks_on_storage;  /* NVM/SSD 上的 block 数 */
   ```

3. `kv_block_t`: `persistent_offset` 字段在 Phase A 已预留，Phase B 启用：
   - 当 block 写入 OrchFS 后，记录文件内偏移
   - 用于后续读回

**估时**: 0.5 天

---

### B5: API 层扩展 — 四级存储操作

**目标**: 在 `orchkv_api` 中新增 DRAM ↔ OrchFS 的 evict/promote 路径。

**新增 API**:

```c
/*
 * Evict to storage: DRAM block → OrchFS (NVM/SSD)
 * 前置条件: block 当前在 TIER_HOST_DRAM
 * 后置条件: block 在 TIER_NVM 或 TIER_SSD (由 OrchFS 决定),
 *           data_ptr = NULL, persistent_offset 已设置
 */
int orchkv_evict_to_storage(kv_request_ctx_t *ctx,
                            uint32_t layer_id,
                            uint32_t head_id,
                            uint32_t block_idx);

/*
 * Promote from storage: OrchFS → DRAM → GPU
 * 前置条件: block 当前在 TIER_NVM 或 TIER_SSD
 * 后置条件: block 在 TIER_GPU_HBM, data_ptr 指向 GPU slab
 */
int orchkv_promote_from_storage(kv_request_ctx_t *ctx,
                                uint32_t layer_id,
                                uint32_t head_id,
                                uint32_t block_idx);

/*
 * Cold evict: GPU → DRAM → OrchFS (两跳，一步到位)
 * 快捷方式: orchkv_evict_to_dram + orchkv_evict_to_storage
 */
int orchkv_evict_cold(kv_request_ctx_t *ctx,
                      uint32_t layer_id,
                      uint32_t head_id,
                      uint32_t block_idx);
```

**实现逻辑**:

```
orchkv_evict_to_storage:
  1. 检查 block.tier == TIER_HOST_DRAM
  2. 计算 OrchFS 文件偏移
  3. orchfs_pwrite(fctx->fd, block->data_ptr, slab_size, offset)
  4. dram_tier_free(block->data_ptr)
  5. block->tier = TIER_SSD  (简化: OrchFS 自动选 NVM/SSD)
  6. block->data_ptr = NULL
  7. block->persistent_offset = offset
  8. block->state = KV_STATE_COLD

orchkv_promote_from_storage:
  1. 检查 block.tier == TIER_SSD 或 TIER_NVM
  2. dram_tier_alloc → dram_ptr
  3. orchfs_pread(fctx->fd, dram_ptr, slab_size, block->persistent_offset)
  4. gpu_tier_alloc → gpu_ptr
  5. transfer_submit(gpu_ptr, dram_ptr, slab_size, H2D) + sync
  6. dram_tier_free(dram_ptr)
  7. block->tier = TIER_GPU_HBM
  8. block->data_ptr = gpu_ptr
  9. block->state = KV_STATE_HOT
```

**orchkv_request_destroy 修改**: 销毁时需额外调用 `orchfs_file_close` 删除请求文件。

**orchkv_init/shutdown 修改**: 初始化/销毁 `orchfs_tier_t` 和 `io_worker_pool_t`。

**估时**: 1.5 天

---

### B6: OrchFS 后端单元测试 (`test_orchfs_tier.c`)

**目标**: 验证 orchfs_tier 的基本 IO 正确性。

**测试用例**:

| # | 测试 | 验证内容 |
|---|------|---------|
| 1 | init/destroy | orchfs_tier 初始化和销毁不崩溃 |
| 2 | file open/close | 创建和关闭请求文件 |
| 3 | write + read | 写入 32KB 数据 → 读回 → 比对 bit-exact |
| 4 | multi-block | 连续写入多个 block → 全部读回 → 验证 |
| 5 | multi-layer-head | 多层多头写入 → 读回 → 验证偏移计算 |
| 6 | bandwidth | 连续写入 1000 个 32KB block，测量吞吐 |
| 7 | io_worker async | 提交 100 个异步写入 → flush → 全部读回验证 |

**估时**: 1 天

---

### B7: 四级存储 E2E 测试 (`test_e2e_4tier.cu`)

**目标**: 验证 GPU → DRAM → OrchFS → DRAM → GPU 完整 round-trip。

**测试用例**:

| # | 测试 | 验证内容 |
|---|------|---------|
| 1 | cold_evict_promote | prefill → evict_cold (GPU→DRAM→OrchFS) → promote_from_storage (OrchFS→DRAM→GPU) → 数据一致 |
| 2 | multi_request | 多个请求并行使用 OrchFS 后端，互不干扰 |
| 3 | mixed_tiers | 同一请求中部分 block 在 GPU、部分在 DRAM、部分在 OrchFS，stats 正确 |
| 4 | decode_with_storage | prefill → decode 100 步 → evict 旧 block 到 OrchFS → 继续 decode → promote 需要的 block |
| 5 | destroy_cleanup | 请求销毁后 OrchFS 文件已删除 |

**性能基准测试** (--bench 模式):

```
模拟 LLaMA-7B 配置 (32 层, 8 头, d_head=128, FP16):
  - Prefill 128 tokens (GPU)
  - Evict layer 0-15 所有 block 到 OrchFS (GPU → DRAM → OrchFS)
  - Promote layer 0 所有 block 回 GPU (OrchFS → DRAM → GPU)
  - 测量:
    - evict_to_storage 延迟 per block (预期: ~50-200 us)
    - promote_from_storage 延迟 per block (预期: ~100-500 us)
    - OrchFS pwrite 吞吐 (预期: 接近 Exp2-S5 数据 ~3 GB/s batch)
    - OrchFS pread 吞吐
```

**估时**: 1.5 天

---

### B8: OrchFS 集成基准测试

**目标**: 收集四级存储的完整延迟数据，为论文提供数据支撑。

**测试矩阵**:

```
存储路径                    | 操作       | 预期延迟      | 数据来源
GPU → DRAM                 | evict      | ~11 us/block  | Phase A E2E (已有)
DRAM → GPU                 | promote    | ~11 us/block  | Phase A E2E (已有)
DRAM → OrchFS (NVM-emul)   | evict      | ~20-50 us     | Phase B 新增
DRAM → OrchFS (SSD)        | evict      | ~50-200 us    | Phase B 新增
OrchFS (NVM-emul) → DRAM   | promote    | ~10-30 us     | Phase B 新增
OrchFS (SSD) → DRAM        | promote    | ~50-300 us    | Phase B 新增
GPU → DRAM → OrchFS        | cold_evict | ~30-60 us     | Phase B 新增
OrchFS → DRAM → GPU        | full_promote | ~70-350 us  | Phase B 新增
```

**注意**: 因为用 DRAM 模拟 NVM，NVM 路径的延迟会偏低（真实 NVM ~300ns vs DRAM ~100ns），论文中需注明。

**产出**: `experiments/exp2_storage_baseline/results/b_4tier_latency.json`

**估时**: 0.5 天

---

## 四、依赖关系与执行顺序

```
B0 (环境准备: DRAM→DAX, OrchFS 编译, kfs_main 启动)
 │
 ├──→ B1 (orchfs_tier.h — 声明)
 │     └──→ B2 (orchfs_tier.c — 实现)
 │           └──→ B6 (单元测试)
 │
 ├──→ B3 (io_worker.h/c — 异步 IO 线程池)
 │           │
 │           └──→ B5 (orchkv_api 扩展) ──→ B7 (E2E 测试) ──→ B8 (基准测试)
 │                     ↑
 └──→ B4 (kv_request/kv_types 扩展) ─┘
```

**推荐执行顺序**:

| Day | 任务 | 产出 |
|-----|------|------|
| Day 0 | B0: 环境准备 (需 root, 重启) | `/dev/dax0.0` 就绪, OrchFS 运行 |
| Day 1 | B1 + B2: orchfs_tier 声明与实现 | OrchFS 读写封装完成 |
| Day 2 | B3: io_worker 线程池 + B4: 扩展 kv_types/kv_request | 异步 IO + 数据结构就绪 |
| Day 3 | B5: orchkv_api 扩展 | evict_to_storage / promote_from_storage 实现 |
| Day 4 | B6: OrchFS 后端单元测试 | 7 个测试用例通过 |
| Day 5 | B7: 四级存储 E2E 测试 | 5 个正确性测试 + 基准测试通过 |
| Day 6 | B8: 基准数据收集 + work3.md 更新 | 延迟数据 JSON 产出 |

---

## 五、CMake 构建修改

Phase B 需要的 CMakeLists.txt 变更：

```cmake
# ---- 新增: OrchFS 链接 ----
set(ORCHFS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/OrchFS)
set(ORCHFS_LIB ${ORCHFS_DIR}/build/libOrchFS.so)
set(ORCHFS_INCLUDE ${ORCHFS_DIR}/LibFS)

# 检查 OrchFS 是否已编译 (Phase B 可选)
if(EXISTS ${ORCHFS_LIB})
    set(HAS_ORCHFS TRUE)
    message(STATUS "OrchFS found: ${ORCHFS_LIB}")
else()
    set(HAS_ORCHFS FALSE)
    message(STATUS "OrchFS not found, Phase B features disabled")
endif()

# 核心库: 追加源文件
add_library(orchkv STATIC
    ... (Phase A 源文件) ...
    src/tiered_store/orchfs_tier.c       # Phase B
    src/tiered_store/io_worker.c         # Phase B
)

if(HAS_ORCHFS)
    target_compile_definitions(orchkv PUBLIC ORCHKV_HAS_ORCHFS=1)
    target_include_directories(orchkv PUBLIC ${ORCHFS_INCLUDE})
    target_link_libraries(orchkv PUBLIC ${ORCHFS_LIB} pmem)
else()
    target_compile_definitions(orchkv PUBLIC ORCHKV_HAS_ORCHFS=0)
endif()

# 测试 (仅当 OrchFS 可用时编译)
if(HAS_ORCHFS)
    add_executable(test_orchfs_tier test/test_orchfs_tier.c)
    target_link_libraries(test_orchfs_tier orchkv)
    add_test(NAME test_orchfs_tier COMMAND test_orchfs_tier)

    add_executable(test_e2e_4tier test/test_e2e_4tier.cu)
    target_link_libraries(test_e2e_4tier orchkv)
    add_test(NAME test_e2e_4tier COMMAND test_e2e_4tier)
endif()
```

**条件编译策略**: 所有 OrchFS 相关代码用 `#if ORCHKV_HAS_ORCHFS` 保护。当 OrchFS 未编译时，Phase A 的功能不受影响。这样 CI/本地开发可以不依赖 OrchFS。

---

## 六、关键设计决策

### 决策 1: OrchFS 文件粒度 — 每请求一个文件

**方案 A**: 全局一个 OrchFS 文件，所有请求共享
**方案 B**: 每请求一个文件

**决策**: 方案 B。理由：
- 请求销毁时直接 `orchfs_unlink` 即可释放所有存储，无需逐 block 标记
- 文件内偏移可以预计算，无需全局分配器
- 不同请求的 IO 互不干扰
- 缺点是文件数可能较多，但 OrchFS 的 inode 管理有足够容量

### 决策 2: NVM 存还是 SSD 存 — 由 OrchFS 自动决定

**方案 A**: 我们显式控制 NVM vs SSD 放置
**方案 B**: 统一用 orchfs_pwrite，让 OrchFS 根据内部策略决定

**决策**: Phase B 先用方案 B。理由：
- OrchFS 设计就是自动管理 NVM/SSD 分层
- 我们的 slab = 32KB = ORCH_BLOCK_SIZE，OrchFS 会倾向于放 SSD
- 如果需要细粒度 NVM 控制（如 sub-block 的 4KB），留给 Phase C 优化
- 简化实现，减少对 OrchFS 内部机制的依赖

### 决策 3: Block tier 标记 — TIER_SSD 统一表示持久化

**Phase B 简化**: 不区分 TIER_NVM 和 TIER_SSD（因为 OrchFS 自动管理），统一用 `TIER_SSD` 表示数据在 OrchFS 中。
- `block->tier = TIER_SSD`
- `block->data_ptr = NULL`
- `block->persistent_offset = orchfs_file_offset`

**Phase C 可能细化**: 如果需要区分 NVM/SSD 以做更精细的调度，可以通过查询 OrchFS 的 `offset_info_t` 来判断实际存储位置。

### 决策 4: 同步 vs 异步 IO — 先同步，io_worker 封装异步

**方案 A**: 直接在调用线程执行同步 `orchfs_pwrite/pread`
**方案 B**: 通过 `io_worker` 线程池异步执行

**决策**: Phase B 实现两种模式：
- `orchkv_evict_to_storage` — 默认同步（测试和功能验证方便）
- `orchkv_evict_to_storage_async` — 通过 io_worker 异步执行 + callback
- E2E 测试先用同步模式验证正确性，基准测试用异步模式测吞吐

### 决策 5: Promote 路径 — 三阶段流水

OrchFS → DRAM → GPU 的 promote 需要两次拷贝：

```
Stage 1: orchfs_pread → DRAM buffer (同步，在 IO worker 中)
Stage 2: cudaMemcpyAsync DRAM → GPU (异步，在 CUDA stream 中)
Stage 3: release DRAM buffer
```

Phase B 实现简单的同步版本（三步串行）。Phase C 可以优化为流水线（Stage 1 和 Stage 2 重叠）。

---

## 七、Phase B 验收标准

Phase B 完成时，必须达到以下目标：

- [ ] B0: `/dev/dax0.0` 存在，OrchFS `kfs_main` 运行正常
- [ ] B0: OrchFS 简单读写测试通过
- [ ] CMake: `cmake .. && make` 在有 OrchFS 时编译通过，无 OrchFS 时 Phase A 功能不受影响
- [ ] `test_orchfs_tier` 通过:
  - init/destroy 正常
  - 写入 32KB → 读回 → bit-exact
  - 多 block 多层多头偏移计算正确
  - IO worker 异步提交 + flush
- [ ] `test_e2e_4tier` 通过:
  - GPU → DRAM → OrchFS → DRAM → GPU round-trip 数据一致
  - 多请求并行不干扰
  - 混合 tier 状态正确
  - decode + storage evict + promote 流程
  - 请求销毁清理正确
- [ ] 基准测试:
  - evict_to_storage 延迟 < 500 us/block (32KB)
  - promote_from_storage 延迟 < 1 ms/block (含 OrchFS read + H2D transfer)
  - 批量 IO 吞吐接近 Exp2-S5 数据 (~1-3 GB/s)
- [ ] 数据保存: `experiments/exp2_storage_baseline/results/b_4tier_latency.json`

---

## 八、Phase B → Phase C 衔接

Phase B 完成后，Phase C（冷热分级与调度）的入口是：

1. **C1: attention_tracker** — 在 decode 循环中采集注意力分数
2. **C2: hotcold_classifier** — 根据 attention + recency + frequency 计算热度
3. **C3: adaptive_threshold** — 动态调整 hot/warm/cold 阈值
4. **C4: eviction_policy** — 自动触发 evict（不再需要手动调用 `orchkv_evict_*`）
5. **C5: prefetch_scheduler** — 预测并提前 promote

Phase B 的 `evict_to_storage` / `promote_from_storage` 是 Phase C 自动调度器调用的底层原语。Phase B 只提供手动接口；Phase C 让它们自动化。

---

## 九、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| DRAM 模拟 NVM 延迟不真实 | 论文数据可信度 | 论文中注明使用模拟，提供真实 NVM 参数的外推分析 |
| OrchFS kfs_main 不稳定 | 测试中断 | 编写 restart 脚本，测试前检查守护进程状态 |
| OrchFS 在高并发下的性能 | 吞吐受限 | 先单线程验证正确性，再逐步增加并发 |
| GRUB memmap 配置错误 | 系统无法启动 | 保留一个正常启动项，测试机上操作 |
| libpmem 版本兼容性 | 编译失败 | 记录 Ubuntu 版本和 libpmem 版本，固定依赖 |

---

## 十、TODO 清单

```
Phase B 总览:
  [B0] 环境准备: DRAM→DAX, OrchFS 编译, kfs_main 启动        ← 起点 (需 root + 重启)
  [B1] orchfs_tier.h — OrchFS 后端声明
  [B2] orchfs_tier.c — OrchFS 后端实现
  [B3] io_worker.h/c — 异步 IO 工作线程池
  [B4] kv_types/kv_request 扩展 — OrchFS 配置字段、文件句柄
  [B5] orchkv_api 扩展 — evict_to_storage / promote_from_storage / evict_cold
  [B6] test_orchfs_tier.c — 后端单元测试 (7 个用例)
  [B7] test_e2e_4tier.cu — 四级存储 E2E 测试 (5 个用例 + 基准)
  [B8] 基准数据收集 — 四级存储延迟 JSON
```

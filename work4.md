# Work4: Phase C — 冷热分级与自动调度

> **前置已完成**: Phase B (四级存储后端 B0-B8, 8/8 ctest pass)
> **本阶段目标**: 实现 KV Block 热度感知 + 自动换入换出 + 预取调度，让系统从"手动迁移"升级为"全自动分级存储管理"
> **预计工期**: 3 周
> **论文价值**: Phase C 是本文最核心的 contribution，对应论文 §4.2 冷热分级算法、§4.3 迁移引擎、§4.4 预取流水线

---

## 〇、Phase B 已有的底层原语

Phase C 直接复用以下 Phase B 接口，**不修改**：

| 函数 | 作用 |
|------|------|
| `orchkv_evict_to_dram(ctx, l, h, bi)` | GPU → DRAM（手动） |
| `orchkv_evict_to_storage(ctx, l, h, bi)` | DRAM → Storage（手动） |
| `orchkv_evict_cold(ctx, l, h, bi)` | GPU → Storage（两跳，手动） |
| `orchkv_promote_from_storage(ctx, l, h, bi)` | Storage → DRAM（手动） |
| `orchkv_promote_to_gpu(ctx, l, h, bi)` | DRAM → GPU（手动） |
| `io_worker_submit / flush` | 异步 IO 线程池 |

Phase C 的任务：在这些原语之上构建**自动决策层**，不再需要调用者手动指定迁移哪个 block。

---

## 一、Phase C 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    orchkv_api (Phase A/B 接口保持不变)            │
├─────────────────────────────────────────────────────────────────┤
│              tiered_manager  [C8] ← 总控调度循环                  │
│       ┌──────────┬──────────────────┬──────────────────┐        │
│  attention_  hotcold_          adaptive_         eviction_       │
│  tracker[C1] classifier[C2]   threshold[C3]     policy[C4]      │
│       └──────────┴──────────────────┴──────────────────┘        │
│              prefetch_scheduler [C5]                             │
│              migration_engine   [C7]  (含流水线 C6)              │
├─────────────────────────────────────────────────────────────────┤
│        Phase B 底层原语: orchfs_tier / io_worker / transfer       │
└─────────────────────────────────────────────────────────────────┘
```

**数据流**：
```
decode step N:
  1. GPU 计算 attention → attention_tracker 采集分数
  2. hotcold_classifier 更新热度 → adaptive_threshold 更新阈值
  3. tiered_manager 周期检查水位:
       GPU  > HWM → eviction_policy 选出最冷 block → migration_engine.demote
       DRAM > HWM → eviction_policy 选出最冷 block → migration_engine.demote_to_storage
  4. prefetch_scheduler 预测下一步需要的 block → migration_engine.promote
```

---

## 二、目录结构变更

Phase C 新增/修改的文件：

```
OrchKvCache/
├── src/
│   ├── scheduler/                    # Phase C 全部在此新目录
│   │   ├── attention_tracker.h       # [C1] 注意力分数采集
│   │   ├── attention_tracker.c
│   │   ├── hotcold_classifier.h      # [C2] 热度计算与三级分类
│   │   ├── hotcold_classifier.c
│   │   ├── adaptive_threshold.h      # [C3] 动态阈值调整
│   │   ├── adaptive_threshold.c
│   │   ├── eviction_policy.h         # [C4] 换出候选选择
│   │   ├── eviction_policy.c
│   │   ├── prefetch_scheduler.h      # [C5] 预取调度
│   │   ├── prefetch_scheduler.c
│   │   ├── migration_engine.h        # [C7] 迁移引擎总控
│   │   ├── migration_engine.c
│   │   ├── tiered_manager.h          # [C8] 分层管理器总控
│   │   └── tiered_manager.c
│   └── api/
│       └── orchkv_api.h/cu           # [C8] 修改: 集成 tiered_manager
├── test/
│   ├── test_hotcold.c                # [C9] 热度分级单元测试
│   ├── test_prefetch.c               # [C10] 预取调度单元测试
│   └── test_e2e_auto.cu              # [C11] 自动调度 E2E 测试
└── CMakeLists.txt                    # 修改: 加入 scheduler/ 源文件
```

---

## 三、任务分解

### C1: 注意力分数采集器 (`attention_tracker.h/c`)

**目标**: 在每次 decode step 中，异步采集各 block 的注意力权重，为热度分类提供原始数据。

**核心设计**:

```c
/*
 * 每个 KV block 的注意力统计（按 block 粒度聚合）
 * 每次 decode step 后更新
 */
typedef struct attn_stats {
    float    sum;        /* 该 block 内所有 token 被 query 到的注意力权重之和 */
    float    max;        /* 该 block 内最大单 token 注意力权重 */
    uint32_t query_hits; /* 被非零注意力 query 到的次数（步数） */
    uint64_t last_hit_step;  /* 上次被访问的 decode step */
} attn_stats_t;

typedef struct attention_tracker {
    attn_stats_t  *stats;          /* 按 block_id 索引 */
    uint32_t       capacity;
    uint64_t       current_step;
    pthread_mutex_t lock;
} attention_tracker_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `attn_tracker_init(t, capacity)` | 初始化，capacity = 最大 block 数 |
| `attn_tracker_destroy(t)` | 销毁 |
| `attn_tracker_update(t, block_id, attn_weight)` | 单 block 更新（decode step 中调用） |
| `attn_tracker_step_done(t)` | 标记一个 decode step 完成，推进 current_step |
| `attn_tracker_get(t, block_id, stats_out)` | 查询 block 的注意力统计 |
| `attn_tracker_reset(t, block_id)` | 清零（block 被驱逐后调用） |

**实现要点**:
- 注意力矩阵在 GPU 上，拷贝到 CPU 需要 `cudaMemcpyAsync` + 独立 CUDA stream（不阻塞计算流）
- 按 block 粒度聚合：`attn_weight[block] = sum(attn_score[token_start:token_end])`
- 使用环形缓冲暂存异步拷贝结果，主线程定期 flush

**依赖**: `kv_block.h`, CUDA runtime

**估时**: 1.5 天

---

### C2: 热度计算与分类器 (`hotcold_classifier.h/c`)

**目标**: 根据注意力分数、时间衰减、访问频次，计算每个 block 的综合热度分数，并将其分为 Hot / Warm / Cold 三级。

**热度公式**:

```
hotness(b, t) = α × attn_ema(b, t)
              + β × recency_score(b, t)
              + γ × frequency_score(b, t)

其中:
  attn_ema(b, t)      = EMA 滑动平均注意力权重 (λ=0.9)
  recency_score(b, t) = exp(-(t - last_hit_step) / τ)  (τ=50 steps)
  frequency_score(b,t)= min(hit_count / N_steps, 1.0)

默认: α=0.5, β=0.3, γ=0.2
```

**分级规则**:

```
hotness ≥ threshold_hot   → Hot  (留 GPU HBM)
hotness ≥ threshold_warm  → Warm (降到 DRAM 或 NVM)
hotness <  threshold_warm → Cold (降到 SSD)

注意: KV_FLAG_ATTN_SINK 标记的 block 永久 Hot（attention sink token）
```

**核心数据结构**:

```c
typedef enum {
    HEAT_HOT  = 0,
    HEAT_WARM = 1,
    HEAT_COLD = 2,
} HeatLevel;

typedef struct hotcold_classifier {
    float    *hotness;       /* 每个 block 的热度分数 */
    HeatLevel *heat_level;  /* 每个 block 的分级结果 */
    uint32_t  capacity;

    float alpha, beta, gamma;
    float ema_lambda;
    float recency_tau;

    attention_tracker_t *tracker;  /* 引用 C1 */
    pthread_mutex_t lock;
} hotcold_classifier_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `hcc_init(c, capacity, tracker, alpha, beta, gamma)` | 初始化 |
| `hcc_destroy(c)` | 销毁 |
| `hcc_update_all(c)` | 根据最新注意力统计更新所有 block 热度（每 decode step 调用） |
| `hcc_get_heat(c, block_id)` | 获取 block 的 HeatLevel |
| `hcc_get_score(c, block_id)` | 获取 block 的 hotness 分数 |
| `hcc_classify(c, block_id, threshold_hot, threshold_warm)` | 单 block 重新分级 |

**实现要点**:
- EMA 更新：`attn_ema = λ × attn_ema_prev + (1-λ) × new_attn`
- Recency 使用指数衰减，避免频繁的浮点 exp 计算（预计算衰减表）
- `hcc_update_all` 是 O(N_blocks)，须在非关键路径上调用

**依赖**: `attention_tracker.h`

**估时**: 1 天

---

### C3: 动态阈值调整器 (`adaptive_threshold.h/c`)

**目标**: 根据各存储层的实时使用率，动态调整 Hot/Warm/Cold 的分级阈值，实现水位线控制。

**水位线机制**:

```
GPU HBM:
  used > HWM (90%) → 触发 demote：将热度最低的 Hot block 降为 Warm
  used < LWM (70%) → 停止 demote

Host DRAM:
  used > HWM (90%) → 触发 demote：将热度最低的 Warm block 降为 Cold
  used < LWM (70%) → 停止 demote
```

**阈值自适应策略**:

```
若 GPU demote 频繁 (>K 次/秒) 且 GPU 使用率 < 80%:
    → threshold_hot 适当调低（允许更多 block 留在 GPU）
若 GPU 使用率持续 > 95%:
    → threshold_hot 适当调高（更激进地驱逐）
```

**核心数据结构**:

```c
typedef struct adaptive_threshold {
    float  threshold_hot;     /* hotness >= 此值 → Hot */
    float  threshold_warm;    /* hotness >= 此值 → Warm */

    float  min_hot,  max_hot;   /* 阈值上下限（防止极端值）*/
    float  min_warm, max_warm;

    /* 水位线 */
    float  gpu_hwm,  gpu_lwm;
    float  dram_hwm, dram_lwm;

    /* 调整步长 */
    float  adjust_step;
    float  adjust_cooldown_s;   /* 最小调整间隔（避免震荡）*/
    double last_adjust_time;

    hotcold_classifier_t *classifier;  /* 引用 C2 */
} adaptive_threshold_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `athresh_init(t, classifier, gpu_hwm, gpu_lwm, dram_hwm, dram_lwm)` | 初始化 |
| `athresh_destroy(t)` | 销毁 |
| `athresh_update(t, gpu_used_ratio, dram_used_ratio)` | 根据当前使用率更新阈值 |
| `athresh_get_hot(t)` / `athresh_get_warm(t)` | 获取当前阈值 |
| `athresh_should_demote_gpu(t, gpu_used)` | 返回是否需要触发 GPU 换出 |
| `athresh_should_demote_dram(t, dram_used)` | 返回是否需要触发 DRAM 换出 |

**估时**: 0.5 天

---

### C4: 换出候选选择策略 (`eviction_policy.h/c`)

**目标**: 当需要换出时，从当前层快速选出最合适的 N 个 block 候选，综合热度分数和 LRU 信息。

**选择策略（热度 + LRU 混合）**:

```
eviction_score(b) = (1 - hotness(b)) × w_heat
                  + lru_rank(b) / N_blocks × w_lru

选出 eviction_score 最高的 K 个 block
默认: w_heat=0.7, w_lru=0.3
```

**核心数据结构**:

```c
typedef struct eviction_policy {
    float w_heat;
    float w_lru;
    uint32_t batch_size;  /* 每次批量换出的 block 数（默认 8）*/

    hotcold_classifier_t *classifier;
    /* LRU 双向链表（按 last_access_step 排序）*/
    kv_block_t *lru_head;
    kv_block_t *lru_tail;
    pthread_mutex_t lru_lock;
} eviction_policy_t;

typedef struct eviction_candidate {
    kv_block_t *block;
    uint64_t    request_id;
    uint32_t    layer, head, block_idx;
    float       score;
} eviction_candidate_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `evpol_init(p, classifier, batch_size, w_heat, w_lru)` | 初始化 |
| `evpol_destroy(p)` | 销毁 |
| `evpol_lru_touch(p, block)` | block 被访问时更新 LRU（promote 或 get_kv 时调用） |
| `evpol_lru_remove(p, block)` | block 被驱逐时从 LRU 链表删除 |
| `evpol_select_gpu_victims(p, n, out_candidates)` | 选出 n 个 GPU 上待驱逐 block |
| `evpol_select_dram_victims(p, n, out_candidates)` | 选出 n 个 DRAM 上待驱逐 block |

**实现要点**:
- LRU 链表用双向链表维护，O(1) 移动节点
- 候选选择扫描 LRU 尾部 2N 个 block，从中按 eviction_score 排序取前 N 个
- 被 `KV_FLAG_PIN` 标记的 block 跳过

**估时**: 1 天

---

### C5: 预取调度器 (`prefetch_scheduler.h/c`)

**目标**: 预测哪些 block 即将被访问，提前从 Storage → DRAM（或 DRAM → GPU），通过 IO 与计算的重叠隐藏换入延迟。

**预取策略（基于注意力模式）**:

Decode step 中注意力分数的分布具有**空间局部性**：
- 若 block B 在当前 step 被高注意力访问，则在 **接下来 W 步内** B 仍可能被访问
- 若 B 的相邻 block（同 layer、同 head、相邻 block_idx）最近有访问记录，也预取

```
预取触发规则:
  对于每个 DRAM 或 Storage 上的 block B:
    if attn_ema(B) > threshold_prefetch × 0.5:
      提交异步 promote(B, target=DRAM)    // Storage → DRAM
    if attn_ema(B) > threshold_prefetch:
      提交异步 promote(B, target=GPU)     // DRAM → GPU（较激进）

  预取数量上限: 每 decode step 最多 prefetch_budget 个 block
```

**核心数据结构**:

```c
typedef struct prefetch_entry {
    kv_block_t  *block;
    uint64_t     request_id;
    uint32_t     layer, head, block_idx;
    StorageTier  target_tier;
    float        priority;
} prefetch_entry_t;

typedef struct prefetch_scheduler {
    /* 优先级队列（max-heap by priority）*/
    prefetch_entry_t *heap;
    uint32_t          heap_size;
    uint32_t          heap_cap;

    uint32_t  prefetch_budget;    /* 每 step 最多预取数 */
    float     threshold_to_dram;  /* attn_ema 触发预取到 DRAM 的阈值 */
    float     threshold_to_gpu;   /* attn_ema 触发预取到 GPU 的阈值 */

    hotcold_classifier_t *classifier;
    io_worker_pool_t     *io_pool;

    /* 统计 */
    uint64_t  total_prefetched;
    uint64_t  prefetch_hits;    /* 预取后实际被访问的次数 */
    uint64_t  prefetch_wasted;  /* 预取后未被访问直接被驱逐 */

    pthread_mutex_t lock;
} prefetch_scheduler_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `prefetch_init(s, classifier, io_pool, budget)` | 初始化 |
| `prefetch_destroy(s)` | 销毁，flush 所有排队预取任务 |
| `prefetch_scan(s, ctx)` | 扫描 request 中所有 block，将候选加入优先级队列 |
| `prefetch_dispatch(s, n_blocks)` | 提交最多 n_blocks 个预取任务到 io_worker |
| `prefetch_notify_hit(s, block_id)` | block 被实际访问，更新命中统计 |
| `prefetch_hit_rate(s)` | 返回预取命中率 |

**估时**: 1.5 天

---

### C6: IO-计算重叠流水线（集成到 C7 migration_engine）

**目标**: 将 Storage → DRAM 的 IO 与 GPU 计算**并行执行**，使 IO 延迟对推理步长不可见。

**三阶段流水线设计**:

```
Decode Step N:
  ├─ Stage A (GPU Compute): attention(Q, K[N-1], V[N-1])   → 产出注意力分数
  │    ↓（异步）
  ├─ Stage B (IO Prefetch): Storage → DRAM for step N+1   ← prefetch_scheduler
  │    └─（与 Stage A 并发执行）
  └─ Stage C (H2D Transfer): DRAM → GPU for blocks needed  ← transfer_engine

Decode Step N+1:
  ├─ Stage A: attention(Q, K[N], V[N])
  │    ...依赖 Stage C(N) 的结果
```

**实现方式**:
- Stage B 用 `io_worker` 线程池，完全独立于 CUDA stream
- Stage C 用独立 CUDA stream（与计算 stream 并发）
- tiered_manager 每步发起 Stage B，确保下一步 block 提前到位

---

### C7: 迁移引擎总控 (`migration_engine.h/c`)

**目标**: 统一封装 Demote（换出）和 Promote（换入）两个方向的迁移，含重试逻辑和并发安全。

**核心数据结构**:

```c
typedef enum {
    MIGRATE_DEMOTE_GPU2DRAM = 0,  /* GPU → DRAM */
    MIGRATE_DEMOTE_DRAM2STOR,     /* DRAM → Storage */
    MIGRATE_DEMOTE_GPU2STOR,      /* GPU → Storage (two-hop) */
    MIGRATE_PROMOTE_STOR2DRAM,    /* Storage → DRAM */
    MIGRATE_PROMOTE_DRAM2GPU,     /* DRAM → GPU */
    MIGRATE_PROMOTE_STOR2GPU,     /* Storage → GPU (two-hop) */
} MigrateOp;

typedef struct migration_engine {
    eviction_policy_t    *evpol;
    prefetch_scheduler_t *prefetch;
    io_worker_pool_t     *io_pool;
    transfer_engine_t    *xfer;

    /* 统计 */
    uint64_t demote_count[4];
    uint64_t promote_count[4];
    uint64_t migrate_errors;

    pthread_mutex_t lock;
} migration_engine_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `mig_init(e, evpol, prefetch, io_pool, xfer)` | 初始化 |
| `mig_destroy(e)` | 销毁 |
| `mig_demote(e, ctx, candidates, n)` | 批量执行换出，选 op 由 block.tier 决定 |
| `mig_promote(e, ctx, block, target_tier)` | 执行换入 |
| `mig_promote_async(e, ctx, block, target_tier, cb, user_data)` | 异步换入 |
| `mig_flush(e)` | 等待所有异步任务完成 |

**实现要点**:
- `mig_demote` 根据 block 当前 tier 自动选择路径（GPU → DRAM 还是直接 GPU → Storage）
- `mig_promote_async` 通过 `io_worker` 提交，callback 中执行 H2D transfer
- block state 在迁移全过程中设为 `KV_STATE_MIGRATING`，防止重复迁移

**估时**: 1.5 天

---

### C8: 分层存储管理器总控 (`tiered_manager.h/c`)

**目标**: 整合 C1-C7 所有组件，提供统一的生命周期管理和调度循环，成为 Phase C 对外的唯一接口。

**核心设计**:

```c
typedef struct tiered_manager {
    bool                  running;
    pthread_t             scheduler_thread;

    attention_tracker_t   tracker;
    hotcold_classifier_t  classifier;
    adaptive_threshold_t  threshold;
    eviction_policy_t     evpol;
    prefetch_scheduler_t  prefetch;
    migration_engine_t    migration;

    /* 调度参数 */
    uint32_t  schedule_interval_us;  /* 调度循环间隔（默认 1ms）*/
    uint32_t  demote_batch_size;     /* 每次最多换出 block 数 */
    uint32_t  prefetch_batch_size;   /* 每次最多预取 block 数 */

    /* 全局请求表（需要遍历所有活跃请求的 block）*/
    kv_request_ctx_t **active_requests;
    uint32_t           n_active;
    uint32_t           max_requests;
    pthread_rwlock_t   req_lock;

    /* 统计快照 */
    orchkv_stats_t     last_stats;
} tiered_manager_t;
```

**API**:

| 函数 | 说明 |
|------|------|
| `tm_init(m, cfg)` | 初始化所有子系统，启动调度线程 |
| `tm_destroy(m)` | 停止调度线程，销毁所有子系统 |
| `tm_register_request(m, ctx)` | 新请求注册到管理器 |
| `tm_unregister_request(m, ctx)` | 请求结束时注销 |
| `tm_notify_attn(m, ctx, layer, head, bi, score)` | decode step 中上报注意力分数 |
| `tm_notify_access(m, block)` | block 被访问时通知（更新 LRU）|
| `tm_get_stats(m, stats_out)` | 获取调度统计 |
| `tm_set_policy(m, alpha, beta, gamma)` | 运行时调整热度权重 |

**调度循环逻辑**:

```c
while (running) {
    usleep(schedule_interval_us);

    // 1. 更新热度
    hcc_update_all(&m->classifier);

    // 2. 更新阈值
    athresh_update(&m->threshold,
                   gpu_used_ratio, dram_used_ratio);

    // 3. GPU 水位检查 → 换出
    if (athresh_should_demote_gpu(...)) {
        n = evpol_select_gpu_victims(..., demote_batch_size, candidates);
        mig_demote(&m->migration, ctx, candidates, n);
    }

    // 4. DRAM 水位检查 → 换出到 Storage
    if (athresh_should_demote_dram(...)) {
        n = evpol_select_dram_victims(..., demote_batch_size, candidates);
        mig_demote(&m->migration, ctx, candidates, n);
    }

    // 5. 预取候选扫描 + 提交
    for each active_request:
        prefetch_scan(&m->prefetch, ctx);
    prefetch_dispatch(&m->prefetch, prefetch_batch_size);
}
```

**与 orchkv_api 的集成**:
- `orchkv_init` 中初始化 `tiered_manager`
- `orchkv_request_create` 调用 `tm_register_request`
- `orchkv_request_destroy` 调用 `tm_unregister_request`
- `orchkv_get_kv_block` 调用 `tm_notify_access`（更新 LRU）
- 新增 `orchkv_report_attn(ctx, layer, head, bi, score)` 供推理引擎上报注意力

**估时**: 2 天

---

### C9: 冷热分级器单元测试 (`test_hotcold.c`)

**测试用例**:

| 测试 | 内容 |
|------|------|
| `test_attn_tracker_basic` | 单 block 更新 + 查询 |
| `test_attn_tracker_ema` | EMA 权重衰减是否正确 |
| `test_classifier_3level` | 热度分数落在三个区间时分级结果 |
| `test_classifier_attn_sink` | KV_FLAG_ATTN_SINK block 永久 Hot |
| `test_adaptive_threshold_hwm` | 超过 HWM 时阈值调高 |
| `test_adaptive_threshold_lwm` | 低于 LWM 时阈值调低 |
| `test_eviction_select_n` | 选出 N 个候选，结果按 eviction_score 降序 |
| `test_eviction_skip_pinned` | KV_FLAG_PIN 的 block 不被选为候选 |
| `test_lru_touch` | touch 后 block 移到 LRU 头部 |

**估时**: 0.5 天

---

### C10: 预取调度器单元测试 (`test_prefetch.c`)

**测试用例**:

| 测试 | 内容 |
|------|------|
| `test_prefetch_scan_empty` | 无候选时 heap 为空 |
| `test_prefetch_priority` | 高热度 block 先被预取 |
| `test_prefetch_budget` | 不超过 prefetch_budget 限制 |
| `test_prefetch_hit_rate` | notify_hit 后命中率统计正确 |
| `test_prefetch_skip_gpu` | 已在 GPU 的 block 不重复预取 |
| `test_prefetch_async` | 异步预取完成后 block 升到目标 tier |

**估时**: 0.5 天

---

### C11: 自动调度 E2E 测试 (`test_e2e_auto.cu`)

**测试场景**:

模拟真实 decode 循环，验证自动调度行为：

| 测试 | 场景 | 验收标准 |
|------|------|---------|
| `test_auto_evict_gpu` | GPU 填满后自动换出到 DRAM | GPU 使用率稳定在 HWM 以下 |
| `test_auto_evict_dram` | DRAM 填满后自动换出到 Storage | DRAM 使用率稳定在 HWM 以下 |
| `test_auto_prefetch` | 注意力高的 block 被提前 promote | prefetch 命中率 > 60% |
| `test_data_integrity` | 经过多次自动 evict + promote 后数据 bit-exact | 所有数据正确 |
| `test_multi_request` | 多个并发请求的 block 互不干扰 | 无数据污染 |

**Benchmark** (输出到 JSON):
- decode 循环中各步平均延迟（含 auto-evict + prefetch 开销）
- tiered_manager 调度开销（CPU time）
- 预取命中率
- 各层存储使用率曲线

**估时**: 1.5 天

---

## 四、依赖关系与执行顺序

```
C1 (attention_tracker)
  └─ C2 (hotcold_classifier)
       ├─ C3 (adaptive_threshold)
       └─ C4 (eviction_policy)
            └─ C7 (migration_engine) ←── C5 (prefetch_scheduler)
                 └─ C8 (tiered_manager)
                      ├─ C9 (test_hotcold)
                      ├─ C10 (test_prefetch)
                      └─ C11 (test_e2e_auto)
```

**推荐执行顺序**: C1 → C2 → C3 → C4 → C5 → C7 → C8 → C9 → C10 → C11

C6（IO-计算重叠流水线）不是独立模块，其实现集成在 C7 的异步路径中。

---

## 五、CMakeLists.txt 修改

在现有基础上添加：

```cmake
# Phase C: scheduler 模块
add_library(orchkv_scheduler STATIC
    src/scheduler/attention_tracker.c
    src/scheduler/hotcold_classifier.c
    src/scheduler/adaptive_threshold.c
    src/scheduler/eviction_policy.c
    src/scheduler/prefetch_scheduler.c
    src/scheduler/migration_engine.c
    src/scheduler/tiered_manager.c
)
target_include_directories(orchkv_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(orchkv_scheduler PUBLIC orchkv pthread)

# Phase C 新增测试
add_executable(test_hotcold test/test_hotcold.c)
target_link_libraries(test_hotcold orchkv_scheduler)
add_test(NAME test_hotcold COMMAND test_hotcold)

add_executable(test_prefetch test/test_prefetch.c)
target_link_libraries(test_prefetch orchkv_scheduler)
add_test(NAME test_prefetch COMMAND test_prefetch)

add_executable(test_e2e_auto test/test_e2e_auto.cu)
target_link_libraries(test_e2e_auto orchkv_scheduler)
add_test(NAME test_e2e_auto COMMAND test_e2e_auto)
```

---

## 六、关键设计决策

### 决策 1: 热度公式的参数默认值

| 参数 | 默认值 | 调整范围 | 影响 |
|------|--------|---------|------|
| α (attention 权重) | 0.5 | [0.3, 0.7] | 越高越依赖注意力分数 |
| β (recency 权重) | 0.3 | [0.1, 0.5] | 越高越偏向 LRU 行为 |
| γ (frequency 权重) | 0.2 | [0.1, 0.4] | 越高越偏向 LFU 行为 |
| λ (EMA 衰减) | 0.9 | [0.8, 0.99] | 越大历史权重越重 |
| τ (recency 时间常数) | 50 steps | [20, 200] | 越小越快遗忘 |

参数将在 E5 消融实验中系统验证。

### 决策 2: 调度线程间隔

默认 1ms 调度间隔，理由：
- decode step 的典型耗时 10~50ms，1ms 调度足够及时
- CPU 开销可接受（< 1% of decode time）
- 可通过配置调整：`cfg.schedule_interval_us`

### 决策 3: 预取激进程度

Phase C 实现**保守预取**（只预取到 DRAM，不主动预取到 GPU），原因：
- GPU 内存珍贵，避免预取造成有效数据被换出
- 到 GPU 的升级由 `get_kv_block` 按需触发
- Phase D 集成 vLLM 后可实现真正的 prefill/decode 感知预取

### 决策 4: 与 Phase A/B API 的向后兼容

Phase C 不破坏已有接口：
- Phase A/B 的手动 evict/promote 仍然有效
- 新增 `orchkv_report_attn` 作为可选接口（不调用时退化为纯 LRU 策略）
- `orchkv_init` 扩展参数兼容旧的 `orchkv_config_default()` 调用

---

## 七、Phase C 验收标准

- [ ] **C1**: attention_tracker 异步采集，每步延迟 < 0.1ms
- [ ] **C2**: hotcold_classifier 分级正确，覆盖三级 + attn_sink 特殊处理
- [ ] **C3**: adaptive_threshold 在压力测试中 GPU 使用率保持在 HWM 以下
- [ ] **C4**: eviction_policy 不选 pinned block，候选按 score 正确排序
- [ ] **C5**: prefetch 命中率 > 50%（合成 decode 序列测试）
- [ ] **C7**: migration_engine 无数据损坏，并发迁移无死锁
- [ ] **C8**: tiered_manager 调度线程 CPU 开销 < 2% of decode time
- [ ] **C9**: test_hotcold 全部通过
- [ ] **C10**: test_prefetch 全部通过
- [ ] **C11**: test_e2e_auto 全部通过，数据 bit-exact，benchmark 数据保存到 JSON

---

## 八、Phase C → Phase D 衔接

Phase C 完成后，Phase D（vLLM 推理引擎集成）的入口是：

1. **D1: Python binding** — 通过 pybind11 暴露 `orchkv_init` / `orchkv_request_create` / `orchkv_report_attn` / `orchkv_get_kv_block`
2. **D2: vLLM BlockManager** — 替换 `BlockSpaceManager`，将 vLLM 的 `PhysicalTokenBlock` 映射到 `kv_block_t`
3. **D3: Attention Hook** — 在 vLLM 的 attention kernel 后插入 `tm_notify_attn`，采集真实注意力分数
4. **D4: 端到端验证** — 在真实模型 (LLaMA-2-7B) 上验证生成质量和性能

Phase C 的 `tiered_manager` 是 Phase D vLLM 集成的核心对接点：vLLM 只需调用 `tm_register_request` / `tm_notify_attn` / `tm_get_stats`，其余调度全部自动。

---

## 九、TODO 清单

```
Phase C 总览:
  [C1] attention_tracker.h/c — 注意力分数异步采集          ← 起点
  [C2] hotcold_classifier.h/c — 热度计算 + 三级分类
  [C3] adaptive_threshold.h/c — 动态水位阈值调整
  [C4] eviction_policy.h/c — 换出候选选择（热度+LRU）
  [C5] prefetch_scheduler.h/c — 预取优先级队列
  [C7] migration_engine.h/c — 迁移总控（含 IO-计算流水）
  [C8] tiered_manager.h/c — 调度器总控 + orchkv_api 集成
  [C9] test_hotcold.c — 分级器单元测试 (9 个用例)
  [C10] test_prefetch.c — 预取器单元测试 (6 个用例)
  [C11] test_e2e_auto.cu — 自动调度 E2E 测试 + benchmark
```

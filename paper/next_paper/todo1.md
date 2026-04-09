# OrchKvCache-Scale: Multi-GPU Disaggregated KV Cache Management

> 下一篇论文规划 — 基于双 A100-80GB NVLink 平台

---

## 定位

| 项目 | 当前论文 (orchkvcache3/5) | 下一篇 (orchkvcache-scale) |
|:-----|:---|:---|
| 范围 | 单 GPU 三层存储编排 | 双/多 GPU KV Cache 池化 |
| 存储层级 | GPU HBM → DRAM → SSD | GPU₀ HBM → GPU₁ HBM → DRAM → SSD |
| 关键技术 | 注意力驱动 block 迁移 | 跨 GPU 热迁移 + NVLink pipeline |
| 存储优化 | OrchFS 集成 | GPU-GPU P2P + OrchFS 分级 |
| 集成层面 | HF/vLLM 单 GPU | vLLM TP=2 / 分离式推理 |
| 目标会议 | FAST / ATC / SC | OSDI / SOSP / EuroSys / SC |

---

## 核心 Idea

**把第二块 GPU 当作 KV Cache 的"高速温存储层"——介于 GPU₀ HBM（热）和 DRAM（冷）之间。**

四层存储层级：

```
GPU₀ HBM (热, 80GB, bandwidth: HBM ~2TB/s)
    ↕ NVLink (600 GB/s bidirectional)
GPU₁ HBM (温, 80GB, latency ~2μs)
    ↕ PCIe Gen4 (23 GB/s)
Host DRAM (冷, 256GB)
    ↕ NVMe (17.8 GB/s read, 5.3 GB/s write)
SSD (归档, 3.5TB)
```

NVLink 比 PCIe 快 **26 倍**（600 vs 23 GB/s），所以 GPU₁ 作为温层比 DRAM 快一个数量级。

---

## 三个应用场景

### 场景 A：TP=2 各管各的（基线，必做）

- 两个 GPU 各持一半 KV heads
- 每个 GPU 跑独立的 OrchKvCache 实例
- **改动最小**：`g_sys` 单例 → 多实例 handle

### 场景 B：KV Disaggregation（核心贡献）

- GPU₀ 负责计算（attention kernel）
- GPU₁ 纯做 KV Cache 扩展池
- 热 block 在 GPU₀，温 block 迁移到 GPU₁，冷 block 继续往 DRAM/SSD 降级
- 调度器决定何时跨 GPU 迁移

### 场景 C：Prefill-Decode 分离（进阶）

- GPU₀ 跑 prefill（生成 KV），GPU₁ 跑 decode（消费 KV）
- KV 通过 NVLink 从 GPU₀ 流式传输到 GPU₁
- 参考 DistServe/Splitwise 的 disaggregated serving 架构

---

## 实现路线

### Phase 1：g_sys 多实例化（3-5 天）

```c
// 之前
static struct { ... } g_sys;
int orchkv_init(const orchkv_config_t *config);

// 之后
typedef struct orchkv_sys orchkv_sys_t;
orchkv_sys_t* orchkv_create(const orchkv_config_t *config);
void orchkv_destroy(orchkv_sys_t *sys);
int orchkv_prefill(orchkv_sys_t *sys, ...);
```

**改动清单：**
- [ ] `src/api/orchkv_api.cu`：`g_sys` → `orchkv_sys_t*` 参数
- [ ] `bindings/orchkv_pybind.cpp`：所有 API 加 `sys_handle` 参数
- [ ] `python/orchkv/kvcache_manager.py`：per-rank 初始化
- [ ] 所有 30-40 个 API 函数签名更新
- [ ] 单元测试适配

### Phase 2：GPU-GPU 传输引擎（3-5 天）

```c
typedef struct gpu_gpu_transfer {
    int src_device;
    int dst_device;
    bool nvlink_available;
    cudaStream_t *streams;
    int num_streams;
    uint64_t total_p2p;
    uint64_t bytes_p2p;
} gpu_gpu_transfer_t;

int gpu_gpu_transfer_init(gpu_gpu_transfer_t *eng, int src_dev, int dst_dev);
int gpu_gpu_submit(gpu_gpu_transfer_t *eng, void *dst, const void *src, size_t bytes);
```

**改动清单：**
- [ ] 新文件 `src/tiered_store/gpu_gpu_transfer.cu`
- [ ] `cudaDeviceCanAccessPeer` 检测 NVLink
- [ ] `cudaMemcpyPeerAsync` 实现 P2P 传输
- [ ] 带宽测试基准（NVLink vs PCIe fallback）

### Phase 3：四层调度策略（5-7 天）

```
kv_block_t.tier 新增：
  GPU_HBM_0    = 0  (计算 GPU, 最热)
  GPU_HBM_1    = 1  (存储 GPU, 温)
  HOST_DRAM    = 2  (冷)
  SSD          = 3  (归档)
```

**调度逻辑：**
- 水位线体系扩展为四层
- 降级路径：GPU₀ → GPU₁ → DRAM → SSD
- 提升路径：SSD → DRAM → GPU₁ → GPU₀
- GPU₁ 充当"预取缓冲区"：即将被 GPU₀ attention 使用的 block 先从 DRAM 预取到 GPU₁

**改动清单：**
- [ ] `src/core/kv_types.h`：tier enum 新增 GPU_REMOTE
- [ ] `src/scheduler/tiered_manager.c`：四层调度逻辑
- [ ] `src/scheduler/hotcold_classifier.c`：新增 GPU₁ 层级阈值
- [ ] `src/scheduler/prefetch_scheduler.c`：GPU₁→GPU₀ 预取

### Phase 4：vLLM TP=2 集成（3-5 天）

- [ ] 在 vLLM 的 TP=2 模式下，每个 worker 运行独立 OrchKvCache 实例
- [ ] 验证 NVLink 带宽利用率
- [ ] 端到端吞吐量对比：vLLM TP=2 baseline vs vLLM TP=2 + OrchKvCache

### Phase 5：Disaggregated Serving（5-7 天，可选）

- [ ] Prefill worker (GPU₀) 生成 KV → NVLink 传输 → Decode worker (GPU₁) 消费
- [ ] 与 DistServe/Splitwise 对比
- [ ] KV 传输与计算重叠的流水线优化

---

## 实验计划

### 硬件
- 2× NVIDIA A100-SXM4-80GB (NVLink 600 GB/s)
- 256 GB DDR4-3200 DRAM
- Samsung PM9A3 NVMe SSD (RAID0)

### 实验矩阵

| 实验 | 配置 | 度量 |
|:-----|:-----|:-----|
| E1: NVLink 带宽基准 | block size sweep (1KB-32MB) | GB/s, 延迟 |
| E2: TP=2 基线 | LLaMA-2-13B TP=2 | 吞吐量, 内存利用率 |
| E3: KV Disagg 吞吐 | GPU₀ 算 + GPU₁ 存 | 吞吐量 vs 单 GPU, vs TP=2 |
| E4: 四层层级对比 | GPU₀-only / +GPU₁ / +DRAM / +SSD | 各层吞吐量开销 |
| E5: 长上下文 | 128K tokens, LLaMA-2-7B | 最大支持 batch size |
| E6: Prefill-Decode 分离 | 分离 vs 不分离 | TTFT, TPOT |

### 对比基线
- vLLM TP=2（原生，无 OrchKvCache）
- Mooncake（datacenter-scale KV pooling）
- DistServe（disaggregated prefill/decode）
- Infinite-LLM（distributed KV cache）

---

## 时间线

| 周 | 任务 |
|:--:|:-----|
| W1 | Phase 1: g_sys 多实例化 + 基本 TP=2 测试 |
| W2 | Phase 2: GPU-GPU 传输引擎 + NVLink 基准 |
| W3 | Phase 3: 四层调度策略实现 |
| W4 | Phase 4: vLLM TP=2 集成 + E2/E3 实验 |
| W5 | Phase 5: Disaggregated serving (可选) |
| W6 | 实验补充 + 论文撰写 |
| W7 | 论文完善 + 内部 Review |
| W8 | 投稿 |

**总周期：~2 个月**

---

## 论文叙事草案

**Title**: OrchKvCache-Scale: NVLink-Aware Disaggregated KV Cache Management for Multi-GPU LLM Inference

**核心 claim**: 利用 NVLink 的 600 GB/s 带宽，将第二块 GPU 作为 KV Cache 的高速温存储层，实现四层存储编排（GPU₀→GPU₁→DRAM→SSD），在双 A100 上支持比单 GPU 大 2 倍的 KV Cache 容量，同时保持无损输出质量。

**与当前论文的关系**: OrchKvCache (当前) 解决了"单 GPU 内如何智能管理 KV Cache"；OrchKvCache-Scale 解决"多 GPU 间如何协调 KV Cache 放置"。前者是后者的基础——四层调度直接复用当前的 EMA 热度评分和自适应阈值机制。

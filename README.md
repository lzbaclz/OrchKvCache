# OrchKvCache

OrchKvCache 是一个面向 LLM decode 阶段的分层 KV-Cache 管理研究原型。项目围绕论文主稿 [`paper/overleaf10.tex`](paper/overleaf10.tex) 展开，目标是在不丢失模型输出质量的前提下，把 KV block 在 GPU HBM、Host DRAM 和 SSD 之间做在线迁移、调度与复用。

这份 README 以论文定稿和当前仓库实现为准来整理信息。需要特别说明的是：

- 论文的正式叙事已经收敛为三层 `GPU / DRAM / SSD`。
- 仓库代码里仍保留部分更早期的 `NVM / 4-tier` 枚举、测试名和实现路径，用于阶段性实验与兼容旧原型。
- 因此，最稳妥的理解方式是：`paper/overleaf10.tex` 是研究结论的主口径，代码仓库展示的是支撑这些实验的原型实现。

从仓库现状看，它更接近“可复现实验和验证机制的系统原型”，而不是一个已经产品化的推理引擎插件。

## 项目概览

### 解决的问题

在 Transformer 自回归解码中，KV-Cache 会随上下文长度和并发请求数线性增长，成为 GPU 显存的主要瓶颈。现有方案大致有三类问题：

- 盲目换出：例如 FIFO，不关心哪些 block 真的会被后续 attention 用到。
- 静态规划：例如离线放置方案，难以适应 decode 阶段每一步变化的访问模式。
- 有损压缩或丢弃：直接减少 KV 数据，可能影响输出质量。

OrchKvCache 试图解决的是一个在线放置问题：在不知道未来访问的前提下，每个 decode step 动态决定哪些 block 留在 GPU、哪些下放到 DRAM、哪些继续溢写到 SSD，同时保持输出完全 lossless。

### 核心思路

论文中的核心思路可以概括为四点：

- 用 attention、recency、frequency 在线估计每个 block 的 hotness。
- 将 hot block 留在 GPU，把 warm block 下放到 DRAM，把 cold block 落到 SSD。
- 用异步 DMA、I/O worker 和 batch write，把 `GPU -> DRAM -> SSD` 的迁移代价尽量隐藏在 decode 计算之后。
- 在多请求场景下探索 shared-pool / time-multiplexing，让 GPU block residency 可以跨请求复用。

### 论文中的主要贡献

按照 [`paper/overleaf10.tex`](paper/overleaf10.tex) 的写法，OrchKvCache 的主要贡献有三项：

- Attention-driven hot/warm/cold classification with adaptive thresholds。
- Multi-granularity I/O adaptation via OrchFS-style aligned batch writes。
- Asynchronous migration pipeline with compute-transfer overlap。

论文同时强调了两个结果导向的结论：

- 完整 `GPU -> DRAM -> SSD -> DRAM -> GPU` 回环下保持 100% bit-exact lossless。
- 在 shared-pool / inter-step time-multiplexing 下，GQA 模型最高可获得 3.66x 吞吐提升。

## 论文与代码的对应关系

| 论文中的模块 | 代码位置 | 当前实现状态 |
| --- | --- | --- |
| KV block / request 元数据 | `src/core/` | 已实现 |
| GPU / DRAM slab pool 与 DMA 传输 | `src/tiered_store/gpu_tier.cu`, `src/tiered_store/dram_tier.cu`, `src/tiered_store/transfer.cu` | 已实现 |
| SSD/OrchFS 后端与异步 I/O worker | `src/tiered_store/orchfs_tier.c`, `src/tiered_store/io_worker.c` | 已实现 |
| Attention tracker / hot-cold classifier / adaptive threshold | `src/scheduler/attention_tracker.c`, `src/scheduler/hotcold_classifier.c`, `src/scheduler/adaptive_threshold.c` | 已实现 |
| Eviction / prefetch / migration / tiered manager | `src/scheduler/` | 已实现 |
| C API | `src/api/orchkv_api.h`, `src/api/orchkv_api.cu` | 已实现 |
| pybind11 绑定 `orchkv_core` | `bindings/orchkv_pybind.cpp` | 已实现 |
| HuggingFace 原型管理器 | `python/orchkv/kvcache_manager.py`, `python/orchkv/fast_kvcache_manager.py`, `python/orchkv/fast_fifo_manager.py` | 已实现 |
| vLLM connector / block scoring / attention hook | `python/orchkv/vllm_integration/` | 研究原型 |
| 论文实验脚本与结果 | `benchmarks/`, `benchmarks/results/` | 已实现 |

## 当前仓库已经实现了什么

### 1. C/CUDA 核心运行时

`src/` 下的实现大致可以分成四层：

- `src/core/`
  - `kv_block_t`、`kv_request_ctx_t`、地址映射和基础类型。
- `src/tiered_store/`
  - GPU/DRAM slab allocator、CUDA transfer engine、OrchFS 或 POSIX 文件后端、异步 I/O worker。
- `src/scheduler/`
  - 论文 Design 中对应的 C1-C8 调度组件。
- `src/api/`
  - 统一的 C API，对外暴露初始化、prefill、append、promote/demote、统计接口。

当前公开接口包括：

- `orchkv_init()` / `orchkv_shutdown()`
- `orchkv_request_create()` / `orchkv_request_destroy()`
- `orchkv_prefill()` / `orchkv_append_token()`
- `orchkv_get_kv_block()`
- `orchkv_evict_to_dram()` / `orchkv_promote_to_gpu()`
- `orchkv_evict_to_storage()` / `orchkv_promote_from_storage()`
- `orchkv_evict_cold()` 两跳迁移

### 2. 调度器机制

Phase C 的调度器包含以下组件：

- Attention Tracker
  - 维护 block 级 attention EMA。
- HotCold Classifier
  - 用 `alpha * attn + beta * recency + gamma * frequency` 计算热度。
- Adaptive Threshold
  - 用 GPU/DRAM 的 HWM/LWM 自适应调整阈值。
- Eviction Policy
  - 当前 C 代码使用 heat-aware + LRU 的候选选择；论文主叙事强调的是基于 composite hotness 的 coldest-first demotion。
- Prefetch Scheduler
  - 基于 EMA 的优先级调度。
- Migration Engine
  - 统一处理 `GPU <-> DRAM`、`DRAM <-> storage`、两跳迁移。
- Tiered Manager
  - 把上述组件串成一次 decode step 的完整调度循环。

### 3. Python 原型与集成

Python 层主要有三类入口：

- `python/orchkv/kvcache_manager.py`
  - HuggingFace 原型版 KV 管理器。
  - 支持 block 级迁移、attention 上报、lossless 验证。
- `python/orchkv/fast_kvcache_manager.py`
  - 预分配 GPU buffer 的快速版本。
  - 用于 time-multiplexing、fair-baseline、overhead 相关实验。
- `python/orchkv/vllm_integration/`
  - 包含 connector、attention hook、block-level swap、engine patch。
  - 用于 vLLM 0.7.x 上的研究性接入与调度诊断。

### 4. 一个容易混淆但很重要的口径说明

论文、C 核心和 Python 原型里都出现过“32KB 对齐 block”，但内部表示并不完全相同：

- 在 C 核心里，block 更接近“单层、单 KV head、连续 token 段”，默认 `tokens_per_block = 64`，在 `d_head = 128`、FP16 下接近 32KB 粒度。
- 在 Python/HuggingFace 原型里，block 常按“单层、所有 KV heads、16 token”建模；对 Qwen2.5-7B 这类 GQA 模型，`16 tokens * 4 KV heads * 128 * 2(K/V) * 2B = 32KB`，也对应论文里的 32KB block。

所以论文中的 16-token block 和 C 核心默认的 64-token block 不一定冲突，它们对应的是不同实现层的内部表示。

### 5. 三层设计与遗留四层代码如何理解

如果你主要关心论文结论，请按下面这个口径理解：

- 正式设计：`GPU HBM -> Host DRAM -> SSD`
- 主要贡献：attention-aware online placement + aligned batch I/O + async migration

如果你主要看代码，需要知道：

- `StorageTier` 里仍有 `TIER_NVM`
- 某些测试名仍保留 `4tier`
- 某些 vLLM/connector 注释仍写成 `GPU -> DRAM -> NVM -> SSD`

这些都来自更早期的阶段性原型，不代表 `overleaf10.tex` 这版论文的最终系统边界。

## 仓库结构

```text
.
├── src/                     # C/CUDA 核心实现
│   ├── core/                # block/request/address map/type definitions
│   ├── tiered_store/        # GPU/DRAM/OrchFS tier + DMA + IO worker
│   ├── scheduler/           # C1-C8 scheduler components
│   └── api/                 # public C API
├── bindings/                # pybind11 binding, module name: orchkv_core
├── python/orchkv/           # Python prototype managers and vLLM integration
├── benchmarks/              # 论文实验脚本与结果
├── test/                    # C/CUDA tests + Python tests
├── paper/                   # 论文主稿、图、plot 脚本
└── experiments/             # 早期实验与硬件/存储基线
```

## 构建

### 依赖

最少需要：

- CMake >= 3.18
- CUDA Toolkit
- 支持 CUDA 的 PyTorch
- `pybind11`，如果要构建 `orchkv_core`

如果要跑 Python 原型和论文实验，通常还需要：

- `transformers`
- `pytest`
- `numpy`
- `vllm`，仅在 vLLM 相关实验中需要

### OrchFS 是可选项

当前 `CMakeLists.txt` 会自动检测：

- `third_party/OrchFS/build/libOrchFS.so`

如果找到，就启用 OrchFS 路径；如果找不到，就自动退化为 POSIX `pwrite/pread` 文件后端。也就是说：

- 没有 OrchFS，仓库也能构建和运行大部分实验。
- 有 OrchFS 时，SSD 写路径更贴近论文中的对齐批量写设计。

如果你把 OrchFS 作为同级仓库放在 `../OrchFS`，建议确认 `third_party/OrchFS` 指向的是正确位置，例如：

```bash
cd /path/to/orchkv/OrchKvCache
rm -f third_party/OrchFS
ln -s ../../OrchFS third_party/OrchFS
```

### 编译命令

```bash
cd /path/to/orchkv/OrchKvCache
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

如果要构建 Python 绑定，请先确保 `pybind11` 能被 CMake 找到，例如：

```bash
pip install "pybind11[global]"
```

### Python 路径

这个仓库当前不是完整的 pip distribution，推荐通过 `PYTHONPATH` 使用：

```bash
export PYTHONPATH=$PWD/build/bindings:$PWD/python:$PYTHONPATH
```

验证方式：

```bash
python -c "import orchkv_core, orchkv; print('ok')"
```

## 测试

### C/CUDA 测试

```bash
ctest --test-dir build --output-on-failure
```

`CMakeLists.txt` 已经注册了较完整的核心测试，包括：

- 基础数据结构：`test_kv_types`, `test_kv_block`, `test_address_map`, `test_kv_request`
- 存储与迁移：`test_gpu_dram`, `test_orchfs_tier`, `test_e2e`, `test_e2e_4tier`
- 调度器：`test_attention_tracker`, `test_hotcold_classifier`, `test_adaptive_threshold`, `test_eviction_policy`, `test_prefetch_scheduler`, `test_pipeline`, `test_migration_engine`, `test_tiered_manager`, `test_e2e_auto`

### Python 测试

```bash
python -m pytest test/test_binding.py -v
python -m pytest test/test_connector.py -v
python -m pytest test/test_attention_hook.py -v
python -m pytest test/test_benchmarks.py -v
```

说明：

- 其中不少测试依赖 CUDA。
- vLLM 相关测试在未安装 vLLM 时会走 fallback stub。

## 快速使用

### 1. C API

```c
orchkv_config_t cfg;
orchkv_config_default(&cfg);
cfg.gpu_pool_bytes = 4ULL << 30;
cfg.dram_pool_bytes = 8ULL << 30;

orchkv_init(&cfg);

kv_request_ctx_t *ctx = orchkv_request_create(req_id, n_layers, n_kv_heads);
orchkv_prefill(ctx, layer_id, k_host_ptr, v_host_ptr, seq_len);
orchkv_append_token(ctx, layer_id, k_token_ptr, v_token_ptr);

void *k_ptr = NULL;
void *v_ptr = NULL;
orchkv_get_kv_block(ctx, layer_id, head_id, block_idx, &k_ptr, &v_ptr);

orchkv_request_destroy(ctx);
orchkv_shutdown();
```

### 2. Python 绑定

```python
import orchkv_core as C

cfg = C.Config()
cfg.gpu_pool_bytes = 4 * (1 << 30)
cfg.dram_pool_bytes = 8 * (1 << 30)

C.init(cfg)

tm = C.tm_create()
C.tm_register_block_id(tm, block_id=0, tier=int(C.GPU_HBM), flags=0)
C.tm_report_attn(tm, block_id=0, attn_weight=0.8)
C.tm_step_done(tm)
C.tm_set_usage(tm, gpu_ratio=0.8, dram_ratio=0.2)
C.tm_schedule_once(tm)

print(C.tm_get_stats(tm))

C.tm_destroy(tm)
C.shutdown()
```

### 3. HuggingFace 原型

更完整的 HuggingFace decode 原型可以直接参考：

- `python/orchkv/kvcache_manager.py`
- `benchmarks/exp_multimodel.py`
- `benchmarks/exp_quality.py`

这些脚本展示了如何在 decode 循环里：

- ingest / append 新生成的 KV
- 上报 attention
- 调用 `step_done()` 和 `schedule()`
- 重建或复用 `past_key_values`

## 主要实验入口

下面这些脚本基本覆盖了 [`paper/overleaf10.tex`](paper/overleaf10.tex) 中的主要评测维度：

| 脚本 | 对应内容 | 备注 |
| --- | --- | --- |
| `benchmarks/exp_multimodel.py` | E2E throughput、quality、ablation | 需要 HuggingFace 模型 |
| `benchmarks/benchmark_prefetch.py` | prefetch dispatch 与调度微基准 | `orchkv_core` 即可 |
| `benchmarks/benchmark_scalability.py` | 调度延迟随 block 数扩展 | `orchkv_core` 即可 |
| `benchmarks/benchmark_storage_bw.py` | GPU<->DRAM、DRAM<->storage 带宽 | 需要 CUDA |
| `benchmarks/exp_trace_driven_cr.py` | competitive ratio / trace-level policy quality | `orchkv_core` 即可 |
| `benchmarks/exp_realistic_workload.py` | ShareGPT-like 和 mixed-length 负载 | 需要 HuggingFace 模型 |
| `benchmarks/exp_signal_ablation.py` | attention vs recency+frequency | 以 trace/simulation 为主 |
| `benchmarks/exp_ssd_ablation_e2e.py` | SSD tier round-trip 与开销 | 需要 CUDA |
| `benchmarks/exp_time_multiplex.py` | inter-step shared-pool / time-multiplexing | 需要 HuggingFace 模型 |
| `benchmarks/exp_subbatch_rotation.py` | sub-batch rotation tradeoff | 需要 HuggingFace 模型 |
| `benchmarks/exp_vllm_production.py` | vLLM 生产运行时诊断 | 需要 vLLM |
| `benchmarks/plot_paper_figures.py` | 从 `benchmarks/results/` 生成论文图 | 依赖已有结果 |

常用命令示例：

```bash
python benchmarks/exp_multimodel.py --quick
python benchmarks/benchmark_prefetch.py
python benchmarks/benchmark_storage_bw.py
python benchmarks/benchmark_scalability.py
python benchmarks/exp_trace_driven_cr.py
python benchmarks/exp_time_multiplex.py
conda run -n orchkv env PYTHONPATH=python python benchmarks/exp_vllm_production.py
```

实验结果默认写入 `benchmarks/results/`。

## 论文中的主要结论

根据当前论文主稿 [`paper/overleaf10.tex`](paper/overleaf10.tex)，项目给出的核心结论包括：

- 相比 FIFO offload，OrchKvCache 将不必要的数据迁移减少了 139x-597x。
- 在完整 `GPU -> DRAM -> SSD -> DRAM -> GPU` 回环下，输出保持 100% bit-exact lossless。
- 调度路径的微基准开销很低：prefetch dispatch 约 5.7-5.9us；4096 blocks 时 P99 调度延迟仍低于 60us。
- 在 shared-pool / inter-step time-multiplexing 场景下，GQA 模型上可获得最高 3.66x 吞吐提升。
- vLLM 研究性集成表明：block-level intelligence 真正带来收益的地方，是 tiered migration 与 cross-request memory sharing，而不是单纯 victim selection。

## 论文实验环境

论文中的代表性实验平台为：

- 2 x NVIDIA A100-SXM4-80GB
- 256GB DDR4-3200
- Samsung PM9A3 NVMe RAID0
- Ubuntu 22.04
- CUDA 12.0
- PyTorch 2.5.1+cu121
- transformers 4.57

这部分是论文评测平台，不等同于仓库的最小运行要求。

## 当前边界与已知局限

- 当前 HuggingFace/Python 原型的主要开销在 `build_past_kv()` 和 attention 提取，不代表原生 C/CUDA 路径的上限。
- 论文已明确指出：prefetch 子系统在当前 per-request 原型中主要完成“机制验证”，还不是端到端吞吐提升的主要来源。
- 当上下文很长时，直接 `output_attentions=True` 会带来 eager attention 额外开销；论文中 8K+ 的部分实验会退化为 recency+frequency 路径。
- vLLM 目录下的 connector / block swap / scoring patch 是研究原型，适合做实验诊断，不应视为生产级补丁。
- 仓库没有完整打包为 pip distribution；当前推荐的使用方式是本地编译 `orchkv_core` 后配合 `PYTHONPATH` 使用。
- 如果你只关心最终论文，请优先参考三层 `GPU / DRAM / SSD` 口径；仓库中的 `NVM / 4-tier` 痕迹不应直接当作论文最终设计。

## 参考

- 论文主稿：[`paper/overleaf10.tex`](paper/overleaf10.tex)
- 架构说明：[`paper/architecture_guide.md`](paper/architecture_guide.md)
- 画图脚本：`paper/figures_plot_code/`

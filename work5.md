# Work5: Phase D — 推理引擎集成（vLLM）

> **前置已完成**: Phase C（冷热分级与自动调度，C1-C11）
> **本阶段目标**: 将 OrchKvCache 嵌入 vLLM，实现在真实 LLM 推理中的端到端四级存储管理
> **预计工期**: 2~3 周
> **论文价值**: Phase D 是 §5 Implementation 和 §6 Evaluation 的数据来源，决定论文的实验说服力
>
> **⚠️ 当前环境状态**: vLLM 和 pybind11 均未安装，需要先完成 D0 环境准备
> **⚠️ D3/D4 依赖 Phase C 完成**: `tm_notify_attn` 等接口待 C8 稳定后补充细节

---

## 〇、Phase D 全局依赖关系

```
Phase A/B (存储四层框架)         ←  D2 依赖（提供 GPU/DRAM/Storage 原语）
Phase C   (tiered_manager)       ←  D3 强依赖（C8 完成后才能写 D3 细节）
vLLM V1 源码理解                 ←  D2/D3 均依赖（V1 架构与 V0 差异极大）
Python/C 混合编译环境 (pybind11) ←  D1 依赖
```

D1、D2 与 Phase C 无强依赖，**可以在 Phase C 实现过程中并行推进**。
D3、D4 需等待 Phase C 完成，本文档 D3/D4 部分为草稿，届时补充细节。

---

## 一、Phase D 架构总览

```
                        Python vLLM V1 推理框架
┌─────────────────────────────────────────────────────────────────────┐
│  EngineCore (独立进程, ZMQ IPC)                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐│
│  │  Scheduler              │    │  Worker (GPU 进程)              ││
│  │  ┌───────────────────┐  │    │  ┌────────────────────────────┐││
│  │  │ KVCacheManager    │  │    │  │  ModelRunner               │││
│  │  │  ├─ GpuBlockPool  │  │    │  │  ┌──────────────────────┐  │││
│  │  │  └─ [D2] OrchKv   │  │    │  │  │ Attention Layer       │  │││
│  │  │     Connector      │  │    │  │  │  [D3] → attn_hook    │  │││
│  │  └───────────────────┘  │    │  │  │    ↓ tm_notify_attn() │  │││
│  └─────────────────────────┘    │  │  └──────────────────────┘  │││
│                                 │  └────────────────────────────┘││
│                                 └─────────────────────────────────┘│
└────────────────────┬──────────────────────────────────────────────┘
                     │  Python binding (pybind11)
                     ▼  [D1]
┌─────────────────────────────────────────────────────────────────────┐
│  OrchKvCache C 库 (liborchkv.so)                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐│
│  │ Phase A          │  │ Phase B          │  │ Phase C            ││
│  │ gpu_tier         │  │ orchfs_tier      │  │ tiered_manager     ││
│  │ dram_tier        │  │ io_worker        │  │ (热度感知 + 调度)  ││
│  │ transfer_engine  │  │                  │  │                    ││
│  └─────────────────┘  └──────────────────┘  └────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**与 Phase A~C 的架构差异**：
- Phase A~C 全部在 C 层，Phase D 是唯一涉及 Python 的阶段
- vLLM V1 采用多进程架构（API Server ↔ EngineCore ↔ Worker），需要注意跨进程状态同步
- KV 数据本身始终在 GPU 上由 C 层管理，Python 侧仅持有不透明句柄

---

## 二、目录结构变更

Phase D 新增的文件：

```
OrchKvCache/
├── bindings/
│   ├── CMakeLists.txt              # [D1] pybind11 模块构建
│   ├── orchkv_pybind.cpp           # [D1] Python binding 实现
│   └── orchkv_pybind.pyi           # [D1] Python 类型存根（IDE 类型提示）
├── python/
│   └── orchkv/
│       ├── __init__.py
│       ├── config.py               # [D1] Python 侧配置封装
│       └── vllm_integration/
│           ├── __init__.py
│           ├── connector.py        # [D2] OffloadingConnector 实现
│           ├── block_pool.py       # [D2] 自定义 BlockPool（映射 slab_alloc）
│           ├── attention_hook.py   # [D3] 注意力分数采集 hook
│           └── engine_patch.py     # [D2/D3] 注册 connector + hook 到 vLLM
├── test/
│   ├── test_binding.py             # [D1] Python binding 单元测试
│   ├── test_vllm_connector.py      # [D2] OffloadingConnector 集成测试
│   └── test_e2e_inference.py       # [D4] 端到端推理测试
├── scripts/
│   ├── benchmark_e2e.py            # 论文 E1-E2 主实验脚本
│   ├── benchmark_ablation.py       # 论文 E4-E7 消融实验脚本
│   └── run_all_benchmarks.sh       # 一键运行所有实验
└── CMakeLists.txt                  # 修改：加入 bindings/ 子目录
```

---

## 三、vLLM 源码关键路径（必读）

在编码 D2/D3 之前，需要深入理解 vLLM V1 的以下源码：

| 模块 | 关键文件 | 需要理解的内容 |
|------|---------|---------------|
| V1 入口 | `vllm/v1/engine/core.py` | `EngineCore` 主循环、Scheduler 调用顺序 |
| KV Cache 管理 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` 的 `allocate` / `free` / `get_computed_blocks` |
| Block 分配 | `vllm/v1/core/block_pool.py` | `BlockPool` 的分配/回收逻辑，理解 block_id 的含义 |
| Offloading | `vllm/v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` 如何协调 GPU ↔ CPU 传输 |
| Connector API | `vllm/kv_transfer/kv_connector/` | `OffloadingConnector` 基类、`AsyncOffloadingConnector` |
| Worker 侧 | `vllm/v1/worker/gpu_worker.py` | Worker 如何执行 `swap_in` / `swap_out` |
| Attention | `vllm/attention/backends/` | `FlashAttention` / `FlashInfer` 后端的 KV 访问方式 |
| 配置 | `vllm/config.py` | `CacheConfig`, `KVTransferConfig` 的字段 |

**重要变化（V0 → V1）**：
- V0 的 `BlockSpaceManager` 已废弃（v0.17+），**不要基于 V0 开发**
- V1 不再区分 prefill/decode 调度阶段，统一为 `{request_id: num_tokens}`
- KV cache offloading 从 v0.11 开始内置支持，v0.18 已有 `--kv-offloading-backend` CLI 参数
- V1 的 `KVCacheManager` 与 V0 的 `BlockSpaceManager` 接口完全不同

---

## 四、任务分解

### D0: 环境准备（前置，非代码）

**当前状态**: vLLM 未安装，pybind11 未安装，系统 Python 3.13

**⚠️ 关键约束**:
- vLLM 需要 Python 3.9~3.12（3.13 不兼容）
- vLLM 需要 PyTorch 与 CUDA 版本匹配
- 当前 CUDA: 12.0，GPU: A100-SXM4-80GB

**步骤**:

```bash
# 1. 创建隔离环境（conda 或 venv）
conda create -n orchkv python=3.11 -y
conda activate orchkv

# 2. 安装 PyTorch（需与 CUDA 12.0 兼容）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 pybind11（构建 binding 用）
pip install pybind11[global]

# 4. 安装 vLLM
#    选择一个稳定版本，避免用最新 rc
pip install vllm==0.17.2

# 5. 验证安装
python -c "import vllm; print(f'vLLM {vllm.__version__} OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# 6. 下载测试模型（LLaMA-2-7B 或同等规模）
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b
```

**vLLM 版本选择建议**:

| vLLM 版本 | 引擎 | KV Cache 管理 | 建议 |
|-----------|------|--------------|------|
| ≤ 0.10.x | V0 only | BlockSpaceManagerV1/V2 | 过时，不推荐 |
| 0.11~0.16 | V0/V1 共存 | V1 引入 KVCacheManager + OffloadingConnector | 可用，但 V1 尚不稳定 |
| **0.17.x** | **V1 默认，V0 废弃** | **KVCacheManager + OffloadingConnector 成熟** | **推荐：稳定，V1 API 成型** |
| 0.18.x | V1 only | 新增 FlexKV 后端 + `--kv-offloading-backend` CLI | 可选：最新特性，但可能有 breaking changes |

**推荐锁定 v0.17.2**：V1 引擎成熟、OffloadingConnector API 稳定、文档齐全。

**估时**: 0.5 天（视网络和依赖冲突情况）

---

### D1: Python Binding（`bindings/orchkv_pybind.cpp`）

**目标**: 通过 pybind11 将 OrchKvCache 的 C API 暴露为 Python 模块 `orchkv_core`，供 vLLM 集成调用。

**暴露的接口清单**:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

extern "C" {
#include "core/kv_types.h"
#include "api/orchkv_api.h"
}

PYBIND11_MODULE(orchkv_core, m) {
    m.doc() = "OrchKvCache: Tiered KV-Cache management for LLM inference";

    /* ---- 枚举 ---- */
    py::enum_<StorageTier>(m, "StorageTier")
        .value("GPU_HBM",    TIER_GPU_HBM)
        .value("HOST_DRAM",  TIER_HOST_DRAM)
        .value("NVM",        TIER_NVM)
        .value("SSD",        TIER_SSD);

    /* ---- 配置结构体 ---- */
    py::class_<orchkv_config_t>(m, "Config")
        .def(py::init<>())
        .def_readwrite("gpu_pool_bytes",  &orchkv_config_t::gpu_pool_bytes)
        .def_readwrite("dram_pool_bytes", &orchkv_config_t::dram_pool_bytes)
        .def_readwrite("slab_bytes",      &orchkv_config_t::slab_bytes)
        .def_readwrite("n_layers",        &orchkv_config_t::n_layers)
        .def_readwrite("n_kv_heads",      &orchkv_config_t::n_kv_heads)
        .def_readwrite("d_head",          &orchkv_config_t::d_head)
        .def_readwrite("orchfs_base_dir", &orchkv_config_t::orchfs_base_dir);

    m.def("config_default", &orchkv_config_default,
          "Return a Config with sensible defaults");

    /* ---- 生命周期 ---- */
    m.def("init",     &orchkv_init,     "Initialize the OrchKvCache system");
    m.def("shutdown", &orchkv_shutdown, "Shutdown and free all resources");

    /* ---- 请求管理 ---- */
    m.def("request_create", &orchkv_request_create,
          py::arg("request_id"), py::arg("n_layers"), py::arg("n_kv_heads"),
          py::return_value_policy::reference,
          "Create a new KV request context");
    m.def("request_destroy", &orchkv_request_destroy,
          "Destroy a request and release its blocks");

    /* ---- 数据操作 ---- */
    m.def("prefill", [](kv_request_ctx_t *ctx, uint32_t layer,
                         uint32_t head, uintptr_t k_ptr, uintptr_t v_ptr,
                         size_t n_tokens) {
        return orchkv_prefill(ctx, layer, head,
                              reinterpret_cast<const void*>(k_ptr),
                              reinterpret_cast<const void*>(v_ptr),
                              n_tokens);
    }, "Prefill KV data from GPU tensor (pass data_ptr as int)");

    m.def("get_kv_block", [](kv_request_ctx_t *ctx, uint32_t layer,
                              uint32_t head, uint32_t block_idx) -> py::tuple {
        void *k_out, *v_out;
        int rc = orchkv_get_kv_block(ctx, layer, head, block_idx,
                                      &k_out, &v_out);
        return py::make_tuple(rc, reinterpret_cast<uintptr_t>(k_out),
                                   reinterpret_cast<uintptr_t>(v_out));
    }, "Get KV block GPU pointers (returned as int addresses)");

    /* ---- 统计信息 ---- */
    m.def("get_stats", []() -> py::dict {
        orchkv_stats_t s;
        orchkv_get_stats(&s);
        py::dict d;
        d["gpu_slabs_used"]  = s.gpu_slabs_used;
        d["gpu_slabs_total"] = s.gpu_slabs_total;
        d["dram_slabs_used"] = s.dram_slabs_used;
        d["dram_slabs_total"]= s.dram_slabs_total;
        d["gpu_to_dram"]     = s.gpu_to_dram;
        d["dram_to_gpu"]     = s.dram_to_gpu;
        d["dram_to_storage"] = s.dram_to_storage;
        d["storage_to_dram"] = s.storage_to_dram;
        return d;
    }, "Return system statistics as a Python dict");

    /* ---- 迁移操作 ---- */
    m.def("evict_to_dram",    &orchkv_evict_to_dram);
    m.def("evict_to_storage", &orchkv_evict_to_storage);
    m.def("evict_cold",       &orchkv_evict_cold);
    m.def("promote_to_gpu",   &orchkv_promote_to_gpu);
    m.def("storage_flush",    &orchkv_storage_flush);

    /* ---- Phase C: 注意力上报（待 C8 稳定后取消注释）---- */
    // m.def("report_attn", &orchkv_report_attn,
    //       py::arg("ctx"), py::arg("layer"), py::arg("head"),
    //       py::arg("block_idx"), py::arg("attn_score"));
}
```

**GPU 指针与 PyTorch Tensor 互操作**:

这是 D1 最关键的技术细节：vLLM 的 KV data 存放在 `torch.Tensor` 中，需要桥接到 OrchKvCache 的 `void*` GPU 指针。

```python
# Python 侧调用示例
import torch
import orchkv_core as _C

k_tensor: torch.Tensor  # shape: [n_tokens, d_head], device="cuda"
v_tensor: torch.Tensor

_C.prefill(ctx, layer=0, head=0,
           k_ptr=k_tensor.data_ptr(),   # torch.Tensor → int (GPU 地址)
           v_ptr=v_tensor.data_ptr(),
           n_tokens=k_tensor.shape[0])
```

**注意事项**:
- `torch.Tensor.data_ptr()` 返回 GPU 指针的整数值，pybind11 用 `uintptr_t` 接收
- 调用期间必须保证 tensor 不被 GC（Python 侧持有引用即可）
- `kv_request_ctx_t*` 在 Python 侧是不透明句柄，`return_value_policy::reference` 避免 GC 误释放
- CUDA stream 同步：OrchKvCache 内部使用自己的 stream，需要在跨框架边界时同步

**CMakeLists.txt 修改**:

```cmake
# Phase D: pybind11 模块
find_package(pybind11 CONFIG)
if(pybind11_FOUND)
    pybind11_add_module(orchkv_core bindings/orchkv_pybind.cpp)
    target_include_directories(orchkv_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(orchkv_core PRIVATE orchkv)
    # Phase C 完成后加入 orchkv_scheduler:
    # target_link_libraries(orchkv_core PRIVATE orchkv orchkv_scheduler)
    message(STATUS "pybind11 found → building orchkv_core Python module")
else()
    message(STATUS "pybind11 not found → skipping Python binding")
endif()
```

**估时**: 1.5 天

---

### D2: vLLM OffloadingConnector 集成

**目标**: 实现 vLLM V1 的 `OffloadingConnector` 接口，将 OrchKvCache 的四级存储接入 vLLM 的 KV cache offloading 框架。

> **⚠️ 重要**: vLLM V0 的 `BlockSpaceManager` 已在 v0.17 废弃。
> 本任务基于 V1 的 `OffloadingConnector` / `KVCacheManager` 架构。

**vLLM V1 的 KV Cache Offloading 机制**:

```
Scheduler (EngineCore)
    │
    ├─ KVCacheManager           # 管理 block 分配/释放/prefix caching
    │   ├─ GpuBlockPool         # GPU 上的 block 池
    │   └─ CpuBlockPool         # CPU 上的 block 池（offloading target）
    │
    └─ OffloadingConnector      # 可插拔的传输后端
        ├─ send_kv_caches_and_hidden_states()   # GPU → CPU (swap-out)
        └─ recv_kv_caches_and_hidden_states()   # CPU → GPU (swap-in)
```

OrchKvCache 替换的是 `CpuBlockPool` + `OffloadingConnector` 层，将"仅 CPU 一级"扩展为"DRAM → NVM → SSD 三级"。

**集成方案：自定义 OffloadingConnector**:

```python
# python/orchkv/vllm_integration/connector.py

from vllm.kv_transfer.kv_connector.base import KVConnectorBase
import orchkv_core as _C

class OrchKvOffloadingConnector(KVConnectorBase):
    """
    将 vLLM 的 KV cache swap-in/swap-out 请求路由到 OrchKvCache C 库。

    替换 vLLM 默认的 CPU-only offloading，提供 DRAM → NVM → SSD 三级存储。
    当 GPU block 需要 swap-out 时，OrchKvCache 的 tiered_manager 自动
    决定数据落到哪一级。
    """

    def __init__(self, config):
        super().__init__()
        orchkv_cfg = _C.config_default()
        orchkv_cfg.gpu_pool_bytes  = 0          # GPU 内存由 vLLM 管理
        orchkv_cfg.dram_pool_bytes = config.cpu_offload_gb * (1 << 30)
        orchkv_cfg.n_layers        = config.num_layers
        orchkv_cfg.n_kv_heads      = config.num_kv_heads
        orchkv_cfg.d_head          = config.head_dim
        _C.init(orchkv_cfg)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input,
        kv_caches,                # List[torch.Tensor], GPU KV cache
        attn_metadata,
    ) -> None:
        """GPU → OrchKvCache (swap-out)"""
        for layer_idx, kv in enumerate(kv_caches):
            # kv shape: [2, num_blocks, block_size, num_heads, head_dim]
            k_cache, v_cache = kv[0], kv[1]
            for block_idx in attn_metadata.blocks_to_swap_out:
                _C.evict_to_dram(...)  # 或由 tiered_manager 自动决策

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input,
        kv_caches,
    ) -> None:
        """OrchKvCache → GPU (swap-in)"""
        for layer_idx, kv in enumerate(kv_caches):
            for block_idx in model_input.blocks_to_swap_in:
                _C.promote_to_gpu(...)

    def close(self):
        _C.shutdown()
```

**注册到 vLLM**:

```python
# python/orchkv/vllm_integration/engine_patch.py

def register_orchkv_backend():
    """
    将 OrchKvCache 注册为 vLLM 的 kv-offloading-backend。

    使用方式 1 — 代码注册:
        from orchkv.vllm_integration.engine_patch import register_orchkv_backend
        register_orchkv_backend()
        llm = LLM(model="...", kv_offloading_backend="orchkv")

    使用方式 2 — CLI:
        python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-2-7b-hf \
            --kv-offloading-backend orchkv
    """
    from vllm.kv_transfer import kv_connector
    from orchkv.vllm_integration.connector import OrchKvOffloadingConnector
    kv_connector.register("orchkv", OrchKvOffloadingConnector)
```

**与 V0 monkey-patch 方案的对比**:

| 维度 | V0 monkey-patch (旧方案) | V1 OffloadingConnector (新方案) |
|------|------------------------|-----------------------------|
| 侵入性 | 高：替换 BlockSpaceManager 类 | 低：实现标准插件接口 |
| 兼容性 | 仅 vLLM ≤ 0.16 | vLLM ≥ 0.11，推荐 ≥ 0.17 |
| 多级存储 | 需自行管理 | Connector 内部路由到 OrchKvCache |
| 多 GPU | monkey-patch 在多进程下不可靠 | Connector 原生支持多 Worker |
| 维护成本 | vLLM 升级必须重写 | 接口稳定，升级成本低 |

**估时**: 2 天

---

### D3: Attention Hook — 注意力分数采集

> **⚠️ 草稿阶段**: 本节细节依赖 Phase C (`tm_notify_attn`) 接口稳定后补充。
> 以下为当前设计方向，实现时以实际 C8 接口为准。

**目标**: 在 vLLM 每次 attention 计算后，将注意力权重汇报给 `tiered_manager`，驱动冷热分级。

**预计 C 层接口**:

```c
// Phase C (C8) 导出的注意力上报 API
int orchkv_report_attn(kv_request_ctx_t *ctx,
                       uint32_t layer, uint32_t head,
                       uint32_t block_idx, float attn_score);
```

**实现方案对比**:

| 方案 | 实现方式 | 额外拷贝 | 开销 | 精度 | 复杂度 | 推荐 |
|------|---------|---------|------|------|--------|------|
| **A: Kernel 内注入** | 修改 vLLM PagedAttention CUDA kernel，在 softmax 后直接写分数到共享 buffer | 零 | 极低 | 精确 | 高（需改 CUDA kernel） | 暂不推荐 |
| **B: Python 层 wrap** | 在 Python 层 wrap attention 函数，对 softmax 输出做 block-wise reduce 后上报 | GPU→CPU 异步拷贝 | 中等 | 精确 | 中 | **推荐方案** |
| **C: 采样近似** | 每 K 步采一次完整注意力分数，中间步用 EMA 外推 | 每 K 步一次拷贝 | 低 | 近似 | 低 | 备选（精度不足时切换到 B） |

**推荐方案 B 的实现草稿**:

```python
# python/orchkv/vllm_integration/attention_hook.py

import torch
import orchkv_core as _C

class AttentionScoreCollector:
    """
    Hook 到 vLLM attention 后端，采集 per-block 注意力权重。

    工作流程:
      1. 在 attention forward 后，获取 softmax(QK^T/√d) 输出
      2. 按 block 粒度做 reduce (mean or max)
      3. 通过 orchkv_core.report_attn() 上报给 tiered_manager
      4. 使用独立 CUDA stream 异步执行 reduce + D2H 拷贝
    """

    def __init__(self, sample_interval: int = 1):
        self.stream = torch.cuda.Stream()
        self.step = 0
        self.sample_interval = sample_interval

    def collect(self, ctx, layer: int, attn_weights: torch.Tensor):
        """
        attn_weights: [batch, n_heads, q_len, kv_len]
        每 sample_interval 步采集一次。
        """
        self.step += 1
        if self.step % self.sample_interval != 0:
            return

        with torch.cuda.stream(self.stream):
            # block-wise reduce: 按 tokens_per_block 分组取 mean
            block_size = 64
            kv_len = attn_weights.shape[-1]
            n_blocks = (kv_len + block_size - 1) // block_size

            for h in range(attn_weights.shape[1]):
                for bi in range(n_blocks):
                    start = bi * block_size
                    end = min(start + block_size, kv_len)
                    score = attn_weights[:, h, :, start:end].mean().item()
                    _C.report_attn(ctx, layer, h, bi, score)
```

**需要在 C8 完成后补充**:
- `tm_notify_attn` 的实际参数类型和调用频率要求
- 接受 per-token score 还是 per-block aggregated score
- GPU→CPU 拷贝的时序约束（与计算 stream 的同步点）
- 是否需要在 C 层提供批量上报接口（减少 Python↔C 调用次数）

**估时**: 1.5 天（待 C8 后）

---

### D4: 端到端推理测试与验证

> **⚠️ 草稿阶段**: 依赖 D2/D3 全部完成。以下为测试框架设计，具体 case 待实现后补充。

**测试目标**:

| 维度 | 目标 | 验收标准 |
|------|------|---------|
| 功能正确性 | OrchKvCache 管理 KV-Cache 时，生成输出与原始 vLLM 一致 | Top-1 token 一致率 ≥ 99.9%，logit 误差 < 1e-4 |
| 显存扩展 | 在固定 GPU 显存下，服务更长序列或更大 batch | 在 vLLM OOM 的配置下正常运行 |
| 性能开销 | TTFT / TPOT 不超过可接受 overhead | 短序列 (≤4K) 退化 < 5%，长序列 (≥16K) 吞吐提升 ≥ 20% |

**测试矩阵**:

```python
# test_e2e_inference.py

MODELS     = ["meta-llama/Llama-2-7b-hf"]
SEQ_LENS   = [1024, 4096, 16384, 32768, 65536]
BATCH_SIZES = [1, 4, 8, 16]
DATASETS    = ["ShareGPT", "LongBench-subset", "Synthetic-uniform"]

def test_output_correctness():
    """
    与 baseline vLLM 逐 token 比对。
    输入: 固定 prompt + 固定随机种子 (temperature=0)
    期望: greedy decoding 下 token ids 完全一致
    """

def test_memory_extension():
    """
    验证显存扩展能力。
    步骤: 逐步增大 batch_size / seq_len，直到 baseline OOM，
          验证 OrchKvCache 仍能正常服务。
    """

def test_throughput():
    """
    吞吐率对比 (tokens/s)。
    输出: JSON 包含 TTFT, TPOT, throughput, GPU utilization。
    """

def test_latency_breakdown():
    """
    延迟分解: 迁移开销 / 调度开销 / 纯计算时间。
    用于论文 E3。
    """
```

**论文实验脚本规划**（待 D2/D3 完成后实现）:

| 脚本 | 对应论文实验 | 测量指标 | 输出格式 |
|------|------------|---------|---------|
| `benchmark_e2e.py` | E1: 端到端吞吐 | tokens/s, TTFT, TPOT | JSON + CSV |
| `benchmark_e2e.py` | E2: Max Batch Size | max_batch @ seq_len | JSON |
| `benchmark_e2e.py` | E3: 延迟分解 | compute / migrate / schedule (μs) | JSON |
| `benchmark_ablation.py` | E4: 存储层消融 | 关闭各层后性能变化 | JSON |
| `benchmark_ablation.py` | E5: 冷热策略对比 | α/β/γ 参数 sweep | JSON |
| `benchmark_ablation.py` | E6: 粒度消融 | block_size sweep | JSON |
| `benchmark_prefetch.py` | E7: 预取效果 | 命中率 / 延迟隐藏率 | JSON |
| `benchmark_storage_bw.py` | E8: 存储带宽 | GB/s per tier | JSON |
| `benchmark_scalability.py` | E9: 可扩展性 | 多请求并发 scaling | JSON |
| `eval_quality.py` | E10: 生成质量 | perplexity, LongBench scores | JSON |

**估时**: 2 天（框架）+ 持续（调试与跑实验数据）

---

## 五、依赖关系与执行顺序

```
D0: 环境准备（conda env + vLLM + pybind11）         ← 最先
  │
  └─ D1: Python binding (orchkv_core.so)
       │
       └─ D2: OffloadingConnector 实现               ← 可在 Phase C 期间并行
            │
            ├─ D3: Attention Hook (注意力采集)        ← 必须等 Phase C(C8) 完成
            │
            └─ D4: E2E 推理测试 + 论文实验脚本        ← 必须等 D2+D3 全部完成
```

**并行策略**:

| 时间段 | Phase C 状态 | Phase D 可做的事 |
|--------|-------------|-----------------|
| Phase C 实现中 | C1~C7 进行 | D0 → D1 → D2（三者与 C 无技术依赖） |
| Phase C 完成后 | C8~C11 完成 | D3 → D4 → 论文实验 |

**关键路径**: D0 → D1 → D2 → D3 → D4（总计约 7.5 天）

---

## 六、关键技术风险与缓解

| 风险 | 影响程度 | 缓解措施 |
|------|---------|---------|
| vLLM V1 OffloadingConnector API 变动 | 高：D2 核心逻辑需重写 | 锁定 vLLM v0.17.2，pin 版本；fork 源码保留可回退 |
| Python 3.13 不兼容 vLLM / PyTorch | 高：D0 环境搭建失败 | 用 conda 创建 Python 3.11 隔离环境 |
| GPU 指针跨框架传递（torch ↔ OrchKvCache） | 中：CUDA stream 不同步导致数据竞争 | 在 Python binding 边界显式调用 `cudaStreamSynchronize`；使用 `torch.cuda.current_stream()` 对齐 |
| vLLM 多进程模式下 Connector 状态不一致 | 中：多 Worker 各持有独立 OrchKvCache 实例 | Phase D 限定单 GPU 验证；多 GPU 支持作为 future work |
| block 大小不对齐（vLLM 16 tokens vs OrchKvCache 64 tokens） | 中：分配逻辑复杂 | 统一 vLLM `block_size=64` 对齐 OrchKvCache；或在 Connector 层做 4:1 映射 |
| `tm_notify_attn` GPU→CPU 拷贝增加 decode 延迟 | 低：每步 < 0.1ms | 独立 CUDA stream 异步拷贝 + 采样频率可调（方案 C 作为 fallback） |
| bit-exact 验证失败（浮点精度差异） | 低：不影响功能 | 验收标准放宽为 top-1 一致 + logit 误差 < 1e-4 |

---

## 七、与论文实验的对应关系

D4 完成后，所有论文核心实验都可以运行：

| 论文实验 | 所需 Phase | 当前状态 | 备注 |
|---------|-----------|---------|------|
| Exp-M1~M4: Motivation 实验 | Phase A/B | **可以现在做** | 不需要 vLLM，仅需 GPU 测量脚本 |
| E1: 端到端吞吐 | D4 | 需等 D4 | |
| E2: Max Batch Size | D4 | 需等 D4 | |
| E3: 延迟分解 | D3/D4 | 需等 D4 | OrchKvCache 内部分解可在 Phase B 后单独做 |
| E4: 消融（存储层） | D4 | 需等 D4 | |
| E5: 冷热策略对比 | C + D4 | 需等 D4 | α/β/γ 参数 sweep |
| E6: 多粒度 vs 单粒度 | C + D4 | 需等 D4 | block_size 消融 |
| E7: 预取效果 | C5 + D4 | 需等 D4 | 预取命中率曲线 |
| E8: 存储带宽 | B | **可以现在做** | orchfs_tier 的带宽已在 Phase B 测量 |
| E9: 可扩展性 | D4 | 需等 D4 | 多请求并发 |
| E10: 生成质量验证 | D4 | 需等 D4 | LongBench + perplexity |

**可以提前做的实验**（不依赖 vLLM 集成）:
- **Exp-M1~M4**: 只需 GPU 测量脚本（Phase A 已有基础）
- **E8 存储带宽**: 只需 `orchfs_tier` 的带宽测试（Phase B `4tier_latency.json` 已有初始数据）
- **E3 延迟分解（部分）**: OrchKvCache 内部迁移延迟分解（不含推理框架部分）

---

## 八、D3/D4 待补充清单（Phase C 完成后填写）

> **本节为占位符，Phase C(C8) 完成后补充以下内容**

- [ ] `tm_notify_attn` 函数签名确认（参数类型、调用频率、是否需要批量接口）
- [ ] `tiered_manager` 的公共 API 列表（哪些需要暴露到 Python）
- [ ] Attention Hook 实现方案最终选择（A / B / C）及理由
- [ ] D4 功能验收标准确认（bit-exact vs top-1 vs perplexity threshold）
- [ ] E1~E10 实验参数完整矩阵（seq_len × batch_size × model）
- [ ] vLLM 版本最终确认（环境搭建后锁定 commit hash）
- [ ] vLLM V1 OffloadingConnector 的 `send` / `recv` 参数精确签名

---

## 九、TODO 清单

```
Phase D 总览:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ [D0] 环境准备                                                          │
  │      conda env (Python 3.11) + vLLM v0.17.2 + pybind11               │
  │      估时: 0.5d    依赖: 无（需网络）                                  │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ [D1] Python binding                                                    │
  │      orchkv_pybind.cpp → orchkv_core.so                               │
  │      GPU 指针 ↔ PyTorch tensor 互操作                                  │
  │      估时: 1.5d    依赖: D0                                           │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ [D2] vLLM OffloadingConnector 集成                                     │
  │      实现 OrchKvOffloadingConnector (send/recv)                       │
  │      注册为 --kv-offloading-backend orchkv                            │
  │      估时: 2d      依赖: D1, vLLM V1 源码理解                         │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ [D3] Attention Hook（⚠️ 依赖 Phase C）                                │
  │      vLLM attention 后端插入 tm_notify_attn                           │
  │      推荐方案 B: Python 层 wrap + 异步 reduce                         │
  │      估时: 1.5d    依赖: D1, C8                                       │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ [D4] E2E 推理测试 + 论文实验                                           │
  │      正确性验证 + 显存扩展 + 吞吐率 benchmark                          │
  │      E1~E10 实验脚本                                                   │
  │      估时: 2d+     依赖: D2, D3                                       │
  └──────────────────────────────────────────────────────────────────────────┘

  关键路径: D0 → D1 → D2 → D3 → D4  (总计 ≈ 7.5 天)
  并行窗口: D0~D2 可与 Phase C 并行推进
```

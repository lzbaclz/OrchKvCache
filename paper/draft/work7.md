# Work 7：端到端实验设计

> 目标：用真实的端到端实验证明 OrchKvCache 的核心价值
> 硬件：2× A100-80GB, 256GB DDR4, Samsung Gen5 NVMe RAID0, **无 NVM**
> 模型：Qwen2.5-7B (14.3GB BF16, 28 layers, 4 KV heads, d=128)
> 框架：vLLM 0.7.3

---

## Phase 0：集成前置工作（跑实验之前必须完成）

不做这些，后面的实验全都是假的。

### P0-1：orchkv_api 与 tiered_manager 打通

**现状**：`orchkv_api.cu` 和 `tiered_manager` 是两个独立子系统。

**要做的事**：
1. 在 `orchkv_init()` 中初始化 `tiered_manager`
2. 在 `orchkv_api` 的 `evict_to_dram` / `promote_to_gpu` 中调用 `tiered_manager` 的调度决策
3. 或者反过来：`tiered_manager` 的 `do_demote` / `do_prefetch` 调用 `orchkv_api` 的迁移函数

**验证方式**：跑现有的 `test_tiered_manager` 测试，确认调度决策能驱动真实的数据搬运。

### P0-2：prefetch 执行路径补全

**现状**：`prefetch_dispatch` 输出候选列表，但 `do_prefetch` 只累加计数器，不搬数据。

**要做的事**：在 `tiered_manager.c` 的 `do_prefetch()` 中，对每个候选调用 `mig_execute_one(PROMOTE, ...)`。

**验证方式**：在 `test_tiered_manager` 中注入几个 DRAM block，触发 prefetch，确认 block 真正回到了 GPU tier。

### P0-3：io_worker 接入主路径

**现状**：`orchkv_api.cu` 的 SSD 读写用同步 `orchfs_tier_write/read`，未走 `io_worker_submit`。

**要做的事**：将 `orchkv_evict_to_storage` / `orchkv_promote_from_storage` 改为提交到 `io_worker_pool`，用 `io_worker_flush` 等待完成。

**验证方式**：跑 `test_orchfs_tier`，确认异步 IO 路径数据正确。

### P0-4：vLLM Connector 适配

**现状**：`connector.py` 实现了 `KVConnectorBase_V1`（vLLM V1 API），但 vLLM 0.7.3 只有旧版接口。

**两条路（选一条）**：

**路线 A（推荐，快）**：不改 Connector，用 **monkey-patch** 方式在 vLLM Worker 的 attention 计算后插入钩子：
```python
# 伪代码
original_forward = model.layers[i].self_attn.forward

def hooked_forward(*args, **kwargs):
    output = original_forward(*args, **kwargs)
    # 拿到 attention output 后，通知 orchkv_core
    orchkv_core.tm_step_done()  # 触发一轮调度
    return output

model.layers[i].self_attn.forward = hooked_forward
```

**路线 B（完整，慢）**：升级 vLLM 到支持 V1 API 的版本（0.8+），然后用现有的 Connector。

**验证方式**：启动 vLLM 推理，在日志中看到 `tm_step_done` 被调用、调度统计数据被打印。

### P0-5：build_vllm_engine 的真集成

**现状**：`orchkv_enabled=True` 仅把 `swap_space` 从 4 改到 32。

**要做的事**：真正初始化 orchkv_core 并挂载到 vLLM：
```python
def build_vllm_engine(model, orchkv_enabled=False, ...):
    engine = LLM(...)
    if orchkv_enabled:
        import orchkv_core
        orchkv_core.init(gpu_pool_gb=gpu_pool_gb, dram_pool_gb=dram_pool_gb, ...)
        # 挂载钩子
        install_orchkv_hooks(engine)
    return engine
```

**验证方式**：在 orchkv_enabled=True 下推理 10 个 token，确认 tiered_manager 统计数据有值（demote_count > 0 或 prefetch_count > 0）。

---

## Phase 1：容量扩展实验 —— 证明"别人 OOM，我能跑"

> 这是最重要的一组实验，直接证明系统的核心价值。

### 实验 E-cap：最大可服务能力

**核心思路**：通过限制 GPU 显存 + 增长上下文/batch，找到每个系统的**极限点**。

**实验矩阵**：

```
固定：model = Qwen2.5-7B, max_new_tokens = 128, block_size = 16

变量 1: gpu_memory_utilization ∈ {0.3, 0.5, 0.7, 0.9}
  0.3 → 可用 24GB (模拟 RTX 4090 级别)
  0.5 → 可用 40GB
  0.7 → 可用 56GB
  0.9 → 可用 72GB (默认)

变量 2: seq_len ∈ {4096, 8192, 16384, 32768}

变量 3: batch_size ∈ {1, 2, 4, 8, 16, 32, 64}

系统：
  A. baseline   = vLLM, swap_space=4
  B. vllm-swap  = vLLM, swap_space=64 (原生 CPU offload 开到最大)
  C. orchkv     = vLLM + orchkv_core 真集成
```

**对每个 (gpu_util, seq_len, batch_size) 三元组**：
1. 尝试启动推理，记录是否 OOM / 被拒绝 / 成功
2. 如果成功，记录吞吐（tokens/s）和延迟（TTFT, TPOT）
3. 如果 OOM，记录为 FAIL

**脚本位置**：`benchmarks/exp_capacity.py`

```python
#!/usr/bin/env python3
"""E-cap: Maximum serving capacity under constrained GPU memory."""

import itertools
import gc, torch, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.bench_utils import (
    save_json, save_csv, CPUTimer, gpu_mem_mb,
    reset_gpu_peak, generate_synthetic_prompts,
)

MODEL = "Qwen/Qwen2.5-7B"
GPU_UTILS = [0.3, 0.5, 0.7, 0.9]
SEQ_LENS  = [4096, 8192, 16384, 32768]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MAX_NEW_TOKENS = 128
N_RUNS = 3

SYSTEMS = {
    "baseline":  {"swap_space": 4,  "orchkv": False},
    "vllm-swap": {"swap_space": 64, "orchkv": False},
    "orchkv":    {"swap_space": 4,  "orchkv": True},
}

def run_one(system_name, system_cfg, gpu_util, seq_len, batch_size):
    """返回 dict: 成功时含 throughput/latency，失败时含 status='OOM'"""
    result = {
        "system": system_name,
        "gpu_util": gpu_util,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "status": "UNKNOWN",
    }

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        result["status"] = "NO_VLLM"
        return result

    prompts = generate_synthetic_prompts(batch_size, seq_len)
    sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    try:
        engine_args = {
            "model": MODEL,
            "max_model_len": seq_len + MAX_NEW_TOKENS,
            "block_size": 16,
            "enforce_eager": True,
            "gpu_memory_utilization": gpu_util,
            "dtype": "auto",
            "swap_space": system_cfg["swap_space"],
        }
        engine = LLM(**engine_args)

        if system_cfg["orchkv"]:
            # TODO: 在 P0 完成后替换为真集成
            # import orchkv_core
            # orchkv_core.init(...)
            # install_orchkv_hooks(engine)
            pass

        # warmup
        try:
            engine.generate(prompt_token_ids=prompts, sampling_params=sampling)
        except Exception:
            result["status"] = "OOM_WARMUP"
            del engine; gc.collect(); torch.cuda.empty_cache()
            return result

        # benchmark
        timer = CPUTimer()
        reset_gpu_peak()
        for _ in range(N_RUNS):
            timer.start()
            outputs = engine.generate(
                prompt_token_ids=prompts, sampling_params=sampling)
            timer.stop()

        n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_tokens = batch_size * seq_len + n_out
        stats = timer.stats()
        mem = gpu_mem_mb()

        result.update({
            "status": "OK",
            "total_tokens": total_tokens,
            "output_tokens": n_out,
            "throughput_tok_s": total_tokens / (stats["avg_us"] / 1e6),
            "avg_latency_ms": stats["avg_us"] / 1e3,
            "p99_latency_ms": stats["p99_us"] / 1e3,
            "gpu_peak_mb": mem["max_allocated_mb"],
        })

        del engine
        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        result["status"] = f"ERROR: {str(e)[:100]}"
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    print("=" * 80)
    print("E-cap: Maximum Serving Capacity Experiment")
    print("=" * 80)

    all_results = []

    for gpu_util in GPU_UTILS:
        for seq_len in SEQ_LENS:
            for batch_size in BATCH_SIZES:
                for sys_name, sys_cfg in SYSTEMS.items():
                    print(f"\n--- {sys_name} | gpu={gpu_util} seq={seq_len} bs={batch_size} ---")
                    r = run_one(sys_name, sys_cfg, gpu_util, seq_len, batch_size)
                    print(f"    → {r['status']}", end="")
                    if r["status"] == "OK":
                        print(f"  {r['throughput_tok_s']:.0f} tok/s  peak={r['gpu_peak_mb']:.0f}MB")
                    else:
                        print()
                    all_results.append(r)

    save_json(all_results, "e_cap_results")
    save_csv(all_results, "e_cap_results")
    print(f"\nDone. {len(all_results)} data points.")


if __name__ == "__main__":
    main()
```

**预期结果（论文中的图表）**：

```
Figure A: 容量地图 (Capacity Heatmap)
  X 轴: seq_len (4K → 32K)
  Y 轴: batch_size (1 → 64)
  分成 3 个子图 (baseline / vllm-swap / orchkv)
  每个格子颜色: 绿=OK, 红=OOM, 黄=太慢
  gpu_memory_utilization = 0.5 (制造最大压力)
  
  预期效果: baseline 在 seq=16K, bs=8 就 OOM
           vllm-swap 能撑到 seq=16K, bs=16 (CPU swap 空间大)
           orchkv 能撑到 seq=32K, bs=32 (三级存储)

Figure B: 最大 Batch Size 对比 (Bar Chart)
  X 轴: seq_len (4K, 8K, 16K, 32K)
  Y 轴: 最大成功 batch_size
  3 组柱子: baseline / vllm-swap / orchkv
  
  预期效果: orchkv 的最大 bs 是 baseline 的 2-4x
```

---

## Phase 2：吞吐与延迟对比 —— 证明"不仅能跑，还不慢"

> 在 Phase 1 确认 orchkv 能处理更大负载后，在公平可比的负载上测速度。

### 实验 E-thru：受压场景下的吞吐对比

**核心思路**：固定在"所有系统都还能跑"的负载下，比较吞吐和延迟。

```
固定: gpu_memory_utilization = 0.5, max_new_tokens = 256

矩阵:
  seq_len    ∈ {2048, 4096, 8192}
  batch_size ∈ {1, 4, 8, 16}

系统 (同上 3 个):
  A. baseline
  B. vllm-swap
  C. orchkv

指标:
  - 吞吐: output tokens/s
  - 延迟: TTFT (首 token 延迟), TPOT (每 token 延迟)
  - GPU 峰值显存
  - 调度开销: orchkv 的 tm_step_done 耗时
```

**脚本位置**：`benchmarks/exp_throughput.py`

（脚本结构类似 exp_capacity.py，但只对三系统都能跑的点测吞吐延迟）

**预期结果（论文中的图表）**：

```
Figure C: 吞吐对比 (Grouped Bar Chart)
  每组 3 柱: baseline / vllm-swap / orchkv
  X 轴: (seq_len, batch_size) 组合
  Y 轴: output tokens/s
  
  预期效果:
    显存充裕时 (seq=2K, bs=1): 三者吞吐几乎相同 (<1% 差异)
    显存紧张时 (seq=8K, bs=16): baseline OOM, vllm-swap 吞吐下降, orchkv 正常

Figure D: 延迟分解 (Stacked Bar Chart)
  每个 bar 分成: GPU compute + scheduling + transfer + IO wait
  X 轴: seq_len
  Y 轴: per-token 延迟 (ms)
  
  预期效果: scheduling 开销 < 1% of total latency
```

---

## Phase 3：组件消融实验 —— 证明"每个组件都有用"

> 逐个关掉组件，看性能怎么变化。

### 实验 E-ablation：四配置消融

```
固定: gpu_util=0.5, seq_len=8192, batch_size=8, max_new_tokens=256

四种配置:
  (1) GPU-only:      全部 KV 在 GPU，不做任何 offload
  (2) GPU+DRAM:      冷数据只下到 DRAM，不写 SSD
  (3) GPU+DRAM+SSD:  冷数据可下到 SSD，但用同步逐块写（模拟 vLLM 式 IO）
  (4) GPU+DRAM+SSD-batched: 冷数据批量对齐写 SSD（我们的完整方案）

指标: 吞吐, 最大 batch, SSD 写带宽利用率
```

**脚本位置**：`benchmarks/exp_ablation.py`

**预期结果**：

```
Figure E: 消融对比

  配置           | 最大 bs | 吞吐 (tok/s) | SSD 写利用率
  GPU-only       |    4    |    高         |    —
  +DRAM          |   12    |    中         |    —
  +SSD (naive)   |   24    |    低 (IO慢)  |   9%
  +SSD (batched) |   24    |    中-高      |  40%+
```

### 实验 E-prefetch：预取策略对比

```
固定: gpu_util=0.5, seq_len=16384, batch_size=4

三种预取策略:
  (1) no-prefetch:  不预取，miss 时同步等待 promote
  (2) ema-prefetch: 现有的基于 EMA 的预取（budget=8）
  (3) clqa-prefetch: CLQA 预取（如果已实现）

指标: 
  - 预取命中率 (hit / dispatched)
  - 平均 miss 阻塞延迟
  - 端到端 TPOT
```

**脚本位置**：`benchmarks/exp_prefetch.py`

**预期结果**：

```
Figure F: 预取策略对比

  策略          | 命中率  | miss 阻塞 (avg) | TPOT
  no-prefetch   |   —     |  1289 μs/miss   | 3.5 ms
  ema-prefetch  |  70%    |   387 μs/miss   | 2.1 ms
  clqa-prefetch |  93%    |    90 μs/miss   | 1.6 ms
```

---

## Phase 4：输出质量验证 —— 证明"完全无损"

> 这个实验你已经有（E10），但需要在新的压力场景下重跑。

### 实验 E-quality：有压力时仍然无损

```
固定: gpu_util=0.5, greedy decoding (temperature=0)

对比:
  A. baseline (gpu_util=0.9, 全 GPU, 无任何 offload) → 作为 ground truth
  B. orchkv   (gpu_util=0.5, 有 offload/promote 发生)

输入: 5 条不同长度的 prompt (1K, 2K, 4K, 8K, 16K tokens)
每条 generate 512 tokens

指标:
  - Token 一致率: orchkv 输出的每个 token 是否与 baseline 完全一致
  - Perplexity 相对差: |PPL_orchkv - PPL_baseline| / PPL_baseline
```

**脚本位置**：`benchmarks/exp_quality.py`

**预期结果**：

```
Table: 输出质量验证

  Prompt 长度 | Token Match | PPL Diff | Offload 发生了?
     1K       |  100.0%     |  0.00%   |  否 (显存够)
     2K       |  100.0%     |  0.00%   |  部分
     4K       |  100.0%     |  0.00%   |  是
     8K       |  100.0%     |  0.00%   |  是 (大量)
    16K       |  100.0%     |  0.00%   |  是 (密集)

结论: 无论 offload 是否发生、多频繁，输出与不 offload 时 bit-exact 一致
```

---

## Phase 5（可选）：真实工作负载

> 如果时间允许，用真实数据集替代合成 prompt。

### 实验 E-real：ShareGPT / LongBench

```
数据集:
  - ShareGPT: 真实多轮对话 trace (输入/输出长度分布不均匀)
  - LongBench: 长文档 QA (长输入 + 短输出)

系统: baseline / vllm-swap / orchkv
指标: 吞吐 (req/s), 延迟 (P50/P99), 显存峰值
```

**脚本位置**：`benchmarks/exp_real_workload.py`

---

## 实验执行顺序和时间估算

```
阶段        任务               预计耗时    前置依赖
─────────────────────────────────────────────────
Phase 0     P0-1 打通 api↔tm    2-3 天     无
            P0-2 prefetch 执行   1 天       P0-1
            P0-3 io_worker 接入  1 天       P0-1
            P0-4 vLLM 钩子       2-3 天     P0-1
            P0-5 build_engine    1 天       P0-4
─────────────────────────────────────────────────
Phase 1     E-cap 容量实验       1-2 天     Phase 0 全部
Phase 2     E-thru 吞吐实验      1 天       Phase 1
Phase 3     E-ablation 消融      1 天       Phase 1
            E-prefetch 预取      1 天       Phase 1
Phase 4     E-quality 质量       半天       Phase 1
Phase 5     E-real 真实负载      1-2 天     Phase 2
─────────────────────────────────────────────────
            合计                 12-16 天
```

---

## 预期的论文级图表清单

| 图表 | 类型 | 来自实验 | 证明什么 |
|------|------|---------|---------|
| Fig A | 容量热力图 | E-cap | OrchKvCache 可服务的范围远大于 baseline |
| Fig B | 最大 batch 柱状图 | E-cap | 最大并发量提升 2-4x |
| Fig C | 吞吐对比柱状图 | E-thru | 显存紧张时吞吐不崩 |
| Fig D | 延迟分解堆叠图 | E-thru | 调度开销可忽略 (<1%) |
| Fig E | 消融对比表 | E-ablation | 每个组件贡献有效 |
| Fig F | 预取策略对比 | E-prefetch | CLQA 命中率远高于 EMA |
| Tab 1 | 质量验证表 | E-quality | 100% 无损 |

**这 7 个图表 + 前面的 motivation 实验（M2 注意力分布、M3 IO 效率），加起来正好撑满 SC 的 10 页正文。**

---

## 最关键的一组数字（论文成败在此）

如果整个实验只能保留一组数字，那就是这个：

```
在 A100-80GB (gpu_util=0.5, 有效 40GB) 上用 Qwen2.5-7B:

             | seq=16K, bs=8          | seq=32K, bs=4
─────────────┼────────────────────────┼──────────────────────
 baseline    | OOM ❌                  | OOM ❌
 vllm-swap   | OK, 但吞吐下降 40% ⚠️  | OOM ❌
 OrchKvCache | OK, 吞吐仅下降 8% ✅    | OK, 吞吐下降 15% ✅
```

这一张表就能说明：**OrchKvCache 在别人做不到的场景下正常工作，在别人勉强能做的场景下效果更好。**

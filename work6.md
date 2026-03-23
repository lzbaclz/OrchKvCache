# Phase E: 论文实验 — 数据采集与分析

> **前置状态**: Phase A/B/C/D 全部完成 ✅
> **目标**: 运行 E1~E10 十组实验，采集论文所需的全部定量数据
> **硬件**: A100-SXM4-80GB / 256GB DRAM / NVMe Gen5 SSD / CUDA 12.0
> **软件**: conda env `orchkv` — Python 3.11 + PyTorch 2.5.1+cu121 + pybind11 3.0.2
> **vLLM**: 需安装 v0.17.2（E1-E4/E6/E10 依赖）
> **模型**: meta-llama/Llama-2-7b-hf（Phase E 完成后可扩展到 13B / 70B）

---

## 一、全局实验参数

```
Models       : ["meta-llama/Llama-2-7b-hf"]
Seq_lens     : [1024, 4096, 8192, 16384, 32768, 65536]
Batch_sizes  : [1, 4, 8, 16, 32]
Datasets     : ["ShareGPT", "LongBench-subset", "Synthetic-uniform"]
Block_sizes  : [16, 32, 64, 128]                        # E6 消融
α/β/γ        : grid {α+β+γ=1.0, 步长 0.1}              # E5 消融
Prefetch     : [off, budget=4, 8, 16, 32]                # E7 消融
Tiers        : [GPU-only, GPU+DRAM, GPU+DRAM+NVM, 4-tier] # E4 消融
Temperature  : 0 (greedy decoding, 保证可复现性)
Seed         : 42
max_new_tokens: 128 (吞吐实验) / 64 (正确性实验)
n_warmup     : 2 轮
n_runs       : 3~5 轮 (取均值与置信区间)
```

---

## 二、实验总览

```
┌──────┬──────────────────────┬───────────────────┬────────────┬──────────┐
│ 编号 │ 实验名称              │ 脚本               │ 需 vLLM？  │ 状态     │
├──────┼──────────────────────┼───────────────────┼────────────┼──────────┤
│ E1   │ 端到端吞吐            │ benchmark_e2e.py  │ ✅ 是      │ TODO     │
│ E2   │ 最大 Batch Size       │ benchmark_e2e.py  │ ✅ 是      │ TODO     │
│ E3   │ 延迟分解              │ benchmark_e2e.py  │ ✅ 是      │ TODO     │
│ E4   │ 存储层消融            │ benchmark_ablation│ ✅ 是      │ TODO     │
│ E5   │ 冷热策略参数 sweep    │ benchmark_ablation│ ❌ 否      │ TODO     │
│ E6   │ Block Size 粒度消融   │ benchmark_ablation│ ✅ 是      │ TODO     │
│ E7   │ 预取效果              │ benchmark_prefetch│ ❌ 否      │ TODO     │
│ E8   │ 存储带宽              │ benchmark_storage │ ❌ 否      │ TODO     │
│ E9   │ 调度可扩展性          │ benchmark_scalabil│ ❌ 否      │ TODO     │
│ E10  │ 生成质量              │ eval_quality.py   │ ✅ 是      │ TODO     │
└──────┴──────────────────────┴───────────────────┴────────────┴──────────┘
```

---

## 三、各实验详细规划

### E1: 端到端吞吐率

**论文定位**: 核心实验 — 证明 OrchKvCache 在不同序列长度/批量下的吞吐表现

**方法**:
- 对 (seq_len × batch_size) 全矩阵，分别运行 baseline vLLM 和 OrchKvCache
- 测量 throughput (tokens/s)、TTFT (Time-To-First-Token)、TPOT (Time-Per-Output-Token)
- 每个配置 warmup 2 轮，跑 3~5 轮取均值

**参数矩阵**:

| 维度 | 取值 |
|------|------|
| seq_len | 1024, 4096, 8192, 16384, 32768 |
| batch_size | 1, 4, 8, 16 |
| backend | baseline, orchkv |

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_e2e.py \
    --seq-lens 1024,4096,8192,16384,32768 \
    --batch-sizes 1,4,8,16 \
    --n-runs 5
```

**输出**: `benchmarks/results/benchmark_e2e.json` + `benchmark_e2e.csv`

**预期图表**:
- 图 1: 吞吐 vs seq_len（固定 bs=4），两条线（baseline / orchkv）
- 图 2: 吞吐 vs batch_size（固定 seq=4096），两条线
- 图 3: TTFT 对比柱状图

**验收标准**:
- 短序列 (≤4K): 退化 < 5%
- 长序列 (≥16K): 吞吐提升 ≥ 20%

---

### E2: 最大 Batch Size（显存扩展）

**论文定位**: 证明 OrchKvCache 的显存扩展能力

**方法**:
- 固定 seq_len，逐步增大 batch_size，直到 baseline OOM
- 验证 OrchKvCache 在 baseline OOM 后仍能正常运行
- 记录各 seq_len 下两种方案的 max_batch_size

**参数矩阵**:

| seq_len | batch_size sweep |
|---------|-----------------|
| 1024 | 1, 2, 4, 8, 16, 32, 64 |
| 4096 | 1, 2, 4, 8, 16, 32 |
| 16384 | 1, 2, 4, 8, 16 |
| 32768 | 1, 2, 4, 8 |

**运行命令**:
```bash
conda run -n orchkv python benchmarks/test_e2e_inference.py --test memory \
    --seq-len 4096
```

**输出**: `benchmarks/results/test_memory_extension.json`

**预期图表**:
- 图 4: 柱状图，baseline vs orchkv max_batch_size @ 各 seq_len
- 期望 orchkv max_batch ≥ 2× baseline

---

### E3: 延迟分解

**论文定位**: 分析 OrchKvCache 的开销来源

**方法**:
- 将总延迟分解为：纯计算时间 / KV 迁移时间 / 调度决策时间
- 对比 baseline（无迁移/调度）与 orchkv 的各部分占比
- 使用 CPU timer 高精度测量各阶段

**参数**: seq_len=4096, batch_size=4, max_new_tokens=128, n_runs=5

**运行命令**:
```bash
conda run -n orchkv python benchmarks/test_e2e_inference.py --test latency \
    --seq-len 4096 --batch-size 4
```

**输出**: `benchmarks/results/test_latency_breakdown.json`

**预期图表**:
- 图 5: 堆叠柱状图，延迟分解 (compute / migrate / schedule)
- 期望 schedule < 1% 总延迟，migrate < 5%

---

### E4: 存储层消融

**论文定位**: 证明多级存储的必要性

**方法**:
- 4 种配置逐一对比：GPU-only → GPU+DRAM → GPU+DRAM+NVM → 完整 4 级
- 固定负载 (seq=4096, bs=4)，测量吞吐和 GPU 峰值内存

**配置矩阵**:

| 配置 | 存储层数 | DRAM pool |
|------|---------|-----------|
| GPU-only | 1 | 0 |
| GPU+DRAM | 2 | 8 GB |
| GPU+DRAM+NVM | 3 | 8 GB |
| GPU+DRAM+NVM+SSD | 4 | 8 GB |

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_ablation.py --exp e4 \
    --seq-len 4096 --batch-size 4
```

**输出**: `benchmarks/results/benchmark_e4_tier_ablation.json`

**预期图表**:
- 图 6: 柱状图，吞吐 vs 存储层配置
- 图 7: GPU 峰值内存 vs 存储层配置
- 期望：层数越多 → GPU 峰值越低，且吞吐不显著下降

---

### E5: 冷热策略参数 sweep

**论文定位**: 分析热度公式 `score = α×attention + β×recency + γ×frequency` 的参数敏感性

**方法**:
- 在 α+β+γ=1.0 约束下做网格搜索
- 纯 orchkv_core 层面评估，不需要模型推理
- 测量各参数组合下的分类结果 (n_hot/n_warm/n_cold) 和迁移次数

**参数矩阵**:
```
α ∈ [0.2, 0.5, 0.8]
β ∈ [0.1, 0.3, 0.5]
γ ∈ [0.1, 0.2, 0.4]
约束: α + β + γ ≈ 1.0
n_blocks = 256, n_steps = 100
```

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_ablation.py --exp e5
```

**输出**: `benchmarks/results/benchmark_e5_policy_sweep.json` + `.csv`

**预期图表**:
- 图 8: 热力图 (α vs β)，颜色 = demote 次数
- 图 9: 三角图或柱状图，不同 α/β/γ 组合的 hot/cold 分布
- 期望：α 越高 → 注意力驱动的分类越准

**⚡ 可以立即运行 — 不依赖 vLLM**

---

### E6: Block Size 粒度消融

**论文定位**: 分析不同 KV block 粒度对传输效率和缓存命中率的影响

**方法**:
- 4 种 block_size (16/32/64/128 tokens per block) 下分别运行
- 固定负载，测量吞吐和传输效率

**参数**: seq_len=4096, batch_size=4, block_size ∈ {16, 32, 64, 128}

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_ablation.py --exp e6 \
    --seq-len 4096 --batch-size 4
```

**输出**: `benchmarks/results/benchmark_e6_block_size.json`

**预期图表**:
- 图 10: 吞吐 vs block_size
- 期望：block_size=16~64 为甜区（太小碎片化严重，太大浪费空间）

---

### E7: 预取效果

**论文定位**: 证明预取机制的有效性和预算的影响

**方法**:
- Sweep prefetch_budget ∈ {0, 4, 8, 16, 32}
- 模拟 decode 负载，25% blocks 为 hot，75% 为 cold
- 测量 prefetch 命中率、调度延迟、迁移次数

**参数**: n_blocks=256, n_steps=100, budget ∈ {0, 4, 8, 16, 32}

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_prefetch.py \
    --n-blocks 256 --n-steps 100 --budgets 0,4,8,16,32
```

**输出**: `benchmarks/results/benchmark_e7_prefetch.json` + `.csv`

**预期图表**:
- 图 11: 预取命中率 vs prefetch_budget（折线图）
- 图 12: 调度延迟 vs prefetch_budget
- 期望：budget=0 时命中率=0；budget=16 时命中率趋稳（边际递减）

**⚡ 可以立即运行 — 不依赖 vLLM**

---

### E8: 存储带宽

**论文定位**: 量化各层存储的物理传输上限

**方法**:
- GPU HBM ↔ Host DRAM: cudaMemcpyAsync (pinned memory)
- Host DRAM ↔ tmpfs/NVM: POSIX pwrite/pread
- 多种数据块大小 (0.5MB ~ 64MB) 下测量双向带宽

**参数**: sizes ∈ {0.5, 1, 2, 4, 8, 16, 32, 64} MB, n_iter=10

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_storage_bw.py \
    --sizes 0.5,1,2,4,8,16,32,64 --n-iter 10
```

**输出**:
- `benchmarks/results/benchmark_e8_storage_bw.json`
- `benchmarks/results/benchmark_e8_gpu_dram.csv`
- `benchmarks/results/benchmark_e8_dram_storage.csv`

**预期图表**:
- 图 13: 带宽 vs 数据块大小，4 条线 (GPU→DRAM, DRAM→GPU, DRAM→tmpfs, tmpfs→DRAM)
- 预期量级：GPU↔DRAM ~20-50 GB/s，DRAM↔tmpfs ~5-15 GB/s

**⚡ 可以立即运行 — 不依赖 vLLM**

---

### E9: 调度可扩展性

**论文定位**: 证明调度器在大规模 block 数量下仍然高效

**方法**:
- 逐步增加 tracked block 数量 (64 → 4096)
- 每个配置跑 50 步，每步做 attention report + schedule_once
- 测量调度延迟的 avg / p50 / p99

**参数**: n_blocks ∈ {64, 128, 256, 512, 1024, 2048, 4096}, n_steps=50

**运行命令**:
```bash
conda run -n orchkv python benchmarks/benchmark_scalability.py \
    --max-blocks 4096 --n-steps 50
```

**输出**: `benchmarks/results/benchmark_e9_scalability.json` + `.csv`

**预期图表**:
- 图 14: 调度延迟 (avg/p99) vs n_blocks（对数 X 轴）
- 期望：延迟随 block 数近线性增长，4096 blocks 下 p99 < 500μs

**⚡ 可以立即运行 — 不依赖 vLLM**

---

### E10: 生成质量验证

**论文定位**: 证明 OrchKvCache 不影响模型输出质量

**方法**:
- **E10a: Token 一致性** — greedy decoding (T=0, seed=42)，对比 baseline 和 orchkv 输出
- **E10b: Perplexity 代理** — 比较 cumulative_logprob，计算近似 perplexity
- 固定 20 个 prompt，每个 512 tokens 输入 + 128 tokens 输出

**运行命令**:
```bash
conda run -n orchkv python benchmarks/eval_quality.py \
    --model meta-llama/Llama-2-7b-hf \
    --seq-len 512 --n-samples 20
```

**输出**: `benchmarks/results/eval_e10_quality.json`

**预期结果**:
- Token 一致率 ≥ 99.9%
- Perplexity 相对误差 < 1%

**验收标准**:

| 指标 | 目标 | 允许偏差 |
|------|------|---------|
| Top-1 token 一致率 | 100% | ≥ 99.9% |
| Logit 绝对误差 | 0 | < 1e-4 |
| Perplexity 相对差 | 0% | < 1% |

---

## 四、执行顺序与依赖

```
Phase 1 — 立即可跑（不依赖 vLLM）:
  ┌─ E5: 冷热策略 sweep         (~2 min)
  ├─ E7: 预取效果               (~1 min)
  ├─ E8: 存储带宽               (~3 min)
  └─ E9: 调度可扩展性           (~2 min)

Phase 2 — 安装 vLLM 后:
  ┌─ pip install vllm==0.17.2
  ├─ huggingface-cli download meta-llama/Llama-2-7b-hf
  └─ 验证: python -c "from vllm import LLM; print('OK')"

Phase 3 — vLLM 实验（需模型）:
  ┌─ E10: 生成质量验证           (~20 min, 最先跑，确认正确性)
  ├─ E1:  端到端吞吐 sweep       (~2 h, 全矩阵)
  ├─ E2:  最大 Batch Size        (~30 min)
  ├─ E3:  延迟分解               (~15 min)
  ├─ E4:  存储层消融             (~30 min)
  └─ E6:  Block Size 消融        (~30 min)
```

**建议执行策略**:
1. 先跑 Phase 1（E5/E7/E8/E9），立即出一批数据
2. 并行安装 vLLM + 下载模型
3. vLLM 就绪后，先跑 E10 确认正确性无问题
4. 正确性确认后，再跑 E1~E4/E6 的大矩阵

---

## 五、输出文件汇总

所有结果输出到 `benchmarks/results/` 目录:

| 文件 | 实验 | 格式 |
|------|------|------|
| `benchmark_e2e.json` / `.csv` | E1-E3 | 全矩阵吞吐/延迟数据 |
| `test_memory_extension.json` | E2 | max_batch 对比 |
| `test_latency_breakdown.json` | E3 | 延迟分解 |
| `benchmark_e4_tier_ablation.json` | E4 | 分层消融结果 |
| `benchmark_e5_policy_sweep.json` / `.csv` | E5 | α/β/γ 网格搜索 |
| `benchmark_e6_block_size.json` | E6 | block_size 消融 |
| `benchmark_e7_prefetch.json` / `.csv` | E7 | 预取命中率数据 |
| `benchmark_e8_storage_bw.json` | E8 | 存储带宽 |
| `benchmark_e8_gpu_dram.csv` | E8 | GPU↔DRAM 带宽表 |
| `benchmark_e8_dram_storage.csv` | E8 | DRAM↔tmpfs 带宽表 |
| `benchmark_e9_scalability.json` / `.csv` | E9 | 调度延迟扩展性 |
| `eval_e10_quality.json` | E10 | 质量验证结果 |

---

## 六、论文图表映射

| 论文图/表 | 数据来源 | 图表类型 |
|----------|---------|---------|
| Fig.1 吞吐 vs seq_len | E1 | 折线图 (2 lines: baseline/orchkv) |
| Fig.2 吞吐 vs batch_size | E1 | 折线图 |
| Fig.3 TTFT 对比 | E1 | 柱状图 |
| Fig.4 Max Batch Size | E2 | 柱状图 (grouped by seq_len) |
| Fig.5 延迟分解 | E3 | 堆叠柱状图 |
| Fig.6 存储层消融-吞吐 | E4 | 柱状图 |
| Fig.7 存储层消融-内存 | E4 | 柱状图 |
| Fig.8 策略参数热力图 | E5 | 热力图 (α vs β → demotes) |
| Fig.9 hot/cold 分布 | E5 | 分组柱状图 |
| Fig.10 Block Size 消融 | E6 | 折线图 |
| Fig.11 预取命中率 | E7 | 折线图 (hit_rate vs budget) |
| Fig.12 预取调度延迟 | E7 | 折线图 |
| Fig.13 存储带宽 | E8 | 折线图 (4 lines, log X) |
| Fig.14 调度扩展性 | E9 | 折线图 (avg/p99, log X) |
| Tab.1 质量验证 | E10 | 表格 (match_rate, PPL diff) |

---

## 七、TODO 清单

```
Phase E 实验执行:

  ┌──────────────────────────────────────────────────────────────┐
  │ [E-prep] 环境准备                                状态: TODO │
  │   pip install vllm==0.17.2                                  │
  │   huggingface-cli download meta-llama/Llama-2-7b-hf         │
  │   估时: 0.5d (视网络)                                       │
  ├──────────────────────────────────────────────────────────────┤
  │ [E5]  冷热策略 sweep        ⚡ 可立即跑         状态: TODO │
  │ [E7]  预取效果              ⚡ 可立即跑         状态: TODO │
  │ [E8]  存储带宽              ⚡ 可立即跑         状态: TODO │
  │ [E9]  调度可扩展性          ⚡ 可立即跑         状态: TODO │
  ├──────────────────────────────────────────────────────────────┤
  │ [E10] 生成质量验证          需 vLLM + 模型      状态: TODO │
  │ [E1]  端到端吞吐            需 vLLM + 模型      状态: TODO │
  │ [E2]  最大 Batch Size       需 vLLM + 模型      状态: TODO │
  │ [E3]  延迟分解              需 vLLM + 模型      状态: TODO │
  │ [E4]  存储层消融            需 vLLM + 模型      状态: TODO │
  │ [E6]  Block Size 消融       需 vLLM + 模型      状态: TODO │
  └──────────────────────────────────────────────────────────────┘

  建议路径: E5+E7+E8+E9 → 安装vLLM → E10 → E1 → E2 → E3 → E4 → E6
  总计估时: ~1 天环境 + ~1 天跑实验 + ~1 天整理数据
```

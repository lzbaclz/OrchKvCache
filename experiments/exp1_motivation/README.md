# Exp1: Motivation 实验 — 用真实数据证明问题存在

> **实验日期**: 2026-03-20
> **目的**: 为 OrchKvCache 论文的 §2 Motivation 部分提供数据支撑
> **模型**: Qwen2.5-1.5B (28 layers, 12 heads, 2 KV heads, d=128)
> **硬件**: 2×A100-SXM4-80GB, 376GB DRAM, Samsung RAID0 Gen5 NVMe

---

## 实验矩阵

| 编号 | 实验 | 脚本 | 结果 | 论文图 |
|------|------|------|------|--------|
| M1 | KV-Cache 显存瓶颈量化 | `scripts/m1_kvcache_memory.py` + `scripts/m1_real_memory.py` | `results/m1_kvcache_theory.json` + `results/m1_real_memory.json` | Fig.2 |
| M2 | 注意力分数分布 (冷热分化) | `scripts/m2_attention_analysis.py` | `results/m2_attention_analysis.json` | Fig.3 |
| M3 | 现有 offloading IO 效率 | `scripts/m3_io_efficiency.py` | `results/m3_io_efficiency.json` | Fig.4 |
| M4 | 存储层级延迟对比 | `scripts/m4_tier_comparison.py` | `results/m4_tier_comparison.json` | Fig.5 |

---

## 核心结论

### M1: KV-Cache 是显存的主要瓶颈

- **LLaMA-2-7B** 在 seq=32768 时, 单请求 KV-Cache = **16 GB**, 超过模型权重 (14 GB)
- A100-80GB 跑 LLaMA-2-7B, seq=4096 时最多服务 **32 个并发请求** (KV-Cache 占 87.5%)
- GQA 模型 (LLaMA-3-8B) KV-Cache 小 4x, 但长上下文仍是瓶颈
- **交叉点**: LLaMA-2-7B 在 seq≈28672 时, KV-Cache 超过模型权重

### M2: 注意力分数呈强烈幂律分布, 冷热分化显著

- **Top-10% token 贡献 90-93% 的注意力权重** (跨 3 种输入长度)
- **Top-20% token 贡献 95-96%** 的注意力权重
- Gini 系数 = -0.93 ~ -0.95 (极度不均匀)
- **Block 粒度仍有效**: block_size=16 时 Top-10% block 覆盖 ~80% 注意力
- **层间差异大**: 中间层 (Layer 4-20) 集中度最高 (Top-10% > 95%), 首层和末层较均匀
- **Attention sink 存在**: 前 5 个 token 吸引了不成比例的注意力
- **跨 decode step 稳定性**: Jaccard 相似度 0.47-0.70, 热 token 集合有一定持续性

### M3: 现有 offloading IO 效率极低

- **Naive (vLLM 式) 逐块同步 eviction**: SSD 带宽利用率仅 **4-26%**
- **Batched 合并写入**: 利用率提升到 **23-41%**, 但仍远低于 SSD 峰值
- **小块 (64KB) 效率最差**: 16-token vLLM block 的 naive eviction 仅 0.23 GB/s (SSD 利用率 4.3%)
- **大块略好**: 64-token block 的 naive eviction 达 1.15 GB/s (利用率 21.6%)
- **Reload (读) 效率也低**: 同步逐层读取仅 1.5-2.9 GB/s (利用率 8-16%)

### M4: DRAM 缓冲层价值巨大

- 4MB 数据: **DRAM 读延迟 18.5 us** vs **SSD 读延迟 349.7 us** → DRAM 快 **18.9x**
- 4MB 数据: **DRAM 写延迟 18.5 us** vs **SSD 写延迟 1359 us** → DRAM 快 **73.5x**
- 16MB 数据: DRAM 快 **46x (读)** 和 **168x (写)**
- DRAM 拷贝带宽接近 GPU D2D (小块时甚至更快, 因为纯内存操作无 PCIe 开销)
- **结论**: DRAM 作为温存储层可将频繁访问的 KV-Cache reload 延迟降低 1-2 个数量级

---

## 四个核心 Insight → Design Decision

| # | Insight (数据支撑) | Design Decision |
|---|-------------------|-----------------|
| 1 | KV-Cache 在长上下文下主导显存 (M1) | 必须将 KV-Cache offload 到外部存储 |
| 2 | 注意力呈幂律分布, 冷热分化显著 (M2) | 按 block 粒度做冷热分级, 仅保留 Top-20% 在 GPU |
| 3 | 现有 IO 路径效率极低 (M3) | 用 OrchFS 的 io_uring + 对齐写入替代 POSIX IO |
| 4 | DRAM 比 SSD 快 19-168x (M4) | 引入 DRAM 温缓冲层, 避免频繁 SSD 访问 |

---

## 文件结构

```
exp1_motivation/
├── README.md
├── scripts/
│   ├── m1_kvcache_memory.py       # M1: 理论 KV-Cache 大小计算
│   ├── m1_real_memory.py          # M1: 真实 GPU 显存测量
│   ├── m2_attention_analysis.py   # M2: 注意力分数分布分析
│   ├── m3_io_efficiency.py        # M3: IO 效率对比
│   └── m4_tier_comparison.py      # M4: 存储层级延迟对比
├── results/
│   ├── m1_kvcache_theory.json
│   ├── m1_kvcache_size.csv
│   ├── m1_real_memory.json
│   ├── m2_attention_analysis.json
│   ├── m3_io_efficiency.json
│   └── m4_tier_comparison.json
└── figures/                       # (待生成) 论文用图
```

## 运行方式

```bash
conda activate orchkv
cd OrchKvCache/experiments/exp1_motivation

# M1
python scripts/m1_kvcache_memory.py
CUDA_VISIBLE_DEVICES=0 python scripts/m1_real_memory.py

# M2 (需要 float32, 约 6GB 显存)
CUDA_VISIBLE_DEVICES=0 python scripts/m2_attention_analysis.py

# M3
CUDA_VISIBLE_DEVICES=0 python scripts/m3_io_efficiency.py

# M4
CUDA_VISIBLE_DEVICES=0 python scripts/m4_tier_comparison.py
```

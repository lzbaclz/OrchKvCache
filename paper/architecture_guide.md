# OrchKvCache 架构图绘制指南

---

## 一、系统架构总览

整个系统分成**三个纵向区域**（左中右），**四个横向存储层**（上到下），核心数据流是一个循环。

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OrchKvCache System                           │
│                                                                      │
│  ┌─────────────┐    ┌────────────────────────┐    ┌──────────────┐  │
│  │  LLM Engine │    │    Scheduler (C1-C8)    │    │ Storage Tiers│  │
│  │  (vLLM)     │    │                        │    │              │  │
│  │             │    │  ┌──────────────────┐   │    │  ┌────────┐ │  │
│  │ ┌─────────┐ │ attn│  │ C1 Attn Tracker  │   │    │  │ T0:GPU │ │  │
│  │ │Attention│─┼─scores→ (EMA per block)  │   │    │  │  HBM   │ │  │
│  │ │ Kernel  │ │    │  └────────┬─────────┘   │    │  │ 80GB   │ │  │
│  │ └─────────┘ │    │           ▼             │    │  └───┬────┘ │  │
│  │             │    │  ┌──────────────────┐   │    │      │      │  │
│  │ ┌─────────┐ │    │  │ C2 Hot/Cold      │   │    │  cudaMemcpy │  │
│  │ │ Block   │─┼─alloc→  Classifier      │   │    │  Async      │  │
│  │ │Allocator│ │ /free│  │ S=α·a+β·R+γ·F  │   │    │      │      │  │
│  │ └─────────┘ │    │  └────────┬─────────┘   │    │  ┌───▼────┐ │  │
│  │             │    │           ▼             │    │  │ T1:DRAM│ │  │
│  │ ┌─────────┐ │    │  ┌──────────────────┐   │    │  │ Pinned │ │  │
│  │ │   KV    │←┼─promote│ C3 Adaptive      │   │    │  │ 256GB  │ │  │
│  │ │Connector│ │ /demote│    Threshold     │   │    │  └───┬────┘ │  │
│  │ └─────────┘ │    │  │ (HWM/LWM水位线)  │   │    │      │      │  │
│  │             │    │  └────────┬─────────┘   │    │  pwrite/    │  │
│  └─────────────┘    │           ▼             │    │  pread      │  │
│                     │  ┌──────────────────┐   │    │      │      │  │
│                     │  │ C4 Eviction      │   │    │  ┌───▼────┐ │  │
│                     │  │    Policy        │   │    │  │T2: NVM │ │  │
│                     │  │ (加权LRU选victim)│   │    │  │ 4KB页  │ │  │
│                     │  └────────┬─────────┘   │    │  │ ~128GB │ │  │
│                     │           ▼             │    │  └───┬────┘ │  │
│                     │  ┌──────────────────┐   │    │      │      │  │
│                     │  │ C5 Prefetch      │   │    │  aligned IO │  │
│                     │  │   Scheduler     │   │    │      │      │  │
│                     │  │ (EMA堆排序预取) │   │    │  ┌───▼────┐ │  │
│                     │  └────────┬─────────┘   │    │  │T3: SSD │ │  │
│                     │           ▼             │    │  │32KB块  │ │  │
│                     │  ┌──────────────────┐   │    │  │ ~4TB   │ │  │
│                     │  │ C7 Migration     │───┼────┼─→└────────┘ │  │
│                     │  │    Engine        │   │    │              │  │
│                     │  └────────┬─────────┘   │    │  ┌────────┐ │  │
│                     │           ▼             │    │  │Transfer│ │  │
│                     │  ┌──────────────────┐   │    │  │Engine  │ │  │
│                     │  │ C8 Tiered Manager│   │    │  │(4 CUDA │ │  │
│                     │  │ (统一调度入口)    │   │    │  │streams)│ │  │
│                     │  └──────────────────┘   │    │  ├────────┤ │  │
│                     │                        │    │  │IO Worker│ │  │
│                     │                        │    │  │Pool     │ │  │
│                     │                        │    │  │(8线程)  │ │  │
│                     └────────────────────────┘    │  └────────┘ │  │
│                                                    └──────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 二、数据流循环（每个 decode step）

```
① vLLM Attention Kernel 产生注意力分数
     ↓
② C1 Tracker 接收并 EMA 平滑
     ↓
③ C2 Classifier 给每个 block 打分 → Hot/Warm/Cold
     ↓
④ C3 检查 GPU 水位，调整阈值
     ↓
⑤ C4 选出要换出的 victim blocks
     ↓                              ⑥ C5 选出要预取的 blocks
     ↓                                   ↓
⑦ C7 Migration Engine 执行搬运：
   - demote: GPU →(cudaMemcpy)→ DRAM →(pwrite)→ NVM/SSD
   - promote: SSD →(pread)→ DRAM →(cudaMemcpy)→ GPU
     ↓
⑧ C8 Tiered Manager 协调以上全部
     ↓
回到 ① 下一个 decode step
```

---

## 三、推荐画图布局

### 整体布局：左中右三列

```
┌─────────┐  ┌────────────────────────┐  ┌─────────────────┐
│  左列    │  │       中列              │  │     右列         │
│ LLM      │  │    OrchKvCache          │  │  Storage         │
│ Engine   │  │    Scheduler            │  │  Hierarchy       │
│          │→ │                        │→ │                  │
│ vLLM     │  │  C1→C2→C3              │  │  [GPU HBM]  蓝   │
│ Attention│  │       ↓                 │  │      ↕           │
│ Kernel   │  │  C4  C5                │  │  [DRAM]     绿   │
│          │← │       ↓                │← │      ↕           │
│          │  │  C7 Migration          │  │  [NVM]      橙   │
│          │  │       ↓                │  │      ↕           │
│          │  │  C8 Tiered Mgr         │  │  [SSD]      紫   │
└─────────┘  └────────────────────────┘  └─────────────────┘
```

### 右列四级存储（从上到下）

- 容量递增：80GB → 256GB → 128GB → 4TB
- 速度递减：274 GB/s → 23 GB/s → ~300ns → 3-18 GB/s
- 每层之间标注传输机制

---

## 四、配色方案

| 存储层 | 底色 (fill) | 边框 (stroke) | 含义 |
|--------|------------|--------------|------|
| GPU HBM | `#E3F2FD` | `#1565C0` | 蓝色系 — 最快最珍贵 |
| DRAM | `#E8F5E9` | `#2E7D32` | 绿色系 — 中间缓冲层 |
| NVM | `#FFF3E0` | `#E65100` | 橙色系 — 温存储 |
| SSD | `#F3E5F5` | `#6A1B9A` | 紫色系 — 冷存储大容量 |
| Scheduler | `#FFF8E1` | `#F57F17` | 浅黄系 — 决策逻辑 |
| vLLM Engine | `#ECEFF1` | `#455A64` | 灰色系 — 外部系统 |

### 箭头规范

- **实线箭头**：数据流（KV block 搬运方向）
- **虚线箭头**：控制流 / 信号（attention scores、alloc events）
- **粗箭头**：主数据路径（demote/promote）
- **细箭头**：辅助信号

---

## 五、关键标注

在图中需要标注的关键信息：

### 左→中 箭头
- `attention scores` (虚线)
- `alloc/free events` (虚线)

### 中→右 箭头
- `demote` (向下实线, 红/橙色)
- `promote` (向上实线, 绿/蓝色)

### 右列每层
- 容量数字：80GB / 256GB / 128GB / 4TB
- 带宽数字：274 GB/s / 23 GB/s / ~300ns / 18 GB/s read
- 传输机制：cudaMemcpyAsync / pwrite-pread / OrchFS aligned IO

### C2 分类器旁
- 小公式：`S = α·attn + β·recency + γ·freq`
- 三级标签：Hot / Warm / Cold

### C3 自适应阈值旁
- HWM/LWM 水位线示意（一个小柱状图）

---

## 六、draw.io 画图步骤

### 第一步：设置画布
1. 打开 draw.io (https://app.diagrams.net)
2. File → Page Setup → 宽度 18cm（单栏）或 8.5cm（双栏），高度 12cm
3. 背景白色

### 第二步：画三列框架
1. 用 Rectangle 画三个大框：LLM Engine / Scheduler / Storage Tiers
2. 给每个框加标题

### 第三步：画四级存储
1. 右列从上到下放四个圆角矩形
2. 填充对应颜色
3. 标注名称、容量、带宽
4. 层与层之间画双向箭头，标注传输机制

### 第四步：画调度组件
1. 中列放 C1-C8 的方块，按从上到下的 pipeline 排列
2. 用箭头连接

### 第五步：画数据流
1. 左→中的虚线箭头（信号）
2. 中→右的实线箭头（demote/promote）
3. 标注箭头名称

### 第六步：字体和格式
- 字体：Times New Roman 或 Helvetica
- 组件标题：11pt 粗体
- 标注文字：9pt 常规
- 确保黑白打印可读

---

## 七、10 篇 CCF-A 优秀架构图参考

| # | 论文 | 会议 | 重点学习 | PDF |
|---|------|------|---------|-----|
| 1 | vLLM / PagedAttention | SOSP'23 | Figure 3: 页表映射图 — 颜色区分请求 | https://arxiv.org/pdf/2309.06180 |
| 2 | FlexGen | ICML'23 | Figure 1: 三级 offloading — 和本系统最像 | https://arxiv.org/pdf/2303.06865 |
| 3 | InfiniGen | OSDI'24 | Figure 3: 预取流水线时序 — pipeline 图参考 | https://www.usenix.org/system/files/osdi24-lee.pdf |
| 4 | Orca | OSDI'22 | Figure 2: 新旧方案对比 — 简洁有力 | https://www.usenix.org/system/files/osdi22-yu.pdf |
| 5 | FlashAttention | NeurIPS'22 | Figure 1: 内存层次 tile — HBM/SRAM 两级 | https://arxiv.org/pdf/2205.14135 |
| 6 | Splitwise | ISCA'24 | Figure 1: Prefill-Decode 分离 | https://arxiv.org/pdf/2311.18677 |
| 7 | DeepSpeed-Inference | SC'22 | Figure 2: 异构推理引擎 — 与本系统最相似 | https://arxiv.org/pdf/2207.00032 |
| 8 | Strata | SOSP'17 | Figure 2: NVM+SSD 分层 — 经典存储层次图 | https://www.cs.utexas.edu/~simon/sosp17-final207.pdf |
| 9 | DistServe | OSDI'24 | Figure 2: 多机部署 + KV 传输 | https://arxiv.org/pdf/2401.09670 |
| 10 | Mooncake | arXiv'24 | Figure 2: KVCache-centric 三部分分离 | https://arxiv.org/pdf/2407.00079 |

### 重点模仿对象

- **存储分层布局** → 学 FlexGen Figure 1 + DeepSpeed Figure 2
- **Pipeline 时序图** → 学 InfiniGen Figure 3
- **新旧对比图** → 学 Orca Figure 2
- **配色和排版** → 学 vLLM Figure 3

---

## 八、额外需要画的图（论文中标注的）

| 图编号 | 内容 | 类型 | 建议工具 |
|--------|------|------|---------|
| Fig. 6 | 系统架构图 | 框图 | draw.io |
| Fig. 7 | KV Block 状态机 | 状态图 | draw.io |
| Fig. 8 | 三阶段 Pipeline 时间线 | Gantt chart | draw.io 或 TikZ |
| Fig. 5 | 四级存储层次概览 | 概念图 | draw.io |

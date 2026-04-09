# OrchKvCache 论文改稿计划 (paper_draft_impro2.md)

> 基于两轮 review 反馈，按优先级排序的改稿行动清单。
> 排除项：vLLM C++/CUDA native 集成（留给下一篇 paper）。

---

## 总体原则

1. **可信度优先**：先修所有数值不一致、TBD、命名冲突——这些会让审稿人直接扣完成度分。
2. **Claim 重排**：把 paper 从"速度论文"改成"机制+边界清楚的系统论文"——最硬的证据（migration reduction、lossless）放前面，throughput 作为后果之一。
3. **约束前置**：凡是影响审稿人解读 headline result 的现实约束，都提前到 Abstract/Intro/Eval 开头说。
4. **设计空间定位**：Related Work 从"分组罗列"改为"定义设计空间+标出我的坐标"。

---

## 第一优先级：修可信度硬伤（P0-Credibility）

### 1.1 λ 统一（Design §3.3 line 257 vs Sensitivity §5.7 line 602/611）

**问题**：
- Design section: `λ = 0.3 is the EMA decay factor`
- Sensitivity section: `defaults (λ=0.9, τ=50, cooldown=0.5s)`
- Fig caption: `accuracy peaks at λ∈[0.7,0.95]`
- 0.3 和 0.9 表示完全不同的系统行为（保守 vs 激进）

**决策**：确认代码中实际默认值是哪个，全文统一。

**改动位置**：
- [x] §3.3: `λ = 0.3` → `λ=0.9` (with ref to §5.7) ✅
- [x] §5.7 (line 602): already `λ=0.9` ✅ (no change needed)
- [x] Fig 8 caption: already consistent ✅
- [x] §5.7 正文: already consistent ✅
- [x] §5.1: **新增** "Default hyperparameters" 段落 ✅

### 1.2 cooldown 统一（Design §3.3 line 298 vs Sensitivity §5.7 line 602/607/611）

**问题**：
- Design section: `cooldown timer (default 100 ms)`
- Sensitivity section: `cooldown=0.5s` 作为 default
- 100ms 和 500ms 差 5 倍，不是"无所谓的小差异"

**决策**：同 1.1，确认代码实际值后全文统一。

**改动位置**：
- [x] §3.3: `100 ms` → `0.5 s` + 补充机制解释 + ref to §5.7 ✅
- [x] §5.7: already `0.5s` ✅ (no change needed)
- [x] §3.3 预埋句已加 ✅

### 1.3 Table XII 补齐 TBD（line 772-775）

**问题**：
- Table XII "Scoring-function ablation" 有 4 行 Accuracy 全是 TBD
- 正文已经在使用"No-attn achieves close to Full EMA"这个未证实的结论
- 8K fallback 的机制解释链未闭环

**改动**：
- [x] 跑实验: synthetic trace acc (0.61/0.10/0.10/0.10) + E2E tok/s (176.2/176.5) ✅
- [x] 补 2 列: E2E tok/s + Evictions ✅
- [x] 结论如实写: trace层面attention 6×更好，E2E单请求无差异，多请求差异在主实验体现 ✅
- [x] 正文引用全部有数字支撑 ✅
- [x] Limitations 引用同步更新 ✅

### 1.4 FIFO / LIFO 命名冲突（line 790）

**问题**：
- Prototype baseline 叫 "FIFO Offload"：oldest block first（真 FIFO）
- vLLM section 也叫 "FIFO"，但描述是 `running_queue.pop()` selects most recently added sequence（实际是 LIFO）
- 两个 baseline 语义完全不同，但用同一个名字

**改动位置**：
- [x] §5.12: 策略改名 "vLLM-default" ✅
- [x] Table XIII caption + 表内行 + Figure caption: 全部同步 ✅
- [x] §5.12 开头新增术语澄清段 ✅
- [x] 全文 "FIFO" 检查: prototype用FIFO(语义正确), vLLM用vLLM-default(已改) ✅
- [x] Summary table: "vs vLLM-default" ✅

### 1.5 8K fallback mode 表述（line 700-728）

**问题**：
- 8K 用 SDPA，没有 per-step attention reporting，只用 recency+frequency
- 但 Table X 和正文的写法让读者以为 2K/4K/8K 是同一个 OrchKvCache
- 这不是坏结果，但需要明确区分

**改动位置**：
- [x] Table X caption: 明确区分 full classifier vs no-attn fallback ✅
- [x] §5.10 正文: 重写，明确 8K 是 fallback 配置 ✅
- [x] 和 Table XII 联动: 引用 Table tail-latency 的 noattn E2E 结果 ✅

---

## 第二优先级：重写主 Claim 和叙事结构（P0-Claim）

### 2.1 重排 Abstract 和 Introduction 的 claim 层级

**现在的问题**：最显眼的是 `1.28-1.77× throughput`，但这个结果受 prototype framework overhead 放大；最硬的证据（migration reduction、lossless 三层路径）反而不突出。

**新的叙事层级**：
1. **核心事实**：Attention hotness 极度偏斜且可预测
2. **核心机制**：三层无损 block orchestration + attention-aware scoring + batch-aligned SSD + overlap pipeline
3. **最干净证据**：139-597× fewer unnecessary migrations
4. **系统效果**：prototype 1.28-1.77×、vLLM under pressure 1.08-1.15×
5. **边界说明**：prototype speedup 在 shared framework overhead regime 下测得；vLLM 1.12× 是更保守但更干净的估计

**改动位置**：
- [x] Abstract: migration reduction + lossless 在前，throughput 在后 ✅ (已在之前版本改过)
- [x] Introduction §1 末尾: claim 重排 + Scope note 段落 ✅
- [x] Positioning sentence 已加入 Related Work 开头 ✅

### 2.2 新增 Scope and Deployment Note（Introduction 末尾）

**现在的问题**：prototype 的角色和限制都在 Discussion §7 才说，太晚了。审稿人在看完 Figure 1/2 后已经形成判断。

**新增内容**（放在 Introduction contributions 之后，约 6-8 句）：
- [x] Prototype 定位 ✅
- [x] Production runtime 限制 ✅
- [x] 两种证据说明 ✅
- [x] Two-level intelligence 说明 ✅
(全部写入 Introduction 末尾 "Scope and deployment context" 段落)

### 2.3 新增 Evaluation Methodological Note（§5.1 末尾或 §5.2 之前）

**改动**：
- [x] §5.1 新增 "Methodological note" 段落 ✅ (manual decode loop + round-robin + 隔离 policy + CB in §5.12)

---

## 第三优先级：Related Work 重构（P0-Positioning）

### 3.1 新增 Design-Space Map 表格

**现在的问题**：Table I 只有 3 个维度（tiers/lossless/limitation），不够展开。Table XVII（feature comparison）出现在 §5.14，太晚了。

**改动**：
- [~] 用户决定不画 design-space map 表格，直接改文字 → 已用 route-based prose 替代 ✅

### 3.2 Related Work 改为"路线式"写法

**现在的问题**：按主题分桶罗列，没有"定义设计空间"。

**新结构**：
- [ ] 开头先引出 design-space map：
  > We organize prior work along N orthogonal axes (Table X)...
- [ ] 然后按路线写，不再按主题：
  1. **Lossy decode-time KV reduction** (H2O, ScissorHands, StreamingLLM, Quest, SnapKV, KIVI) → 和我们的分界线：discard vs migrate
  2. **Two-tier lossless offloading** (vLLM, InfiniGen) → 和我们的分界线：no SSD / no proactive eviction / reactive only
  3. **Offline three-tier planning** (FlexGen) → 和我们的分界线：offline vs online
  4. **Prefix-KV storage** (IMPRESS, ContiguousKV) → 和我们的分界线：prefix vs decode phase
  5. **Datacenter-scale KV management** (Mooncake, Infinite-LLM, vTensor) → 和我们的分界线：单节点 decode-phase placement vs 集群调度
  6. **Heterogeneous storage** (OrchFS, Strata, SPFS) → 我们的 I/O 基础设施

- [x] 每段结尾明确写差异 ✅ (7条路线全部完成)
- [x] vLLM / FlexGen / InfiniGen / IMPRESS 作为前两段重点展开 ✅

---

## 第四优先级：Overhead Diagnosis 前移（P1-Structure）

### 4.1 把 Discussion 中的 overhead analysis 前移到 Evaluation

**现在的问题**：Table XI (overhead breakdown) 和 Table XII (fair baseline) 在 Discussion §7 才出现，但它们是理解整个 paper 结果的关键——它们解释了为什么 prototype 和 GPU-Only 有大 gap、瓶颈在哪。

**改动**：
- [x] Eval 新增 §"Throughput Overhead Diagnosis" (含 Table XI + XII + Key finding + Decomposing the gap) ✅
- [x] Discussion 瘦身: 仅保留摘要引用 + future-looking 内容 ✅

---

## 第五优先级：补充实验和指标（P1-Evidence）

### 5.1 补 stronger baselines（不涉及 vLLM native）

**审稿人可能的质疑**："你只赢了弱 baseline (FIFO)"

**可补的 baseline**：
- [x] **No-attention heuristic (orchkv-noattn)**: E2E 实验完成 → 176.5 tok/s, 与 full OrchKv 几乎一致 ✅
  - 写入 Table tail-latency + Table signal-ablation
- [~] **Two-tier strong baseline**: 未单独跑，但 Table tier-throughput (GPU+DRAM vs GPU+DRAM+SSD) 已展示 SSD 增量仅 1.5-4.3%
- [~] **No-batching ablation**: microbenchmark 数据已有 (Table II: 4.3% → 40.8%), E2E 需改 C 代码 batch_size → 留为 future work

### 5.2 补 tail latency / QoS 指标

**审稿人可能的质疑**："只给 throughput 不够，serving 系统需要看延迟分布"

**可补的指标**：
- [x] TPOT P50/P95/P99/Max: 新 Table tail-latency 写入 §5.2 ✅
  - FIFO P99=568ms, OrchKvCache P99=192ms (2.95× reduction)
  - Finding 3a + Finding 3b 写入正文 ✅
  - Summary table 新增 P99 行 ✅
- [~] Realistic workload 下的 CDF: 留为 optional (当前 table 已足够)

### 5.3 SSD batching end-to-end ablation

**审稿人可能的质疑**："SSD batching 在 microbenchmark 上有效，end-to-end 呢？"

**可补的实验**：
- [~] Microbenchmark 数据已有 (Table II + exp_ssd_io_ablation.json)。E2E ablation 需改 C 代码 → 留为 future work

---

## 第六优先级：升格 production insight（P1-Insight）

### 6.1 把 "block-level intelligence in production" 写成 paper takeaway

**现在的位置**：Discussion §7，偏"讨论性质"

**改动**：
- [ ] 在 Introduction 或 §5.12 结论处加入明确的 systems takeaway：
  > Current production runtimes expose a structural abstraction barrier: their attention kernels assume that all blocks of a running sequence are GPU-resident, preventing true intra-request partial swap. OrchKvCache shows that block-level intelligence is useful, but fully exploiting it requires mixed-tier block tables and on-demand block fetch in the attention kernel.
- [ ] 这会让 paper 不只是"提一个方法"，而是在指出现有 runtime abstraction 的结构性限制

---

## 不做的事项（留给下一篇 paper）

| 建议 | 为什么不做 |
|---|---|
| vLLM C++/CUDA native scheduling loop | 这是下一篇 paper 的核心工作 |
| 修改 FlashAttention 支持 mixed GPU/CPU block table | 工程量太大，是 future work |
| 移植到 FlexGen 直接和 InfiniGen 同框架对比 | 性价比低，方案 A 已经够用 |
| 多 GPU / CXL / NVM 实验 | 没有硬件 |

---

## 改稿执行顺序

```
Phase 1: 修硬伤（1-2天）
  1.1 λ 统一
  1.2 cooldown 统一
  1.3 Table XII 补齐（需要跑实验）
  1.4 FIFO/LIFO 改名
  1.5 8K fallback 表述

Phase 2: 结构重组（1-2天）
  2.1 重排 Abstract/Intro claim 层级
  2.2 新增 Scope note
  2.3 新增 Eval methodological note
  4.1 Overhead diagnosis 前移
  4.2 Discussion 瘦身

Phase 3: Related Work 重构（1天）
  3.1 Design-space map 表格
  3.2 路线式重写

Phase 4: 补充实验（2-3天）
  5.1 Stronger baselines
  5.2 Tail latency
  5.3 SSD batching ablation
  1.3 Table XII 实验数据（如果 Phase 1 还没完成）

Phase 5: 升格 insight（0.5天）
  6.1 Production insight takeaway
```

---

## 核心改稿原则（贯穿全文检查）

> **让审稿人在任何一个关键结果旁边，都能立刻知道：
> 这是什么配置下的结果、和什么 baseline 比、这个结果真正说明什么、不说明什么。**

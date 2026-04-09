# OrchKvCache 改稿计划 Round 3 (paper_draft_impro3.md)

> 基于第三轮 review 反馈。审稿人判断：从 Weak Reject 提升到 Borderline，但仍卡在 production path 和 baseline 强度上。

---

## 审稿人认可的改进（不需要再动）

- [x] Scope and deployment context 段落
- [x] 参数一致性（λ=0.9, cooldown=0.5s）
- [x] Methodological note
- [x] Tail latency P50/P95/P99
- [x] 8K fallback 表述
- [x] FIFO/LIFO 命名修正
- [x] Related Work 路线式重构

---

## 第一优先级：必须修的硬伤（P0，改文字即可）

### P0-1. Signal Ablation "single-request" 与 Table XIII "4 req" 冲突

**问题**：Table XIII caption 写 "4 requests"，但正文写 "On real single-request inference"。4 req ≠ single-request。

**修改方案**：
- [ ] 正文从 "single-request" 改为 "per-request sequential decode"
- [ ] 解释清楚：虽然是 4 个请求，但它们是 round-robin 逐个 decode 的，每个请求独享 GPU budget 的 1/4，block 竞争有限
- [ ] 核心论点不变：在低竞争配置下 attention signal 的 E2E 差异不显著

**改动位置**：§5.11 (line ~800)

### P0-2. Competitive Ratio 节降低 framing

**问题**：审稿人认为 "theoretical grounding" 的帽子太大——实际上是 synthetic trace replay，不是真正的理论证明。

**修改方案**：
- [ ] 标题从 "Competitive Ratio Analysis" 不变，但开头段从 "To provide a theoretical grounding" 改为 "To compare scheduling policies under controlled contention"
- [ ] 明确说这是 "empirical policy comparison on synthetic traces"，不是 formal theoretical bound
- [ ] 保留 Belady baseline（它仍然是有意义的参照）
- [ ] 可选：补一句 caveat，说 synthetic trace 用 eviction count 近似 cost，未考虑 tier-dependent cost

**改动位置**：§5.6 (line ~588-610)

### P0-3. Abstract/Intro 进一步弱化 prototype throughput 的权重

**问题**：审稿人说 1.28-1.77× 仍然太显眼，prototype throughput 的代表性不够。

**修改方案**：
- [ ] Abstract：考虑把语序调整为 "139-597× fewer migrations ... 100% lossless ... translating to 1.28-1.77× in prototype; 1.12× in vLLM confirms production transferability"
- [ ] 最好加一个修饰语，如 "In our block-level prototype (where shared framework overhead amplifies relative differences), these savings translate to..."
- [ ] 确保 "1.12× in vLLM" 不是附属品，而是和 prototype 结果并列的独立证据

**改动位置**：Abstract (line ~71), Intro summary (line ~138)

---

## 第二优先级：可以在这轮改的改进（P1）

### P1-1. native critical path 的展望写得更具体

**问题**：审稿人最大的质疑是 "block-level idea 没有在 native path 验证"。

**现实**：完整 native 化是下一篇 paper 的工作，但可以在这轮做的是：
- [ ] 在 Discussion Future directions 中给出更具体的 projected throughput
  - 已有数据：Python scheduling loop = 9.42ms/step, C/CUDA classifier = 0.04ms
  - 预计 native 后: 81.1ms → ~71.7ms/step → ~292 tok/s (vs 258 currently)
  - 这个数据已经在 Overhead Diagnosis 节了，在 Discussion 中显式引用并做 projection
- [ ] 可选：如果时间允许，把 `build_past_kv` 的 pre-allocation 优化做进 prototype，跑一个 "partial native" result

**改动位置**：Discussion Future directions

### P1-2. Classification accuracy vs E2E behavior 的解释加强

**问题**：Table XIII 显示 trace accuracy 差 6×，但 E2E 完全一样——审稿人觉得两者脱节。

**修改方案**：
- [ ] 在 Signal Ablation 文字中加一段更精确的解释：
  - Trace accuracy 衡量的是"在 GPU capacity < hot set 时的分类正确率"
  - E2E eviction count 衡量的是"实际触发的搬迁次数"
  - 两者脱节是因为 E2E 实验的 GPU capacity 对于 4 个 round-robin 请求来说仍然足够装下每个请求的热集
  - 真正的区分出现在 16 请求竞争时（主实验 Table III 已证明）
- [ ] 考虑补一个 16-request 的 signal ablation E2E 数据点（如果时间允许跑一轮 GPU 实验）

**改动位置**：§5.11 (line ~796-802)

---

## 第三优先级：留给下一轮/下一篇的工作（P2）

### P2-1. Native critical path 实现

- 把 Python scheduling loop 移到 C/CUDA
- 把 `build_past_kv` 改为 pre-allocated buffer + zero-copy
- 这是下一篇 paper 的核心工作

### P2-2. Stronger head-to-head baseline

- 同 stack 下的 two-tier proactive prefetch baseline
- DRAM+SSD priority heuristic baseline
- 需要额外实现工作

### P2-3. SSD batching end-to-end ablation

- no-batch vs batch vs batch+OrchFS 的 E2E 对比
- 需要修改 C 代码的 batch_size 参数
- microbenchmark 数据已有 (Table II)，E2E 闭环需要更多工程

### P2-4. 16-request signal ablation E2E

- 跑 orchkv vs orchkv-noattn 在 16 请求下的 E2E 对比
- 预期：16 请求竞争时 attention signal 的 E2E 差异应该出现
- 如果来得及在这轮跑，可以升级为 P1

---

## 改稿执行顺序

```
立即可做（改文字，0.5天）：
  P0-1: 修 "single-request" 措辞
  P0-2: 降低 Competitive Ratio framing
  P0-3: Abstract/Intro 弱化 prototype throughput

尽量做（1天）：
  P1-1: Discussion 补 projected throughput
  P1-2: Signal Ablation 加强解释

如果时间允许（1-2天 GPU 实验）：
  P2-4: 16-request signal ablation E2E（升级为 P1）

留给下一篇：
  P2-1: Native critical path
  P2-2: Stronger baseline
  P2-3: SSD batching E2E ablation
```

---

## 审稿人的核心卡点

> "block-level idea 的 production-grade 证据还不够硬"

**我们的应对策略**：
1. 不硬撑 production-ready claim，而是把 paper 定位为 **mechanism validation + production insight**
2. 在 Abstract/Intro/Conclusion 中反复明确：prototype 验证机制，vLLM 验证策略迁移，native path 是下一步
3. 用 overhead diagnosis 证明瓶颈在 Python orchestration 而非 policy quality
4. 用 projected throughput 给出 native 化后的预期

**底线**：这篇 paper 的价值不在于"我已经做了一个 production-ready 系统"，而在于"我证明了 attention-driven block-level three-tier orchestration 这个方向是对的，机制有效，约束清楚，production path 明确"。

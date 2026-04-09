# Improvement Plan v7 — Final Push: Borderline → Accept

## 审稿共识（4 位 reviewer，含自审）

| Reviewer | 评分 | 核心意见 |
|----------|------|---------|
| R1 (自审) | Accept (弱) | "2.64× 是决定性拼图，但只有 1 个配置点" |
| R4 (最新) | Borderline (偏正面) | "2.64× 很有价值，但证据偏窄；还没进 production" |

**共识**：2.64× time-multiplexing 是质变，但需要更多 data points 来从 "promising" 变成 "convincing"。

## R4 明确说了最值钱的一件事

> "把 shared-pool time-multiplexing 至少做一个更接近 production 的实现或评测——多模型、多请求数、latency/fairness 指标"

这是从 Borderline → Accept 的**唯一**必须做的事。其他问题（prefetch、8K fallback、native build_past_kv）已经被诚实 acknowledge 了，不再是 blocking issue。

---

## 唯一的 P0：扩展 Time-Multiplexing 实验矩阵

### 当前只有 1 个点
```
Qwen2.5-7B, 4 requests, seq=1024, budget=50MB → 2.64×
```

### 需要扩展到的矩阵

| 维度 | 当前 | 扩展到 |
|------|------|--------|
| 模型 | Qwen2.5-7B (GQA, 56KB/tok) | + LLaMA-2-7B (MHA, 512KB/tok) |
| 请求数 | 4 | 4, 8, 16 |
| 序列长度 | 1024 | 1024, 2048 |
| Budget | 50MB | 50MB, 100MB |

**最小可行矩阵**（ROI 最高的 6 个点）：

| # | Model | N_req | seq | budget | 预期 |
|---|-------|-------|-----|--------|------|
| 1 | Qwen2.5-7B | 4 | 1024 | 50MB | 已有: 2.64× |
| 2 | Qwen2.5-7B | 8 | 1024 | 50MB | 预期更高（更紧的 isolated budget） |
| 3 | Qwen2.5-7B | 4 | 2048 | 50MB | 预期更高（更大 KV per req） |
| 4 | LLaMA-2-7B | 4 | 1024 | 50MB | MHA 模型验证 |
| 5 | LLaMA-2-7B | 4 | 2048 | 100MB | 更大规模 |
| 6 | Qwen2.5-7B | 16 | 1024 | 100MB | 高并发 |

### 预期结果
- 随着 N_req 增大，Isolated 的 per-req budget 更小 → speedup 应该更大
- LLaMA-2-7B (512KB/tok vs Qwen 56KB/tok) → 内存压力更大 → speedup 更大
- 这些点组合起来变成一张 Table 或 Figure，展示 time-multiplexing 的 scaling behavior

### 产出
- 1 张新表（替换当前的 2-row Table XVI）
- 或 1 张图（speedup vs N_requests × model）

### 难度
低。修改 `exp_time_multiplex.py` 加入 for loop，跑 6 个配置 × 约 2 min each = 12 min 总计。

---

## P1：文字微调（如果空间允许）

### P1.1 — Time-multiplexing baseline 公平性讨论（R1 W1, R4 问题 1）

> "为什么不能简单通过减少并发数来增大 per-request budget？"

答案：total KV = N × per_req_KV。固定 GPU memory 下，N requests 的 total KV 就是 N×。减少 N 可以增大 per-req budget，但也降低了并发数 → 降低了 throughput/concurrency tradeoff。Time-multiplexing 的价值是 "不降低并发数，但每个 request 获得更大有效 budget"。

在 Table 正文加 1-2 句解释。

### P1.2 — 在 §time-multiplex 中讨论 latency/fairness

R4 提到 "latency/fairness 指标"。在 round-robin 下，fairness 是自然保证的（每个 request 轮流获得 full GPU）。可以加 1 句：

> "Under round-robin scheduling, each request receives equal GPU time per cycle; P99 per-request latency is bounded by N × per-step time."

---

## 执行计划

| 步骤 | 做什么 | 时间 |
|------|--------|------|
| 1 | 修改 exp_time_multiplex.py 支持多配置 | 10 min |
| 2 | 跑 6 个配置 | ~15 min |
| 3 | 更新 Table XVI 为多行表 | 10 min |
| 4 | 加 baseline fairness 讨论 (P1.1, P1.2) | 10 min |
| 5 | 更新 Abstract/Conclusion 的 headline | 5 min |
| **总计** | | **~50 min** |

## 预期效果

扩展后的 time-multiplexing 表（6+ 配置，2 模型，3 并发级别）直接回答 R4 的核心 concern："证据偏窄"。如果 LLaMA-2-7B + 高并发下 speedup 更大（很可能，因为 MHA 的 KV/tok 是 GQA 的 9×），则 headline 可以从 "2.64×" 升级为 "2.64–X×"。

预期评分变化：R4 从 Borderline → Weak Accept, R1 维持 Accept (弱)。

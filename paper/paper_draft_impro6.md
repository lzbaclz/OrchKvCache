# Improvement Plan v6 — Synthesized from Three Reviews

## Review Consensus

Three reviewers (all High Confidence) converge on the same verdict:

| Reviewer | Rating | Key phrase |
|----------|--------|-----------|
| R1 (你) | Borderline Accept (weak) | "strong prototype + systems diagnosis, not yet fully convincing production-grade systems paper" |
| R2 (老师1) | Borderline / Weak Reject | "高质量、边界清楚的机制验证与诊断型论文，而不是已经完整闭环到 production benefit 的系统论文" |
| R3 (老师2) | Borderline | "technically interesting, substantially improved, potentially acceptable depending on committee taste" |

**共识**：mechanism-level evidence 很强 (139-597×, lossless, SSD)，vLLM diagnostic 有价值，但 production path 没有闭环。三位都认为论文已经 "serious and discussable"，差的是最后一步。

---

## 三位 Reviewer 共同提出的弱点（按频率排序）

| # | 弱点 | R1 | R2 | R3 | 可修？ |
|---|------|----|----|----|----|
| **W-A** | Strongest results only in prototype, not production | ✓ | ✓ | ✓ | 部分（native-path） |
| **W-B** | Native-path projection 是估算，不是实测 | ✓ | ✓ | ✓ | **可做实验** |
| **W-C** | Attention signal 的 E2E 价值未被直接证明（No-attn 更快） | ✓ | ✓ | | **可重新 frame** |
| **W-D** | SSD ablation 规模太小（128KB） | ✓ | | | **可跑大规模** |
| **W-E** | Competitive ratio 用 synthetic trace 而非 real attention trace | ✓ | ✓ | ✓ | **可做实验** |
| **W-F** | InfiniGen 对比不够 head-to-head | ✓ | | ✓ | 难（需重新实现） |
| **W-G** | Prefetch 是 mechanism not E2E contributor | | ✓ | | **可收缩 claim** |
| **W-H** | 32K/128K claim 证据不足 | | ✓ | | **可加 caveat** |
| **W-I** | 论文仍 slightly split between mechanism/diagnosis | ✓ | | ✓ | **可进一步收紧** |

---

## 从 Borderline → Accept 的策略

三位 reviewer 都说了同一句话的变体："a strong revision response and minor additional evidence could push it over the line." 关键是选对 **哪几个** evidence 补上去。

### 核心判断：不要什么都补，选 ROI 最高的 3 件事

1. **W-B: 做一个 partial native-path 实测**（所有 reviewer 都提了）
2. **W-E: 用 real attention trace 跑 competitive ratio**（R1+R2+R3 都提了）
3. **W-D: 大规模 SSD ablation**（R1 提了，且是最容易做的）

其余弱点通过 **文字修改 / reframing** 处理，不需要新实验。

---

## P0: 必须做（新实验 + 关键修改）

### P0.1 — Partial native-path 实测（回答 W-B, 所有 reviewer 的 Q2）

**问题**：Native-path projection 是用 overhead breakdown 算的，不是实测。Reviewer 问 "once Python is removed, what other bottlenecks emerge?"

**方案**：不需要重写整个 framework。做一个 **targeted measurement**：

1. 用 `orchkv_core` C library 的 `tm_create` / `tm_report_attn` / `tm_step_done` / `tm_schedule_once` 在一个 tight C loop 中运行 classification + eviction + prefetch dispatch，输入是从真实 decode 中 dump 的 attention trace。
2. 测量：N blocks × M steps 的 wall-clock 时间。
3. 对比：Python scheduling loop (9.4ms/step) vs C native loop (?μs/step)。
4. 这已经有了 microbenchmark（<40μs at 4096 blocks），但 reviewer 要的是 "full scheduling loop" 而不只是 classifier。

**产出**：1 行新数据（"Full C scheduling loop: X μs/step at N blocks"），加入 Table X 或作为 inline number。

**预期**：全 C loop < 100μs/step（classifier 40μs + eviction scan + prefetch heap），vs Python 9.4ms → **~100× speedup**，projected step time 从 81.1ms 降到 ~20ms 得到实测支持。

**难度**：低。`orchkv_core` 已经有 pybind 接口，直接写一个 C benchmark 调用 `tm_*` 函数即可。

### P0.2 — Real attention trace competitive ratio（回答 W-E, 三位都提了）

**问题**：Table VI 用 synthetic Zipf trace。Reviewer 说 "real attention traces from model inference would be more convincing."

**方案**：

1. 从现有的 E2E runs 中 dump per-block attention scores（`report_attn` 已经在每步调用）。
2. 保存为 trace 文件：(step, block_id, attn_weight)。
3. 用 trace-driven simulator（已有 `exp_competitive_ratio.py`）替换 synthetic trace 为 real trace。
4. 跑 FIFO / LRU / LFU / EMA / OPT 对比。

**产出**：1 张新 table 或替换 Table VI 的数据列。

**预期**：Real trace 上 EMA 仍优于 FIFO/LRU，但差距可能不同于 synthetic。Reviewer 会更信。

**难度**：中。需要修改 prototype 来 dump traces，然后修改 CR simulator 来读取。

### P0.3 — 大规模 SSD ablation（回答 W-D, R1 的 Q1）

**问题**：当前 SSD ablation 只有 128KB SSD traffic，per-block vs batched 差异 <0.2%。

**方案**：用 LLaMA-2-7B seq=4096 budget=10MB 跑 SSD ablation。

- LLaMA-2-7B 的 KV/tok = 512KB → seq=4096 → 2GB KV per request
- budget=10MB → 大量 block 需要 spill 到 SSD
- 预期 SSD write volume: 数十 MB → batching 效果应该可观

**产出**：Figure 11 扩展或补充数据点。

**难度**：低。改 `exp_ssd_ablation_e2e.py` 的参数即可。

---

## P1: 文字修改（不需要新实验）

### P1.1 — 收缩 Prefetch claim（回答 R2 的 W-G）

**现状**：Contribution #3 说 "asynchronous migration pipeline with compute-transfer overlap"，其中包含 prefetch。但 E2E 中 prefetch dispatched=0。

**修改**：在 §5.5 的 prefetch 段落末尾加一句：

> "In the current per-request prototype, speculative prefetch does not fire (§5.5). The prefetch subsystem is therefore validated at the mechanism level but is not yet an E2E throughput contributor; it becomes meaningful when multiple requests share a pooled block budget, as in a native production integration."

### P1.2 — 进一步收紧 thesis（回答 W-I, R2 Q for authors, R3 Q1）

**现状**：Scope paragraph 已经很好，但三位 reviewer 仍觉得 split。

**修改**：在 Introduction 最后加 **一句话** 显式回答 "primary takeaway"：

> "The primary contribution is (1)—the mechanism; the production diagnosis (§5.10) identifies the specific runtime barrier that prevents current systems from deploying it."

### P1.3 — 收缩 32K/128K claim（回答 R2 的 W-H）

**现状**："Extending to 128K requires no algorithmic changes."

**修改**：加 caveat：

> "Extending to 128K requires no algorithmic changes, though end-to-end validation at that scale remains future work pending sufficient DRAM+SSD capacity."

### P1.4 — 重新 frame No-attn > Full EMA 的 E2E 结果（回答 W-C）

**现状**：Table IX 显示 No-attn 更快，文本说是 SDPA vs eager artifact。

**修改**：在 signal ablation 段落加强措辞：

> "The 7% throughput advantage of No-attn reflects the use of SDPA (which fuses attention and does not output weights) vs. eager attention (which materializes the full weight matrix for score extraction). This is a cost of the *extraction mechanism*, not of the *scoring policy*: any native integration that obtains attention scores as a kernel side-effect (e.g., FlashAttention's partial output mode) would eliminate this gap entirely. The contention-dependent identity analysis (Fig. 8) isolates the policy's value from the extraction cost."

### P1.5 — Minor issues（三位都提了零散小问题）

- Abstract 末尾精简 mixed-tier kernels 那句（R1）
- "shaped by" → "amplified by"（R1）
- Figure 11 caption 加具体数字 "175.4→179.1 (+2.1%)"（R1）
- "Native-path projection (Q:...)" → "Native-path projection."（R1）
- Table XIII 加系统最大吞吐量参照（R1）
- InfiniGen 解释技术障碍而非只说 "future work"（R1）

---

## P2: 可选（如果空间和时间允许）

### P2.1 — 减少 table/figure 密度（R1 W4）
砍掉 Table III (model specs → 合入 Setup text) 或 Table V (quality → 合入 1 句话) 腾出空间。

### P2.2 — 回答 "when does OrchKvCache become necessary at full GPU capacity?"（R1 W5）
加 1 句：16 requests × 4K context = 32GB KV on LLaMA-2-7B → 已经 40% 的 80GB GPU。128K context × 4 requests → 256GB → 远超单 GPU。

### P2.3 — Mixed-tier kernel 的初步设计草图（R2 Questions #1）
如果有空间，在 Discussion 加 2-3 句话描述 mixed-tier attention kernel 的 conceptual design：block table 中某些 slot 指向 DRAM/SSD buffer，attention kernel 在遇到 non-GPU slot 时 issue an async fetch + compute partial attention on available blocks。

---

## 执行优先级

| 优先级 | 项目 | 预计时间 | 产出 |
|--------|------|---------|------|
| **1** | P0.1 C native loop 实测 | 1-2 小时 | 1 个数字 + 1 段文字 |
| **2** | P0.3 大规模 SSD ablation | 1 小时 | 更新 Figure 11 + 文字 |
| **3** | P1.1-P1.5 文字修改 | 1 小时 | ~20 行改动 |
| **4** | P0.2 Real attention trace CR | 2-3 小时 | 替换 Table VI |
| **5** | P2.1-P2.3 可选 | 1 小时 | 空间允许才做 |

**总预计**：核心修改 5-7 小时，全部完成 8-10 小时。

---

## 预期效果

如果 P0.1 + P0.2 + P0.3 + P1 全部完成：

- W-A (production gap): 由 native-path **实测** 而非估算部分弥合
- W-B (projection → measurement): **直接解决**
- W-C (attention signal value): 更好的 framing
- W-D (SSD ablation scale): **直接解决**
- W-E (synthetic trace): **直接解决**
- W-F (InfiniGen): 已 acknowledged, 不可解决
- W-G (prefetch): 收缩 claim
- W-H (128K): 加 caveat
- W-I (thesis split): 进一步收紧

预期评分变化：R2 从 Weak Reject → Borderline Accept, R1/R3 从 Borderline → Accept。

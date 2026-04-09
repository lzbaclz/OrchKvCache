# OrchKvCache 论文改稿计划 Round 4 (paper_draft_impro4.md)

> 基于 Review Round 4 (清华/UIUC/Wisconsin 视角) 的改稿行动清单。
> 目标文件: `overleaf5.tex` → 就地修改

---

## Review 评分: 5.5/10 (Weak Reject)
## 目标: 解决所有可在 paper 层面修复的问题 → 6.5-7.0

---

## P0: 补充 16-request ablation 数据 ✅

**问题 (W1)**: Table 12 (Signal Ablation) 在 4-request 设置下 Full EMA 和 No-attn E2E 完全一致。

**实验结果** (exp_p0_16req_ablation.json):
- nreq=4:  Full EMA 164.8 tok/s, No-attn 181.3 tok/s, evictions identical (2,048)
- nreq=8:  Full EMA 168.7, No-attn 181.3, evictions identical (4,096)
- nreq=16: Full EMA 169.2, No-attn 181.9, evictions identical (8,192)

**关键发现**: Per-request manager 架构下，attention signal 不减少 eviction (per-request hot set 太小)。
No-attn 反而快 ~7%（避免 eager attention 开销）。Attention signal 的价值在 shared-pool contention 下体现 (Table contention-identity)。

---

## P3: 恢复 Competitive Ratio Analysis [Paper 编辑] ✅

**问题 (W5)**: Lines 590-617 将整段 Competitive Ratio Analysis 注释掉了。缺少算法质量的理论支撑。

**改动**:
- [x] 取消注释 → 新建 §5.X "Empirical Competitive Ratio" with \label{sec:competitive-ratio}
- [x] 加入方法论 caveat: "Methodology note" 段落说明 synthetic traces + eviction-count 近似
- [x] 明确定位为 "complement the E2E migration reduction"
- [x] 在 Signal Ablation 中交叉引用 Table tab:cr

---

## P4: 删除 5 个未引用的 bibliography entries [Paper 编辑] ✅

**问题 (m1)**: 以下 5 个 \bibitem 在正文中从未被 \cite{} 引用:
1. `deepspeed-inf` (DeepSpeed-Inference, SC 2022)
2. `deepspeed` (DeepSpeed, KDD 2020)
3. `zero` (ZeRO, SC 2020)
4. `speculative` (Speculative Decoding, ICML 2023)
5. `awq` (AWQ, MLSys 2024)

**改动**:
- [x] 删除这 5 个 \bibitem entries → 50 → 46 references
- [x] 更新 \begin{thebibliography}{46}

---

## P5: 补全 Table 5 (Quality Verification) [Paper 编辑] ✅

**问题 (m2)**: 声称 4 模型验证但 Table 5 只有 Qwen 和 Mistral，缺 LLaMA-2-7B 和 LLaMA-2-13B。

**改动**:
- [x] 从 llama_quality.json 获取实验数据
- [x] Table 5 添加 6 行 (LLaMA-2-7B × 3 + LLaMA-2-13B × 3)，全部 100% match
- [x] 正文更新: "across all four models and three prompt lengths (12 configurations)"

---

## W1: 强化 Signal Ablation 论证 [Paper 编辑] ✅

**问题**: Table 12 ablation 未在高竞争配置下展示 attention signal 的价值。

**改动**:
- [x] "When does attention matter?" 段落重写为 "Three converging lines of evidence"
- [x] 引用 (1) Table contention-identity (identity precision under 8-req), (2) E2E migration scaling data, (3) Table cr (competitive ratio)
- [x] 明确 4-request ablation 是 low-contention regime

---

## W3: 优化 vLLM Block-Scoring 讨论 [Paper 编辑] ✅

**问题 (W3)**: Block-level scoring 在 32-req 下 0.91x，论文需要更精确地 frame 这个结果。

**改动**:
- [x] Root-cause 段落重写: 量化 Python vs C 开销差异 (128 blocks: ~333μs native vs ~4ms Python, 12× gap)
- [x] 明确 16-req 下 block-scoring 为 1.06× (算法有效，Python 不 scale)
- [x] Implication 段落更新: progress-aware 作为 lightweight proxy, block-scoring 在 moderate concurrency 有效
- [x] Abstract/Intro/Conclusion 中 vLLM 数字从 "1.12×" 统一为 "1.08--1.15×"

---

## W4: 增加长上下文 (32K+) 实验 + 讨论 ✅

**问题 (W4)**: 2K-8K 在 2025/2026 太短。32K+ 才是 KV offloading 真正有价值的场景。

**实验结果** (exp_w4_32k_context.json):
| Seq   | GPU-Only | FIFO  | OrchKv | Orch/FIFO | Match |
|-------|----------|-------|--------|-----------|-------|
| 8K    | 4,431    | 253   | 381    | 1.51x     | 100%  |
| 16K   | 7,404    | 240   | 367    | 1.53x     | 100%  |
| 32K   | 7,207    | 237   | 367    | 1.55x     | 100%  |

**改动**:
- [x] Table scale 扩展: 新增 Qwen2.5-7B 16K 和 32K 行
- [x] 段落 "Projection to 32K" → "Scaling to 32K context" (有实验数据支撑)
- [x] Abstract/Conclusion 更新为 "2K--32K"
- [x] Limitation (1) 更新: "validated lossless up to 32K"

---

## m3: 添加 Sampling/Temperature 讨论 [Paper 编辑] ✅

**问题**: 所有 lossless 验证在 greedy decoding (temperature=0)。未讨论 sampling 场景。

**改动**:
- [x] §5.3 末尾新增 "Note on sampling" 段落
- [x] 说明 distribution-preserving (logits 一致 → sampling 分布一致)
- [x] 引用 KL divergence < 10^-12 的经验验证

---

## m4: 修复 \label{sec:infinigen-comparison} 位置 [Paper 编辑] ✅

**问题**: \label 在 \textbf{} 段落内而非 \subsection{}。\S\ref{} 会解析为 Section 7 编号。

**改动**:
- [x] 删除 \label{sec:infinigen-comparison}
- [x] Related Work 中 \S\ref{sec:infinigen-comparison} → \S\ref{sec:discussion}

---

## m5: 扩展 Multi-GPU 讨论 [Paper 编辑] ✅

**问题**: 论文仅用单 GPU 评估，对 SC 会议审稿人来说 scalability 是关注点。

**改动**:
- [x] Future direction (3) 扩展为完整段落: TP 下每 rank 独立管理 KV heads
- [x] 新增 Limitation (3): 明确单 GPU 限制 + forward reference
- [x] CXL 四层扩展讨论

---

## 执行顺序

```
1. P4: 删除未引用 bibliography (5分钟)
2. P5: 补全 Table 5 (5分钟)
3. m4: 修复 \label (2分钟)
4. m3: 添加 sampling 讨论 (5分钟)
5. P3: 恢复 Competitive Ratio (15分钟)
6. W1: 强化 Signal Ablation (10分钟)
7. W3: 优化 vLLM 讨论 (10分钟)
8. W4: 长上下文讨论 (10分钟)
9. m5: Multi-GPU 讨论 (5分钟)
10. Final pass (10分钟)
```

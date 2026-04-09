# Paper Draft Improvement 1: Trace-Driven Competitive Ratio + Signal Ablation

## 导师 Review 要点

1. **Table XII (Signal Ablation) 留着 TBD** — 致命，投稿级大雷
2. **Table VII (Competitive Ratio) 用 synthetic trace** — 说服力不够
3. **"theoretical grounding" 措辞太大** — 应该是 empirical policy evaluation
4. **只看 eviction count** — 应该加 weighted migration cost
5. **只测 30% capacity** — 应该 sweep 多个压力点
6. **缺 variance/CI** — 需要 mean±std
7. **synthetic trace 参数未解释** — Zipf exponent / shift period / hot set size

## 行动计划

### Phase 1: 真实 Trace 收集
- 在 Qwen2.5-7B 和 LLaMA-2-7B 上跑 E2E inference
- 每步 dump `(step, block_id, attn_score)` 序列
- 配置: seq=2048, budget=50MB, 4 requests, 64 decode steps
- 输出: trace JSON 文件

### Phase 2: Trace-Driven Replay Study
- 用真实 trace replay 5 种策略: FIFO, LRU, LFU, EMA, OPT(Belady)
- **Weighted cost metric**:
  `cost = c_gd × N_GPU→DRAM + c_ds × N_DRAM→SSD + c_sg × N_SSD→GPU`
  其中 c_gd = 1.4μs, c_ds = 139μs, c_sg = 156μs (from microbenchmark)
- **Capacity sweep**: 10%, 20%, 30%, 50%, 70% GPU capacity
- **Variance**: 每配置 3 runs (不同随机种子 for OPT tiebreaking), 报 mean±std
- 输出: Table VII (替换 synthetic 版)

### Phase 3: Signal Ablation (填 Table XII TBD)
- 用真实 trace 跑 4 种权重配置:
  - Full EMA (α=0.7, β=0.2, γ=0.1)
  - No-attn (α=0.0, β=0.6, γ=0.4)
  - Recency-only (α=0.0, β=1.0, γ=0.0) = LRU
  - Frequency-only (α=0.0, β=0.0, γ=1.0) = LFU
- Metric: classification accuracy vs ground-truth (Belady-optimal hot set)
- 输出: Table XII (填上真实数据, 不再 TBD)

### Phase 4: 更新 Paper
- 措辞: "theoretical grounding" → "trace-driven policy evaluation"
- Table VII: 替换为真实 trace + weighted cost + capacity sweep + mean±std
- Table XII: 填上真实数据
- 新增图: competitive ratio vs capacity (折线图, 5 策略 × 5 capacity)
- 文字: 解释 cost metric, 引用 microbenchmark 数据

## 文件输出
- benchmarks/exp_trace_driven_cr.py — trace dump + replay 脚本
- benchmarks/results/exp_trace_driven_cr.json — 结果
- paper/plot_figures_code_data/ — 数据副本 + 画图代码
- paper/overleaf5.tex — 更新 Table VII, Table XII, 相关文字

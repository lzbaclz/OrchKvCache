# OrchKvCache 论文改进方案

> 针对审稿意见中每一个 Weakness 的具体解决方案
> 优先级：🔴 Must-fix（不改会被拒） | 🟡 Should-fix（改了分数明显上升） | 🟢 Nice-to-fix（锦上添花）

---

## W1 🔴 实验基线不公平：自建 FIFO 而非对比真实系统

### 问题本质

主实验的三个系统（GPU-Only / FIFO / OrchKvCache）共享同一个低效的推理框架
（HuggingFace transformers + eager attention + manual decode loop），而非与
vLLM、FlexGen 等真实系统对比。FIFO 基线被人为弱化，导致 1.28-1.77x 的数字
说服力不足。

### 解决方案

#### 方案 A：在 vLLM 中实现 block-level partial swap（推荐，工作量大但一劳永逸）

```
目标：让 vLLM 支持 intra-request partial eviction
修改点：
  1. vllm/core/scheduler.py
     - _preempt() 方法当前整体 swap 一个 SequenceGroup
     - 改为：遍历 seq_group 的 block_table，对每个 block 查询 orchkv 的
       hotness score，只 swap score 最低的 N 个 block
  2. vllm/core/block_manager.py
     - swap_out() 当前移动全部 block
     - 改为：swap_out_partial(seq_group, block_ids) 只移动指定 block
  3. vllm/worker/worker.py
     - execute_model() 在 attention 计算时需要处理"部分在 GPU、部分在 CPU"
       的 block_table → 在 attention 前先 prefetch 需要的 block
预期结果：
  - 如果在 vLLM 内实现 block-level partial swap，预计可看到 >1.1x 的吞吐提升
  - 这个结果远比现在的 1.28-1.77x（弱基线）更有说服力
```

#### 方案 B：增加 FlashAttention 基线对比（工作量中等，性价比高）

```
在当前框架中同时开启 FlashAttention：
  1. 在 HuggingFace transformers 中设置 attn_implementation="flash_attention_2"
  2. 三个系统（GPU-Only / FIFO / OrchKvCache）全部使用 FlashAttention
  3. 重新跑全部实验矩阵

注意事项：
  - FlashAttention 不返回 attention weights（出于效率考虑）
  - OrchKvCache 需要注意力分数来做冷热分类
  - 解决办法：
    a) 用 QK inner-product norm 近似（不需要完整 softmax）
       具体：在 FlashAttention 前，用 Q @ K^T 的行 L2-norm 作为 proxy
       已有论文证明这与 full attention 高度相关（参考 Quest 的做法）
    b) 每 N 步做一次 eager attention 采样（N=5 或 10），其余步骤用 EMA 插值
    c) 在 prefill 阶段收集一次完整的 attention map 作为 warm-start

这个方案的好处：
  - 不需要修改 vLLM 源码
  - 直接消除"没用 FlashAttention"的质疑
  - 如果加速比从 1.77x 降到 1.3x，仍然是有意义的结果
```

#### 方案 C：重新框定论文定位（最小工作量，修改叙事）

```
如果时间不够做 A 或 B，至少要在论文中：
  1. 明确承认基线限制：
     "Our evaluation isolates the scheduling policy by controlling all other
      variables. The reported speedups reflect the pure benefit of attention-
      driven block selection over FIFO, independent of kernel optimizations
      (FlashAttention) or batching strategies (continuous batching)."
  2. 补充一个 "overhead breakdown" 表格：
     - 分解 GPU-Only 到 OrchKvCache 之间的差距来源
     - 例如：eager attention 收集 X%，block 重构 Y%，调度 Z%
     - 让读者清楚地看到哪些开销是可消除的
  3. 把 vLLM 实验重新叙述为"论文的核心发现之一"而非"附加实验"
```

**建议：优先做方案 B，同时在论文中使用方案 C 的叙事策略。**

---

## W2 🔴 缺少与 InfiniGen 的定量对比

### 问题本质

InfiniGen（OSDI'24）是 KV Cache 管理的 SOTA，论文多次提及但无定量对比。
审稿人会认为你在回避与最强竞品的比较。

### 解决方案

#### 方案 A：直接用 InfiniGen 开源代码做端到端对比（推荐，最有说服力）

```
InfiniGen 开源仓库：https://github.com/snu-comparch/InfiniGen
来自首尔国立大学 Wonbeom Lee 等人，OSDI 2024。

步骤：
  1. clone InfiniGen 仓库，按其 README 搭建环境
     git clone https://github.com/snu-comparch/InfiniGen.git
  2. 在相同硬件（A100-80GB）和相同模型（LLaMA-2-7B, Qwen2.5-7B 等）上
     跑 InfiniGen 的 benchmark
  3. 用 OrchKvCache 跑相同配置（相同 prompt、相同 budget、相同输出长度）
  4. 对比维度：
     - 端到端吞吐量（tok/s）
     - 预取命中率（prefetch hit rate / accuracy）
     - 每步调度开销（μs/step）
     - GPU 内存利用率
     - 输出质量（token match rate）
  5. 在论文中呈现为 head-to-head 对比表

关键对比叙事（OrchKvCache 的差异化优势）：
  - InfiniGen 只支持两层（GPU+DRAM），OrchKvCache 支持三层（+SSD）
  - InfiniGen 不主动 evict cold blocks，GPU 内存利用率可能更低
  - InfiniGen 的跨层预测可能更精准（95%+），但 OrchKvCache 的 EMA 开销更低
  - OrchKvCache 在 DRAM 不足时可 spill 到 SSD（InfiniGen 不行）
  - 两者是互补的：InfiniGen 的跨层预取 + OrchKvCache 的分层存储可以组合

注意事项：
  - InfiniGen 可能依赖特定 PyTorch/CUDA 版本，需要检查兼容性
  - 如果 InfiniGen 的 benchmark 脚本不支持我们的模型，
    需要适配其接口（通常只需改 model config）
  - 确保对比公平：相同 block_size、相同 dtype、相同硬件
```

#### 方案 B：子组件级对比 + 定性特征表（方案 A 的补充）

```
即使做了端到端对比，也应补充子组件分析和定性特征表：

1. 定性特征对比表（论文中必须有）

   | Feature          | OrchKvCache | InfiniGen | Quest  | H2O    |
   |-----------------|-------------|-----------|--------|--------|
   | Tiers           | 3 (GPU/DRAM/SSD) | 2 (GPU/DRAM) | 1 (GPU) | 1 (GPU) |
   | Lossless?       | Yes         | Yes       | No     | No     |
   | Proactive evict | Yes         | No        | N/A    | N/A    |
   | Prediction      | EMA-history | Cross-layer| Query  | Cumulative |
   | Granularity     | Block       | Block     | Page   | Token  |
   | IO optimization | SSD-aligned | No        | No     | No     |

2. 预取准确率子组件对比（可选，锦上添花）
   - 用相同的 attention trace 评估不同预取策略的 precision@K
   - 可以直接从 InfiniGen 代码中提取其预取算法的核心逻辑
```

#### 方案 C：引用 + 定性分析（最小工作量，仅作 fallback）

```
如果实在无法实现定量对比，至少在论文中：
  1. 加一个 "Comparison with InfiniGen" 小节到 Evaluation 中
  2. 引用 InfiniGen 论文中的数字（95% prefetch accuracy, 3.6x throughput on LLaMA-7B）
  3. 分析差异的来源（两层 vs 三层、被动 vs 主动 evict）
  4. 明确说明 OrchKvCache 和 InfiniGen 是互补关系而非替代关系：
     "InfiniGen's cross-layer prefetching and OrchKvCache's attention-driven
      tiered eviction address orthogonal aspects of KV-cache management and
      can be composed: InfiniGen predicts which blocks to prefetch, while
      OrchKvCache decides where to place them."
```

**建议：至少做方案 B（子组件对比），如果时间允许做方案 A。**

---

## W3 🔴 Eager Attention 收集的 7.2x 开销未解决

### 问题本质

OrchKvCache 87 tok/s vs GPU-Only 627 tok/s，说明注意力分数收集本身就是一个
巨大的瓶颈。论文声称"采样可降至 <15%"但没有实验证据。

### 解决方案

#### 实验：注意力采样间隔 vs 精度/吞吐量 trade-off

**实验脚本：`benchmarks/exp_attn_sampling.py`**

```
新增实验 "E10: Attention Sampling Sensitivity"

═══ Part A: 分类准确率（orchkv_core 追踪模拟，无需 GPU） ═══

原理：
  attention_tracker.c 的 step_done() 中，未被上报的 block 每步 EMA 衰减：
    ema *= lambda (默认 0.9)
  对 lambda=0.9、采样间隔 N，两次采样之间 EMA 衰减到 0.9^(N-1)：
    N=2  → 0.90x     N=5  → 0.66x
    N=10 → 0.39x     N=20 → 0.14x
  衰减越大，热 block 越可能被错判为 warm/cold → 触发不必要迁移

设计：
  - 生成仿真注意力追踪（Zipf 分布 + 缓慢漂移的 hot set，Gini≈0.9）
  - 对 N ∈ {1, 2, 5, 10, 20, 50}，各创建独立 tiered_manager
  - 采样步上报全部 attention，非采样步只调用 step_done (EMA 衰减)
  - 每步记录 (n_hot, n_warm, n_cold)
  - 对比 N=K 与 N=1 (ground truth) 的聚合分类分布一致性
  - 记录 gpu_demotes / migration_ratio

度量：
  a) classification_agreement = 1 - Σ|class_diff|/(2·n_blocks)
  b) migration_ratio = total_demotes(N) / total_demotes(N=1)

运行：python benchmarks/exp_attn_sampling.py --part a --n-blocks 256 --n-steps 200

═══ Part B: 端到端吞吐量（HF 模型 + SDPA/Eager 切换） ═══

原理：
  模型以 attn_implementation="sdpa" 加载。非采样步：
    output_attentions=False → PyTorch SDPA (Flash 内核，~627 tok/s)
  采样步：
    output_attentions=True  → fallback 到 eager attention (返回 softmax 权重)
  平均每步耗时 = ((N-1)·T_sdpa + T_eager) / N，随 N 增大趋近 T_sdpa

设计：
  - 扫描 N ∈ {0, 1, 2, 5, 10, 20, 50}
    N=0: 纯 SDPA，无 KV 管理 → GPU-Only 上限
    N=1: 全 eager + 全采集 → 当前 OrchKvCache
    N≥2: SDPA 为主 + 间隔 eager → 提议改进
  - 每个 N 跑 warmup + 3 次正式运行
  - 度量：throughput(tok/s), evictions, promotions, speedup_vs_N=1

运行：python benchmarks/exp_attn_sampling.py --part b --model Qwen/Qwen2.5-7B

═══ 论文呈现 ═══

  Fig.15: dual-axis 折线图
    - 左轴: throughput (tok/s)，来自 Part B
    - 右轴: classification accuracy (%)，来自 Part A
    - X 轴: sampling interval N
    - GPU-Only 虚线标注上限吞吐量
  Fig.16: migration_ratio 柱状图
    - X 轴 N，Y 轴迁移倍率（相对 N=1）

  画图：python benchmarks/plot_paper_figures.py --only e10s

  结论模板（用实际数据填充）：
  "Sampling every K steps achieves X× throughput improvement
   over always-eager with only Y% classification accuracy loss
   and Z% additional migrations."
```

#### 方案 B：QK Norm 作为 Lightweight Proxy

```
不收集完整 attention weights，而是用 QK inner-product norm 近似：

实现：
  1. 在每个 decode step，计算 q_current @ K_block^T 的 L2-norm
     这只需要一次矩阵乘法，不需要 softmax
  2. 用这个 norm 作为 a_raw(b) 的近似值
  3. 输入到相同的 EMA 和分类流水线

优势：
  - 与 FlashAttention 兼容（FA 不输出 attn weights，但可以在 FA 之前提取 Q）
  - 计算量：O(n_kv * d * n_blocks) 而非 O(n_kv * seq * seq)
  - Quest 论文已证明 QK norm 与真实 attention 的 Spearman 相关 > 0.9

实验设计：
  对比 "EMA on full attention" vs "EMA on QK norm proxy"
  度量：分类一致性、吞吐量、迁移次数
```

**建议：两个方案都做。方案 A 是必须的（审稿人会直接追问），方案 B 是锦上添花。**

---

## W4 🟡 三层路径缺少吞吐量数据

### 问题本质

SSD 验证实验（§5.7）只证明了正确性，没有报告性能。审稿人想知道：
走 SSD 路径的实际吞吐量损失是多少？

### 解决方案

```
修改 W4 实验，补充性能数据：

实验设置扩展：
  - 在现有 W4 设置（10MB GPU budget）的基础上
  - 增加两个对比配置：
    a) 两层模式：GPU(10MB) + DRAM（无 SSD spill）→ DRAM 不够时 OOM 或 FIFO
    b) 三层模式：GPU(10MB) + DRAM(50MB) + SSD（当前 W4）
    c) 三层模式：GPU(10MB) + DRAM(200MB) + SSD
  - 在 Qwen2.5-7B 和 LLaMA-2-7B 上各跑

补充度量：
  - 端到端吞吐量 (tok/s)
  - SSD 读/写次数和数据量
  - SSD tier 平均驻留时间（冷 block 在 SSD 上待多久才被 promote）
  - 分层时间分解（classify + evict + prefetch 各占多少）

论文呈现方式：
  新增 Table: "Two-tier vs Three-tier Performance"

  | Config           | Throughput | SSD reads | SSD writes | Max seq supported |
  |-----------------|------------|-----------|------------|-------------------|
  | GPU only (80GB) | 627 tok/s  | 0         | 0          | limited by GPU    |
  | GPU+DRAM        | ~85 tok/s  | 0         | 0          | limited by DRAM   |
  | GPU+DRAM+SSD    | ~60 tok/s  | 128       | 128        | virtually unlimited|

  关键叙事：
  "三层模式的吞吐量比两层低 X%，但它支持的最大序列长度/并发数是两层模式的 Y 倍。
   对于 DRAM 不足以容纳全部 KV Cache 的场景（如 128K 上下文），SSD 层是唯一的
   无损解决方案。"
```

---

## W5 🟡 分类器参数敏感性分析不足

### 问题本质

论文只对 α 做了敏感性分析，其余多个超参数（λ, τ, cooldown, δ, θ 初始值）
均使用默认值且未分析。

### 解决方案

```
新增实验 "E11: Hyperparameter Sensitivity"

需要分析的参数（按重要性排序）：

1. λ（EMA decay factor）—— 当前固定 0.3
   实验：λ ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
   度量：分类准确率 + 吞吐量
   预期：
   - λ 太小（0.1）→ 反应太慢，该降级的 block 不及时降级
   - λ 太大（0.9）→ 噪声敏感，频繁误分类
   - Sweet spot 在 0.2-0.4

2. cooldown（阈值调节冷却时间）—— 当前固定 100ms
   实验：cooldown ∈ {10, 50, 100, 200, 500}ms
   度量：阈值振荡次数 + GPU 内存利用率方差
   预期：
   - 太短 → 阈值频繁震荡
   - 太长 → 响应内存压力不及时

3. τ（时间衰减常数）—— 当前未明确给出
   实验：τ ∈ {0.01, 0.05, 0.1, 0.2, 0.5}
   这控制 R(b) 衰减的半衰期，即"多久前访问的 block 开始被认为不重要"

论文呈现方式：
  用一个 2x2 的子图面板：
  (a) λ sensitivity: throughput vs λ
  (b) cooldown sensitivity: threshold oscillation count vs cooldown
  (c) τ sensitivity: classification accuracy vs τ
  (d) Combined: 3D scatter（λ, τ, accuracy）

  结论示例：
  "OrchKvCache 在 λ ∈ [0.2, 0.5], τ ∈ [0.05, 0.2] 范围内表现稳健，
   吞吐量波动 < 5%。推荐默认值 λ=0.3, τ=0.1。"
```

---

## W6 🟡 Prefill 阶段未考虑

### 问题本质

论文只关注 decode 阶段，但 prefill 也会生成大量 KV Cache，尤其是长上下文。

### 解决方案

```
两种处理策略（论文中讨论即可，不一定需要完整实验）：

策略 1: Prefill 后 Warm-Start 分类
  - 在 prefill 完成后，用 prefill 阶段的 attention map 初始化 EMA
  - 具体：prefill 的最后 N 层的 attention distribution → 直接作为 ema_0(b)
  - 好处：decode 第一步就有准确的冷热信息，可以立即开始 evict
  - 实现复杂度低：只需在 KVCacheManager.init_from_prefill() 中添加
    一次 attention 聚合

策略 2: Prefill 期间流式分类
  - Prefill 本身也是分 chunk 执行的（chunked prefill, Sarathi-Serve 方式）
  - 每处理一个 chunk 后，更新对应 block 的 EMA
  - 在 prefill 过程中就开始 evict 冷 block
  - 好处：超长 prefill（128K tokens）不会一次性占满 GPU
  - 实现复杂度中等

论文中的处理：
  - 在 §7 Discussion 中新增一段 "Prefill-Phase Integration"
  - 描述上述两种策略
  - 指出当前实现使用策略 1（成本最低）
  - 如果有时间，补一个简单实验：
    有 vs 没有 prefill warm-start 的 decode 前 50 步分类准确率对比
```

---

## W7 🟢 Motivation 和 Evaluation 使用不同模型

### 问题本质

Motivation 用 Qwen2.5-1.5B 分析注意力分布，但 Evaluation 用 7B-13B 模型。
审稿人质疑分布特征的可迁移性。

### 解决方案

```
最直接的方案：在 Evaluation 的任一模型上补充 Motivation 分析

具体步骤：
  1. 选 Qwen2.5-7B（和 Motivation 的 1.5B 同系列，最有可比性）
  2. 复用 profiling 脚本，收集以下数据：
     - Top-10% 注意力占比
     - Gini 系数
     - Jaccard 相似度
     - Block 级集中度
  3. 在 §2.3 末尾加一段：
     "We validate these findings on Qwen2.5-7B (28 layers, 4 GQA KV heads):
      top-10% attention concentration is 88-95% (vs 90-96% on 1.5B),
      Gini coefficient 0.85-0.95 (vs 0.87-0.97), and Jaccard similarity
      0.45-0.68 (vs 0.47-0.70). The distributions are qualitatively
      identical, confirming that attention skewness generalizes across
      model scales within the same architecture family."

工作量：
  - 只需跑一次 profiling（~30 分钟）
  - 加一段文字 + 可选的一个补充表格
```

---

## W8 🟢 Sequential Processing 限制需澄清

### 问题本质

论文说"our prototype processes requests sequentially"，但又测了 nreq=4,8,16。
读者困惑：到底是怎么跑的？

### 解决方案

```
需要在 §5.1 Experimental Setup 中明确澄清：

替换当前的 "request counts ∈ {1, 4, 8, 16}" 描述为：

  "Requests are processed in a round-robin fashion: the scheduler cycles
   through all active requests, executing one decode step per request before
   moving to the next. All requests share the same GPU KV budget, creating
   memory contention. This models a simplified continuous-batching scenario
   where the KV caches of multiple requests coexist in GPU memory but
   attention computation is serialized. True concurrent batching (where
   multiple requests are batched into a single attention kernel) requires
   integration with production engines such as vLLM, which we analyze
   in §5.8."

同时在 §7 Limitations 中修改第 (3) 条为：

  "(3) Our prototype interleaves requests in round-robin order rather than
   true concurrent batching. The reported throughput includes context-
   switching overhead between requests. With concurrent batching, the
   absolute throughput would be higher for all three systems, but the
   relative advantage of OrchKvCache over FIFO is expected to persist
   because the core benefit—fewer unnecessary migrations—is orthogonal
   to batching strategy."
```

---

## W9 🟢 写作上的小问题

### 解决方案

```
1. 删除所有占位符注释
   搜索 "% [insert" 并全部删除，涉及的行：
   - §2.1: "% [insert Table 1/2]"
   - §2.3: "% [insert Table 3]", "% [insert figure: attention CDF/sink]"
   - §2.4: "% [insert Table 4]", "% [insert figure: SSD bandwidth]"
   - §2.5: "% [insert architecture figure: storage hierarchy]"
   - §3.1: "% [insert architecture figure: system architecture]"
   - §3.2: "% [insert figure: state machine]"
   - §3.6: "% [insert figure: pipeline timeline]"

2. 压缩摘要
   当前 overleaf1.tex 的摘要已经比 orchkvcache1.tex 短很多且更聚焦，
   但仍可以进一步：
   - 删掉 "as its size grows linearly with context length and quickly
     exhausts GPU memory"（冗余，前一句已暗示）
   - 压缩三个 idea 的描述为一句话
   目标：从约 180 词压缩到 150 词

3. 清理未引用的参考文献
   以下 bibitem 在正文中未被 \cite{} 引用：
   - roformer, mamba, rwkv, gemini, qwen2, mistral, mqashazeer
   使用工具搜索确认后删除

4. 统一 overleaf1.tex 的摘要与正文数据
   当前摘要只报告 Qwen2.5-7B 的 1.24x，但正文仍有四模型 1.28-1.77x 的数据。
   建议统一为：
   - 摘要中报告所有模型的范围：1.24-1.77x
   - 或如果只想在摘要中聚焦单模型，在首段末加 "with gains up to 1.77x on
     MHA architectures"
```

---

## Q1-Q6 具体问题的回答建议

### Q1: 为什么不在 vLLM 中实现 block-level partial swap？

```
Rebuttal 回答：
  "Block-level partial swap requires modifying vLLM's block_manager (to
   track per-block tier placement within a request), scheduler (to select
   individual blocks rather than entire requests for swap), and attention
   kernel (to handle mixed GPU/CPU block tables). This is a significant
   engineering effort (~2000 LOC across 5+ files) that we plan as the
   primary follow-up work. Our current prototype validates the algorithmic
   benefit of block-level scheduling, while the vLLM experiment identifies
   the architectural gap that must be bridged."

  如果来得及实现（哪怕是 prototype），加到 revision 里将极大提升分数。
```

### Q2: Eager attention 开销采样后能降到多少？

```
跑 E10 实验（见上方 W3 的解决方案），用数据回答。
```

### Q3: 多请求实验的执行方式？

```
在论文中澄清（见上方 W8 的解决方案）。
```

### Q4: 1.5B vs 7B 模型的分布可迁移性？

```
补充一组 7B 上的 profiling 数据（见上方 W7 的解决方案）。
```

### Q5: 三层路径吞吐量？

```
补充性能数据（见上方 W4 的解决方案）。
```

### Q6: GQA 模型中 KV head 共享的注意力聚合方式？

```
需要在 §3.3 或 §4 中补充说明：

  "For GQA models, multiple query heads share the same KV head. OrchKvCache
   aggregates attention scores at the KV-head granularity: for a KV head
   shared by G query heads, the reported attention score is the average
   across all G query heads' softmax weights for that block:
   
   a_raw(b, kv_head) = (1/G) * Σ_{g=1}^{G} attn_weight(q_head_g, b)
   
   This ensures that a KV block is considered hot if *any* of its sharing
   query heads attends to it strongly. We found averaging to be more stable
   than max-pooling across query heads."
```

---

## 改进优先级排序

| 优先级 | 改进项 | 预计工作量 | 预期分数提升 |
|:------:|--------|:---------:|:-----------:|
| 1 | W3: 采样间隔实验 | 2-3 天 | +1.0 |
| 2 | W1-B: FlashAttention 基线 | 3-5 天 | +1.5 |
| 3 | W2-A: 用 InfiniGen 开源代码做端到端对比 | 2-3 天 | +1.5 |
| 4 | W4: 三层吞吐量数据 | 1 天 | +0.5 |
| 5 | W7: 7B 上的 Motivation 验证 | 0.5 天 | +0.3 |
| 6 | W8: 澄清执行方式 | 0.5 天 | +0.2 |
| 7 | W5: 参数敏感性实验 | 2 天 | +0.5 |
| 8 | W9: 写作清理 | 0.5 天 | +0.2 |
| 9 | W6: Prefill 讨论 | 0.5 天 | +0.2 |
| 10 | W1-A: vLLM partial swap | 7-14 天 | +2.0 |

**如果只有两周时间：做 1+2+3+4+5+6+8+9（约 11 天），分数预计从 5.5 → 7.5+**

**如果有一个月时间：再加上 7+10，分数预计 → 8.0+，有竞争力冲击 OSDI/EuroSys**

---

## 摘要修订建议（overleaf1.tex 版本）

当前 overleaf1.tex 摘要只报告 Qwen2.5-7B 单模型，正文却有四模型数据。
建议修订摘要为如下（保守但完整）：

```
Evaluated on four models (Qwen2.5-7B, Mistral-7B, LLaMA-2-7B, LLaMA-2-13B)
spanning GQA and MHA architectures with A100-80GB GPUs, OrchKvCache improves
throughput by 1.28-1.77x over FIFO-based offloading, with the advantage
scaling with per-token KV-cache size. It reduces unnecessary data migrations
by 139-597x and maintains stable per-token latency as concurrency increases.
Under realistic variable-length workloads, OrchKvCache achieves 1.53x
speedup. Across all migration paths including the full GPU→DRAM→SSD→DRAM→GPU
round trip, OrchKvCache preserves lossless correctness.
```

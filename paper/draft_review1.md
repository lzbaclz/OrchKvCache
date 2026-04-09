# OrchKvCache overleaf5.tex 过度数学化问题审查

> 审查原则：CCF-A 系统会议（SOSP/OSDI/FAST/ATC）的审稿人是系统工程师，不是数学家。公式应该只在**增加清晰度**时使用，而不是"看起来严谨"。对比 vLLM (SOSP'23)、FlexGen (ICML'23)、InfiniGen (OSDI'24)、IMPRESS (FAST'25) 的写作风格：它们极少使用独立编号公式，大量用内联文字 + 数据表格。

---

## 一、可以直接删掉的公式

### 1. Eq 1: Attention 公式（L146-148）

```latex
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
```

**问题**：这是 Transformer 最基础的公式，2017 年至今被引用 13 万次。投 CCF-A 系统会议的审稿人**不可能不知道这个公式**。vLLM 的论文里没写这个公式，InfiniGen 也没写，IMPRESS 也没写。

**建议**：删掉。用一句话带过："LLM 通过 scaled dot-product attention 计算 KV cache 的查询结果 [Vaswani et al., 2017]。"

---

### 2. Eq 3: B_max 最大 batch size（L157-159）

```latex
B_{\max} = \left\lfloor \frac{M_{\text{GPU}} - M_{\text{model}}}{M_{\text{KV}}(s)} \right\rfloor
```

**问题**：这就是一个除法。"GPU 剩余内存除以每个请求的 KV cache 大小"——任何工程师看一眼就懂。给它一个独立编号公式，像在写操作系统教科书。

**建议**：删掉公式。改成内联文字："GPU 能容纳的最大 batch 为 $(M_\text{GPU} - M_\text{model}) / M_\text{KV}(s)$。"

---

## 二、可以简化的公式

### 3. Eq 2: M_KV 内存公式（L151-153）

```latex
M_{\text{KV}} = 2 \times L \times n_{kv} \times s \times d \times \text{sizeof(dtype)}
```

**问题**：Introduction（L84）已经用内联形式写过一遍 `$2\,L\,n_{kv}\,s\,d\;\text{sizeof(dtype)}$`。这里又用独立公式写了第二遍——**重复**。

**建议**：保留 Introduction 的内联形式，删掉 Background 的独立公式。或者只保留这个公式但删 Introduction 中的重复。不要写两遍。

---

### 4. Eq 6: SSD offset 公式（L328-330）

```latex
\text{offset} = (\text{layer} \cdot n_{kv} \cdot B_{\max} + \text{head} \cdot B_{\max} + \text{idx}) \times \text{slab\_size}
```

**问题**：这就是一个三维数组的线性索引。对于写过文件系统或内存分配器的审稿人来说，这个 "公式" 就是 `array[layer][head][idx]` 的 flat offset 计算。不需要独立编号。

**建议**：改为内联表述或用伪代码一行带过。重点是"相同 (layer, head) 的块连续存放，保证顺序写"，这个设计思想比公式本身更重要。

---

## 三、算法伪代码与正文重复

### 5. Algorithm 1: Classification（L281-309）与正文（L260-279）

**问题**：正文 C1+C2 已经用自然语言完整描述了整个分类流程：EMA 更新 → 复合评分 → 阈值比较 → 返回 Hot/Warm/Cold。Algorithm 1 把**完全相同的逻辑**又用伪代码重写了一遍，没有增加任何新信息。

对比 InfiniGen 的做法：它的 cross-layer speculation 算法只在 Algorithm 中描述一次，正文只做高层次总结，不重复。

**建议**：二选一。要么保留 Algorithm 1（它确实更精确），精简正文的 C2 段落为 2-3 句高层次概述；要么保留正文描述，删掉 Algorithm 1。不要两者都完整保留。

---

### 6. C2 分项公式过重（L272-276）

```
â(b) = ema(b) / max_{b'} ema(b')        — normalized attention
R(b) = e^{-τ·(t - t_last(b))}            — recency decay
F(b) = min(f(b) / f_max, 1)              — normalized frequency
```

**问题**：三个子信号每个都给了完整数学定义 + 用斜体解释 "capturing *how strongly/recently/often*"。这种写法像论文的 Related Work 在引用自己的方法论。对于系统论文读者来说，"EMA 归一化"、"指数时间衰减"、"频率归一化" 这三个概念不需要独立公式来定义——它们是信号处理/缓存的基本套路。

**建议**：保留 Eq 5（复合评分 S(b) 是核心贡献），但三个子信号的完整数学展开可以压缩。例如："其中 $\hat{a}(b)$ 为归一化 EMA attention score，$R(b)$ 为指数时间衰减（预计算 256 项查找表），$F(b)$ 为归一化访问频率。" 不需要每个都写出完整的数学表达式。

---

## 四、Online Paging 形式化重复

### 7. §II-A "The Online Tiered KV-Cache Placement Problem"（L161）

> "We formalize this as an instance of the online paging problem: n blocks compete for k < n GPU slots; each eviction incurs a tier-dependent cost..."

### 8. §V-F Competitive Ratio Analysis（L603-607）

> "we model KV-cache management as an instance of the online paging problem: n pages (KV blocks) compete for k < n cache slots (GPU blocks)..."

**问题**：同一个 formalization **说了两遍**。§II-A 在 Background 里形式化了一次，§V-F 在 Evaluation 里又形式化了一次，连措辞都几乎一样。

**建议**：§II-A 只需一句话点明 "这本质上是 online paging 问题"，不需要展开 n、k、cost 的定义。完整的形式化放在 §V-F（那里才需要它来引出 competitive ratio 实验）。

---

## 五、数字密度过高（非公式但影响可读性）

### 9. Scale validation 段落（L190）

> "Gini coefficient 0.45--0.98 (mean 0.91, vs. 0.87--0.97 on 1.5B); top-10% token concentration 44--98% (mean 90%, vs. 90--96%); block-level top-10% concentration 23--97% (mean 81%, vs. ~80%); Jaccard similarity 0.01--0.83 (mean 0.64, vs. 0.47--0.70)."

**问题**：一句话里塞了 **16 个数字**，读者根本记不住。这段想说的就是 "7B 模型和 1.5B 模型的 attention 分布特性基本一致"，但被数字淹没了。

**建议**：要么做成一个小表格（1.5B vs 7B 各指标对比），要么压缩成："7B 模型的所有指标均值与 1.5B 结果一致（Gini 0.91 vs 0.87-0.97，Jaccard 0.64 vs 0.47-0.70），证实 attention skewness 在不同模型规模间具有通用性。" 只保留 2 个最关键的对比数字。

---

### 10. Introduction L84 的内联公式

> "Its footprint scales as $2\,L\,n_{kv}\,s\,d\;\text{sizeof(dtype)}$, where $L$ is the layer count, $n_{kv}$ the number of KV heads, $s$ the sequence length, and $d$ the head dimension."

**问题**：Introduction 是论文最重要的 "hook"，应该用具体数字打动审稿人（"LLaMA-2-7B 每 token 0.5MB"），而不是让他们在第一段就解析一个 6 变量的乘法公式。后面 §II-A 还有独立公式 Eq 2 重复这个内容。

**建议**：Introduction 只写具体数字（"每 token 0.5MB，16×4K batch 就要 32GB"），删掉通用公式。通用公式如果需要，放 Background 即可。

---

## 六、C3 自适应阈值的符号过载

### 11. C3 Adaptive Threshold（L311-319）

文中用了大量符号：$\theta_{\text{hot}}$, $\theta_{\text{cold}}$, $\delta$, HWM$_{\text{gpu}}$, LWM$_{\text{gpu}}$, HWM$_{\text{dram}}$, $\theta_{\min}$, $\theta_{\max}$, cooldown timer。

**问题**：这段描述的机制其实很简单——就是 OS 里经典的 watermark-based 双阈值调节（类似 Linux 的 kswapd 的 high/low watermark）。但引入了 8+ 个符号让它看起来像一个优化问题的约束条件。

**建议**：用更系统化的语言描述："阈值通过水位线反馈循环动态调整：当 GPU 利用率超过上水位线时，降低热阈值以增加驱逐；低于下水位线时，提高热阈值以减少驱逐。DRAM 同理。阈值变化步长固定，受上下限钳制和 100ms 冷却计时器保护。" 不需要 8 个数学符号。

---

## 七、Discussion 的 overhead 可以精简

### 12. Overhead decomposition（§VII, L979-1038）

这部分有两个独立表格（Table XIII overhead breakdown + Table XIV fair-baseline），加上 "Level 1"、"Level 2" 的分层分析结构和三个独立的 gap 因素（batching gap / framework gap / scheduling-policy gap）。

**问题**：分析本身是有价值的（诚实地展示开销），但写法像一篇独立的性能分析报告。对于 Discussion 章节，这些内容的展开程度过深。两张表格 + 1000 字的分析对于本质上是 "Python 开销大" 这个结论来说，篇幅过多。

**建议**：合并两张表格为一张，或保留一张最关键的（fair-baseline），精简文字到 1 段。核心信息就三句：(1) 主要瓶颈是 build_past_kv 占 59.5%；(2) Python scheduling loop 23-49ms >> C classifier <40μs；(3) vLLM 集成证实算法本身有效（1.12×）。

---

## 总结：按优先级排序的修改建议

| 优先级 | 位置 | 当前状态 | 建议 |
|--------|------|----------|------|
| 🔴 高 | Eq 1 (Attention) | 独立编号公式 | **删掉** |
| 🔴 高 | Eq 3 (B_max) | 独立编号公式 | **删掉**，改内联 |
| 🔴 高 | Algorithm 1 vs 正文 C2 | 完全重复 | **二选一** |
| 🔴 高 | Online paging formalization | §II-A 和 §V-F 说了两遍 | §II-A 只点一句，§V-F 展开 |
| 🟡 中 | Eq 2 (M_KV) vs Introduction | 写了两遍 | **只保留一处** |
| 🟡 中 | C2 子信号展开 | 3 个完整子公式 | 压缩为 1 句描述 |
| 🟡 中 | Scale validation 数字墙 | 一句 16 个数字 | 改表格或只留 2 个关键对比 |
| 🟡 中 | C3 符号过载 | 8+ 符号 | 用系统语言重写 |
| 🟢 低 | Eq 6 (SSD offset) | 独立编号公式 | 改内联或伪代码 |
| 🟢 低 | Discussion overhead | 2 张表 + 长文 | 精简为 1 张表 + 1 段 |

### 预计效果
- 删掉/精简以上内容可以**节省约 0.5-0.75 页**
- 论文风格从"偏理论/数学化"转向"系统工程实践"，更符合 FAST/ATC/OSDI 审稿人的阅读习惯
- 核心公式（Eq 4 EMA + Eq 5 Hotness Score + Algorithm 2 Scheduling Loop）保留，这些是真正的贡献

---
---

# 第二部分：文字段落中的"数学味"——让人读不下去的地方

> 审查原则：好的系统论文读起来像**讲故事**，不好的读起来像**实验记录本**。审稿人每篇论文只花 1-2 小时，如果一段话里塞 6+ 个数字、每句话都是 "X achieves Y× over Z (A vs B tok/s)"，读到第三段就会跳读。

---

## 八、核心数据点全文重复过多

以下关键数字在论文中出现的次数统计：

| 数据点 | 出现位置 | 次数 |
|--------|----------|------|
| "139--597× fewer migrations" | Abstract, Intro, §V-B, §V-D, Summary Table, Discussion, Conclusion | **7次** |
| "1.28--1.77× throughput" | Abstract, Intro, §V-B, Summary Table, Discussion ×2, Conclusion | **7次** |
| "100% lossless" | Abstract, Intro, §V-C, §V-I, §V-J, Summary Table, Conclusion | **7次** |
| "4.3% SSD write util → 40.8% batched" | Intro L1, L4, §II-D, §III-D, §V-E caption | **5次** |
| "23 GB/s PCIe Gen4" | Intro, §II-D ×2, §III-D, §V-E, Discussion | **6次** |
| "Python 23--49ms vs C <40μs" | Discussion ×3, Summary Table, Conclusion | **5次** |

**问题**：审稿人第三次看到 "139--597×" 的时候已经记住了，第七次看到的时候会觉得"这篇论文是不是只有这一个结果？"。过度重复同一组数字会给审稿人一种**内容单薄、在凑篇幅**的印象。

**建议**：每个核心数字最多出现 3 次：Abstract 一次（hook），Evaluation 对应 section 一次（详细），Conclusion 一次（总结）。其他引用用 "as shown in §X" 替代具体数字。

---

## 九、段落中数字密度过高——"数字轰炸"

### 13. Introduction 最后一段（L138）

> "demonstrates: (i) **1.28--1.77×** throughput over FIFO offloading with **139--597×** fewer migrations, where the advantage grows with KV-cache size; (ii) sub-**60μs** scheduling latency at **4,096** blocks with sub-linear scaling; (iii) GPU↔DRAM transfer at **23 GB/s** saturating PCIe Gen4; and (iv) bit-exact lossless output on all four models."

一句话里 **7 个加粗数字 + 4 个要点**。读者的大脑在 (ii) 就已经开始滑动了。

**建议**：Introduction 只突出 2-3 个最震撼的数字（throughput + migration reduction + lossless），其他细节留给 Evaluation。例如："OrchKvCache achieves 1.28--1.77× throughput with 139--597× fewer migrations, while maintaining 100% lossless accuracy. The scheduling subsystem adds <60μs per step."

---

### 14. §V-B Finding 1（L481）

> "Table~\ref{tab:e2e} shows that speedup grows from **1.28×** (Qwen, **56 KB**/tok) to **1.77×** (LLaMA-13B, **800 KB**/tok). This is because each unnecessary FIFO eviction moves more data on MHA models, making the cost of blind decisions proportionally higher."

**问题**：Table 已经清楚列出了这些数字，正文再把表格数据逐行复述一遍是冗余的。审稿人不会同时忽略表格和正文。

**建议**：正文只总结趋势："Table X shows a clear trend: speedup scales with per-token KV size, from 1.28× on the smallest (Qwen, GQA) to 1.77× on the largest (LLaMA-13B, MHA)." 不需要重复 56KB 和 800KB——表格里有。

---

### 15. §V-H Realistic Workload 段落（L678）

> "OrchKvCache achieves **1.53×** speedup over FIFO on LLaMA-2-7B (**235.6** vs **153.9** tok/s, Fig.~X) while reducing evictions by **123×** (**4,904** vs **603,972**, Fig.~Y). Under the LongContext-mix workload (which includes requests up to **3,877** tokens), OrchKvCache achieves **1.57×** speedup on LLaMA-2-7B (**246.8** vs **156.8** tok/s) with **170×** fewer evictions."

两句话里 **14 个数字**。而且 "(235.6 vs 153.9 tok/s)" 这种精确到小数点后一位的对比在正文里完全没必要——图表已经展示了这些数据。

**建议**：正文只写核心倍数和趋势："Under both distributions, OrchKvCache achieves 1.5× throughput and 120--170× migration reduction on LLaMA-2-7B. Variable-length requests amplify the benefit of hotness-aware scheduling." 精确数字让图表说话。

---

### 16. §V-H 段落重复（L678-680）

**L678-679 和 L680 是完全一样的段落复制粘贴了两遍！** 这是一个明显的编辑失误，必须删掉一个。

---

### 17. §V-E Scalability（L598）

> "sub-linear scaling (exponent **0.749**): a **64×** increase in blocks (**64→4,096**) yields only **22.5×** increase in latency. At **4,096** blocks (**65K tokens**), P99 is **57.88μs**."

**问题**：
- "exponent 0.749" —— 精确到三位小数的幂律指数。审稿人不关心是 0.749 还是 0.75。说 "sub-linear (roughly $O(n^{0.75})$)" 足够。
- "57.88μs" —— 精确到小数点后两位的微秒数。"~58μs" 或 "<60μs" 已经足够精确。
- 一句话 7 个数字。

**建议**："Scheduling latency scales sub-linearly: at 4K blocks (~65K tokens), P99 remains under 60μs."

---

## 十、解释显而易见的事情

### 18. §V-C Quality Verification 的 downstream task 段落（L561）

> "This is the expected consequence of bit-exact lossless output: since every generated token is identical under greedy decoding, any downstream metric computed from that output (accuracy, F1, BLEU, etc.) is also identical **by construction**. The result provides empirical confirmation that OrchKvCache's three-tier migration introduces *zero* quality degradation on real reasoning tasks, not just on synthetic token-match tests."

**问题**：花了 3 行解释 "如果 token 完全一样，那下游指标也完全一样" ——这是一个**不需要解释的自明逻辑**。说 "by construction" 然后又说 "provides empirical confirmation" 是矛盾的（要么是 by construction 不需要 empirical confirmation，要么需要 empirical 那就不是 by construction）。

**建议**：删到一句话："Since output is bit-exact, all downstream metrics are identical by construction (Table X confirms this on four NLP tasks)."

---

### 19. §V-C Quality Verification 的 lossless 解释（L537）

> "The lossless property is by construction: all data movement uses `torch.Tensor.copy_` (CUDA DMA for GPU↔DRAM) and standard file I/O---both data-preserving operations. The per-block `rwlock` prevents partial reads during migration."

**问题**：这段解释了"为什么是无损的"——因为 copy 操作本身不改数据。这在 Design 里的 Migration Engine (C7) 已经详细描述了（write lock + atomic update）。Evaluation 里再解释一遍是重复的。

**建议**：直接引用 Design："The lossless property follows from the migration engine's data-preserving copy semantics and per-block locking (§III-E)."

---

## 十一、Design 段落像实现手册

### 20. §III-E Demote Path（L340-345）

> "(1) the engine acquires a write lock on each victim, transitioning its state to *migrating*; (2) for GPU→DRAM, an asynchronous DMA copy is issued on a dedicated CUDA stream (round-robin across N=4 streams); (3) for DRAM→SSD, an asynchronous write is submitted to the IO thread pool---for batch operations, k blocks are serialized into a contiguous staging buffer and issued as a single large write, improving SSD utilization from ~9% to ~41%; (4) for two-hop GPU→SSD cold eviction, the engine chains steps (2) and (3) via an intermediate DRAM staging buffer; (5) upon completion, the source-tier slot is freed and the block's metadata is updated atomically under the write lock."

**问题**：5 个编号步骤读起来像 API 文档或代码注释。审稿人想知道**设计思想**（"async + batch + chain"），不是想读 step-by-step 的调用顺序。"round-robin across N=4 streams" 这种实现细节可以放 Implementation 里。

**建议**：压缩为 2-3 句设计层面的描述：
"Demotion uses asynchronous DMA (GPU→DRAM) and batched pwrite (DRAM→SSD), chaining both for two-hop cold eviction. Each victim is write-locked during migration to prevent partial reads. Batch writes aggregate multiple 32KB blocks into a single contiguous I/O, raising SSD utilization from 9% to 41%."

---

### 21. §III-E Atomicity 段落（L349）

> "A per-block reader-writer lock ensures that no concurrent reader observes partially-migrated data. If a transfer fails (DMA error or I/O error), the engine rolls back: the block remains at its source tier in active state and any destination allocation is freed. Per-direction migration statistics (count, bytes, errors) are tracked for monitoring."

**问题**：最后一句 "Per-direction migration statistics (count, bytes, errors) are tracked for monitoring" 是纯实现细节，不影响 correctness 也不影响性能。不该出现在 Design 里。

**建议**：删掉最后一句。

---

## 十二、Evaluation 的 Finding 格式太机械

### 22. §V-B Findings 1-4 的模板化写法

每个 Finding 都严格遵循同一个模板：
```
**Finding N: [Bold claim with ⇒ arrow].** Table/Figure X shows that [exact numbers]. 
This is because [causal explanation].
```

**问题**：四个 Finding 读下来非常单调。每个都是 "Table shows... This is because..." 的三段式。审稿人到 Finding 3 就开始跳读了。

**建议**：不需要每个都用 "Finding N:" 的编号格式。前 1-2 个关键发现可以保留加粗标题，后面的可以用自然段落衔接。更好的做法是把 Findings 组织成一个连贯的故事："OrchKvCache 在所有模型上都优于 FIFO（Fig.1），且优势随 KV 大小增长（Table IV）。这种趋势的根因是... TPOT 也保持稳定（Fig.3）..."

---

## 十三、vLLM Integration 写得像调试报告

### 23. §V-L Root-cause analysis 段落（L844-845）

> "Block-level scoring underperforms at 32 concurrent requests because `score_request()` computes per-block hotness scores in Python for *every* block of each candidate victim: with **32 sequences × ~50 blocks** each, this is **~1,600 Python-level `get_score()` calls** per preemption event. Additionally, the EMA decay in `step_done()` iterates over all tracked blocks every step."

**问题**：读起来像 profiling 日志。"32 sequences × ~50 blocks = ~1,600 calls" 这种乘法审稿人不需要你算给他看。

**建议**："Block-level scoring's Python implementation does not scale: per-block score computation grows linearly with total blocks across all candidate victims, adding milliseconds per preemption event. A C-native implementation (§V-E: 2.6μs) would eliminate this overhead."

---

### 24. §V-L Partial swap 的数学化解释（L846-847）

> "Formally, evicting $N$ blocks frees $N$ GPU slots, but the subsequent restore of those $N$ blocks requires $N$ GPU slots, leaving zero headroom for the new allocation that triggered preemption."

**问题**：给一个 "释放 N 块，恢复 N 块，净释放 0 块" 的显然结论用 "Formally" 开头，像在做数学证明。

**建议**：直接说："The freed GPU memory is immediately consumed by restoring the evicted blocks, yielding zero net gain—confirmed by a `NoFreeBlocksError` in our prototype."

---

## 十四、Discussion 多处自我重复

### 25. §VII Attention collection overhead（L1048-1049）

> "pure SDPA achieves 1,658 tok/s; N=1 drops to 180 tok/s; N=50 stays at 183 tok/s. Attention extraction itself is only 3.8% of per-step time at N=10. The gap is dominated by KV-cache reconstruction (59.5%) and the Python scheduling loop (11.6%)."

**问题**：这些数字在以下位置已经完整出现过：
- §V-G Hyperparameter Sensitivity 的 E2E validation 段（L656）
- Table XVI overhead breakdown（L983-1001）
- §VII Level 1 component breakdown（L1003-1004）

这是**第四次**叙述同样的信息。Discussion 应该是提出新的 insight，不是重复 Evaluation 的数据。

**建议**：删掉这整段。直接引用："The overhead analysis (Tables XV--XVI) shows that scheduling overhead (3.8%) is negligible; the bottleneck is KV reconstruction (59.5%)."

---

### 26. §VII Block-level vs request-level（L1043-1044）

这段的内容和 §V-L vLLM Integration 的 Implication 段落（L848-849）高度重叠：

- §V-L: "the benefit of attention-driven scheduling operates at two levels: (1) victim selection level... (2) data migration level..."
- §VII: "attention-driven scheduling benefits LLM inference at two distinct levels. At the victim selection level... At the data migration level..."

**几乎一模一样的结构和措辞。**

**建议**：§VII 只保留"下一步怎么做"（extend FlashAttention），删掉对 §V-L 的重述。

---

## 十五、Hyperparam 小节过度展开

### 27. §V-G 四段超参分析（L641-647）

λ、τ、cooldown、joint sensitivity 各一段，每段都是：
1. "Figure X(y) sweeps [parameter] ∈ {具体数值}."
2. "At [极端值], [坏事发生]."
3. "At [另一个极端], [另一个坏事]."
4. "The optimal range is [区间]."

**问题**：这个四段的结论就是 "默认参数在一个很大的 sweet spot 里，不需要精调"。但花了 ~20 行说这件事。对于一篇空间紧张的会议论文，这个 subsection 的 ROI（投入产出比）很低。

**建议**：压缩成一段 + 图："Figure X shows that accuracy is stable (< 5% variation) across a broad parameter range (λ ∈ [0.7, 0.95], τ ∈ [25, 100]). The defaults (λ=0.9, τ=50, cooldown=0.5s) lie in the center of this sweet spot. E2E validation on Qwen2.5-7B confirms: throughput varies by <1.6% across all sampling intervals (Fig. Y)."

---

## 总结更新：完整优先级列表

| 优先级 | # | 问题 | 类型 | 建议 |
|--------|---|------|------|------|
| 🔴 | 16 | §V-H 段落完整重复（复制粘贴 bug） | **Bug** | 删掉重复段落 |
| 🔴 | 1 | Eq 1 Attention 公式 | 删掉的公式 | 删 |
| 🔴 | 2 | Eq 3 B_max 除法 | 删掉的公式 | 删 |
| 🔴 | 5 | Algorithm 1 vs 正文重复 | 重复 | 二选一 |
| 🔴 | 7-8 | Online paging 说两遍 | 重复 | §II-A 精简 |
| 🔴 | 8b | 核心数据 7 次重复 | 全文重复 | 每个数字≤3次 |
| 🟡 | 13 | Intro 结尾 7 数字轰炸 | 数字密度 | 减到 2-3 个 |
| 🟡 | 14 | Finding 复述表格数据 | 冗余 | 只写趋势 |
| 🟡 | 15 | Realistic 14 个数字 | 数字密度 | 精确数让图说话 |
| 🟡 | 17 | Scalability 0.749/57.88μs | 过精确 | 约数 |
| 🟡 | 18 | downstream "by construction" | 解释显然 | 删到一句 |
| 🟡 | 19 | lossless 原因重复解释 | 重复 | 引用 Design |
| 🟡 | 20 | Demote 5 步操作手册 | 实现细节 | 压缩到 2-3 句 |
| 🟡 | 22 | Findings 1-4 模板化 | 读感单调 | 自然段落叙事 |
| 🟡 | 25-26 | Discussion 重述 Evaluation | 重复 | 删重述，只写新 insight |
| 🟡 | 27 | Hyperparam 4 段展开 | 过度展开 | 压缩到 1 段 |
| 🟢 | 9 | Scale validation 16 数字 | 数字密度 | 表格或精简 |
| 🟢 | 10 | Intro 内联公式重复 | 重复 | 删 Intro 公式 |
| 🟢 | 11 | C3 符号过载 | 符号密度 | 用系统语言 |
| 🟢 | 12 | Discussion overhead 过长 | 过度展开 | 精简 |
| 🟢 | 21 | Atomicity 监控统计 | 实现细节 | 删最后一句 |
| 🟢 | 23 | Root-cause 32×50=1600 | 调试报告 | 概括 |
| 🟢 | 24 | Partial swap "Formally" | 数学化 | 直接说结论 |

### 预计总效果
- 修复 Part 1 公式问题：**节省 ~0.5-0.75 页**
- 修复 Part 2 文字问题：**额外节省 ~0.75-1 页**
- 合计可省 **1.25-1.75 页**，用于放更多实验图表或补充遗漏的内容
- 论文读感从"实验记录本 + 数学教材"变成"讲清楚一个系统故事"

---
---

# 第三部分：审稿人 Review 回应方案

---

## W1：1.28–1.77× 的 headline claim 在不公平环境下测得

### 审稿人的核心逻辑

1. Table XVII 显示 Fast-FIFO（608 tok/s）> Fast-OrchKv（258 tok/s）
2. 所以 1.28–1.77× 的加速来自"共享的 build_past_kv 开销稀释了 FIFO 的额外延迟"
3. 这是比值放大效应，不是调度策略的净收益

### 这个攻击有没有道理？

**部分有道理，但不完全成立。** 需要拆清楚两件事：

- Fast-OrchKv 比 Fast-FIFO **慢**的原因是 **Python scheduling loop 开销**（23-49ms），不是调度策略本身差。Table XVII 的 "Fast-OrchKv" 在每步要跑完整的 Python → pybind11 → C classifier → Python iteration，而 Fast-FIFO 只做一个 deque.pop()。这是**实现开销**，不是**算法开销**（C classifier 本身只要 <40μs）。
- 1.28–1.77× 在原型框架里**确实被共享开销放大了**——这一点论文自己承认了。但 migration reduction（139-597×）是**不受框架开销影响的独立指标**，而 vLLM 集成（1.12×）是在**生产引擎中不受 Python 开销影响的测量**。

### 修改方案：重组叙事（选方案 a，不需要新实验）

#### 修改 1：Abstract 重写 headline

**现在的写法：**
> "OrchKvCache achieves **1.28–1.77×** throughput over FIFO offloading, reduces unnecessary migrations by **123–374×**..."

**改为：**
> "OrchKvCache reduces unnecessary data migrations by **139–597×** over FIFO while maintaining **100% lossless** accuracy across the full GPU→DRAM→SSD path. In a block-level prototype, these savings translate to **1.28–1.77×** measured throughput improvement; integration into vLLM's production engine delivers **1.12×** throughput under memory pressure, free of prototype framework effects."

**关键变化**：migration reduction 变成 lead claim；1.28-1.77× 降级为 "prototype 环境下的测量值"；1.12× vLLM 数据作为 "production-grade 验证"。

#### 修改 2：Introduction contributions 段重写

在 contributions 列表后加一段上下文：

> "**Transparency note on throughput measurement.** The 1.28–1.77× prototype speedup is measured in a regime where shared framework overhead (Python KV-cache reconstruction, 59.5% of per-step time) amplifies relative throughput differences. We present a systematic overhead decomposition in §VII that disentangles three independent factors: batching strategy, framework overhead, and scheduling-policy benefit. The scheduling algorithm's intrinsic benefit is most cleanly visible in two metrics: (1) 139–597× migration reduction (§V-B), which is independent of framework overhead; and (2) 1.12× throughput in vLLM's C++ engine (§V-L), which is free of Python amplification."

#### 修改 3：Conclusion 对齐

**现在的写法：**
> "OrchKvCache achieves **1.28–1.77×** throughput over FIFO with **139–597×** fewer migrations..."

**改为：**
> "OrchKvCache's attention-driven classifier reduces data migrations by **139–597×**, achieving a competitive ratio within **1.51×** of the offline optimal. These savings translate to **1.28–1.77×** throughput in our prototype (where shared framework overhead amplifies the relative gain) and **1.12×** in vLLM's production engine."

#### 修改 4：§VII Discussion 加一段"预估吞吐"（方案 b 的简化版，不需要新实验，只做 back-of-envelope）

用 Table XVI 数据做简单计算：

- Fast-OrchKv 当前：258 tok/s（Qwen）
- Python scheduling loop 每步增加的开销：9.42ms（step_done + schedule）
- C classifier 等价开销：<0.04ms
- 如果替换 Python loop → C：每步节省 ~9.4ms → 每步从 81.1ms 降到 ~71.7ms
- 预估吞吐：258 × (81.1/71.7) ≈ **292 tok/s**
- Fast-FIFO：608 tok/s（无 attention 开销）

加一段：

> "**Projected throughput with native scheduling loop.** If the Python scheduling loop (9.42 ms/step) were replaced by the C/CUDA classifier (0.04 ms/step)—following the same porting pattern used for the core classifier—per-step time would decrease from 81.1 ms to ~71.7 ms, projecting Fast-OrchKv throughput to ~292 tok/s on Qwen2.5-7B. The remaining gap to Fast-FIFO (608 tok/s) is dominated by `report_attention` (3.06 ms) and the inherent cost of maintaining per-block EMA state; eliminating eager-attention extraction via FlashAttention's partial output would close this further."

这让审稿人看到：(1) 你知道这个问题，(2) 你量化了改进空间，(3) 路径是清晰的。

---

## W5：InfiniGen 对比不是同条件 benchmark

### 审稿人的核心逻辑

1. OrchKvCache 用 HuggingFace + Qwen/LLaMA，InfiniGen 用 FlexGen + OPT-1.3B
2. 不同框架 + 不同模型 + 不同硬件配置 = 不是公平对比
3. "complementary" 的声明未经验证

### 这个攻击完全成立吗？

**基本成立。** 论文确实把一个设计空间分析写成了实验对比的语气。Table XVI 的 InfiniGen 数据是直接引用的，不是复现的。

### 修改方案：降级表述（方案 a，不需要新实验）

#### 修改 1：改 subsection 标题

**现在：** `\subsection{Comparison with InfiniGen}`

**改为：** `\subsection{Design-Space Positioning: OrchKvCache vs. InfiniGen}`

#### 修改 2：开头段加 disclaimer

**现在的写法：**
> "We compare OrchKvCache with InfiniGen, the state-of-the-art KV cache management system (OSDI '24), using data from their published evaluation on the same models and datasets."

**改为：**
> "We position OrchKvCache relative to InfiniGen~\cite{infinigen} (OSDI '24) in the KV-cache design space. **We do not claim throughput superiority**, as the two systems target different frameworks (HuggingFace vs. FlexGen), different models (Qwen/LLaMA vs. OPT), and different offloading paths (three-tier attention-driven vs. two-tier cross-layer prefetch). Instead, we analyze their complementary design choices and reproduce InfiniGen's published numbers to contextualize the throughput landscape."

#### 修改 3：删除/改写暗示优于 InfiniGen 的句子

审查全文，确保没有 "OrchKvCache outperforms InfiniGen" 或 "achieves higher throughput than InfiniGen" 的句子。当前论文原文在这方面还算克制（写的是 "orthogonal and could be combined"），但末尾段：

> "OrchKvCache achieves 1.28–1.77× over FIFO on LLaMA-2-7B through 13B in its native framework."

这句紧跟在 InfiniGen 吞吐数据之后，暗示了对比。改为：

> "In its native framework, OrchKvCache achieves 1.28–1.77× over FIFO, demonstrating the effectiveness of attention-driven scheduling in a different region of the design space."

#### 修改 4：Table XV（Feature comparison）保留但加注

在 Table XV 的 caption 后加：

> "Design-space comparison; not a head-to-head benchmark. InfiniGen data from [5]."

#### 修改 5：强化互补性叙事

保留 "complementary" 段落但更具体地说明 how：

> "A concrete composition: InfiniGen's cross-layer speculator predicts *which* blocks to prefetch with higher accuracy than EMA alone; OrchKvCache's three-tier engine decides *where* each block resides and *when* to batch-migrate cold blocks to SSD—a capability InfiniGen does not provide. Testing this composition requires integrating InfiniGen's speculator into OrchKvCache's tiered manager, which we leave to future work."

---

## W6：SDPA 回退模式的分类精度未验证

### 审稿人的核心逻辑

1. OrchKvCache 的核心贡献是 "attention-driven" 分类
2. 在 8K+（最需要 KV 管理的场景），attention 信号不可用，退化为 recency+frequency
3. 从未展示 SDPA fallback 的分类精度 vs full EMA 的对比
4. 如果 recency+frequency 就够用了，attention 信号就是不必要的开销

### 这个攻击有多严重？

**很严重——但可以化解。** 有两条辩护线：

1. **§V-K Selective Restore 已经间接回答了**：Table XI 展示 EMA top-1% blocks 覆盖 100% attention weight，而 Random 只覆盖 2%。这证明 EMA（含 attention 信号）的预测质量远超随机/纯频率信号。但确实没有直接对比 "full EMA vs recency+frequency only"。

2. **§V-G Hyperparameter Sensitivity 的 trace simulation 有相关数据**：α=0.7（attention 占主导）时精度最高，α=0（无 attention）时精度会怎样？如果有这个数据点就是直接证据。

### 修改方案：两步走

#### 步骤 1：补一个简单的消融实验（推荐，工作量小）

在 2K 或 4K 上下文下（不会 OOM），跑两组配置：

| 配置 | α (attention) | β (recency) | γ (frequency) | 描述 |
|------|---------------|-------------|---------------|------|
| Full EMA | 0.7 | 0.2 | 0.1 | 默认（有 attention 信号） |
| No-attn | 0.0 | 0.6 | 0.4 | SDPA 回退模拟（无 attention 信号） |
| Recency-only | 0.0 | 1.0 | 0.0 | 纯 LRU 等价 |
| Frequency-only | 0.0 | 0.0 | 1.0 | 纯 LFU 等价 |

测量：
- 分类精度（vs ground-truth hot/cold labels from full attention）
- 迁移次数
- 吞吐

这个实验只需要改 α/β/γ 参数，其他代码不变。**估计 1-2 小时可以跑完。**

如果结果是 "Full EMA 显著优于 No-attn"，则证明 attention signal 确实关键，需要讨论 8K+ 如何获取它。

如果结果是 "No-attn 接近 Full EMA"，则可以说 "recency+frequency 在短期上下文中已足够，attention signal 在注意力模式更复杂的长上下文或多轮对话中可能更有价值"——把它变成一个正面发现（系统的优雅降级）。

#### 步骤 2：修改论文文字

无论实验结果如何，在 §V-J 末尾加一段：

**如果 Full EMA >> No-attn（attention 信号关键）：**

> "**Attention signal vs. fallback signals.** To quantify the marginal contribution of the attention signal, we compare full EMA (α=0.7) with a recency+frequency-only configuration (α=0, simulating SDPA fallback) at 2K context on Qwen2.5-7B. Full EMA achieves X% classification accuracy vs. Y% for the fallback, with Z× more migrations under the fallback. This confirms that the attention signal provides substantial prediction quality beyond recency and frequency alone. At 8K+ context, where eager attention triggers OOM, integrating partial attention export from FlashAttention (e.g., per-block QK-norm statistics) would recover this signal; we leave this integration to future work."

**如果 Full EMA ≈ No-attn（attention 信号锦上添花）：**

> "**Graceful degradation without attention signals.** At 2K context, a recency+frequency-only configuration (α=0) achieves Y% classification accuracy, close to full EMA's X%. This explains why SDPA-fallback mode at 8K still outperforms FIFO (Table X): the scheduling framework's recency and frequency signals alone capture most of the block-level hotness structure. The attention signal becomes increasingly valuable in scenarios with rapid attention-pattern shifts (e.g., long-range retrieval), where recency alone cannot track the hot set fast enough—precisely the regime targeted by future FlashAttention integration."

#### 步骤 3：Limitations 重排

把 SDPA fallback 提到 Limitations 第一条（而不是第四条）：

> "(1) At 8K+ context, eager attention triggers OOM, forcing a fallback to recency+frequency-only classification. [实验结果的一句话总结]. Integrating partial attention statistics from FlashAttention would restore the full classifier at long contexts."

---

## 三个 Weakness 修改的优先级和工作量

| Weakness | 严重程度 | 修改方案 | 工作量 | 需要新实验？ |
|----------|----------|----------|--------|-------------|
| **W1** | 🔴 致命（影响 headline claim） | 重组叙事 + back-of-envelope 预估 | **2-3 小时**（纯文字修改） | 否 |
| **W5** | 🟡 中等（影响一个 subsection） | 降级表述 + 改标题 | **1 小时**（纯文字修改） | 否 |
| **W6** | 🔴 致命（动摇核心 contribution） | 补消融实验 + 修改文字 | **3-4 小时**（1-2h 跑实验 + 2h 改文字） | **是（但很简单）** |

### 建议执行顺序

1. **先做 W6 的消融实验**——这决定了后续文字怎么写（attention 信号到底关不关键）
2. **然后改 W1 的叙事**——最影响审稿人对论文的整体印象
3. **最后改 W5**——工作量最小，改几个词就行

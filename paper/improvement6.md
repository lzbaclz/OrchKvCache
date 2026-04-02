# OrchKvCache SC'26 改进方案：Fix 三个 Major Weaknesses

> 目标会议：SC'26 (截稿约 2026 年 4 月中旬)
> 只聚焦三个 Major Weaknesses，不涉及 Minor 修复
> 每个 Weakness 给出可落地的实现路径 + 预期产出

---

## 核心诊断：9.2x 开销的根因

在讨论三个 Weakness 之前，先定位根因。E10 的数据表明：

```
N=0 (SDPA only, 无 KV 管理)：1658 tok/s
N=1 (每步 eager attention)：  180 tok/s    → 9.2x gap
N=50 (50步采样一次)：          183 tok/s    → 仍然 9.0x gap
```

N=50 vs N=1 几乎无差别，说明 attention 收集不是瓶颈。
真正的瓶颈在 `build_past_kv()`：

```python
# kvcache_manager.py:270-311 —— 每步 decode 都执行
def build_past_kv(self):
    for layer_idx in range(self.n_layers):       # 28-40 层循环
        total_tokens = sum(b.token_count ...)     # Python 求和
        key_out = torch.zeros(1, n_kv, total, d)  # 每步重新分配 GPU tensor
        val_out = torch.zeros_like(key_out)        # 又一次分配
        for blk in blocks:                         # 逐 block Python 循环
            if blk.tier != TIER_GPU:
                self._promote_block(blk)           # 可能触发 DRAM→GPU copy
            key_out[0,:,pos:pos+n,:] = blk.gpu_data[0,:,:n,:]  # 逐 block copy
```

**每步 decode 的开销**：
- 28-40 次 `torch.zeros()` GPU 分配 (每次 ~1 ms)
- 28-40 层 x N_blocks 次 Python for 循环 + tensor slice copy
- Python GIL + 对象创建 + GC 压力

这个开销是 **O(n_layers x n_blocks)** 的纯 Python 开销，与 attention 采集无关。

---

## MW1: 消灭 9.2x Python 开销

> 对应审稿意见 W3: "9.2x overhead blocks deployment"
> 这是最关键的改进——不解决这个，其他数字都没有说服力

### 方案：C/CUDA Persistent Block Table（零重构路径）

#### 核心思路

不再每步重建 past_key_values tensor。改为维护一个 **GPU 上的持久化分页 KV 缓存**，
attention 计算直接从 block table 读取，不需要拼接成连续 tensor。

#### 具体实现

```
新增文件：src/paged_kv/paged_kv_cache.cu

数据结构：
  // GPU 端常驻：block table
  struct PagedKVCache {
      half **block_ptrs;       // [max_blocks] GPU 指针数组
      int  *block_tokens;      // [max_blocks] 每 block 实际 token 数
      int  *seq_block_ids;     // [max_seq_len / block_size] 当前序列的 block 映射
      int   n_blocks;
      int   block_size;        // 16
      int   n_kv_heads;
      int   head_dim;
  };

  // 关键接口
  __host__ void paged_kv_append_token(PagedKVCache *cache, int layer,
                                       const half *new_k, const half *new_v);
  __host__ void paged_kv_evict_block(PagedKVCache *cache, int block_id,
                                      half *dram_dst);
  __host__ void paged_kv_promote_block(PagedKVCache *cache, int block_id,
                                        const half *dram_src);

Attention 集成（两个方案任选其一）：
  方案 A：自定义 Paged Attention Kernel
    - 改写 attention 计算，直接从 block_ptrs[] 读取 K/V
    - 不需要 build_past_kv()，完全消除重建开销
    - 参考 vLLM 的 PagedAttention kernel，但简化为单请求版本
    - 工作量：~2000 行 CUDA

  方案 B：Persistent Contiguous Buffer（折中方案）
    - 维护一个 GPU 上的大 contiguous buffer [n_layers, 2, n_kv, max_seq, d]
    - 每个 block 直接映射到 buffer 的固定偏移位置
    - 新 token append：直接写入 buffer 对应位置，无需重建
    - evict：只需更新元数据（标记该段无效），buffer 空间不释放
    - promote：从 DRAM copy 回 buffer 对应偏移
    - 优势：HF transformers 的 attention 不需要改，直接传 buffer slice
    - 工作量：~500 行 C/CUDA + ~200 行 Python

推荐：先做方案 B（1 周），拿到数字后评估是否需要方案 A。
```

#### 预期效果

```
当前 (Python build_past_kv):
  GPU-Only: 627 tok/s → OrchKvCache: 87 tok/s (7.2x gap)
  SDPA-Only: 1658 tok/s → OrchKvCache+N=1: 180 tok/s (9.2x gap)

方案 B 后（消除 tensor 重建）:
  预期 OrchKvCache 吞吐：~500-600 tok/s（接近 GPU-Only 的 80-95%）
  调度开销本身 < 40 μs/step << 1-10 ms decode step
  预期 OrchKvCache vs FIFO 加速比不变（1.28-1.77x），但绝对数字有说服力

  关键验证指标：
  - overhead_ratio = (GPU-Only - OrchKvCache) / GPU-Only
  - 目标：< 10%（当前 86%）
```

#### 实施计划

```
Week 1:
  - Day 1-2: 实现 PersistentKVBuffer C/CUDA 数据结构
    - GPU 端大 buffer 分配/释放
    - append_token / evict / promote 的 C 函数
  - Day 3-4: pybind11 暴露接口，改造 KVCacheManager
    - 删除 build_past_kv() 中的 torch.zeros + for 循环
    - 改为：直接返回 buffer 的 view/slice
  - Day 5: 集成测试
    - 验证 token match 仍为 100%
    - 基准吞吐测试

Week 2:
  - Day 1-2: 重新跑全部实验矩阵（4 模型 x 配置）
  - Day 3: 如果吞吐仍有 gap，profile 定位并优化
  - Day 4-5: 更新论文数字和图表
```

---

## MW2: InfiniGen 端到端对比

> 对应审稿意见 W2: "Missing head-to-head comparison with InfiniGen"
> InfiniGen 是 OSDI'24，最直接的竞争者

### 方案：三层面对比（端到端 + 子组件 + 定性表）

#### 层面 1：端到端吞吐对比

```
InfiniGen 开源仓库：https://github.com/snu-comparch/InfiniGen

步骤：
  1. 部署 InfiniGen
     - clone 仓库，按 README 配环境
     - InfiniGen 基于 FlexGen，支持 OPT/LLaMA 系列
     - 关键：确认它能在 A100-80GB 上跑 LLaMA-2-7B

  2. 统一测试条件
     - 硬件：A100-80GB
     - 模型：LLaMA-2-7B（两个系统都支持）
     - 序列长度：{1024, 2048, 4096}
     - GPU KV 内存限制：{50%, 80%, 100%} of full cache
     - 生成长度：64-128 tokens

  3. 度量指标
     - 端到端吞吐量 (tok/s)
     - GPU 内存峰值
     - 输出质量：perplexity (WikiText-2) + token match rate
     - 每步调度开销 (μs)

  4. 如果 InfiniGen 不支持 LLaMA-2-7B
     - 用 InfiniGen 支持的 OPT-6.7B 做对比
     - 在 OrchKvCache 侧也加 OPT-6.7B 支持
     - 或只做 OPT-13B（InfiniGen 论文的主要评估模型）
```

#### 层面 2：子组件级对比

```
即使端到端对比因框架差异不完全公平，也可做子组件对比：

  预取准确率对比：
    - 从真实模型推理中收集 attention trace (100 decode steps)
    - 用 InfiniGen 的跨层预测算法 和 OrchKvCache 的 EMA 算法
      分别预测下一步需要的 block
    - 计算 precision@K, recall@K, F1@K
    - InfiniGen 论文报告 95%+ 准确率
    - OrchKvCache 的 EMA 预测准确率预计较低，但调度开销更低

  调度开销对比：
    - InfiniGen 需要 cross-layer inference（上一层的 Q 预测下一层的 KV 需求）
    - OrchKvCache 只需 EMA 更新 + 阈值检查
    - 在相同 block 数下对比 per-step μs
```

#### 层面 3：定性特征对比表

```
更新论文中的 Table（已有雏形，需要加数据列）：

| Feature              | OrchKvCache | InfiniGen | H2O  | vLLM  |
|---------------------|-------------|-----------|------|-------|
| Storage tiers       | 3 (GPU/DRAM/SSD) | 2 (GPU/DRAM) | 1    | 2     |
| Lossless?           | Yes         | Partial*  | No   | Yes   |
| Proactive evict     | Yes         | No        | N/A  | No    |
| Prediction          | EMA+recency | Cross-layer | Cumul.| None  |
| Pred. accuracy      | ~80-88%†    | 95%+      | N/A  | N/A   |
| Scheduling overhead | <40μs       | ~Xms‡     | N/A  | N/A   |
| SSD I/O opt.        | Yes         | No        | No   | No    |
| Framework           | Custom+vLLM | FlexGen   | HF   | vLLM  |

† 来自 E10 Part A trace 仿真
‡ 需要实测
* InfiniGen 在 <100% capacity 时丢弃 cold entries
```

#### 论文叙事策略

```
核心定位：OrchKvCache 和 InfiniGen 是互补的，不是替代关系。

  - InfiniGen 的优势：跨层预测更精准 (95% vs 80-88%)
  - OrchKvCache 的优势：
    1. 三层存储（InfiniGen 止步于 DRAM）
    2. 主动驱逐（InfiniGen 不 evict，DRAM 满了就丢）
    3. SSD 对齐 IO（InfiniGen 无 SSD 优化）
    4. 调度开销更低（EMA vs cross-layer inference）

  组合潜力：
  "InfiniGen's cross-layer prefetcher predicts WHICH blocks to fetch;
   OrchKvCache's tiered engine decides WHERE to place them.
   A natural composition: InfiniGen drives OrchKvCache's three-tier migration."
```

#### 实施计划

```
Week 1:
  - Day 1-2: 部署 InfiniGen，验证能在 A100 上跑
  - Day 3-4: 统一测试脚本，跑端到端对比
  - Day 5: 子组件级对比（预取准确率 + 调度开销）

Week 2:
  - Day 1: 整理数据，制作对比表和图
  - Day 2: 更新论文 §5.9 Comparison with InfiniGen
```

---

## MW3: 公平基线（在 MW1 基础上自然解决）

> 对应审稿意见 W1: "Unfair baseline — toy framework with 7-9x Python overhead"
> 如果 MW1 成功消除 Python 开销，这个问题自然缓解

### 方案：分两步走

#### Step 1：在 MW1 的 C/CUDA 路径上重新测量（MW1 自然产出）

```
MW1 完成后，OrchKvCache 的绝对吞吐应从 87 tok/s 提升到 ~500-600 tok/s。
此时 GPU-Only 仍为 627 tok/s，overhead < 10%。

重新跑全矩阵：
  - 4 模型 x budget {50,100,200} MB x seq {2048,4096} x nreq {1,4,8,16}
  - 新的 OrchKvCache vs FIFO 加速比（两者都用 C/CUDA 路径）
  - 新的 overhead breakdown：
    - C/CUDA 调度 X%
    - EMA 更新 Y%
    - 数据迁移（仅在驱逐/提升时）Z%
  - 预期加速比仍为 1.2-1.7x（核心逻辑不变），但绝对数字有意义了
```

#### Step 2：补充 FlashAttention 基线

```
在 MW1 的 C/CUDA 路径基础上，启用 SDPA/FlashAttention：

  attn_implementation = "sdpa"   # PyTorch 2.5 自动选择 FlashAttention-2

  问题：FlashAttention 不输出 attention weights
  解决：两种 proxy（任选一种）

  Proxy A：QK Norm（推荐，与 Quest ICML'24 方法一致）
    每 N 步：
      q = model.get_current_query()       # [1, n_heads, 1, d]
      for block in kv_blocks:
        k_block = block.key_data          # [n_kv, block_size, d]
        qk_norm = torch.norm(q @ k_block.transpose(-1,-2), dim=-1).mean()
        report_score(block.id, qk_norm)
    复杂度：O(n_blocks * block_size * d)，远小于 full attention O(seq^2 * d)
    每步 ~10-50 μs（在 CUDA 上）

  Proxy B：稀疏采样
    每 N=10 步强制 eager attention（output_attentions=True），其余步 SDPA
    E10 已证明 N=10 时分类准确率 81%，token match 100%

  预期结果：
    - GPU-Only (SDPA): 1658 tok/s
    - OrchKvCache (SDPA + QK proxy): 预计 ~1400-1500 tok/s（overhead ~10-15%）
    - FIFO (SDPA): 预计 ~800-1000 tok/s
    - OrchKvCache vs FIFO: 仍然 1.3-1.7x，但在 FlashAttention 框架下
```

#### 实施计划

```
Week 2-3（与 MW1/MW2 并行）:
  - Day 1-2: 实现 QK Norm proxy
    - 在 FlashAttention 前抽取 Q tensor
    - 计算 Q @ K_block^T 的 norm
    - 输入 EMA 分类流水线
  - Day 3: 集成到 SDPA 路径
  - Day 4-5: 跑 FlashAttention 基线实验
  - Day 6-7: 更新全部实验数据和论文
```

---

## 总时间线与优先级

```
            Week 1          Week 2          Week 3          Week 4
         ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
MW1      │ C/CUDA   │   │ 重跑实验 │   │          │   │          │
(开销)   │ 持久化   │   │ + 优化   │   │          │   │          │
         │ Buffer   │   │          │   │          │   │          │
         └──────────┘   └──────────┘   │          │   │          │
MW2      │ 部署     │   │ 数据     │   │          │   │          │
(InfiniGen)│InfiniGen│   │ 整理     │   │          │   │          │
         │ + 对比   │   │ + 写作   │   │          │   │          │
         └──────────┘   └──────────┘   │          │   │          │
MW3                     │ QK Norm  │   │ FA 基线  │   │          │
(公平基线)              │ proxy    │   │ 实验     │   │          │
                        │ 实现     │   │          │   │          │
                        └──────────┘   └──────────┘   │          │
论文修订                                │ 数字更新 │   │ 全文    │
                                        │ + 图表   │   │ 定稿    │
                                        └──────────┘   └──────────┘

总工作量：约 3-4 周
```

### 优先级排序

| 优先级 | 改进项 | 工作量 | 预期影响 | 依赖关系 |
|:------:|--------|:------:|:-------:|:-------:|
| **P0** | MW1: C/CUDA 持久化 Buffer | 1-2 周 | 消灭 9.2x → <10% overhead | 无 |
| **P1** | MW2: InfiniGen 端到端对比 | 1 周 | 补齐最大竞品对比 | 无 |
| **P2** | MW3: FlashAttention 基线 | 1 周 | 在生产级框架下验证数字 | MW1 |

**MW1 是关键路径**：不解决 9.2x 开销，MW3 的 FlashAttention 基线也没意义
（因为 Python overhead 仍然主导）。MW2 与 MW1 可以并行推进。

### 完成后的论文预期数字

```
修改前（当前 v2）:
  OrchKvCache: 87-183 tok/s
  GPU-Only:    627-1658 tok/s
  Overhead:    7.2-9.2x
  vs FIFO:     1.28-1.77x (弱基线)
  vs InfiniGen: 定性对比

修改后（目标）:
  OrchKvCache (C/CUDA + SDPA): ~1400-1500 tok/s
  GPU-Only (SDPA):             ~1658 tok/s
  Overhead:                     <10%
  vs FIFO (SDPA):              1.3-1.7x (公平基线)
  vs InfiniGen:                端到端吞吐 + 子组件对比 + 定性表
  vLLM block scoring:          1.12x (已有，保持)
```

### 对 SC'26 审稿的回应力度

| 审稿问题 | 修改前 | 修改后 |
|---------|--------|--------|
| "基线不公平" | 7-9x Python 开销，弱基线 | <10% overhead，FlashAttention 基线 |
| "InfiniGen 对比缺失" | 1 个 PPL 点 + 特征表 | 端到端吞吐 + 预取准确率 + 开销对比 |
| "部署不可行" | 183 tok/s (9.2x gap) | ~1400-1500 tok/s (<10% gap) |
| 综合竞争力 | Borderline (6.5/10) | **Solid Accept (7.5-8.0/10)** |

---

## Evaluation 4.5 分详细诊断与修复方案

> 审稿意见原文：「基线不公平、缺竞品对比、overhead 主导数字」
> 这三句话拆开来是三个独立的致命问题，每一个都能让审稿人给 reject。
> 下面逐个解释到底为什么扣分，以及精确的修复路径。

---

### 问题 1: "基线不公平" — 到底哪里不公平？

#### 一句话总结

论文声称 OrchKvCache 比 FIFO 快 1.28-1.77x，但这个数字是在一个**两者都
被 Python 开销拖慢 5 倍的玩具框架里**测的。审稿人看到 OrchKvCache 绝对
吞吐 168 tok/s vs GPU-Only 823 tok/s，第一反应是："你的系统让推理慢了 5 倍，
然后声称比同样慢的 FIFO 快了 1.77 倍——这说明什么？"

#### 不公平的三层含义

**第一层：框架层面的不公平**

```
当前实验框架：
  HuggingFace transformers + eager attention + Python manual decode loop

问题：
  - 没有使用 FlashAttention（SDPA），eager attention 本身就比 SDPA 慢 15-20%
  - 没有使用 continuous batching，而是手动 for 循环逐步 decode
  - 没有使用 CUDA graph，每步都有 Python→CUDA 的 launch overhead

结果：
  GPU-Only (eager): 823 tok/s
  GPU-Only (SDPA):  953 tok/s     ← 真实 serving 系统的起点
  vLLM (SDPA+CB):  ~2000-3000 tok/s ← 生产系统的水平

  论文的 FIFO 基线 (168 tok/s) 比真实 vLLM FIFO (~2000 tok/s) 慢 12 倍！
  在这个基础上声称 1.77x 加速，审稿人自然不信。

对比：如果你在 vLLM 框架内实现同样的 FIFO 基线和 OrchKvCache，
  FIFO 的吞吐大约 2000 tok/s，OrchKvCache 可能 2200 tok/s，
  加速比可能只有 1.1x——但这个 1.1x 远比当前的 1.77x 有说服力。
```

**第二层：Overhead 污染的不公平**

```
当前的 overhead breakdown (Table 8):
  build_past_kv:   48.3 ms (59.5%)
  step_schedule:    9.4 ms (11.6%)
  forward:         19.2 ms (23.6%)
  report_attn:      3.1 ms (3.8%)
  append_token:     1.2 ms (1.4%)

问题：OrchKvCache 和 FIFO 都使用相同的 build_past_kv（48 ms）。
  也就是说，两个系统共享 59.5% 的固定开销。
  在剩下的 40.5% 中，两者的差异才是真正的"调度收益"。

  打个比方：两个人赛跑，都背着 60 公斤的沙袋。
  A 跑了 100 秒，B 跑了 128 秒。
  你说 "A 比 B 快 1.28 倍"——对，但这 1.28 倍有多少是 A 腿快、
  多少是沙袋随机晃动的差异？

  审稿人要的是：把沙袋去掉，看裸奔时 A 和 B 的差距。
```

**第三层：缺乏绝对性能参照的不公平**

```
论文没有给出 "ceiling"（天花板）参照：

  - 没有 "Oracle" 基线：如果用未来信息做驱逐决策（上帝视角），吞吐是多少？
    → 用来量化 EMA 预测 vs 最优策略的 gap
  - 没有 "Zero-overhead" 基线：如果调度逻辑开销为 0（只有数据迁移开销），
    吞吐是多少？
    → 用来分离 "调度收益" 和 "调度开销"
  - 没有 vLLM native 基线：在不修改 vLLM 的情况下，相同模型相同配置的
    原生 vLLM 吞吐是多少？
    → 给读者一个"真实世界"参照点

  没有这些参照，审稿人无法判断 1.28-1.77x 的数字在实际场景中有多大意义。
```

#### 如何使基线"公平"？—— 四层修复方案

**Fix 1（必须）：在 FastKVCacheManager 上跑 OrchKvCache vs Fast-FIFO 完整对比**

```
当前论文只展示了：
  Original FIFO (168 tok/s) vs Original OrchKvCache (168 tok/s) → 1.28-1.77x
  GPU-Only (823) vs Original (168) vs Fast (265) → 只看了绝对吞吐

缺失的关键实验：
  Fast-FIFO vs Fast-OrchKvCache

  如果 Fast-FIFO = 250 tok/s, Fast-OrchKvCache = 350 tok/s → 1.40x ← 有说服力
  如果 Fast-FIFO = 260 tok/s, Fast-OrchKvCache = 265 tok/s → 1.02x ← 说明收益来自 overhead 差异

  这个实验是最重要的：它回答了"去掉 Python 沙袋后，调度策略本身的收益到底有多大"

实施方式：
  1. 用 FastKVCacheManager 的架构实现 Fast-FIFO
     （pre-allocated buffer + 零重建，但驱逐策略用 FIFO 而非 attention-based）
  2. 在相同 4 模型 x 配置矩阵上跑两者对比
  3. 报告 Fast-OrchKvCache / Fast-FIFO 加速比

工作量：1 天（复用 FastKVCacheManager 代码，只改驱逐逻辑）
```

**Fix 2（必须）：补充 GPU-Only (SDPA) 作为天花板参照**

```
在每组实验的第一行加上 GPU-Only (SDPA) 的数字，让读者清楚：
  - OrchKvCache 相对于"不做任何 KV 管理"的开销是多少
  - 这个开销是否在可接受范围内

展示方式（论文中每个吞吐表格都加这一行）：
  | 系统                 | tok/s | vs GPU-Only (SDPA) |
  |---------------------|-------|-------------------|
  | GPU-Only (SDPA)     | 953   | 1.00x (ceiling)   |
  | GPU-Only (eager)    | 823   | 0.86x             |
  | Fast FIFO           | XXX   | 0.XXx             |
  | Fast OrchKvCache    | XXX   | 0.XXx             |

  关键指标：OrchKvCache overhead = 1 - (OrchKvCache / GPU-Only SDPA)
  目标：< 20%（即 OrchKvCache ≥ 760 tok/s on Qwen）
```

**Fix 3（推荐）：补充 vLLM native baseline 作为"真实世界"参照**

```
即使不在 vLLM 内部实现 OrchKvCache，也应该报告 vLLM 原生性能：

  | 系统                        | tok/s | 说明 |
  |----------------------------|-------|------|
  | vLLM native (SDPA+CB)      | ~2500 | 参照点 |
  | GPU-Only (SDPA, 本文框架)   | 953   | 本文 ceiling |
  | Fast OrchKvCache (SDPA)     | ~760  | 本文系统 |
  | vLLM + block scoring (§5.8) | 243   | 已有 vLLM 集成 |

  这让审稿人理解：
  - 本文框架 vs vLLM 的 gap 来自 continuous batching（而非 OrchKvCache 的问题）
  - OrchKvCache 的 调度开销 < 20% 是合理的
  - vLLM 集成的 1.12x 是在真实引擎中的数字

工作量：半天（跑几组 vLLM benchmark 即可）
```

**Fix 4（锦上添花）：Oracle 上界实验**

```
用离线 attention trace 做 oracle 调度：
  - 收集 100 步的完整 attention weights
  - 回放时用 "未来信息" 做驱逐决策（驱逐接下来 N 步都不会被用到的 block）
  - 对比 Oracle vs EMA vs FIFO 的迁移次数和分类准确率

作用：量化 OrchKvCache 的 EMA 策略离理论最优有多远
  如果 Oracle 减少 500x 迁移，EMA 减少 300x → EMA 达到 Oracle 的 60%
  这为 EMA 策略的合理性提供理论支撑

工作量：1 天（纯 trace 仿真，不需要 GPU）
```

---

### 问题 2: "缺竞品对比" — 为什么 InfiniGen 必须比？

```
InfiniGen (OSDI'24) 的地位：
  - KV cache offloading 方向的 SOTA（被引 100+）
  - 也是 GPU+DRAM 两层管理 + 预测驱动预取
  - 已开源
  - 审稿人如果做 LLM serving 方向，大概率读过这篇

当前论文的对比：
  - 1 个 perplexity 数据点 (5.82 vs 5.69)
  - 1 张 6 行特征表 (Table 7)
  - 一段话 "direct comparison is not possible"

审稿人的反应："InfiniGen 开源了，你说无法对比？那你是不愿意跑还是跑不赢？"

修复方案：见上方 MW2 部分。核心是端到端吞吐 + 子组件级对比两条线都要有。
如果确实无法端到端对比（框架差异），至少要做子组件级（预取准确率 + 调度开销）
的 apple-to-apple 对比。
```

---

### 问题 3: "overhead 主导数字" — 具体什么意思？

```
这个问题的本质：论文报告的加速比（1.28-1.77x）有多少是"真实调度收益"，
有多少是"Python overhead 的随机放大效应"？

具体数学分析：

  设每步 decode 耗时 = T_forward + T_overhead + T_scheduling
  
  对 GPU-Only:       T = T_forward = 19 ms
  对 FIFO:           T = T_forward + T_overhead_fifo + T_schedule_fifo
  对 OrchKvCache:    T = T_forward + T_overhead_orchkv + T_schedule_orchkv

  当前实测 (Qwen2.5-7B, N=10):
    FIFO:       ~100 ms/step → throughput ~168 tok/s
    OrchKvCache: ~81 ms/step → throughput ~265 tok/s (Fast 版)
    GPU-Only:    ~19 ms/step → throughput ~823 tok/s

  分解：
    FIFO overhead:       100 - 19 = 81 ms (Python build_past_kv + FIFO schedule)
    OrchKv overhead:     81 - 19  = 62 ms (Python build_past_kv + 调度 + 采样)
    
    差值：81 - 62 = 19 ms ← 这才是 OrchKvCache 调度策略的真实收益
    
    OrchKvCache 的"真实调度收益" = 19 ms / 81 ms ≈ 23%
    但论文报告的加速比 = 265/168 = 1.58x → 58% improvement

    为什么 23% 的真实收益变成了 58% 的报告数字？
    因为 overhead 放大了差异：
      FIFO: 19 ms forward + 81 ms overhead = 100 ms
      OrchKv: 19 ms forward + 62 ms overhead = 81 ms
      100/81 = 1.23x ← 这才是扣除共享 overhead 后的真实比值

  结论：当前的 1.58x (Fast) 或 1.28-1.77x (Original) 被 overhead 放大了。
  审稿人如果自己做上述分析，会认为数字不可信。

修复方法：
  1. Fix 1（Fast-FIFO vs Fast-OrchKvCache）直接给出消除 overhead 后的数字
  2. 在论文中主动做上述分解分析（显示诚实和严谨）
  3. 用两种方式报告加速比：
     - "Amortized speedup" = 在相同框架中的相对加速（当前数字）
     - "Policy-only speedup" = 排除共享 overhead 后的纯策略收益
     - "Absolute overhead" = OrchKvCache 相对 GPU-Only 的开销
```

---

### 修复优先级排序

| 优先级 | 修复项 | 工作量 | 影响 | 说明 |
|:------:|--------|:------:|:----:|------|
| **P0** | Fix 1: Fast-FIFO vs Fast-OrchKvCache | 1 天 | 致命 | 回答"纯调度收益是多少" |
| **P0** | Fix 2: GPU-Only (SDPA) 天花板参照 | 已有数据 | 致命 | 展示绝对 overhead |
| **P1** | Fix 3: vLLM native baseline | 0.5 天 | 重要 | 给"真实世界"参照 |
| **P1** | InfiniGen 端到端对比 (MW2) | 1 周 | 重要 | 补齐最大竞品 |
| **P2** | Fix 4: Oracle 上界实验 | 1 天 | 锦上添花 | 量化 EMA vs 最优 gap |
| **P2** | 论文中主动做 overhead 分解分析 | 写作 | 加分 | 展示严谨性 |

**最低要求（不做会被 reject）：Fix 1 + Fix 2 + InfiniGen 至少子组件级对比**
**推荐做到（冲 accept）：全部 Fix + InfiniGen 端到端 + 论文中的 overhead 分解叙事**

---

## Eval-Fix 5: InfiniGen 端到端对比

> 状态：InfiniGen 已 clone 在 /home/lzq/codes/InfiniGen，conda env `infinigen` 已存在
> InfiniGen 基于 FlexGen (ICML'23)，使用 OPT 系列模型 (6.7B/13B/30B) + LLaMA-2 (7B/13B)

### 问题本质

InfiniGen (OSDI'24) 是最直接的竞品。当前论文只有 1 个 PPL 数据点 + 1 张特征表。
审稿人会认为："InfiniGen 开源了你都不跑，说明你心虚。"

### 可行的对比路径

InfiniGen 的 speedup 评估用 FlexGen 框架 + OPT-13B 模型，与 OrchKvCache 的
HF transformers + LLaMA/Qwen 框架不同。完全 apple-to-apple 端到端对比不可行。
但有三条可行路径：

**路径 A：Perplexity 对比（最可行，扩展当前 1 个点到完整矩阵）**

```
InfiniGen accuracy/perplexity/table2.sh 已提供完整脚本：
  - 模型：OPT-{6.7B,13B,30B} + LLaMA-2-{7B,13B}
  - 数据集：WikiText-2 + PTB
  - 容量：100% (full cache) + 80% (with eviction)
  - 驱逐策略：FIFO / LRU / Counter (InfiniGen)

OrchKvCache 侧需要：
  1. 跑相同模型+数据集+容量的 PPL
  2. 已有 LLaMA-2-7B 的 WikiText-2 结果 (PPL=5.82)
  3. 需补充：LLaMA-2-13B + OPT-6.7B/13B 的 PPL

对比表扩展为：
  | Model | Dataset | Full | 80% FIFO | 80% InfiniGen | 80% OrchKv |
  |-------|---------|------|----------|---------------|------------|
  | LLaMA-2-7B  | WikiText-2 | 5.82 | 22.26† | 5.69† | 5.82 |
  | LLaMA-2-7B  | PTB        | ?    | ?†     | ?†    | ?    |
  | LLaMA-2-13B | WikiText-2 | ?    | 21.41† | 5.25† | ?    |
  | OPT-6.7B    | WikiText-2 | ?    | ?†     | ?†    | ?    |
  † = InfiniGen 论文报告值

关键叙事：
  OrchKvCache 在所有配置下 PPL = Full Cache（因为无损迁移，不丢弃任何 token）
  InfiniGen 在 100% 容量下也无损，但在 80% 容量下使用 counter-based eviction（有损）
  差异化：OrchKvCache 是 ALWAYS lossless，InfiniGen 只在满容量时 lossless
```

**路径 B：调度开销微基准对比（可行，不依赖框架一致性）**

```
两个系统的调度开销可以独立测量：

  InfiniGen 调度开销：
    - 跨层预测：上一层 attention output → 预测下一层需要的 KV
    - 涉及 matrix multiply (Q_partial @ K_partial^T)
    - 论文报告 prefetch accuracy 95%+，但没报告 per-step overhead

  OrchKvCache 调度开销：
    - C/CUDA 核心分类器：<40 μs（已有数据）
    - Python 调度循环：23-49 ms（已诊断的瓶颈）

  对比方式：
    - 在相同硬件 (A100) 上
    - 模拟相同 block 数 (e.g., 64/256/1024)
    - 测量 per-step 调度延迟
    - 对比 prefetch accuracy（precision@K, recall@K）

  意义：说明 OrchKvCache 在调度 overhead 上有数量级优势（40μs vs ms级）
  即使 InfiniGen 预测更准（95% vs 80-88%），OrchKvCache 的低开销在
  高频 decode 场景下可能更有优势（latency-accuracy tradeoff）
```

**路径 C：引用 InfiniGen 论文数字做间接对比（最低要求）**

```
如果时间不够跑实验，至少在论文中：
  1. 引用 InfiniGen 论文的 Figure 14/15 数字
     - OPT-13B: InfiniGen 3.00x speedup over FlexGen baseline
     - LLaMA-2-7B: InfiniGen 2.5x speedup (estimated from their figure)
  2. 分析差异来源：
     - InfiniGen 优化的是 prefetch volume（减少从 DRAM 取的数据量）
     - OrchKvCache 优化的是 migration frequency（减少 GPU-DRAM 迁移次数）
     - 两者是正交的
  3. 给出组合预测：
     "InfiniGen 的跨层预取 + OrchKvCache 的三层分级
      可以组合：InfiniGen 预测 WHICH blocks to fetch，
      OrchKvCache 决定 WHERE to place them。"
```

### 实施计划

```
Day 1: 路径 A — 扩展 PPL 对比
  - 在 orchkv conda env 中跑 LLaMA-2-7B PTB + LLaMA-2-13B WikiText-2
  - 整理成完整的 PPL 对比表
  - 从 InfiniGen 论文 Table 2 提取对应数字

Day 2: 路径 B — 调度开销微基准（如果时间允许）
  - 在 infinigen conda env 中跑 InfiniGen 的 prefetch 延迟
  - 在 orchkv env 中用 orchkv_core 跑相同 block 数的调度延迟
  - 制作对比表

Day 2 alt: 至少完成路径 C — 引用数字做间接对比
  - 从 InfiniGen 论文提取 speedup 和 accuracy 数字
  - 更新论文中的对比叙事
```

---

## Eval-Fix 6: vLLM 实验统计显著性

> 当前问题：1.12x 基于 gpu_util=0.25、16 请求、恰好 1 次 preemption
> 审稿人质疑：1 次 preemption 的差异可能是噪声

### 解决方案：多压力级别 x 多重复

```
扩展 vLLM 实验矩阵：

  固定参数：
    - 模型：LLaMA-2-7B (MHA-32, 512 KB/token)
    - swap_space=32 GB
    - max_tokens=64

  变化参数：
    - gpu_memory_utilization: {0.15, 0.20, 0.25, 0.30}
      → 越低 = KV budget 越小 = 更多 preemption
    - num_prompts: {8, 16, 32, 64}
      → 越多 = 更大并发 = 更多内存竞争
    - 每个 (gpu_util, num_prompts) 跑 3 次取平均

  三个策略：
    1. FIFO (vLLM default)
    2. Progress-aware
    3. Block-level scoring (OrchKvCache)

  预期结果：
    - 低 gpu_util (0.15-0.20) 下 preemption 频率更高（可能 5-10 次）
    - Block-level scoring 的优势在高 preemption 频率下应更明显
    - 可以画 "preemption frequency vs speedup" 曲线

  论文呈现：
    1. 主表：(gpu_util x num_prompts) → throughput for 3 strategies
    2. 图：preemption count vs OrchKv/FIFO speedup（证明收益随 preemption 增加）
    3. 置信区间：3 次重复的 std bar

代码：复用现有 exp_vllm_partial_swap.py，扩展参数扫描
```

### 实施计划

```
Day 1:
  - 修改 exp_vllm_partial_swap.py 支持批量参数扫描
  - gpu_util={0.15, 0.20, 0.25, 0.30} x num_prompts={16, 32}
  - 3 策略 x 3 重复 = 72 个数据点
  - 估计运行时间：每个点 ~30s，总计 ~40 分钟

Day 2:
  - 整理数据，制作对比表和图
  - 更新论文 §5.8 vLLM Integration Analysis
```

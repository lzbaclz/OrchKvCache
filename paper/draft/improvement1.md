# Improvement Plan 1: 审稿人意见修改清单

> 基于模拟审稿的 6 个不足（W1-W6）和 3 个提问，按优先级排列

---

## 必须做的实验（Must Fix）

### 1. 增加 vLLM 原生 swap 作为基线 [W1]

**审稿人的问题**：你只和自己写的 FIFO 对比，为什么不直接和 vLLM 对比？

**需要做的实验**：
- 用 vLLM（`swap_space=32GB`）跑相同的 4 个模型 × 相同的 seq_len/nreq 矩阵
- 和 OrchKvCache 的结果放在同一张图里对比
- 这样审稿人就能看到"OrchKvCache vs 真正的 vLLM"而不是"vs 自己写的 FIFO"

**预计工作量**：1-2 天
- vLLM 已经装好（0.7.3），直接调用 `LLM()` API 就行
- 不需要改 OrchKvCache 代码，只需要写一个新的 benchmark 脚本
- 参数矩阵和现有实验完全一致，方便直接对比

**脚本**：新建 `benchmarks/exp_vllm_baseline.py`

---

### 2. 增加 ShareGPT 真实工作负载实验 [W3]

**审稿人的问题**：所有实验都是合成 prompt，真实请求长度分布不均匀时表现如何？

**需要做的实验**：
- 下载 ShareGPT 数据集（~50MB）
- 从中抽取真实的用户 prompt（长度自然分布：几十到几千 token 不等）
- 在至少 2 个模型上（Qwen + LLaMA-7B）跑 OrchKvCache vs FIFO
- 不需要跑完整矩阵，选一个 budget 配置跑 50-100 个请求即可

**预计工作量**：1 天
- 数据集下载一行命令
- 修改现有脚本的 prompt 来源即可

**脚本**：新建 `benchmarks/exp_sharegpt.py`

**下载命令**：
```bash
wget -O /raid/models/ShareGPT_V3.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

---

## 应该做的实验（Should Fix）

### 3. "baseline OOM, OrchKvCache 能跑"的容量演示 [W6]

**审稿人的问题**：你声称能扩展 KV-Cache 容量，但没有一个实验展示 baseline OOM 的场景。

**需要做的实验**：
- 用 `torch.cuda.set_per_process_memory_fraction()` 限制 GPU 可用显存到 30-40GB
- 在 LLaMA-2-7B（512KB/token）上跑 seq=4096, batch=16
- 预期结果：baseline OOM，orchkv 正常完成

**预计工作量**：半天
- 改一下现有脚本加上显存限制即可

**脚本**：新建 `benchmarks/exp_capacity_demo.py`

---

### 4. 开销来源分解 [W2 + 审稿人提问 3]

**审稿人的问题**：87 tok/s vs 627 tok/s 的差距到底来自哪里？

**需要做的实验**：
- 分别测量每步的时间分解：
  - (a) GPU forward pass 本身
  - (b) eager attention 的额外开销（vs FlashAttention）
  - (c) block 管理（ingest/append/build_past_kv）
  - (d) orchkv_core 调度（tm_schedule_once）
  - (e) 实际迁移（GPU↔DRAM copy）
- 用 `torch.cuda.Event` 精确计时每个阶段

**预计工作量**：1 天
- 在现有的 `run_decode` 函数里插入计时点
- 不需要新模型或新数据

**脚本**：新建 `benchmarks/exp_overhead_breakdown.py`

**预期结论**：
```
eager attention 开销:  ~70% ← 这是最大的来源，切换到 FlashAttention 可消除
block 重建开销:        ~20% ← 每步重建 past_key_values 的 tensor 拼接
调度开销:              ~5%  ← 已经证明只有 38us
迁移开销:              ~5%  ← GPU↔DRAM copy
```

如果能证明 70% 开销来自 eager attention（而不是 OrchKvCache 本身），审稿人就能接受——因为 eager attention 是为了采集注意力分数的临时方案，生产环境用 FlashAttention + 采样就能大幅降低。

---

## 锦上添花（Nice to Have）

### 5. SSD 层端到端验证 [W4]

**审稿人的问题**：SSD 层在设计中描述了，但端到端实验中没触发。

**需要做的实验**：
- 把 GPU budget 设得极小（10-20MB），同时把 DRAM budget 也限制住
- 迫使数据不得不写入 SSD
- 验证从 SSD 读回后输出仍然正确

**预计工作量**：1 天
- 需要修改 KVCacheManager 支持 DRAM budget 限制和 SSD 写入
- 目前 SSD 写入路径在 C 代码里有（orchfs_tier），但 Python 层只做了 GPU↔DRAM

**是否必须**：不是必须。论文可以在 Discussion 中说明"当前评估聚焦 GPU↔DRAM 迁移，SSD 层验证留作 future work"。但如果能做，会更完整。

### 6. NVM 硬件验证 [W5]

**当前情况**：没有 NVM 硬件，无法做

**论文中的处理**：已经在 Discussion 中诚实说明了，并指出 CXL memory 是未来替代方案。审稿人可以接受这个 limitation。

**不需要做新实验。**

---

## 不需要做实验的修改（纯文字改动）

### 7. Discussion 中补充开销分析 [W2]

如果做了实验 4（开销分解），在 Discussion 中加一段：

> "The throughput gap between OrchKvCache and GPU-Only (Table X) is dominated by eager attention overhead (Y%), which is required for collecting attention weights in our current prototype. In production, FlashAttention does not output attention weights; attention sampling every N=50 steps would reduce this overhead to <Z%. The scheduling and migration overhead of OrchKvCache itself accounts for only W% of total per-step time."

### 8. Limitations 部分更新

把现有的三点 limitation 扩充，加上审稿人关心的点。

---

## 总结：工作量估算

| 优先级 | 任务 | 需要新实验？ | 预计时间 |
|:---:|------|:---:|:---:|
| **必须** | vLLM 原生基线对比 | 是 | 1-2 天 |
| **必须** | ShareGPT 真实负载 | 是 | 1 天 |
| **应该** | OOM 容量演示 | 是 | 半天 |
| **应该** | 开销来源分解 | 是 | 1 天 |
| 锦上添花 | SSD 层端到端 | 是 | 1 天 |
| 锦上添花 | NVM 验证 | 不可能 | — |
| 不需要 | 文字润色 | 否 | 半天 |
| **总计** | | | **4-5.5 天** |

**建议执行顺序**：
1. 先做实验 4（开销分解）—— 半天就出结果，可以立刻更新 Discussion
2. 再做实验 1（vLLM 基线）—— 最重要，直接回应 W1
3. 然后做实验 2（ShareGPT）—— 回应 W3
4. 最后做实验 3（OOM 演示）—— 补充核心叙事
5. 如果还有时间做实验 5（SSD 层）

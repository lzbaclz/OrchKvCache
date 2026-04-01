# Improvement 2: 将 OrchKvCache 调度逻辑集成到 vLLM

> 目标: 实现 vLLM(FIFO) vs vLLM(OrchKvCache) 的公平对比
> 预计工作量: 2-3 天
> 风险: 中等（修改 vLLM 内部代码，但改动量小）

---

## vLLM 0.7.3 Swap 机制总结

```
当 GPU 放不下新请求的 KV block 时：
  scheduler._schedule_running()
    → running_queue.pop()     ← 选队列尾部的请求（FCFS）
    → _preempt(victim)
    → _swap_out(victim)       ← 把整个请求的所有 KV block 搬到 CPU
    → block_manager.swap_out(seq_group)
    → CacheEngine.swap_out()  ← 实际 GPU→CPU memcpy
```

**关键发现**：swap 是请求级的，不是 block 级的。选择哪个请求被 swap 出去的逻辑只有一行代码：
```python
victim_seq_group = running_queue.pop()  # scheduler.py ~line 730
```

## 修改方案：请求级注意力感知 Swap

### 原理

不改 block 级别的任何代码，只修改"选哪个请求被 swap 出去"：

```
原来（FIFO）：选最后进来的请求 → 可能选到一个正在被关注的请求
修改后：     选所有 running 请求中"平均注意力分数最低"的 → 选到真正没人关注的请求
```

### 需要修改的文件

```
文件 1: vllm/core/scheduler.py
  位置: _schedule_running() 函数
  改动: 把 running_queue.pop() 改成按注意力分数选 victim

文件 2: vllm/core/scheduler.py
  位置: Scheduler 类
  改动: 加一个 attention_scores 字典，存每个 seq_group 的平均分数

文件 3: vllm/worker/model_runner.py（或 attention hook）
  位置: forward pass 之后
  改动: 采集注意力分数回传给 scheduler
```

### 具体改动

#### 改动 1: scheduler.py — 加注意力分数存储

```python
class Scheduler:
    def __init__(self, ...):
        ...
        # OrchKvCache: per-request attention scores
        self.orchkv_scores: Dict[str, float] = {}  # req_id → avg_attn_score
```

#### 改动 2: scheduler.py — 修改 victim 选择逻辑

```python
# 原来:
victim_seq_group = running_queue.pop()

# 改为:
if self.orchkv_enabled:
    # 选注意力分数最低的请求
    min_score = float('inf')
    min_idx = len(running_queue) - 1
    for i, sg in enumerate(running_queue):
        score = self.orchkv_scores.get(sg.request_id, 0.0)
        if score < min_score:
            min_score = score
            min_idx = i
    victim_seq_group = running_queue[min_idx]
    del running_queue[min_idx]
else:
    victim_seq_group = running_queue.pop()
```

#### 改动 3: 注意力分数回传

在 model forward 之后，把 attention 分数汇总到 scheduler：

```python
# 在 LLMEngine.step() 或 model_runner 中:
if orchkv_enabled and output.attentions:
    for seq_group in scheduler_output.scheduled_seq_groups:
        avg_score = compute_avg_attention(output.attentions, seq_group)
        scheduler.orchkv_scores[seq_group.request_id] = avg_score
```

### 实验设计

```
两种 vLLM 配置：
  A. vLLM-FIFO:    原版 vLLM，swap 按 FCFS 选 victim
  B. vLLM-OrchKv:  修改版 vLLM，swap 按注意力分数选 victim

其他完全相同：
  - 同一个 vLLM 代码库
  - 同样的 continuous batching
  - 同样的 FlashAttention / PagedAttention
  - 同样的 CUDA graph
  - 同样的模型、prompt、硬件

唯一区别：选谁被 swap 出去

这是无可挑剔的对比。
```

### 执行步骤

1. 备份 vLLM 原始文件
2. 修改 scheduler.py（加分数存储 + 改 victim 选择）
3. 修改 model_runner.py 或 engine（加分数回传）
4. 加一个配置开关 `--orchkv-swap` 控制是否启用
5. 跑相同的实验矩阵（4 模型 × 2 配置）
6. 对比 vLLM-FIFO vs vLLM-OrchKv

### 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| vLLM 启动失败 | 低 | 高 | 先备份，改动最小化 |
| 注意力分数采集影响性能 | 中 | 中 | 只每 N 步采集一次 |
| eager attention 在 vLLM 中不可用 | 中 | 高 | 用 proxy 信号替代（QK norm） |
| deque 中间删除效率低 | 低 | 低 | running_queue 通常很短 |

# Improvement 4: 统一实验体系 + 解决 vLLM 无差异问题

## 问题分析

现在有两套实验体系，结论矛盾：
```
体系 1 (HF transformers): orchkv 比 naive 快 1.28-1.77x → 效果显著
体系 2 (vLLM A/B):        orchkv 和 fifo 持平 0.97-1.02x → 效果为零
```

原因：vLLM 在 gpu_util=0.5/0.7 + A100-80GB 下显存依然充裕，swap 几乎不触发。
策略再好，不触发就没有效果。

## 解决方案

不做两套体系，只做一套：**全部在 vLLM 里跑，但把 gpu_memory_utilization 压得更低，迫使 swap 频繁触发。**

```
关键参数：gpu_memory_utilization = 0.25~0.35
  → A100-80GB × 0.3 = 24GB 可用
  → 模型 14GB → 只剩 10GB 给 KV-Cache
  → Qwen-7B seq=2048 batch=16: KV 需要 16 × 2048 × 56KB = 1.75GB → 够用
  → LLaMA-7B seq=2048 batch=16: KV 需要 16 × 2048 × 512KB = 16GB → 超了！
  → vLLM 被迫频繁 swap → FIFO vs OrchKv 差异就出来了
```

这样论文里只需要一套实验：
```
vLLM (FIFO swap, gpu_util=0.3) vs vLLM (OrchKv swap, gpu_util=0.3)
→ 同一个 vLLM，同一切配置，唯一区别是 swap 策略
→ 审稿人无话可说
```

## 执行计划

1. 在 exp_vllm_ab.py 中加入 gpu_util=0.25 和 0.30 的配置
2. 跑 LLaMA-2-7B（最容易触发 swap）
3. 对比 FIFO vs OrchKv

## 预计工作量：2-3 小时

# Improvement 8: Scale + Kernel-Level Selective Restore

## 目标
将 SC 中稿率从 ~25% 提升到 50%+。解决两个核心弱点：
1. **Scale**: 最大只测了 13B / 4K context → 需要 8K-16K context
2. **Kernel-level**: 没有证据证明 selective block placement 真的有效 → 需要原型验证

## 改进 1: 大规模 FlashAttention 实验

### 方案
- 模型: LLaMA-2-13B (已缓存, 40 layers, 40 KV heads, MHA)
- Context: 8192 tokens (之前 OOM，现在有 FlashAttention 不需要 output_attentions)
- 使用 QK-norm proxy 替代 output_attentions 做 hotness scoring
- 对比: GPU-Only (FlashAttn) vs FIFO Offload vs OrchKvCache

### 预期结果
- 证明 OrchKvCache 在 8K context 下仍然有效
- 证明 FlashAttention 集成解决了 OOM 限制
- 在更大 KV cache 压力下，migration reduction 的优势应该更明显

## 改进 2: Selective Restore 原型

### 核心思路
不修改 attention kernel，而是在 build_past_kv 阶段做 **选择性恢复**：

1. 查询 C 端 `tm_get_block_score()` 获取每个 evicted block 的 hotness score
2. 按 score 降序恢复（hot first, cold last）
3. 测量：如果只恢复 top-K% 的 block，能捕获多少 attention weight？
4. 对比：full-restore vs selective-restore 的恢复量和吞吐

### 实验设计
**实验 A: Attention Weight Coverage**
- 在 Qwen2.5-7B / LLaMA-2-7B 上，收集每步的 per-block attention weight
- 按 OrchKvCache EMA score 排序 block
- 计算 top-50%, top-30%, top-10% blocks 覆盖的 attention weight 比例
- 预期: top-50% blocks 覆盖 >95% attention weight

**实验 B: Selective Restore Throughput**
- 修改 FastKVCacheManager：
  - 只恢复 high-score blocks (hot)
  - Cold blocks 填零（用于 attention 计算时贡献微小）
  - 测量吞吐提升和 token match degradation
- 展示 OrchKvCache 调度器能准确预测哪些 block 是 attention-critical

## 需要修改/创建的文件
- benchmarks/exp_scale_flashattn.py — 大规模实验
- benchmarks/exp_selective_restore.py — 选择性恢复实验
- paper/orchkvcache5.tex — 更新论文

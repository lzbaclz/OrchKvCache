# Improvement 3: 修复 W3 (真实工作负载) + W4 (SSD 层端到端)

## W3: ShareGPT 真实工作负载实验

1. 下载 ShareGPT 数据集
2. 从中抽取真实用户 prompt（长度自然分布）
3. 在 Qwen2.5-7B + LLaMA-2-7B 上跑 OrchKvCache vs Naive
4. 用 budget=50MB 制造压力

## W4: SSD 层端到端验证

1. 在 KVCacheManager 中设置极小的 DRAM budget
2. 迫使冷 block 从 DRAM 进一步写入 SSD 文件
3. 验证从 SSD 读回后输出仍然 100% 一致
4. 在 Qwen2.5-7B 上跑质量验证

## 预计工作量：1.5 天

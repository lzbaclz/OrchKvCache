# Improvement 9: LM-Eval Benchmark Validation

## 目标
用标准 NLP benchmark 验证 OrchKvCache 的 lossless 保证在真实任务上成立。
堵住审稿人 "你只测了合成文本" 的质疑。

## 实验设计
- **Benchmark**: LM-Evaluation-Harness
- **Tasks**: PIQA (常识推理), RTE (逻辑推理), COPA (因果推理), OpenBookQA (科学问答)
- **Models**: Qwen2.5-7B, LLaMA-2-7B
- **Systems**: GPU-Only (baseline), OrchKvCache (tiered management)
- **Metric**: Accuracy on each task
- **Expected**: 两者准确率完全一致 (lossless by construction)

## 数据点
4 tasks × 2 models × 2 systems = 16 data points

## 预期结果
GPU-Only 和 OrchKvCache 在所有 task 上准确率完全一致,
验证 "100% token match → downstream task accuracy preserved"

## 文件输出
- benchmarks/exp_lm_eval.py — 实验脚本
- benchmarks/results/exp_lm_eval.json — 结果数据
- paper/plot_figures_code_data/exp_lm_eval.json — 副本
- paper/plot_figures_code_data/plot_exp_lm_eval.py — 画图代码
- 论文 Section 5.3 (Quality) 新增 Table + 一段话

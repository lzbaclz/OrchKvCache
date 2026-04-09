# OrchKvCache 论文完成计划

> 基于 Abstract.md 中识别的差距和 paper_outline.md 的论文结构，制定完成这篇 CCF-A 系统论文的详细计划。
>
> 总时间估算: **8 周**（假设全职投入）

---

## 阶段总览

| 阶段 | 时间 | 核心目标 | 交付物 |
|------|------|---------|--------|
| **P0: 系统打通** | Week 1-2 | 将已有组件拼装成完整数据通路 | 可运行的端到端系统 |
| **P1: vLLM 集成** | Week 3-4 | 真正将 OrchKvCache 嵌入 vLLM 推理路径 | 集成后的 vLLM + OrchKvCache |
| **P2: 实验设计与执行** | Week 4-6 | 在压力场景下收集全部实验数据 | 所有图表数据 |
| **P3: 论文撰写** | Week 5-8 | 完成论文初稿 | 12 页完整论文 |
| **P4: 打磨与投稿** | Week 8 | 内部审阅、修改、投稿 | 最终版本 |

---

## P0: 系统打通 (Week 1-2)

**目标**: 让 orchkv_api + tiered_manager + prefetch + pipeline + io_worker 形成完整数据通路。

### Week 1: 核心路径打通

- [ ] **T0.1** 在 `orchkv_api.cu` 中集成 `tiered_manager`
  - 在 `orchkv_init()` 中创建并初始化 `tiered_manager_t`
  - 在 `orchkv_step_done()` 中触发 `tm_step_done()`，驱动自动调度循环
  - 在 `orchkv_report_attention()` 中调用 `tm_report_attention()`
  - 使 `tm_step_done` 的 demote 决策自动调用 `orchkv_evict_to_dram()` / `orchkv_evict_to_storage()`
  - 使 `tm_step_done` 的 promote 决策自动调用 `orchkv_promote_to_gpu()`

- [ ] **T0.2** 补全 prefetch 执行路径
  - 在 `tiered_manager.c` 的 `do_prefetch()` 中，将 `prefetch_dispatch` 的输出连接到 `mig_execute_one()`
  - 确保预取操作真正执行 promote（从 DRAM/Storage → GPU）
  - 添加预取完成的回调/状态更新

- [ ] **T0.3** 接入 pipeline 组件
  - 在调度主循环中插入 `pipeline_step_begin()` / `pipeline_compute_done()` / `pipeline_transfer_done()`
  - 收集计算-传输-IO 三阶段的时间统计

- [ ] **T0.4** 将 `orchkv_api.cu` 的存储 IO 改为异步
  - `orchkv_evict_to_storage()` 改用 `io_worker_submit()` 提交异步任务
  - `orchkv_promote_from_storage()` 同理
  - 添加同步等待点（确保 promote 完成后再返回 GPU 地址）

### Week 2: 端到端验证

- [ ] **T0.5** 编写端到端集成测试
  - 模拟完整推理流程：prefill → 多步 decode → 触发 eviction → 触发 promote
  - 验证：evict 到 SSD 的 block 被 promote 回 GPU 后数据一致
  - 验证：pipeline 统计数据正确记录
  - 验证：预取操作确实在 GPU 计算之前完成

- [ ] **T0.6** 压力测试
  - 配置极小的 GPU 和 DRAM 池（迫使大量 eviction/promote 发生）
  - 跑多请求并发场景，检测死锁/数据损坏
  - 测量端到端延迟和正确性

- [ ] **T0.7** 修复 bug，确保通过所有现有测试 + 新增测试

---

## P1: vLLM 集成 (Week 3-4)

**目标**: 让 OrchKvCache 真正嵌入 vLLM 推理路径，替代 vLLM 原生的 swap 机制。

### Week 3: Connector 适配

- [ ] **T1.1** 适配 vLLM API 版本
  - 选项 A: 将 connector.py 改为适配 vLLM 0.7.3 的 `KVConnectorBase`
  - 选项 B: 升级到 vLLM ≥ 0.8 以使用 V1 API
  - 评估两个选项的工作量后选择

- [ ] **T1.2** 实现 Connector Worker 的 C API 调用
  - `save_kv_layer()`: 调用 `orchkv_evict_to_dram()` 而非简单 `copy_()`
  - `load_kv_layer()`: 调用 `orchkv_promote_to_gpu()`
  - 集成注意力分数采集（`orchkv_report_attention()`）

- [ ] **T1.3** 实现 Connector Scheduler 的状态管理
  - `update_state_after_alloc()`: 注册新的 KV block 到 tiered_manager
  - `update_state_after_free()`: 从 tiered_manager 中注销
  - 显存 watermark 监控 → 触发调度

- [ ] **T1.4** 注意力钩子修复
  - 修复 `attention_hook.py` 中 `_get_all_modules()` 返回空列表的问题
  - 对 vLLM 的 FlashAttention 模块注册 forward hook
  - 采样式注意力采集（每 K 步采集一次，降低开销）

### Week 4: 集成测试与调试

- [ ] **T1.5** 端到端集成测试
  - 用 Qwen2.5-7B 跑真实推理，OrchKvCache 全路径激活
  - 验证 greedy decoding 输出与 baseline 一致
  - 收集并分析运行日志（eviction/promote 次数、延迟分布）

- [ ] **T1.6** 性能 profiling
  - 找到并消除性能瓶颈
  - 确保调度开销 < 1% decode 延迟

- [ ] **T1.7** 修复所有集成 bug

---

## P2: 实验设计与执行 (Week 4-6)

**目标**: 在有说服力的场景下，收集所有论文需要的实验数据。

### Week 4-5: 实验环境与压力场景设计

- [ ] **T2.1** 设计压力场景（关键！）
  - **方案 A**: 限制 `gpu_memory_utilization` 到 0.3-0.5（模拟小 GPU）
  - **方案 B**: 使用更大模型（13B/30B）或更长上下文（32K-128K）
  - **方案 C**: 同时运行多个请求，使总 KV-Cache 超过 GPU 显存
  - 目标：创造 "vLLM OOM 但 OrchKvCache 能跑" 的场景

- [ ] **T2.2** 准备强基线
  - **vLLM-swap**: vLLM 原生 GPU↔CPU swap（swap_space=32GB）
  - **vLLM-offload**: vLLM 的 `cpu_offload_gb` 参数
  - **FlexGen**: 按其论文配置复现（或使用开源实现）
  - **InfiniGen**: 如有开源实现则复现，否则按论文数据引用

- [ ] **T2.3** 准备评估 workload
  - ShareGPT 真实对话 trace
  - 合成长序列：8K / 16K / 32K / 64K / 128K tokens
  - 不同 batch size: 1 / 4 / 16 / 64

### Week 5-6: 实验执行

- [ ] **T2.4** E1 端到端吞吐实验（重新跑，使用真 OrchKvCache 路径）
  - OrchKvCache vs vLLM-swap vs vLLM-offload
  - 多序列长度 × 多 batch size
  - 生成 Fig.7

- [ ] **T2.5** E2 最大 Batch Size 扩展实验
  - 在受限 GPU 下，找到各系统的最大可服务 batch size
  - 生成 Fig.8

- [ ] **T2.6** E3 延迟分解实验
  - TPOT breakdown: GPU 计算 / 调度 / GPU↔DRAM 传输 / DRAM↔Storage IO
  - 生成 Fig.9

- [ ] **T2.7** E4 存储层消融实验
  - GPU-only → GPU+DRAM → GPU+DRAM+NVM → GPU+DRAM+NVM+SSD
  - 在压力场景下展示每增加一层的边际收益
  - 生成 Fig.10

- [ ] **T2.8** E5/E6/E7/E8/E9 组件级实验
  - 可复用现有数据（已使用 orchkv_core C 库），补充新数据
  - 生成 Fig.11-15

- [ ] **T2.9** E10 质量验证实验（重新跑）
  - 使用真 OrchKvCache 路径，greedy decoding
  - 多模型、多序列长度
  - 生成 Tab.1

- [ ] **T2.10** 生成所有论文级图表
  - 更新 `plot_paper_figures.py`
  - 所有图表统一风格、字体、颜色
  - PDF + PNG 双格式输出

---

## P3: 论文撰写 (Week 5-8)

**目标**: 完成 12 页完整论文初稿。

### Week 5: Motivation + Design

- [ ] **T3.1** 撰写 §2 Background & Motivation
  - KV-Cache 背景 + 大小计算
  - 现有方案分析（vLLM, FlexGen, H2O）
  - Exp-M2 注意力分析结果 + 图表
  - Exp-M3/M4 存储带宽 gap 分析
  - 估计 2.5 pages

- [ ] **T3.2** 撰写 §3 System Design
  - 架构总览 + 架构图
  - 冷热分级算法（公式、伪代码）
  - 分层存储管理
  - 迁移引擎 + 预取流水线
  - 估计 4 pages

### Week 6: Implementation + Evaluation

- [ ] **T3.3** 撰写 §4 Implementation
  - 代码结构、行数统计
  - 关键实现决策
  - 与 vLLM 的集成方式
  - 估计 1 page

- [ ] **T3.4** 撰写 §5 Evaluation
  - 实验设置（硬件、软件、模型、基线、workload）
  - 端到端性能（E1-E3）
  - 显存扩展能力（E4）
  - 组件分析（E5-E9）
  - 质量保证（E10）
  - 估计 4 pages

### Week 7: Introduction + Related Work + Polish

- [ ] **T3.5** 撰写 §1 Introduction
  - 在有了完整实验数据后再写 Introduction
  - 确保 intro 中的数据与 evaluation 一致
  - 估计 2.5 pages

- [ ] **T3.6** 撰写 §6 Related Work
  - 5 个子类别全面覆盖
  - 每个类别突出差异化
  - 估计 1.5 pages

- [ ] **T3.7** 撰写 §7 Discussion + §8 Conclusion
  - 局限性、未来工作
  - 估计 1 page

- [ ] **T3.8** 论文通读与修改
  - 检查逻辑链完整性
  - 检查前后一致性（数据、术语）
  - 检查页数是否超限

### Week 8: References + Final Polish

- [ ] **T3.9** 整理参考文献
  - BibTeX 格式化
  - 检查所有引用是否正确
  - 确保覆盖所有重要相关工作

- [ ] **T3.10** 图表最终优化
  - 所有图表的 caption 完善
  - 字体大小统一（≥8pt）
  - 黑白打印可读性检查

---

## P4: 打磨与投稿 (Week 8)

- [ ] **T4.1** 内部审阅
  - 导师审阅
  - 同学交叉审阅
  - 模拟审稿人提问

- [ ] **T4.2** 根据审阅意见修改

- [ ] **T4.3** 格式检查
  - 页数限制
  - 字体、边距
  - 匿名化（双盲会议）

- [ ] **T4.4** 投稿

---

## 时间轴 (甘特图视角)

```
Week:  1        2        3        4        5        6        7        8
       |--------|--------|--------|--------|--------|--------|--------|
P0:    [████████████████]                                              系统打通
P1:                      [████████████████]                            vLLM 集成
P2:                               [██████████████████████████]         实验执行
P3:                                        [██████████████████████████] 论文撰写
P4:                                                                [██] 投稿

具体:
W1:  T0.1-T0.4  orchkv_api集成tiered_manager, prefetch执行, pipeline接入, 异步IO
W2:  T0.5-T0.7  端到端验证, 压力测试, bug修复
W3:  T1.1-T1.4  vLLM Connector适配, C API调用, 注意力钩子
W4:  T1.5-T1.7  集成测试 | T2.1-T2.3 实验设计, 基线准备
W5:  T2.4-T2.7  核心实验执行 | T3.1-T3.2 Motivation+Design 撰写
W6:  T2.8-T2.10 补充实验+图表 | T3.3-T3.4 Implementation+Evaluation 撰写
W7:  T3.5-T3.8  Introduction+RelatedWork 撰写, 通读修改
W8:  T3.9-T3.10 References+图表优化 | T4.1-T4.4 审阅+投稿
```

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| vLLM API 适配困难 | 高 | 高 | 优先评估 T1.1 的两个选项；如都不行，考虑独立推理 demo 替代 vLLM 集成 |
| NVM 硬件不可用 | 中 | 中 | 使用 tmpfs/ramdisk 模拟 NVM 层，在论文中注明；或使用 CXL 内存 |
| 实验数据不显著 | 中 | 高 | 关键在 T2.1 的压力场景设计——必须让 GPU 显存成为瓶颈；使用更大模型或限制 GPU 利用率 |
| 论文写作时间不足 | 中 | 中 | P3 与 P2 并行推进；先写 Design 章节（不依赖实验数据） |
| FlexGen/InfiniGen 基线复现困难 | 中 | 低 | 如无法复现，引用其论文中的数据做间接对比；重点对比 vLLM 基线（可完全控制） |

---

## 质量检查清单 (投稿前)

### 系统实现
- [ ] orchkv_api ↔ tiered_manager 完全打通
- [ ] prefetch 调度→执行路径完整
- [ ] pipeline 三阶段统计正常工作
- [ ] io_worker 异步 IO 正常工作
- [ ] 所有现有测试通过
- [ ] 端到端 greedy decoding bit-exact 验证

### 实验
- [ ] 至少一个压力场景展示 "OrchKvCache 能跑 / baseline 不能跑"
- [ ] 至少对比 vLLM-swap 和 vLLM-offload 两个基线
- [ ] 所有图表有 error bar 或多次运行统计
- [ ] E10 质量验证在真集成路径下通过

### 论文
- [ ] Abstract 准确反映实验结果
- [ ] Introduction 的数据与 Evaluation 一致
- [ ] Design 章节有足够的技术深度（算法伪代码 + 公式）
- [ ] Related Work 覆盖所有 34 篇引用
- [ ] 所有图表可黑白打印
- [ ] 页数 ≤ 12 页正文
- [ ] 无语法错误

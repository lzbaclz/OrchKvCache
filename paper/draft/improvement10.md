# Improvement 10: 论文排版优化

## 问题诊断

### 问题 1：图片不跟随文字，堆积在一起（p2-p3）
- **根因**：Evaluation 节有 13 个独立 `figure` 浮动体 + 10 个 `table` 浮动体，LaTeX 浮动体队列溢出，无法在引用位置就近放置，被推到后续页面堆积
- **加剧因素**：所有 figure 都用 `[t]` placement，没有 `!` 强制修饰符，LaTeX 的保守策略容易拒绝放置

### 问题 2：PDF 图片与标题之间空白过大（p5-p6）
- **根因**：matplotlib 导出的 PDF 有较大的 bounding box 留白（上下约 10-20pt），图片内容实际只占 bounding box 的 ~85%
- **加剧因素**：全部使用 `width=\columnwidth`，没有用 `trim`/`clip` 裁剪，也没有用负 vspace 补偿

### 问题 3：图片可以更紧凑（参考 IMPRESS 等论文）
- **根因**：许多逻辑相关的图片（如 throughput + eviction、TPOT + scalability）各自独立为 figure 环境，每个浮动体都有固定的 caption + floatsep 开销
- **改进方向**：使用 subfloat 合并为子图并排，减少浮动体总数

## 修改方案

### A. Preamble 全局参数调优

```latex
% 添加 subfig 支持（IEEEtran 推荐用法）
\usepackage[caption=false,font=footnotesize]{subfig}

% 压缩浮动体间距
\setlength{\textfloatsep}{8pt plus 2pt minus 4pt}   % 正文与浮动体间距（默认~20pt）
\setlength{\floatsep}{6pt plus 2pt minus 2pt}       % 相邻浮动体间距（默认~12pt）
\setlength{\intextsep}{6pt plus 2pt minus 2pt}      % [h] 浮动体与正文间距
\setlength{\abovecaptionskip}{4pt}                   % 图片与 caption 间距（默认~10pt）
\setlength{\belowcaptionskip}{-2pt}                  % caption 下方间距（默认~0pt）
```

### B. 图片合并方案（13 → 9 个浮动体）

| 原 Figure | 操作 | 新 Figure |
|-----------|------|-----------|
| Fig 1 (throughput) | 保留，缩至 0.92\columnwidth | Fig 1 |
| Fig 2 (eviction) | 保留，缩至 0.95\columnwidth | Fig 2 |
| Fig 3 (TPOT) + Fig 10 (scalability) | **并排 subfloat**，各 0.48\columnwidth | Fig 3 |
| Fig 4 (speedup heatmap) | 保留 | Fig 4 |
| Fig 6 (quality) | 缩至 0.82\columnwidth | Fig 5 |
| Fig 5 (ablation) | 缩至 0.95\columnwidth | Fig 6 |
| Fig 9 (policy) + Fig 12 (prefetch) | **并排 subfloat**，各 0.48\columnwidth | Fig 7 |
| Fig 17 (hyperparam) | 保留，4面板已紧凑 | Fig 8 |
| Fig w3_throughput + w3_eviction | **垂直堆叠单浮动体** | Fig 9 |
| Fig w4 (SSD tier) | 缩至 0.82\columnwidth | Fig 10 |

> **效果**：从 13 个独立 figure 减少到 10 个，减少 3 个浮动体的排队压力

### C. 所有 figure placement `[t]` → `[!t]`
- `!` 修饰符让 LaTeX 忽略部分内部限制（如单页最大浮动体数），允许更积极地放置图片

### D. 单面板图片尺寸缩减

| 图片 | 原宽度 | 新宽度 | 节省 |
|------|--------|--------|------|
| fig1_throughput | \columnwidth | 0.92\columnwidth | ~8% |
| fig6_quality | \columnwidth | 0.82\columnwidth | ~18% |
| fig_w4_ssd | \columnwidth | 0.82\columnwidth | ~18% |
| fig9_policy | \columnwidth → 合并后 0.48 | ~52% |
| fig3_tpot | 0.85\columnwidth → 合并后 0.48 | ~44% |
| fig10_scalability | 0.85\columnwidth → 合并后 0.48 | ~44% |

### E. PDF 白边裁剪

对所有 `\includegraphics` 添加 `clip` 选项。对已知白边较大的图片，可使用 `trim={left bottom right top}` 参数进一步裁剪。

## 预期效果

- 浮动体队列压力减少 ~30%（13→10 个 figure）
- 图片紧跟文字排列，不再堆积到 p2-p3
- 图片与标题间空白减少约 50%（通过 abovecaptionskip + trim）
- 整体节省约 0.5-1 页空间，可用于增加正文内容

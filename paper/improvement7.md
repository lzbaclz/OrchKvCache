# OrchKvCache Novelty Enhancement: Direction B + C

> 目标：将 Novelty 评分从 5.5 提升到 7.0+
> 方向 B：形式化调度理论 + 经验竞争比分析
> 方向 C：SSD IO 优化消融 + 独立于 OrchFS 的 IO 贡献量化

---

## 方向 B：Online Tiered Paging 理论分析

### B.1 问题形式化

将 KV cache 分层调度建模为 **Weighted Online Paging** 问题：

```
定义：Online Tiered Paging Problem

  输入：
    - k 个 cache 槽位（GPU blocks）
    - n 个 page（KV blocks）
    - 请求序列 σ = (r_1, r_2, ..., r_T)，每步请求一个 page 子集
    - 三层存储，miss cost:
        GPU (tier 0): 0
        DRAM (tier 1): c₁ (e.g., 0.1 ms for 32KB @ 23 GB/s)
        SSD (tier 2):  c₂ (e.g., 2.2 ms for 32KB @ 14.4 GB/s read)

  目标：最小化总 miss cost = Σ c_tier(miss) 

  约束：
    - 每步至多有 k 个 page 在 GPU
    - 驱逐决策必须 online（不知道未来请求）

对应的 offline optimal：
  Belady's Algorithm (OPT)：驱逐"未来最晚被使用"的 page
  推广到三层：贪心地驱逐 cost-weighted 最远距离的 page

Competitive Ratio 定义：
  CR(A) = max_σ { cost_A(σ) / cost_OPT(σ) }
  即算法 A 在最坏输入上相对于 OPT 的代价倍率
```

### B.2 已知理论结果（可直接引用）

```
经典 Paging (Sleator-Tarjan, 1985):
  - LRU competitive ratio = k（紧界）
  - FIFO competitive ratio = k（紧界）
  - RAND (随机 marking) competitive ratio = H_k = O(log k)

Weighted Paging (Bansal-Buchbinder-Naor, 2007):
  - 不同 page 有不同 miss weight
  - 存在 O(log k)-competitive 随机算法
  - 确定性下界 = k

对本文的映射：
  - 三层存储 = 两级权重：c₁ (DRAM miss) 和 c₂ (SSD miss)
  - OrchKvCache 的 EMA 策略 ≈ "weighted LRU with decay"
  - EMA 利用了 temporal locality（不是 adversarial），所以实际 CR 远好于最坏界
```

### B.3 经验竞争比实验（核心实验）

```
实现 Belady 的推广版本（三层 OPT）：
  对每个时刻，如果 GPU 满了需要驱逐：
  - 对每个 GPU-resident block，计算 "下一次被使用的时刻"
  - 驱逐最远的那个到 DRAM
  - 如果 DRAM 满了，将 DRAM 中最远的驱逐到 SSD

在合成 attention trace 上对比五种策略：
  1. FIFO：驱逐最老的
  2. LRU：驱逐最久未使用的
  3. LFU：驱逐使用频率最低的
  4. EMA (OrchKvCache)：按 hotness score 驱逐最冷的
  5. OPT (Belady)：离线最优

度量：
  - 总迁移次数（GPU→DRAM + DRAM→SSD 降级总数）
  - 经验竞争比 = 策略迁移次数 / OPT 迁移次数
  - 分类准确率 vs ground truth

trace 参数：
  - n_blocks ∈ {64, 128, 256, 512}
  - GPU capacity = n_blocks * 0.3（30% 拟合率，制造 eviction 压力）
  - n_steps = 500
  - Zipf + shifting hot set
  - 3 runs 取平均

预期结果：
  FIFO CR ~ 5-10x
  LRU CR ~ 3-5x
  EMA CR ~ 1.5-2.5x（接近 OPT）
  OPT CR = 1.0x（定义）
```

### B.4 论文呈现

```
新增 §3.X "Theoretical Foundation" 或在 §5 末尾新增：

  1. 问题形式化（半页：定义 + 一个定理框）
  2. 经验竞争比实验（Figure: 5 策略 x 4 scale 的 CR 曲线）
  3. 关键叙事：
     "Under temporally-correlated attention patterns (Zipf + slow drift),
      OrchKvCache's EMA-based policy achieves empirical competitive ratio
      1.X--2.Xx, substantially closer to the offline optimal than FIFO
      (X--Xx) or LRU (X--Xx)."
```

---

## 方向 C：SSD IO 优化消融

### C.1 现有 IO 优化清单

```
当前 OrchKvCache 的 SSD IO 优化（已实现）：

  1. 确定性偏移布局 (Deterministic Offset Layout)
     - orchfs_tier.h: offset = (layer * n_kv * B_max + head * B_max + idx) * slab_size
     - 同一 (layer, head) 的 blocks 在文件中连续
     - 使得批量驱逐产生顺序写模式

  2. 32KB Slab 对齐 (SSD Page Alignment)
     - slab_size = 32KB = SSD page size
     - 避免 read-modify-write 放大

  3. 异步 IO 线程池 (Async IO Worker Pool)
     - io_worker.c: N=4 worker threads + circular queue
     - 非阻塞提交，flush 做 barrier

  4. 批量写合并 (Batch Write Coalescing)
     - migration_engine: batch 多个 block 的写入
     - 序列化到 staging buffer 后单次 pwrite
```

### C.2 消融实验设计

```
逐个关闭每项优化，测量 SSD 写/读带宽利用率和端到端吞吐：

  配置矩阵：
  | 编号 | 偏移布局 | 对齐 | 异步IO | 批量写 | 预期 SSD 写利用率 |
  |------|---------|------|--------|--------|-----------------|
  | C0   | random  | off  | sync   | off    | ~4% (naive)     |
  | C1   | determ. | off  | sync   | off    | ~8%             |
  | C2   | determ. | 32KB | sync   | off    | ~15%            |
  | C3   | determ. | 32KB | async  | off    | ~20%            |
  | C4   | determ. | 32KB | async  | batch  | ~41% (full)     |

  对每个配置：
  - 强制 SSD 路径（GPU budget = 10MB）
  - 模型：Qwen2.5-7B + LLaMA-2-7B
  - seq=1024, gen=64
  - 测量：SSD 写带宽 (MB/s), 读带宽 (MB/s), 端到端 tok/s

  这是一个 clean ablation：从最差的 C0 到最优的 C4，
  每步只开启一项优化，量化每项的独立贡献。
```

### C.3 论文呈现

```
新增 Table 或 Figure：
  "SSD IO Optimization Ablation"

  | Optimization              | Write BW | Read BW | Tok/s | Contribution |
  |--------------------------|----------|---------|-------|--------------|
  | None (random, sync, single) | X MB/s | Y MB/s | Z    | baseline     |
  | + Deterministic layout      | +A%    | +B%    | +C%  | layout       |
  | + 32KB alignment            | +D%    | +E%    | +F%  | alignment    |
  | + Async IO pool             | +G%    | +H%    | +I%  | concurrency  |
  | + Batch coalescing          | +J%    | +K%    | +L%  | coalescing   |

  叙事：
  "Each IO optimisation contributes independently: deterministic layout provides
   X%, alignment adds Y%, async IO adds Z%, and batch coalescing adds W%.
   Together they lift SSD write utilisation from 4% to 41%---a 10x improvement
   that is a contribution of this work independent of OrchFS."
```

---

## 实施计划

```
Phase 1 (方向 B，约 3-4 小时):
  1. 实现 Belady OPT 算法（三层版本）
  2. 实现 FIFO / LRU / LFU / EMA 五策略在 trace 上的模拟
  3. 生成合成 trace，跑经验 CR 实验
  4. 制作 CR 对比图 + 表

Phase 2 (方向 C，约 2-3 小时):
  1. 在 Python 层实现 SSD IO 消融框架
     - 用 Python 文件 IO 模拟不同配置
     - 对照组：随机偏移 + 逐 block 同步写
     - 实验组：确定性偏移 + 对齐 + 异步 + 批量
  2. 跑 SSD 消融实验
  3. 整理数据

Phase 3 (论文更新，约 2 小时):
  1. 新增 §理论分析 子节（CR 定义 + 实验）
  2. 新增 IO 消融 table
  3. 更新 Summary + Conclusion
```

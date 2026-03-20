# 硬件环境盘点与存储层性能基线

> 测试日期: 2026-03-20
> 测试机器: Ubuntu 24.04.3 LTS, kernel 6.8.0-101-generic
> Conda 环境: orchkv (Python 3.11.15, PyTorch 2.5.1+cu121)

---

## 1. 硬件清单

### 1.1 GPU

| 项目 | 值 |
|------|-----|
| 型号 | NVIDIA A100-SXM4-80GB × 2 |
| 显存 (HBM2e) | 80 GB × 2 = 160 GB |
| PCIe | Gen 4 (当前运行) |
| 互联 | SXM4 (NVLink) |
| CUDA Driver | 535.288.01 |
| CUDA Version | 12.2 |
| HBM 带宽 (理论) | ~2 TB/s |
| Bus-ID | GPU0: A8:00.0, GPU1: B8:00.0 |

### 1.2 CPU

| 项目 | 值 |
|------|-----|
| 型号 | Intel Xeon Gold 6430 |
| Socket 数 | 2 |
| 每 Socket 核心数 | 32 |
| 总线程数 | 128 (32×2×2 HT) |
| NUMA 节点 | 2 (node0: 0-31,64-95; node1: 32-63,96-127) |

### 1.3 内存 (DRAM)

| 项目 | 值 |
|------|-----|
| 总容量 | 376 GB |
| 可用 | ~360 GB |
| Swap | 8 GB |

### 1.4 NVM (非易失性内存)

| 项目 | 值 |
|------|-----|
| 状态 | **未配置** |
| /dev/dax* | 不存在 |
| /dev/pmem* | 不存在 |
| ndctl | 未安装 |

> **影响**: 没有硬件 NVM，OrchKvCache 的 NVM 中间层需要通过 DRAM 模拟或直接跳过，采用 GPU HBM → DRAM → SSD 的两层架构。

### 1.5 存储设备

| 设备 | 型号 | 容量 | 接口 | PCIe | 挂载点 | 用途 |
|------|------|------|------|------|--------|------|
| nvme0n1 | KIOXIA EXCERIA BASIC | 931.5 GB | NVMe | Gen 4 x4 (16 GT/s) | / (系统盘, LVM) | 系统 + 代码 |
| nvme1n1 | Samsung MZWLO1T9HCJR | 1.7 TB | NVMe | Gen 5 x4 (32 GT/s) | /raid (RAID0) | **实验主存储** |
| nvme2n1 | Samsung MZWLO1T9HCJR | 1.7 TB | NVMe | Gen 5 x4 (32 GT/s) | /raid (RAID0) | **实验主存储** |
| sda | Samsung MZ7L31T9 | 1.7 TB | SATA | N/A | /public | 公共数据/对比基线 |

> **RAID0 总容量**: 3.5 TB (2×1.7TB Samsung Gen5 NVMe), 挂载于 `/raid`
> **推荐实验路径**: `/raid/orchkv_bench` (最高带宽)

---

## 2. GPU HBM ↔ Host DRAM 传输性能

### 2.1 完整数据表

#### Pageable Memory (非锁页)

| 数据大小 | D2H 延迟 (us) | D2H 带宽 (GB/s) | H2D 延迟 (us) | H2D 带宽 (GB/s) |
|----------|--------------|-----------------|--------------|-----------------|
| 512 B | 32.8 | 0.02 | 14.9 | 0.03 |
| 256 KB | 44.2 | 5.93 | 43.1 | 6.08 |
| 1 MB | 127.6 | 8.22 | 119.2 | 8.80 |
| 4 MB | 345.8 | 12.13 | 358.7 | 11.69 |
| 16 MB | 1180.5 | 14.21 | 1199.8 | 13.98 |
| 64 MB | 4526.8 | 14.82 | 5675.8 | 11.82 |
| 128 MB | 8925.0 | 15.04 | 13181.1 | 10.18 |
| 256 MB | 18137.7 | 14.80 | 28350.9 | 9.47 |
| 512 MB | 36064.0 | 14.89 | 57896.8 | 9.27 |

#### Pinned Memory (锁页)

| 数据大小 | D2H 延迟 (us) | D2H 带宽 (GB/s) | H2D 延迟 (us) | H2D 带宽 (GB/s) |
|----------|--------------|-----------------|--------------|-----------------|
| 512 B | 14.7 | 0.03 | 14.9 | 0.03 |
| 256 KB | 25.7 | 10.20 | 26.4 | 9.93 |
| 1 MB | 58.1 | 18.04 | 57.9 | 18.12 |
| 4 MB | 188.5 | 22.25 | 181.0 | 23.17 |
| 16 MB | 743.4 | 22.57 | 678.7 | 24.72 |
| 64 MB | 2943.0 | 22.80 | 2657.1 | 25.26 |
| 128 MB | 5684.2 | 23.61 | 5330.4 | 25.18 |
| 256 MB | 11573.1 | 23.19 | 10601.0 | 25.32 |
| 512 MB | 22557.9 | 23.80 | 21289.5 | 25.22 |

#### GPU HBM 内部 (D2D Baseline)

| 数据大小 | 延迟 (us) | 带宽 (GB/s) |
|----------|-----------|-------------|
| 512 B | 12.0 | 0.04 |
| 256 KB | 12.0 | 21.83 |
| 1 MB | 12.5 | 83.90 |
| 4 MB | 15.1 | 276.91 |
| 16 MB | 28.8 | 583.18 |
| 64 MB | 96.9 | 692.66 |
| 128 MB | 181.7 | 738.80 |
| 256 MB | 350.2 | 766.44 |
| 512 MB | 694.3 | 773.30 |

### 2.2 关键发现

1. **Pinned memory 是必须的**: 大块传输时，pinned 比 pageable 快 **~1.7x** (D2H: 23.8 vs 14.9 GB/s; H2D: 25.2 vs 9.3 GB/s)
2. **Pageable H2D 大块退化严重**: 512MB 时 H2D 仅 9.3 GB/s，因为需要额外的页面锁定+拷贝
3. **GPU 内部带宽是 PCIe 的 30x**: D2D ~773 GB/s vs D2H pinned ~24 GB/s
4. **小块 (<256KB) 延迟主导**: 传输延迟约 15-26 us，带宽利用率低
5. **甜区在 4-16 MB**: 此区间 pinned 传输已达 22-25 GB/s (接近 PCIe Gen4 x16 理论 ~32 GB/s)

---

## 3. SSD I/O 性能

### 3.1 fio 标准测试 (Direct I/O, libaio, iodepth=32)

| 测试项 | KIOXIA Gen4 | Samsung RAID0 Gen5 | SATA Samsung |
|--------|------------|-------------------|-------------|
| **顺序读 1M** | 7.30 GB/s | **17.80 GB/s** | 0.57 GB/s |
| **顺序写 1M** | 6.79 GB/s | 5.26 GB/s | 0.54 GB/s |
| **随机读 4K** | 276K IOPS (1.13 GB/s) | 202K IOPS (0.83 GB/s) | 95K IOPS (0.39 GB/s) |
| **随机写 4K** | 225K IOPS (0.92 GB/s) | 33K IOPS (0.14 GB/s) | 73K IOPS (0.30 GB/s) |
| **随机读 64K** | 108K IOPS (7.11 GB/s) | 127K IOPS (8.35 GB/s) | 8.4K IOPS (0.55 GB/s) |
| **随机写 64K** | 23K IOPS (1.50 GB/s) | 61K IOPS (4.00 GB/s) | 7.9K IOPS (0.52 GB/s) |
| **随机读 1M** | 7.43 GB/s | **17.19 GB/s** | 0.57 GB/s |
| **随机写 1M** | 6.45 GB/s | 5.58 GB/s | 0.54 GB/s |

### 3.2 fsync 延迟

| 磁盘 | avg (us) | p50 (us) | p99 (us) |
|------|----------|----------|----------|
| KIOXIA Gen4 | 407.8 | 412.0 | 483.9 |
| **Samsung RAID0 Gen5** | **53.4** | **52.5** | **70.2** |
| SATA Samsung | 219.4 | 218.7 | 234.8 |

### 3.3 Python Buffered I/O

| 磁盘 | 顺序写 1M (GB/s) | 顺序读 1M (GB/s) |
|------|-----------------|-----------------|
| KIOXIA Gen4 | 1.34 | 5.57 |
| Samsung RAID0 Gen5 | 2.87 | 7.74 |
| SATA Samsung | 0.43 | 5.75 |

### 3.4 关键发现

1. **RAID0 Samsung Gen5 是最优实验盘**: 顺序读 17.8 GB/s，fsync 仅 53 us
2. **读远强于写**: Samsung RAID0 读 17.8 GB/s vs 写 5.3 GB/s (RAID0 写放大 + 控制器特性)
3. **KV-cache 大块读取有利**: 1M 随机读 17.2 GB/s，说明大块 KV-cache reload 的 SSD 端不是瓶颈
4. **4K 随机写是瓶颈**: RAID0 仅 33K IOPS，小块频繁 eviction 需避免
5. **Python buffered I/O 损失严重**: 仅 fio 性能的 ~30-50%，说明 IO 路径优化(如 io_uring、直接对齐写) 很有价值
6. **SATA SSD 差距巨大**: ~30x 慢于 NVMe Gen5，可作为极端慢速存储对比

---

## 4. 端到端 KV-Cache Offload 路径延迟

> 测试存储: Samsung RAID0 Gen5 NVMe (/raid)
> 传输方式: Pinned memory, synchronous copy + buffered file I/O

### 4.1 GPU ↔ DRAM (Warm Tier, 无磁盘)

| KV-Cache 配置 | 数据大小 | Offload D2H (us) | BW (GB/s) | Reload H2D (us) | BW (GB/s) |
|---------------|---------|------------------|-----------|-----------------|-----------|
| vLLM block 16tok | 256 KB | 26.0 | 10.1 | 26.1 | 10.0 |
| vLLM block 64tok | 1 MB | 58.7 | 17.9 | 57.5 | 18.2 |
| Llama-70B seq256 (GQA8) | 1 MB | 58.9 | 17.8 | 57.7 | 18.2 |
| Llama-7B seq256 | 4 MB | 189.3 | 22.2 | 181.7 | 23.1 |
| Llama-70B seq2048 (GQA8) | 8 MB | 362.9 | 23.1 | 347.2 | 24.2 |
| Llama-7B seq2048 | 32 MB | 1478.6 | 22.7 | 1338.9 | 25.1 |
| Llama-7B seq4096 | 64 MB | 2933.5 | 22.9 | 2659.4 | 25.2 |

### 4.2 GPU → DRAM → SSD (Cold Eviction 路径)

| KV-Cache 配置 | 数据大小 | D2H (us) | 磁盘写 (us) | **总延迟 (us)** | **E2E BW (GB/s)** |
|---------------|---------|----------|------------|----------------|-------------------|
| vLLM block 16tok | 256 KB | 30 | 178 | **209** | 1.26 |
| vLLM block 64tok | 1 MB | 70 | 803 | **873** | 1.20 |
| Llama-70B seq256 | 1 MB | 70 | 458 | **528** | 1.99 |
| Llama-7B seq256 | 4 MB | 202 | 2114 | **2316** | 1.81 |
| Llama-70B seq2048 | 8 MB | 378 | 3675 | **4053** | 2.07 |
| Llama-7B seq2048 | 32 MB | 1514 | 35593 | **37108** | 0.90 |
| Llama-7B seq4096 | 64 MB | 3120 | 79275 | **82395** | 0.81 |

### 4.3 SSD → DRAM → GPU (Cold Loading 路径)

| KV-Cache 配置 | 数据大小 | 磁盘读 (us) | H2D (us) | **总延迟 (us)** | **E2E BW (GB/s)** |
|---------------|---------|------------|----------|----------------|-------------------|
| vLLM block 16tok | 256 KB | 73 | 35 | **109** | 2.42 |
| vLLM block 64tok | 1 MB | 299 | 99 | **398** | 2.64 |
| Llama-70B seq256 | 1 MB | 279 | 100 | **379** | 2.77 |
| Llama-7B seq256 | 4 MB | 987 | 302 | **1289** | 3.26 |
| Llama-70B seq2048 | 8 MB | 1958 | 525 | **2483** | 3.38 |
| Llama-7B seq2048 | 32 MB | 44551 | 2837 | **47389** | 0.71 |
| Llama-7B seq4096 | 64 MB | 93245 | 3544 | **96789** | 0.69 |

### 4.4 关键发现与设计启示

#### 4.4.1 层级延迟差距

```
GPU HBM 内部:    ~15 us    (4MB, 277 GB/s)     → 基线
GPU → DRAM:      ~189 us   (4MB, 22 GB/s)      → 12.6x 慢
GPU → DRAM → SSD: ~2316 us (4MB, 1.8 GB/s)     → 154x 慢
```

**结论**: 三级存储之间存在 **1-2 个数量级** 的延迟差距，冷热分级是值得做的。

#### 4.4.2 瓶颈分析

| 路径 | 瓶颈在哪 | 占比 |
|------|----------|------|
| Eviction (4MB) | **磁盘写** (2114 us / 2316 us) | **91%** |
| Loading (4MB) | **磁盘读** (987 us / 1289 us) | **77%** |
| Warm offload (4MB) | PCIe 传输 | 100% (无磁盘) |

**结论**: 冷路径的瓶颈在 **磁盘 IO**，不在 PCIe。优化 IO 路径 (io_uring, O_DIRECT, batch write) 是最有价值的方向。

#### 4.4.3 对 OrchKvCache 的设计启示

1. **必须用 pinned memory**: 比 pageable 快 1.5-1.7x，且 async copy 需要 pinned
2. **DRAM 缓冲层至关重要**: Warm tier (GPU↔DRAM ~190us) 比 Cold tier (GPU↔SSD ~2.3ms) 快 **12x**
3. **大块 eviction 需要异步**: 32MB KV-cache 写 SSD 要 37ms，必须与计算 overlap
4. **SSD 写比读慢**: eviction 比 loading 慢 ~2x，需要 write-back 而非 write-through
5. **小块频繁 eviction 不可取**: 256KB×N 的离散写效率低于一次性 batch 大块写
6. **IO 路径优化空间大**: Python buffered I/O 仅达到 fio 性能的 30-50%，OrchFS 的 io_uring + O_DIRECT 可望接近 fio 性能
7. **没有 NVM → 两层架构**: 当前硬件下 OrchKvCache 采用 GPU HBM ↔ Host DRAM ↔ NVMe SSD 三级，无 NVM 中间层

#### 4.4.4 关键延迟数字 (设计参考)

```
Decode 一个 token (A100):        ~0.5-2 ms
Prefill 256 tokens:              ~5-20 ms
KV block evict to DRAM (4MB):    ~189 us     ← 可在 1 个 decode step 内完成
KV block evict to SSD (4MB):     ~2.3 ms     ← 约等于 1 个 decode step
KV block reload from DRAM (4MB): ~182 us     ← 可隐藏在 prefill 中
KV block reload from SSD (4MB):  ~1.3 ms     ← 需要预取来隐藏
```

---

## 5. 存储层级带宽汇总图

```
                     带宽 (GB/s, 大块传输)
GPU HBM (D2D):       ████████████████████████████████████████  773
GPU↔DRAM (pinned):   █████                                      25
GPU↔DRAM (pageable): ███                                        15
SSD Seq Read (RAID0):██                                        17.8
SSD Seq Write(RAID0):█                                          5.3
E2E Cold Load (4MB): ▏                                          3.3
E2E Cold Evict(4MB): ▏                                          1.8
SATA SSD:            ▏                                          0.57
```

---

## 6. 下一步建议

1. **构建 pinned memory pool**: 预分配锁页内存作为 GPU↔DRAM staging buffer
2. **实现异步 eviction pipeline**: 利用 CUDA stream + 异步磁盘写，与 decode 计算 overlap
3. **设计 batch eviction**: 将多个小 KV block 合并为大块一次写入 SSD
4. **集成 OrchFS**: 利用其 io_uring 引擎和对齐写入，有望将 buffered IO 性能提升 2-3x
5. **实现预取调度器**: SSD reload 1.3ms 不算慢，但需要提前 1-2 个 decode step 预取
6. **NVM 模拟方案**: 可用 DRAM 限速模拟 (~5-10 GB/s) 来验证三层架构设计

---

*数据文件:*
- `results/bench_gpu_dram.json` — GPU↔DRAM 传输详细数据
- `results/bench_ssd_io.json` — SSD I/O 性能详细数据
- `results/bench_e2e_offload.json` — 端到端 offload 路径详细数据

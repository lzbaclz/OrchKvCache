# Exp0: 硬件环境盘点与存储层性能基线

> **实验日期**: 2026-03-20
> **目的**: Step 0 — 摸清硬件能力，建立各存储层级的延迟/带宽基线，为后续 OrchKvCache 系统设计提供数据支撑

---

## 实验内容

| 编号 | 测试项 | 脚本 | 结果 |
|------|--------|------|------|
| T1 | GPU HBM ↔ Host DRAM 传输延迟/带宽 | `scripts/bench_gpu_dram.py` | `results/bench_gpu_dram.json` |
| T2 | SSD I/O 性能 (fio + Python IO) | `scripts/bench_ssd_io.py` | `results/bench_ssd_io.json` |
| T3 | 端到端 KV-Cache Offload 路径延迟 | `scripts/bench_e2e_offload.py` | `results/bench_e2e_offload.json` |

## 文件结构

```
exp0_hardware_baseline/
├── README.md                          # 本文件
├── docs/
│   └── hardware_inventory.md          # 完整硬件清单 + 全部实验数据汇总 + 设计启示
├── scripts/
│   ├── bench_gpu_dram.py              # T1: GPU↔DRAM 传输 (pinned/pageable/async)
│   ├── bench_ssd_io.py                # T2: SSD IO (seq/rand, buffered/direct, fio)
│   └── bench_e2e_offload.py           # T3: 端到端 KV-cache offload 路径
└── results/
    ├── bench_gpu_dram.json            # T1 原始数据
    ├── bench_ssd_io.json              # T2 原始数据
    └── bench_e2e_offload.json         # T3 原始数据
```

## 关键结论

- **GPU↔DRAM (pinned)**: D2H ~23.8 GB/s, H2D ~25.2 GB/s
- **GPU 内部 D2D**: ~773 GB/s
- **Samsung RAID0 Gen5 NVMe**: 顺序读 17.8 GB/s, 顺序写 5.3 GB/s
- **E2E Cold Eviction (4MB)**: ~2.3ms, 瓶颈在磁盘写 (91%)
- **E2E Cold Loading (4MB)**: ~1.3ms, 预取 1-2 个 decode step 可隐藏
- **无 NVM 硬件**: 采用 GPU HBM ↔ DRAM ↔ SSD 两级架构

详细数据和设计启示见 `docs/hardware_inventory.md`。

## 运行方式

```bash
conda activate orchkv
CUDA_VISIBLE_DEVICES=0 python scripts/bench_gpu_dram.py
python scripts/bench_ssd_io.py
CUDA_VISIBLE_DEVICES=0 python scripts/bench_e2e_offload.py
```

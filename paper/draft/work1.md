# Work1: 第一阶段动手任务 —— 环境验证 + Motivation 实验

## 你现在最该做什么？

**不是直接写 OrchKvCache 的代码**，而是先做三件事：

```
Step 0: 确认硬件环境（你手上有什么牌）
Step 1: Motivation 实验（用真实数据证明问题存在）
Step 2: 存储基线测量（搞清楚 OrchFS / NVM / SSD 到底多快）
```

### 为什么这个顺序？

很多研究生的常见错误是：**假设问题存在 → 闷头写系统 → 最后发现假设不成立或效果不显著**。

正确的做法是：
1. **先证明问题确实严重**（M1: KV-Cache 真的吃光显存了吗？）
2. **再证明机会确实存在**（M2: 冷热分化真的很明显吗？）
3. **再证明现有方案确实低效**（M3: 现有 offloading 带宽利用率真的差吗？）
4. **再证明你的路线确实可行**（M4/S1~S3: NVM 和 OrchFS 是否真的能带来预期的延迟/带宽优势？）

这些数据有三个作用：
- **论文 §2 的内容**（Motivation 部分，审稿人第一个看的地方）
- **指导你的设计参数**（KV Block 该多大？阈值该多少？NVM 容量分配多少？）
- **风险前置**（如果某个假设不成立，现在发现比写完系统再发现好 100 倍）

---

## Step 0: 硬件环境盘点（Day 1，半天）

在做任何实验之前，你必须搞清楚手上有什么硬件。这直接决定了哪些实验能做、哪些需要模拟。

### 0.1 需要确认的硬件

```bash
# ===== GPU =====
nvidia-smi
# 记录：型号、显存大小、CUDA 版本

# ===== CPU & DRAM =====
lscpu | grep -E "Model name|Socket|Core|Thread"
free -h
# 记录：CPU 型号、核数、DRAM 总量

# ===== NVM (Intel Optane PM) =====
# 检查是否有 DAX 设备
ls /dev/dax*
# 检查是否有 PMEM 设备
ndctl list
# 如果都没有，说明没有 NVM 硬件 —— 这是一个关键风险点

# ===== SSD =====
lsblk -d -o NAME,SIZE,MODEL,ROTA,TRAN
# ROTA=0 表示 SSD, TRAN=nvme 表示 NVMe SSD
# 记录：SSD 型号、容量、接口类型

# ===== PCIe 带宽 =====
lspci | grep -i nvme
lspci | grep -i nvidia
# 确认 PCIe 版本（Gen3/Gen4/Gen5）
```

### 0.2 产出：硬件清单

在项目根目录创建 `docs/hardware_inventory.md`，记录所有硬件信息。格式：

```markdown
# 实验环境硬件清单
- GPU: [型号] [显存]GB, CUDA [版本], PCIe [Gen]
- CPU: [型号], [核数] cores, [线程数] threads
- DRAM: [容量]GB
- NVM: [有/无], [型号], [容量]GB, 设备路径 [/dev/dax?.?]
- SSD: [型号], [容量]TB, [NVMe/SATA], PCIe [Gen]
- OS: [发行版] [内核版本]
```

### 0.3 关键决策点

| 情况 | 影响 | 应对 |
|------|------|------|
| **有 NVM + SSD + GPU** | 最佳，所有实验都能做 | 按计划进行 |
| **无 NVM，有 SSD + GPU** | 无法做 M4 和 NVM 相关实验 | 用 DRAM 模拟 NVM（限制带宽和延迟），论文中说明 |
| **无 SSD，有 NVM + GPU** | 无法做 M3 和 SSD 相关实验 | 几乎不可能，服务器都有 SSD |
| **只有小显存 GPU（<40GB）** | 不能跑 70B 模型 | 聚焦 7B/13B 模型 |

---

## Step 1: Motivation 实验（Week 1~2，核心产出）

### M1: KV-Cache 显存瓶颈量化

**测什么**：不同模型、不同序列长度下 KV-Cache 到底吃多少显存。

**为什么测**：这是论文 §2 的**第一张图**——告诉审稿人 "问题确实严重"。如果 KV-Cache 只占显存的 10%，那你的整个项目就没必要了。你需要用真实数据证明：**长上下文场景下，KV-Cache 是显存的主要消耗者，严重限制了 batch size**。

**怎么测**：

```bash
# 安装 vLLM（你的基线系统）
pip install vllm

# 下载模型（先用小模型，后面再测大的）
# LLaMA-2-7B 权重需从 HuggingFace 下载
```

```python
# scripts/motivation/m1_kvcache_memory.py

"""
Exp-M1: KV-Cache 显存占用量化
目标：绘制 "序列长度 vs KV-Cache 大小 vs GPU显存容量" 对比图
"""

import torch
import json
import csv

# KV-Cache 大小的理论计算（无需跑模型，纯数学）
def calc_kvcache_size_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    """计算 KV-Cache 的理论大小（bytes）
    dtype_bytes=2 表示 FP16, =1 表示 INT8
    """
    # 2 是因为 K 和 V 各一份
    return 2 * n_layers * n_kv_heads * seq_len * d_head * dtype_bytes

# 模型配置
models = {
    "LLaMA-2-7B":   {"n_layers": 32, "n_kv_heads": 32, "d_head": 128},
    "LLaMA-2-13B":  {"n_layers": 40, "n_kv_heads": 40, "d_head": 128},
    "LLaMA-2-70B":  {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128},  # GQA
    "LLaMA-3-8B":   {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128},  # GQA
    "Mistral-7B":   {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128},  # GQA
}

# GPU 显存容量
gpu_memory = {
    "A100-40GB": 40 * 1024**3,
    "A100-80GB": 80 * 1024**3,
    "H100-80GB": 80 * 1024**3,
}

# 序列长度
seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# 计算并输出
results = []
for model_name, cfg in models.items():
    for seq_len in seq_lens:
        size = calc_kvcache_size_bytes(**cfg, seq_len=seq_len)
        results.append({
            "model": model_name,
            "seq_len": seq_len,
            "kvcache_GB": size / 1024**3,
            "per_token_MB": size / seq_len / 1024**2,
        })
        print(f"{model_name}, seq={seq_len:>7d}: KV-Cache = {size/1024**3:.2f} GB, "
              f"per token = {size/seq_len/1024**2:.4f} MB")

# 存为 CSV
with open("results/m1_kvcache_size.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
```

```python
# scripts/motivation/m1_plot.py
# 绘图：柱状图/折线图，X轴序列长度，Y轴KV-Cache大小，不同模型不同颜色
# 叠加 GPU 显存容量的水平参考线
# 这张图直接用于论文 Figure 2
```

**然后还要做一个真实验证**（不能只算理论值，需要 vLLM 跑真实推理来验证）：

```python
# scripts/motivation/m1_real_memory.py

"""
真实测量：用 vLLM 跑推理，通过 nvidia-smi 或 torch.cuda.memory_allocated()
测量不同 seq_len 和 batch_size 下的实际 GPU 显存占用
"""

from vllm import LLM, SamplingParams

def measure_memory(model_name, seq_len, batch_size):
    """测量指定配置下的 GPU 显存占用"""
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(model=model_name, max_model_len=seq_len)
    # 构造 batch_size 个长度为 seq_len 的请求
    prompts = ["Hello " * (seq_len // 2)] * batch_size  # 粗略构造
    params = SamplingParams(max_tokens=1)  # 只生成 1 个 token，关注 prefill 阶段的 KV-Cache
    outputs = llm.generate(prompts, params)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    return peak_mem

# 扫描不同 batch_size，找到 OOM 的临界点
# 记录：模型权重显存 + KV-Cache显存 + 其他显存
```

**预期产出**：
- `results/m1_kvcache_size.csv` — 理论 KV-Cache 大小数据
- `results/m1_real_memory.csv` — 实际显存占用数据
- 论文 Figure 2：KV-Cache 大小 vs 序列长度 vs GPU 显存容量线

---

### M2: 注意力分数分布分析（冷热分化验证）

**测什么**：在真实推理中，注意力分数在 token 之间的分布是否真的呈幂律分布？是否存在明显的冷热分化？

**为什么测**：这是你整个项目的**理论根基**。如果注意力分数在所有 token 上均匀分布，那冷热分级就没有意义——所有 KV-Cache 都同等重要，你不能换出任何一个。你需要用真实数据证明：**少量 token 贡献了绝大部分注意力，大量 token 几乎不被关注。** H2O 论文做过类似分析，但你需要在你自己的模型和数据上重现，并且分析 OrchKvCache 特有的维度（按 block 粒度聚合后是否仍然有明显分化）。

**怎么测**：

```python
# scripts/motivation/m2_attention_analysis.py

"""
Exp-M2: 注意力分数分布分析
需要修改模型推理代码来导出注意力分数
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def collect_attention_scores(model_name, input_text, device="cuda"):
    """收集每层每头的注意力分数矩阵"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        output_attentions=True,  # 关键：输出注意力分数
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (batch, n_heads, seq_len, seq_len)
    # 每层一个张量
    return outputs.attentions

def analyze_distribution(attentions, block_size=64):
    """分析注意力分数分布"""
    results = {}

    for layer_idx, attn in enumerate(attentions):
        # attn: (1, n_heads, seq_len, seq_len)
        # 取最后一个 query token 的注意力分布
        last_query_attn = attn[0, :, -1, :]  # (n_heads, seq_len)

        # 1. 按 token 级别：计算每个位置的平均注意力（跨头平均）
        token_attn = last_query_attn.mean(dim=0)  # (seq_len,)

        # 2. 计算 Top-K% 占总注意力的比例
        sorted_attn, _ = token_attn.sort(descending=True)
        total = sorted_attn.sum()
        cumsum = sorted_attn.cumsum(dim=0) / total
        # 找 Top-10%, Top-20% 的覆盖率
        seq_len = len(token_attn)
        top10_coverage = cumsum[int(seq_len * 0.1)].item()
        top20_coverage = cumsum[int(seq_len * 0.2)].item()

        # 3. 按 block 级别聚合（模拟 KV Block 粒度）
        n_blocks = seq_len // block_size
        block_attn = token_attn[:n_blocks * block_size].reshape(n_blocks, block_size).sum(dim=1)
        block_attn = block_attn / block_attn.sum()
        sorted_block, _ = block_attn.sort(descending=True)
        block_cumsum = sorted_block.cumsum(dim=0)

        results[layer_idx] = {
            "token_attn": token_attn.cpu(),
            "top10_coverage": top10_coverage,
            "top20_coverage": top20_coverage,
            "block_attn": block_attn.cpu(),
            "block_cumsum": block_cumsum.cpu(),
        }

        print(f"Layer {layer_idx:2d}: Top-10% tokens cover {top10_coverage:.1%} attention, "
              f"Top-20% cover {top20_coverage:.1%}")

    return results

# 使用长上下文输入（如 LongBench 的某个样本）
# 或者用一段长文章作为 prompt
```

```python
# scripts/motivation/m2_plot.py
# 绘制：
# 1. 热力图：某层某头的注意力矩阵可视化
# 2. CDF 图：X轴 = token 比例 (0~100%), Y轴 = 累积注意力覆盖率
#    预期：20% 的 token 覆盖 >90% 的注意力 → 曲线急剧上升后平坦
# 3. 跨层对比：不同层的 Top-20% 覆盖率柱状图
# 这些图直接用于论文 Figure 3
```

**额外需要分析的维度**（对你的设计非常重要）：

```python
# M2 额外分析

# (a) 跨 decode step 的重要性持续性（验证 ScissorHands 假说）
# 在连续 10 个 decode step 中，Top-K token 集合的 Jaccard 相似度
# → 如果相似度高，说明你的 EMA 热度更新是有道理的

# (b) 层间差异
# 底层 vs 高层的 Top-20% 覆盖率对比
# → 如果差异大，说明 SqueezeAttention 的层级自适应阈值是必要的

# (c) Block 粒度 vs Token 粒度
# 按 block_size = {8, 16, 32, 64} 聚合后的覆盖率变化
# → 帮你确定 KV Block 大小：block 太大会导致 "block 内有冷有热，但整体被判为热"
```

**预期产出**：
- `results/m2_attn_distribution.pkl` — 原始注意力分数数据
- `results/m2_coverage_stats.csv` — 各层 Top-K% 覆盖率统计
- 论文 Figure 3：CDF 图 + 热力图

---

### M3: 现有 offloading 方案的 IO 效率分析

**测什么**：vLLM 和 FlexGen 在做 KV-Cache swap/offloading 时，SSD 的实际写带宽利用率是多少。

**为什么测**：你的论文核心故事之一是 "现有方案的 IO 效率低"。你需要用真实数据证明：**FlexGen/vLLM 使用 POSIX IO 写 SSD 时，实际带宽远低于 SSD 的理论峰值**。然后在后续实验中证明 OrchFS 能大幅提升这个利用率。如果现有方案 IO 已经够快了，那你用 OrchFS 就没有优势——所以这个实验也是在验证你的技术路线是否有意义。

**怎么测**：

```python
# scripts/motivation/m3_io_profiling.py

"""
Exp-M3: 存储 IO 效率分析
方法 1: 用 fio 测 SSD 理论峰值带宽
方法 2: 用 strace / blktrace 追踪 vLLM swap 的实际 IO 模式
方法 3: 写一个模拟 KV-Cache offloading 的简单程序，对比不同写入方式的带宽
"""

import os
import time
import mmap
import numpy as np

def bench_write_bandwidth(file_path, block_size, total_size, method="posix"):
    """测量不同写入方式的带宽"""
    data = np.random.bytes(block_size)
    n_blocks = total_size // block_size

    if method == "posix":
        fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        start = time.perf_counter()
        for _ in range(n_blocks):
            os.write(fd, data)
        os.fsync(fd)
        elapsed = time.perf_counter() - start
        os.close(fd)

    elif method == "direct_io":
        # O_DIRECT: 绕过页缓存，直接写 SSD
        fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
        # O_DIRECT 要求内存对齐
        aligned_data = mmap.mmap(-1, block_size)
        aligned_data.write(data)
        aligned_data.seek(0)
        start = time.perf_counter()
        for _ in range(n_blocks):
            os.write(fd, aligned_data.read(block_size))
            aligned_data.seek(0)
        elapsed = time.perf_counter() - start
        os.close(fd)
        aligned_data.close()

    bw = total_size / elapsed / 1024**3  # GB/s
    return bw, elapsed

# 测试矩阵
block_sizes = [4096, 32768, 65536, 262144, 1048576]  # 4KB ~ 1MB
total_size = 1 * 1024**3  # 1GB 总量
methods = ["posix", "direct_io"]

print(f"{'Block Size':>12} {'Method':>12} {'Bandwidth':>12} {'Time':>10}")
print("-" * 50)
for bs in block_sizes:
    for method in methods:
        bw, elapsed = bench_write_bandwidth("/tmp/bench_test", bs, total_size, method)
        print(f"{bs:>12} {method:>12} {bw:>10.2f} GB/s {elapsed:>8.2f}s")
    os.remove("/tmp/bench_test")
```

```bash
# 先用 fio 测 SSD 理论峰值（作为参照线）
# 顺序写峰值
fio --name=seq_write --rw=write --bs=128k --size=4G --numjobs=4 \
    --ioengine=libaio --iodepth=32 --direct=1 \
    --filename=/dev/nvme0n1  # 改成你的 SSD 设备

# 4KB 随机写
fio --name=rand_write_4k --rw=randwrite --bs=4k --size=1G --numjobs=4 \
    --ioengine=libaio --iodepth=32 --direct=1 \
    --filename=/dev/nvme0n1

# 32KB 随机写（对应 OrchFS 的 SSD Block）
fio --name=rand_write_32k --rw=randwrite --bs=32k --size=2G --numjobs=4 \
    --ioengine=libaio --iodepth=32 --direct=1 \
    --filename=/dev/nvme0n1
```

**预期产出**：
- `results/m3_ssd_peak_bandwidth.txt` — fio 测得的 SSD 理论峰值
- `results/m3_write_bandwidth.csv` — 不同块大小 × 写入方式的实际带宽
- 论文 Figure 4：柱状图，对比各方案的 SSD 带宽利用率

---

### M4: NVM 中间层价值量化

**测什么**：NVM 读写的延迟和带宽，对比 SSD，证明 NVM 作为中间层的价值。

**为什么测**：你的四级架构（GPU→DRAM→NVM→SSD）比 FlexGen 的三级（GPU→CPU→Disk）多了 NVM 层。审稿人一定会问："**多加一层 NVM 真的有用吗？为什么不直接 DRAM→SSD？**"你需要数据证明：NVM 的读延迟（~300ns）比 SSD（~10μs）低 30 倍以上，对于温数据的快速换入具有不可替代的价值。

**如果你有 NVM 硬件**：

```bash
# 测 NVM 延迟和带宽
# 假设 NVM 挂载在 /dev/dax0.0 或 /mnt/pmem0

# 4KB 随机读延迟（最关键的指标——对应温数据换入）
fio --name=nvm_rand_read_4k --rw=randread --bs=4k --size=1G \
    --ioengine=dev-dax --iodepth=1 \
    --filename=/dev/dax0.0

# 4KB 顺序写
fio --name=nvm_seq_write_4k --rw=write --bs=4k --size=1G \
    --ioengine=dev-dax --iodepth=1 \
    --filename=/dev/dax0.0

# 对比 SSD 同样的操作
fio --name=ssd_rand_read_4k --rw=randread --bs=4k --size=1G \
    --ioengine=libaio --iodepth=1 --direct=1 \
    --filename=/dev/nvme0n1
```

**如果你没有 NVM 硬件**（Intel Optane 已停产，这种情况很常见）：

```bash
# 方案 A: 用 DRAM 模拟 NVM (通过 GRUB 限制 DRAM 区域)
# 需要配置 memmap 内核参数，将部分 DRAM 划为 PMEM
# 参考: https://pmem.io/blog/2016/02/how-to-emulate-persistent-memory/

# 方案 B: 使用理论数据 + 软件延迟注入
# 在程序中人为插入 300ns 延迟来模拟 NVM 读延迟
# 这种方法可以在论文中说明是 emulation
```

```python
# scripts/motivation/m4_tier_latency.py

"""
Exp-M4: 各存储层延迟对比
测量 4KB 和 32KB 数据块在不同存储层的读/写延迟
"""

import time
import torch
import numpy as np

def bench_gpu_to_dram(size_bytes, iterations=1000):
    """GPU HBM → Host DRAM 传输延迟"""
    gpu_tensor = torch.randn(size_bytes // 4, device="cuda", dtype=torch.float32)
    cpu_tensor = torch.empty_like(gpu_tensor, device="cpu").pin_memory()

    # warmup
    for _ in range(10):
        cpu_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        cpu_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iterations * 1e6
    bw_gbps = size_bytes * iterations / elapsed / 1e9
    return avg_us, bw_gbps

def bench_dram_to_dram(size_bytes, iterations=10000):
    """DRAM 内存拷贝延迟"""
    src = np.random.bytes(size_bytes)
    dst = bytearray(size_bytes)

    start = time.perf_counter()
    for _ in range(iterations):
        dst[:] = src
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iterations * 1e6
    return avg_us

# 测量不同大小
for size_label, size in [("4KB", 4096), ("32KB", 32768)]:
    gpu_lat, gpu_bw = bench_gpu_to_dram(size)
    dram_lat = bench_dram_to_dram(size)
    print(f"{size_label}: GPU→DRAM = {gpu_lat:.1f} μs ({gpu_bw:.1f} GB/s), "
          f"DRAM copy = {dram_lat:.1f} μs")
    # NVM 和 SSD 用 fio 数据补充
```

**预期产出**：
- `results/m4_tier_latency.csv` — 各存储层读写延迟数据
- 论文 Figure 5：柱状图，GPU HBM / DRAM / NVM / SSD 的读写延迟对比

---

## Step 2: OrchFS 存储基线测量（Week 2，与 M3/M4 并行）

如果你有 NVM+SSD 硬件，需要验证 OrchFS 的性能是否符合预期。

### S1: OrchFS 编译和基础功能验证

```bash
# 编译 OrchFS
cd /home/lzq/codes/orchkv/OrchFS
python config_parameter.py /dev/dax0.0 /dev/nvme0n1 4 16 32k
mkdir -p build && cd build
cmake .. && make

# 运行 OrchFS 自带的微基准
cd ../scripts/micro
sudo sh run_micro.sh
# 记录顺序写/随机写带宽
```

### S2: OrchFS vs POSIX IO 带宽对比

**为什么测**：你的论文声称 OrchFS 的异构 IO 编排能带来更高的存储带宽利用率。这个声称需要数据支撑。

```bash
# 对比 OrchFS 路径写入 vs 普通 POSIX 写入
# 在 OrchFS 的 /Or 路径下写入
# 在普通文件系统路径下写入
# 对比不同块大小下的带宽
```

### S3: OrchFS 的 NVM→SSD 迁移性能

**为什么测**：你的设计依赖 OrchFS 的迁移机制将冷数据从 NVM 下刷到 SSD。如果迁移速度太慢，你的系统在 NVM 容量用尽时会阻塞推理。

```bash
# 写满 NVM 触发迁移，测量迁移吞吐
# 记录迁移延迟和对前台读写的影响
```

---

## 任务执行清单（按天排列）

### Day 1: 环境准备

- [ ] 运行 Step 0 的所有硬件检查命令
- [ ] 创建 `docs/hardware_inventory.md`
- [ ] 安装 vLLM: `pip install vllm`
- [ ] 安装 HuggingFace Transformers: `pip install transformers accelerate`
- [ ] 下载 LLaMA-2-7B 模型权重（或你能获取的最小 LLaMA 模型）
- [ ] 创建 `scripts/motivation/` 和 `results/` 目录
- [ ] 如果有 NVM 设备，验证 OrchFS 编译和基本运行

### Day 2~3: M1 实验

- [ ] 编写 `m1_kvcache_memory.py`（理论计算，30 分钟）
- [ ] 编写 `m1_real_memory.py`（真实测量，根据模型大小 1~3 小时）
- [ ] 编写 `m1_plot.py`（绘图，1 小时）
- [ ] 分析数据，确认 KV-Cache 显存瓶颈确实严重

### Day 4~6: M2 实验（最重要、最耗时）

- [ ] 编写 `m2_attention_analysis.py`（注意力分数收集）
- [ ] 准备长上下文输入文本（从 LongBench 数据集获取，或用长文章）
- [ ] 运行不同模型、不同输入的注意力分析
- [ ] 分析冷热分化程度：Top-K% 覆盖率统计
- [ ] 分析 block 粒度聚合后的覆盖率变化
- [ ] 分析跨层差异
- [ ] 编写 `m2_plot.py`（热力图 + CDF 图）

### Day 7~8: M3 实验

- [ ] 用 fio 测 SSD 理论峰值
- [ ] 编写 `m3_io_profiling.py`（不同方式的写带宽对比）
- [ ] 如有条件，profile vLLM swap 的 IO 模式
- [ ] 分析 POSIX IO vs Direct IO vs 理论峰值的差距

### Day 9~10: M4 + 存储基线

- [ ] 编写 `m4_tier_latency.py`（GPU↔DRAM 延迟测量）
- [ ] 如有 NVM：用 fio 测 NVM 延迟和带宽
- [ ] 如无 NVM：用 DRAM 模拟或查文献数据
- [ ] 如有 OrchFS 硬件环境：运行 S1~S3
- [ ] 汇总所有数据，绘制存储层延迟对比图

---

## 产出检查清单

完成 Step 1~2 后，你应该手上有以下东西：

| 产出 | 文件 | 用途 |
|------|------|------|
| 硬件清单 | `docs/hardware_inventory.md` | 了解自己的实验能力边界 |
| M1 数据 | `results/m1_*.csv` | 论文 Figure 2 |
| M2 数据 | `results/m2_*.csv`, `results/m2_*.pkl` | 论文 Figure 3，冷热分级参数设定依据 |
| M3 数据 | `results/m3_*.csv`, `results/m3_ssd_peak.txt` | 论文 Figure 4，OrchFS 优势论证 |
| M4 数据 | `results/m4_*.csv` | 论文 Figure 5，NVM 层必要性论证 |
| 实验脚本 | `scripts/motivation/m{1,2,3,4}_*.py` | 可复现的实验代码 |
| OrchFS 状态 | OrchFS 是否编译通过、能否正常使用 | 决定 Phase B 实现方案 |

**更重要的是，你应该能回答这四个问题**：

1. KV-Cache 的显存瓶颈有多严重？→ 数字来自 M1
2. 冷热分化有多明显？block 粒度管理是否可行？→ 数字来自 M2
3. 现有 IO 方案的效率有多低？OrchFS 能带来多大提升？→ 数字来自 M3
4. NVM 中间层能省多少延迟？→ 数字来自 M4

**如果某个问题的答案不如预期**（比如冷热分化不明显、NVM 延迟优势不大），你需要在这个阶段就调整设计方向，而不是写完系统再发现。

---

## 关于你提到的"找 bottleneck"——一个纠正

你说的"做测试找 bottleneck"这个思路是对的，但要明确：**你现在找的不是你自己系统的 bottleneck（因为系统还没写），而是找现有方案的 bottleneck**。

具体来说：
- **M1 找的是 GPU 显存的 bottleneck**：证明显存是瓶颈
- **M2 找的是 KV-Cache 管理策略的 bottleneck**：证明"一刀切"的管理浪费了大量资源
- **M3 找的是存储 IO 的 bottleneck**：证明 POSIX IO + 固定粒度没有用满 SSD 带宽
- **M4 找的是存储层次的 bottleneck**：证明缺少 NVM 中间层导致温数据换入延迟过大

这四个 bottleneck 对应你设计中的四个核心组件：分层存储管理、冷热分级、OrchFS 集成、NVM 中间层。每一个 bottleneck 都是你论文的一个 **Insight → Design Decision** 的支撑。

等这些 Motivation 实验做完，你就有充足的数据和信心开始写系统代码了。

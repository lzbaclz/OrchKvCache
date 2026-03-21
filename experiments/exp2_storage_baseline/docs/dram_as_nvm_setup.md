# DRAM 模拟 NVM (Persistent Memory) 配置指南

> 当前机器没有真实 NVM 硬件 (Intel Optane PM)。
> 本文档说明如何用 DRAM 模拟 NVM，生成 /dev/dax* 设备，以便编译和运行 OrchFS。

---

## 前置条件

- **需要 root 权限**
- **需要重启机器**
- 当前内存: 376 GB DRAM
- 建议模拟 NVM 大小: 32-64 GB (从 DRAM 中划出)

## 步骤

### 1. 修改 GRUB 启动参数

```bash
sudo vim /etc/default/grub

# 在 GRUB_CMDLINE_LINUX_DEFAULT 中添加 memmap 参数
# 格式: memmap=<size>!<start_offset>
# 例如: 从物理地址 320GB 处划出 32GB 作为 PMEM
GRUB_CMDLINE_LINUX_DEFAULT="... memmap=32G!320G"
```

**注意**: `start_offset` 必须大于系统使用的最大物理内存地址。
当前系统有 376GB DRAM，建议用 `memmap=32G!340G` 或类似配置。
最终可用 DRAM = 376 - 32 = 344 GB (足够)。

### 2. 更新 GRUB 并重启

```bash
sudo update-grub
sudo reboot
```

### 3. 验证 PMEM 区域

```bash
# 重启后检查
dmesg | grep -i pmem
ls /dev/pmem*
# 应该看到 /dev/pmem0
```

### 4. 安装 ndctl 和 daxctl

```bash
sudo apt-get update
sudo apt-get install -y ndctl daxctl
```

### 5. 创建 DAX 命名空间

```bash
# 查看 region
ndctl list --regions

# 创建 devdax 模式的命名空间
sudo ndctl create-namespace --mode=devdax --region=region0 --size=32G

# 验证
ls /dev/dax*
# 应该看到 /dev/dax0.0
```

### 6. 配置和编译 OrchFS

```bash
cd /home/lzq/codes/orchkv/OrchFS

# 使用 Samsung RAID0 Gen5 NVMe 作为 SSD 后端
# /dev/nvme1n1 或 /dev/nvme2n1 (RAID0 组件)
python config_parameter.py /dev/dax0.0 /dev/nvme1n1 4 16 32k

# 编译
mkdir -p build && cd build
cmake ..
make
```

### 7. 运行 OrchFS 测试

```bash
# 格式化 (KernelFS)
sudo ./kfs_mkfs

# 启动 KernelFS 后台线程
sudo ./kfs_main &

# 运行 LibFS 测试
cd ../test
make
./simple_test
```

---

## 性能预期 (DRAM 模拟 vs 真实 NVM)

| 指标 | DRAM 模拟 | 真实 Optane PM | 差异 |
|------|-----------|---------------|------|
| 随机读 4KB 延迟 | ~1.2 us | ~0.3 us | 4x |
| 顺序读带宽 | ~10 GB/s | ~6.6 GB/s | 1.5x |
| 顺序写带宽 | ~1.7 GB/s | ~2.3 GB/s | 0.7x |

> DRAM 模拟的延迟**低于**真实 NVM，带宽**高于**（读）或**接近**（写）。
> 因此 DRAM 模拟的 OrchFS 性能是**乐观上界**。
> 论文中应说明使用 DRAM 模拟，并引用 Optane 文献数据作为修正参考。

---

## 替代方案

如果无法重启机器，可以：

1. **手写 config/config.h**: 跳过 `config_parameter.py` 的设备检查，直接编译 OrchFS（可编译但无法运行）
2. **使用 emul_dram_bench**: 运行 `OrchFS/orchkvtest/benchmarks/emul_dram_bench` 做软件延迟注入基准（不需要 DAX 设备）
3. **使用文献数据**: 引用 Intel Optane 和 OrchFS 论文中的性能数据

---

## OrchFS config_parameter.py 参数说明

```
python config_parameter.py <dax_device> <ssd_device> <nvm_threads> <ssd_threads> <split_size>

参数:
  dax_device:   NVM DAX 设备路径, e.g. /dev/dax0.0
  ssd_device:   SSD 块设备路径, e.g. /dev/nvme1n1
  nvm_threads:  NVM IO 线程数 (建议 4)
  ssd_threads:  SSD IO 线程数 (建议 16)
  split_size:   SSD 拆分粒度 (必须为 32KB 倍数, 建议 32k)
```

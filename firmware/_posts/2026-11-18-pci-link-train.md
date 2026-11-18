---
title: "PCIE Link Train"
date: 2026-11-18
---

## Intel 平台

系统刚上电（Reset 释放后、BIOS 执行早期初始化代码时），**不能通过 BDF 操作 PCIe 设备**，原因有三：

### 1. PCIe Link 未训练
- 设备物理上处于 **Detect → Polling → Configuration** 状态机中，Link 还没 Up
- LTSSM 没达到 **L0** 状态，Transaction Layer 根本不可用
- 此时发送 Config Cycle 相当于对空槽操作，返回 `0xFFFFFFFF`（Master Abort）

### 2. PCI Bus Number 未分配
- Host Bridge（RC）内部的 **Bus Number Register**（BDF 中的 Bus 域）默认值为 0
- 只有 Bus 0（Root Bus）可达，其下的 Secondary/Subordinate Bus 都没配
- 没有 Bus Number → 无法路由到任何下游设备

### 3. MMIO/IO Decode 未使能
- 各 Root Port 的 **Memory/IO Base/Limit** 寄存器默认 Disabled
- 即使知道某个设备的 BDF，Host Bridge 也不会 Forward 任何 Config/Memory 事务过去

### 但早期代码确实需要操作设备——怎么办？

Intel 平台提供了 **BDF 之外的访问途径**：

| 方法 | 原理 | 典型用途 |
|------|------|----------|
| **RCRB** (Root Complex Register Block) | 固定物理地址（如 `FB000000h`）映射到 RC 内部寄存器空间 | 配 PCIe Root Port 的 Link Control、Target Link Speed 等 |
| **P2SB** (Primary to Sideband Bus) | 通过固定 MMIO 窗口转发 Sideband Message | 访问 PCH 内部设备（SPI、GPIO、UART） |
| **Sideband / PCODE** | 通过 Mailbox 寄存器发送 Sideband 命令 | 读 DIMM SPD、配置 System Agent 等 |
| **早期 PCI 枚举** | BIOS 先配 Host Bridge Bus Number + 发 Config Cycle 做 Discovery | 找到设备后配 MMIO Base/Limit，然后才能正常访问 |

### 典型流程

```
Power On
  │
  ├─ RCRB 访问 Root Port (配 Link Speed、Width)
  ├─ Sideband 读 SPD / PCODE 读 DIMM 信息
  │
  ├─ PCIe Link Training (LTSSM → L0)
  ├─ 分配 Bus Number (PCI Bus Walk)
  ├─ 分配 MMIO Base/Limit
  │
  └─ ✅ 此时才能通过 BDF (CF8/CFC 或 MMCFG) 正常操作设备
```

## ARM 平台

| 方面 | x86 (Intel) | ARM (常见 SoC) |
|------|-------------|----------------|
| **Config 访问机制** | CF8/CFC (legacy) + MMCFG (ECAM) | **只有 ECAM**（MMIO 映射），无 legacy IO |
| **RC 内部寄存器** | **RCRB**（固定 MMIO 窗口） | **SoC 自有 MMIO 寄存器**（在 SoC memory map 中） |
| **早期 Sideband** | P2SB / PCODE Mailbox | **APB / AHB 侧总线**、SCP (System Control Processor) Mailbox |
| **Firmware 架构** | UEFI PI (SEC/PEI/DXE) | **U-Boot / TF-A (Trusted Firmware-A) / UEFI** |
| **设备发现** | PCI Bus Walk (Enumeration) | 同左，但常配合 **Device Tree** 预置拓扑信息 |

## ARM 上电早期如何操作 PCIe 控制器

```
Power On
  │
  ├─ SoC Boot ROM → TF-A (BL2/BL31) → U-Boot / UEFI
  │
  ├─ 通过 SoC MMIO 寄存器配 Root Complex:
  │   - 配 PHY (SerDes) 参数
  │   - 配 Link Control (Target Speed, Width)
  │   - 这些寄存器在 SoC memory map 中，地址固定
  │
  ├─ 触发 PCIe Link Training (LTSSM)
  │
  ├─ 枚举 Bus (ECAM 访问)
  │
  ├─ 分配 MMIO 空间
  │
  └─ ✅ 此时才能通过 ECAM (BDF → MMIO offset) 操作设备
```

### 关键区别举例

**x86 上读 Root Port 的 Link Status：**
```c
// RCRB 方式
MmioRead32(RCRB_BASE + RootPortIndex * 0x1000 + LINK_STATUS_OFFSET)
```

**ARM 上读同样的信息：**
```c
// SoC 特定 MMIO 寄存器
MmioRead32(SOC_PCIE_CTRL_BASE + RootPortIndex * 0x10000 + LINK_STATUS_OFFSET)
// 或者通过 SCP 发 Mailbox 命令查询
```
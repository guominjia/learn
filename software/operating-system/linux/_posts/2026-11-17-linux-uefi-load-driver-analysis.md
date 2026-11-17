---
layout: post
title: "Linux VS Uefi load driver"
date: 2026-11-17
---

**整个自动匹配链条的起点，就是驱动里那几行 `pci_device_id` 表**。没有它，内核不知道这个驱动能管哪些硬件。

### 驱动源码中的声明

以 `igc` 驱动为例：

```c
// drivers/net/ethernet/intel/igc/igc_main.c

/* PCI Device ID Table */
static const struct pci_device_id igc_pci_tbl[] = {
    { PCI_VDEVICE(INTEL, 0x15F3), 0 },  // I225-V
    { PCI_VDEVICE(INTEL, 0x15F4), 0 },  // I225-I
    { PCI_VDEVICE(INTEL, 0x15F5), 0 },  // I226-V
    { PCI_VDEVICE(INTEL, 0x15F6), 0 },  // I226-I
    { PCI_VDEVICE(INTEL, 0x0DC6), 0 },  // I226-LM
    { }
};
MODULE_DEVICE_TABLE(pci, igc_pci_tbl);
```

### 编译时发生了什么

```
igc_pci_tbl[] 写在源码里
       │
       ▼
编译时，depmod 工具提取这个表
       │
       ▼
生成 modules.alias 条目:
alias pci:v00008086d00000DC6sv*sd*bc*sc*i* igc
       │
       ▼
内核发现 PCI 设备 8086:0DC6
       │
       ▼
request_module("pci:v00008086d00000DC6sv*sd*bc*sc*i*")
       │
       ▼
modprobe 匹配 modules.alias → 加载 igc.ko
```

### `MODULE_DEVICE_TABLE` 宏

这个宏是关键。它把 `pci_device_id` 表编译到 `.ko` 文件的 **ELF 段**中：

```bash
# 查看 .ko 文件里的设备 ID 表
modinfo igc
# filename:       /lib/modules/.../igc.ko
# alias:          pci:v00008086d00000DC6sv*sd*bc*sc*i*
# alias:          pci:v00008086d000015F3sv*sd*bc*sc*i*
# ...

# 或者直接看 ELF 段
hexdump -C igc.ko | grep ...  # __mod_pci_device_table 段
```

`depmod` 扫描所有 `.ko` 的这个段，汇总成 `modules.alias`。

### 不只是 PCI

同样的机制也用于其他总线：

```c
// USB
MODULE_DEVICE_TABLE(usb, rtl8150_table);
// → modules.alias: usb:v0BDAp8150d*dc*dsc*dp*ic*isc*ip*

// I2C
MODULE_DEVICE_TABLE(i2c, bmp280_i2c_id);
// → modules.alias: i2c:bmp280

// Platform (非可枚举设备)
MODULE_DEVICE_TABLE(platform, dwmac_driver_ids);
```

### 总结

```
驱动源码写 { Vendor ID, Device ID }
       ↓
MODULE_DEVICE_TABLE 编译进 .ko
       ↓
depmod 提取 → modules.alias
       ↓
内核发现硬件 → 查 alias → modprobe 加载
```
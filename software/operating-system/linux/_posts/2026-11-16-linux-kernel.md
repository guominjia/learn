---
layout: post
title: "Linux kernel"
date: 2026-11-16
categories: [linux]
---

## 内核代码流程

```
------ shim ----------------------------------------------
 efi_main -> init_grub -> should_use_fallback
------ grub2 ---------------------------------------------
 multiboot ->
   grub_cmd_multiboot
   grub_cmd_module
------ txt-stat ------------------------------------------
 main ->
   display_config_regs
   display_tboot_log
------ tboot ---------------------------------------------
 memlog_init
 memlog_write
 launch_kernel
------ kernel --------------------------------------------
 startup_32 -> startup_64 -> startup_64 ->
   paging_prepare
   cleanup_trampoline
   initialize_identity_maps
 startup_32 -> startup_64 -> startup_64 ->
   reset_early_page_tables
   kasan_early_init
   x86_64_start_reservations ->
     start_kernel ->
       smp_setup_processor_id
       boot_cpu_init
       pr_notice -> _printk -> vprintk -> vprintk_default -> vprintk_emit ->
         printk_delay
         vprintk_store
         preempt_disable
         console_unlock
       setup_arch ->
         x86_init_noop
         early_reserve_memory
         setup_initial_init_mm
         probe_roms
       [x86_init.paging.pagetable_init] paging_init ->
         sparse_init
         zone_sizes_init -> free_area_init -> subsection_map_init -> __nr_to_section
       arch_call_rest_init -> reset_init -> kernel_init -> kernel_init_freeable -> prepare_namespace -> mount_root -> mount_root_generic -> do_mount_root ->
         init_mount -> path_mount
         init_chdir
------ file ----------------------------------------------
 arch/x86/boot/compressed/head_64.S      |75|  startup_32               function
 arch/x86/boot/compressed/head_64.S      |336| startup_64               function
 arch/x86/boot/compressed/head_64.S      |606| trampoline_32bit_src     function
 arch/x86/boot/compressed/pgtable_64.c   |193| cleanup_trampoline       function
 arch/x86/boot/compressed/ident_map_64.c |110| initialize_identity_maps function
 arch/x86/kernel/head_64.S               |46|  startup_64               function
------ func ----------------------------------------------
 1. trampoline_32bit_src will **keep cr3** when 5-level page table been enabled and **set cr3 to trampoline_pgtable** when 5-level page table been disabled.
 2. cleanup_trampoline have defect that when 5-level page table been enabled, CR3 will not set to trampoline_pgtable so will not update it to desired pgtable
 3. initialize_identity_maps will invoke memset to clear memory. there are some problem in here when page table is incorrect. 
```

## 目录创建

### 重启丢失的（虚拟文件系统）

```
├── /proc/     procfs     — 进程、CPU、内存信息
├── /sys/      sysfs      — 设备拓扑、驱动绑定
├── /dev/      devtmpfs   — 设备节点 (内核创建，udev 管理)
├── /run/      tmpfs      — 运行时数据 (PID 文件、socket)
├── /tmp/      tmpfs      — 有些发行版用 tmpfs (如 Fedora)
└── /dev/shm/  tmpfs      — 共享内存
```

### 保留的（磁盘上）

```
/          → 根文件系统 (ext4/xfs/btrfs...)  — 整个根都在磁盘上
├── /bin/      磁盘
├── /boot/     磁盘  — 内核、initramfs
├── /dev/      磁盘  — 但 udev 启动时会清空重建 (devtmpfs)
├── /etc/      磁盘  — 配置文件
├── /home/     磁盘
├── /lib/      磁盘  — 内核模块、动态库
├── /opt/      磁盘
├── /root/     磁盘
├── /sbin/     磁盘
├── /srv/      磁盘
├── /tmp/      磁盘  — 但很多发行版启动时清空
├── /usr/      磁盘
└── /var/      磁盘
```

### 特殊情况

| 目录 | 存储 | 重启后 |
|---|---|---|
| `/dev/` | devtmpfs (内存) | **清空重建** — 内核重新枚举硬件，udev 重新创建设备节点 |
| `/run/` | tmpfs (内存) | **清空** — systemd 和各服务重新创建 |
| `/tmp/` | 磁盘 或 tmpfs | 看发行版 — Debian/Ubuntu 默认磁盘保留，Fedora/Arch 默认 tmpfs 清空 |

### 验证

```bash
df -h -t proc -t sysfs
# Filesystem     Type   Size  Used Avail Use% Mounted on
# proc           proc      0     0     0    - /proc
# sysfs          sysfs     0     0     0    - /sys
```

## 内核操作命令

- `lsmod/modprobe/moddep/
- `/lib/modules/$(uname -r)/`

## sysfs 创建流程

```
内核启动
  ↓
start_kernel() → vfs_caches_init() → sysfs_init()
  ↓
挂载 sysfs 到 /sys/ (rootfs 阶段)
  ↓
各子系统初始化时往里填内容:
  ├─ bus_register()     → /sys/bus/
  ├─ class_register()   → /sys/class/
  ├─ device_add()       → /sys/devices/
  ├─ driver_register()  → /sys/bus/<bus>/drivers/
  ├─ module_init()      → /sys/module/
  └─ ...
```

### sysfs 整体架构

```
/sys/
├── bus/          ← 总线类型 (PCI, USB, I2C, SPI...)
├── class/        ← 设备分类 (net, input, sound, tty...)
├── devices/      ← 所有设备的真实拓扑
├── firmware/     ← 固件信息 (ACPI, DMI, EFI...)
├── module/       ← 已加载的内核模块
├── power/        ← 电源管理
└── kernel/       ← 内核参数
```

**全部由内核在启动时创建**，具体是内核的 **kobject 框架** + **sysfs 文件系统** 驱动的结果。

### /sys/bus/ 的创建时机

```
内核启动
  → 初始化 drivers/base/bus.c
  → 注册 "pci" 总线类型 (pci_driver_init)
  → 注册 "usb" 总线类型 (usb_init)
  → 注册 "i2c" 总线类型 (i2c_init)
  → ...
  → 每个总线类型调用 bus_register()
      → 创建 /sys/bus/<name>/
          ├── devices/    ← 该总线上的所有设备 (符号链接)
          └── drivers/    ← 该总线的所有驱动
```

### /sys/bus/pci/ 举例

```
/sys/bus/pci/
├── devices/
│   ├── 0000:00:1f.6 → ../../../devices/pci0000:00/0000:00:1f.6
│   └── ...
└── drivers/
    ├── igc/
    │   ├── 0000:00:1f.6 → ../../../../devices/pci0000:00/0000:00:1f.6
    │   ├── bind
    │   ├── unbind
    │   ├── uevent
    │   └── ...
    └── e1000e/
        └── ...
```

- `devices/` 下的链接指向 `/sys/devices/` 里的真实设备
- `drivers/` 下列出所有注册到该总线的驱动
- 驱动目录下的 PCI 地址链接表示**这个驱动绑定了这个设备**

### 内核代码

```c
// drivers/base/bus.c
int bus_register(struct bus_type *bus)
{
    // 创建 /sys/bus/<name>/
    bus->p->subsys.kobj = kobject_create_and_add(bus->name, &bus_subsys);
    // 创建 /sys/bus/<name>/devices/
    bus->p->devices_kset = kset_create_and_add("devices", ...);
    // 创建 /sys/bus/<name>/drivers/
    bus->p->drivers_kset = kset_create_and_add("drivers", ...);
}
```

### 总结

| 路径 | 谁创建 | 何时 |
|---|---|---|
| `/sys/` | 内核 (sysfs) | 启动时挂载 |
| `/sys/bus/` | 内核 (bus_register) | 各总线子系统初始化时 |
| `/sys/class/` | 内核 (class_register) | 各设备类注册时 |
| `/sys/devices/` | 内核 (device_add) | 设备被发现时 |
| `/sys/class/net/eth0` | 内核 (register_netdevice) | 网卡驱动 probe 时 |

## procfs 创建流程

```
内核启动
  ↓
start_kernel() → vfs_caches_init() → proc_init()
  ↓
挂载 procfs 到 /proc/
  ↓
各子系统往里填内容:
  ├─ /proc/1/, /proc/2/...  ← 进程创建时内核添加
  ├─ /proc/cpuinfo          ← 内核读取 CPU 寄存器生成
  ├─ /proc/meminfo          ← 内核读取内存管理结构生成
  ├─ /proc/stat             ← 内核统计
  ├─ /proc/net/             ← 内核网络栈
  └─ /proc/sys/             ← 内核参数 (sysctl)
```

## 网络设备

Linux 中网络接口**不是设备文件**，不遵循 `open/read/write/ioctl` 这种字符/块设备模型。它们通过 **socket** 接口（`AF_INET`、`AF_PACKET`）和 **netlink** 来操作。

### 对比

```
硬盘:  /dev/sda       ← 块设备，在 /dev/ 下有节点
串口:  /dev/ttyS0     ← 字符设备，在 /dev/ 下有节点
声卡:  /dev/snd/...   ← 字符设备，在 /dev/ 下有节点
```

### 网卡在哪？

```
/sys/class/net/eth0     ← sysfs 暴露的设备信息
/proc/net/dev           ← 内核网络统计
```

操作网卡通过 **socket**：

```c
int sock = socket(AF_INET, SOCK_DGRAM, 0);  // 不是 open("/dev/eth0")
ioctl(sock, SIOCGIFADDR, &ifr);             // 查 IP
```

### 唯一例外

有些网络设备确实在 `/dev/` 下有节点，但那是**控制面**用的，不是数据面：

```
/dev/net/tun     ← TUN/TAP 虚拟网卡 (字符设备, 10:200)
/dev/ppp         ← PPP 协议 (字符设备, 108:0)
```

这些是给用户态程序创建虚拟网络接口用的，物理网卡（eth0、wlan0）不在 `/dev/` 下。

## 网络创建流程

### 创建链路

```
驱动 probe() 成功
    │
    ▼
alloc_etherdev() / alloc_netdev()    ← 分配 net_device 结构体
    │
    ▼
register_netdevice(netdev)           ← 注册到内核网络子系统
    │
    ├─ 内核创建 /sys/class/net/eth0/
    │     ├─ addr_assign_type
    │     ├─ address              ← MAC 地址
    │     ├─ carrier              ← 链路状态 (0/1)
    │     ├─ device → ../../pci0000:00/...  ← 指向 PCI 设备
    │     ├─ duplex
    │     ├─ speed
    │     ├─ mtu
    │     ├─ operstate
    │     ├─ statistics/          ← 收发统计
    │     ├─ subsystem → ../../class/net
    │     └─ uevent
    │
    ├─ 发送 uevent (ACTION=add, INTERFACE=eth0)
    │
    └─ 发送 RTM_NEWLINK (rtnetlink 消息)
```

### 代码层面

在 Linux 内核源码 `net/core/net-sysfs.c` 中：

```c
int register_netdevice(struct net_device *dev)
{
    // ... 初始化 ...
    ret = netdev_register_kobject(dev);  // ← 创建 /sys/class/net/<name>
    // ...
}
```

`netdev_register_kobject()` 调用 `device_add()`，后者通过 **kobject** 框架在 sysfs 中创建目录和属性文件。

### 所以整个链路是

```
硬件插上
  → PCI 枚举 (内核)
  → modprobe 加载驱动 (内核触发)
  → 驱动 probe() (驱动代码)
  → register_netdevice() (内核网络核心)
  → /sys/class/net/eth0 出现 (sysfs/kobject)
  → uevent 通知用户态 (udev)
  → NetworkManager/systemd-networkd 收到事件
```

`/sys/class/net/` 是内核暴露网络接口信息的**窗口**，用户态程序（ip、NetworkManager、ifconfig）都是通过读这个目录来发现网卡的。

### 确认网卡工作

```
# 1. 看网卡接口是否存在
ip link show

# 2. 看驱动绑定了哪些设备
ls /sys/bus/pci/drivers/igb/
# 输出: 0000:02:00.0  0000:02:00.1  bind  uevent  unbind
# 有 PCI 地址 = 有设备绑定

# 3. 看设备用了哪个驱动
lspci -k -s 02:00.0
# Kernel driver in use: igb

# 4. 看驱动统计信息
cat /sys/class/net/eth0/device/driver/module/refcnt
# 或者
lsmod | grep igb
```
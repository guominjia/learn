---
title: "机器鸭电路学习：IMU TO DXL"
categories: [hardware]
tags: [imu]
---

Linux 下有几个纯命令行的电路仿真器，最常用的是 **ngspice**。

| 工具 | 说明 |
|---|---|
| **ngspice** | 开源 SPICE，最常用，支持 batch 模式；KiCad 内置的仿真用的就是它 |
| **Xyce** | Sandia 实验室开发，支持并行，适合大电路 |
| **gnucap** | 轻量级仿真器 |
| **PySpice** | 用 Python 驱动 ngspice，方便扫参数和画图 |

## 用 ngspice 跑这个热插拔电路

安装：

```bash
sudo apt install ngspice
```

网表 `hotplug.cir`：

```spice
* VDD_BUS hot-plug: cable L + series R + C6
.param Rs=0.1
V1  in 0  PWL(0 0 1n 8.4)
L1  in n1 1u
R1  n1 vf {Rs}
C6  vf 0  10u

.control
foreach r 1m 0.05 0.1 0.3 0.63 1 2
  alterparam Rs=$r
  reset
  tran 1n 200u
  meas tran vpk MAX v(vf)
  meas tran tpk MAX_AT v(vf)
end
.endc
.end
```

运行：

```bash
ngspice -b hotplug.cir
```

几点说明：

- 0 Ω 电阻在 SPICE 里可能报错，所以第一个点用 `1m`（1 mΩ）来近似。
- `foreach` 加 `alterparam` 实现扫参数，`meas` 直接输出每个阻值下的峰值电压和峰值时间，可以和前面的 RK4 结果对照。
- 想看波形，可以用 `asciiplot v(vf)` 在终端里画；也可以用 `wrdata out.txt v(vf)` 导出数据，再拿 gnuplot 或 Python 画图。

## 加入 D1

把 TVS 也放进模型，就能看到钳位效果。最简单是用一个击穿电压约 13.3 V 的二极管来近似：

```spice
D1 0 vf DTVS
.model DTVS D(BV=13.3 IBV=1m RS=0.6)
```

`RS=0.6` 对应前面估算的动态电阻 $R_d$。更准确的做法是去厂家官网下载 SMF12A 的 SPICE 模型。

## 基本规则

每一行描述一个元件，格式是：

```
元件名  节点1  节点2  值
```

- **元件名的首字母决定类型**：`V` 电压源，`L` 电感，`R` 电阻，`C` 电容，`D` 二极管。
- **节点**就是电路里的连接点，名字可以随便起；`0` 固定代表 GND。
- 两个元件写了同一个节点名，就表示它们在这一点连在一起。

## 逐行对照

先给电路里的点起名字：

```
 in        n1        vf
  ●──L1────●───R1────●
  │                  │
  V1                 C6
  │                  │
  0 ─────────────────┘ (GND)
```

| 网表行 | 含义 |
|---|---|
| `* VDD_BUS hot-plug ...` | `*` 开头是注释 |
| `.param Rs=0.1` | 定义一个变量 Rs，默认 0.1 Ω |
| `V1 in 0 PWL(0 0 1n 8.4)` | 电压源 V1 接在 in 和 GND 之间。`PWL` 是分段线性波形：0 时刻 0 V，1 ns 时升到 8.4 V，用来模拟"插上电源" |
| `L1 in n1 1u` | 电感 L1 接在 in 和 n1 之间，1 µH，代表舵机线缆的电感 |
| `R1 n1 vf {Rs}` | 电阻 R1 接在 n1 和 vf 之间，阻值取变量 Rs，代表 F1 加走线的电阻 |
| `C6 vf 0 10u` | 电容 C6 接在 vf 和 GND 之间，10 µF |

`vf` 就是原理图里的 VDD_F，我们要观察的是这个点的电压 `v(vf)`。

## 控制部分（`.control` 到 `.endc`）

这一段是"仿真时要做什么"的脚本：

| 语句 | 含义 |
|---|---|
| `foreach r 1m 0.05 ...` | 循环，r 依次取后面列出的每个值 |
| `alterparam Rs=$r` | 把 Rs 改成当前的 r |
| `reset` | 让参数修改生效 |
| `tran 1n 200u` | 瞬态仿真：步长 1 ns，仿真 200 µs |
| `meas tran vpk MAX v(vf)` | 测量 vf 电压的最大值，结果命名为 vpk |
| `meas tran tpk MAX_AT v(vf)` | 测量最大值出现的时刻，结果命名为 tpk |
| `end` | 循环结束 |

## TVS 那两行

```spice
D1 0 vf DTVS
.model DTVS D(BV=13.3 IBV=1m RS=0.6)
```

- `D1 0 vf DTVS`：二极管 D1 的阳极接 GND，阴极接 vf，使用名为 DTVS 的模型。这和原理图上 D1 的方向一致：阴极朝上、阳极接地。
- `.model` 这一行定义 DTVS 模型的参数：击穿电压 `BV` = 13.3 V，`IBV` = 1 mA 是到达击穿电压时的电流，`RS` = 0.6 Ω 是串联电阻，对应前面估算的动态电阻 $R_d$。
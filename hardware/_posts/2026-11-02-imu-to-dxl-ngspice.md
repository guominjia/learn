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

## tran

`tran tstep tstop [tstart [tmax]] [uic]`：

| 参数 | 值 | 含义 |
|---|---|---|
| `tstep` | 10n | 输出/打印步长。开了 `.options interp` 时，结果会插值到这个间隔上 |
| `tstop` | 200u | 仿真到 200 µs 结束 |
| `tstart` | 0 | 从 0 开始保存数据。设成比如 50u，就会丢掉 50 µs 之前的输出，但仍然从 0 开始计算 |
| `tmax` | 20n | 内部求解器允许的最大步长 |

要点：

- ngspice 内部用**自适应步长**：波形变化快时自动缩小步长，平缓时放大，但不会超过 `tmax`。在 PWL 拐点（0 和 100 ns）这样的断点处，它也会自动加密。
- 不写 `tmax` 时，默认取 `min(tstep, (tstop-tstart)/50)`。
- 要写 `tmax`，就必须把 `tstart` 也写上（这里写 0 占位），因为参数是按位置识别的。

## 电路修正

电路本身根本不需要这么细：

$$f_0=\frac{1}{2\pi\sqrt{LC}}=\frac{1}{2\pi\sqrt{1\mu\cdot10\mu}}\approx 50.3\text{ kHz},\quad T\approx 19.9\ \mu s$$

步长 10–20 ns 时，一个振荡周期仍有约 1000 个点，测峰值完全够用。

**2. 1 ns 时就到 8.4 V 是 `v(in)`，不是 `v(vf)`**

`PWL(0 0 1n 8.4)` 本来就是一个近似理想的阶跃，用来模拟热插拔，所以 `v(in)` 在 1 ns 时等于 8.4 V 是正常的。

`v(vf)` 不会这样跳：电感限制了电流上升速度，$di/dt = 8.4/1\mu = 8.4\text{ A}/\mu s$，所以 1 ns 时 `v(vf)` 几乎还是 0。

另外，1 ns 的上升沿对结果影响很小。只要上升时间远小于 $T/4 \approx 5\ \mu s$，激励就相当于一个阶跃。改成更接近实际插拔的 100 ns，峰值基本不变。

## 修改后的脚本

```spice
* VDD_BUS hot-plug: cable L + series R + C6
.param Rs=0.1
V1  in 0  PWL(0 0 100n 8.4)
L1  in n1 1u
R1  n1 vf {Rs}
C6  vf 0  10u
.options interp

.control
foreach r 1m 0.05 0.1 0.3 0.63 1 2
  alterparam Rs=$r
  reset
  tran 10n 200u 0 20n
  meas tran vpk MAX v(vf)
  meas tran tpk MAX_AT v(vf)
  wrdata out_{$r}.txt v(vf)
end
.endc
.end
```

- `tran 10n 200u 0 20n`：输出步长 10 ns，内部最大步长 20 ns。ngspice 会自动在 PWL 拐点处加密，所以上升沿不会丢。
- `.options interp`：把输出插值到 `tstep` 网格上，每次输出约 2 万行，而不是记录所有内部计算点。
- 只关心峰值的话，`tran` 跑到 50 µs 就够了。

## 用解析解核对

这是一个二阶 RLC 电路，$Z_0=\sqrt{L/C}\approx0.316\ \Omega$，阻尼比 $\zeta = R_s/(2Z_0)=R_s/0.632$。所以 0.63 Ω 正好约等于临界阻尼。当 $\zeta<1$ 时：

$$V_{pk}=8.4\left(1+e^{-\pi\zeta/\sqrt{1-\zeta^2}}\right),\qquad t_{pk}=\frac{\pi\sqrt{LC}}{\sqrt{1-\zeta^2}}\approx\frac{9.93\ \mu s}{\sqrt{1-\zeta^2}}$$

| Rs | ζ | Vpk 估算 | tpk 估算 |
|---|---|---|---|
| 1 mΩ | 0.0016 | ≈16.8 V | ≈9.9 µs |
| 0.1 Ω | 0.158 | ≈13.5 V | ≈10.1 µs |
| 0.3 Ω | 0.474 | ≈9.9 V | ≈11.3 µs |
| ≥0.63 Ω | ≥1 | 无过冲，趋近 8.4 V | — |
---
layout: post
title: "锁相环：锁相与锁频的数学原理"
date: 2026-10-08
categories: [hardware, math]
tags: [pll, phase-locked-loop, frequency, control-theory]
---

# 锁相环：锁相与锁频的数学原理

锁相环（Phase-Locked Loop, PLL）是通信、时钟恢复、频率合成等领域最核心的模拟/混合信号电路之一。本文从数学角度推导 PLL 的**锁频（frequency lock）** 与 **锁相（phase lock）** 两个概念，厘清它们的区别与联系。

## 1. PLL 基本结构

一个经典 PLL 由三部分组成：

| 模块 | 功能 | 典型实现 |
|---|---|---|
| 鉴相器（Phase Detector, PD） | 比较输入信号与反馈信号的相位差，输出误差信号 | 模拟乘法器、XOR、PFD |
| 环路滤波器（Loop Filter, LF） | 滤除高频分量和噪声，决定环路动态特性 | RC 低通滤波器 |
| 压控振荡器（Voltage-Controlled Oscillator, VCO） | 根据控制电压调整输出频率 | LC 振荡器、环形振荡器 |

反馈通路将 VCO 输出回送至鉴相器，形成闭环。

## 2. 信号模型与相位定义

设输入参考信号为：

$$
v_{in}(t) = A \sin\!\bigl(\theta_i(t)\bigr) = A \sin\!\bigl(\omega_i t + \phi_i(t)\bigr)
$$

VCO 输出信号为：

$$
v_{vco}(t) = B \cos\!\bigl(\theta_o(t)\bigr) = B \cos\!\bigl(\omega_o t + \phi_o(t)\bigr)
$$

其中 $\theta_i(t)$ 和 $\theta_o(t)$ 是**瞬时总相位**（total phase），$\phi_i(t)$ 和 $\phi_o(t)$ 是相对于载波的**残余相位**（excess phase）。

定义**相位误差**：

$$
\theta_e(t) = \theta_i(t) - \theta_o(t)
$$

PLL 的目标就是使 $\theta_e(t) \to 0$（或常数）。

## 3. 鉴相器输出

对于模拟乘法器型鉴相器，将两路信号相乘后经过低通滤波：

$$
v_{in}(t) \cdot v_{vco}(t) = \frac{AB}{2}\Bigl[\sin\!\bigl(\theta_i + \theta_o\bigr) + \sin\!\bigl(\theta_i - \theta_o\bigr)\Bigr]
$$

环路滤波器滤除和频项 $\sin(\theta_i + \theta_o)$（高频），保留差频项：

$$
v_d(t) = \frac{AB}{2}\sin\!\bigl(\theta_e(t)\bigr)
$$

这就是鉴相器的**正弦特性**。当相位误差较小时（$\theta_e \ll 1$ rad），有线性化近似：

$$
v_d(t) \approx K_d \,\theta_e(t), \quad K_d = \frac{AB}{2}
$$

其中 $K_d$ 为**鉴相增益**（rad/V 或 V/rad，取决于具体定义）。

## 4. VCO 模型

VCO 的输出频率随控制电压 $v_c(t)$ 线性变化：

$$
\omega_o(t) = \omega_{free} + K_{vco}\, v_c(t)
$$

其中 $K_{vco}$ 为 VCO 增益（rad/s/V）。由于频率是相位的导数：

$$
\frac{d\theta_o(t)}{dt} = \omega_o(t) = \omega_{free} + K_{vco}\, v_c(t)
$$

对相位取拉普拉斯变换：

$$
\Theta_o(s) = \frac{K_{vco}}{s}\, V_c(s)
$$

VCO 在频域中相当于一个**积分器**（$1/s$），这是 PLL 能实现零稳态相位误差的关键。

## 5. 线性化环路方程

### 5.1 各模块的频域模型

在小信号线性化下（$\sin\theta_e \approx \theta_e$），三个模块在频域中的输入输出关系为：

**鉴相器（PD）**：输出误差电压正比于相位误差。

$$
V_d(s) = K_d \,\Theta_e(s) = K_d\bigl[\Theta_i(s) - \Theta_o(s)\bigr]
$$

**环路滤波器（LF）**：对误差电压做线性滤波。

$$
V_c(s) = F(s)\,V_d(s) = F(s)\,K_d\,\Theta_e(s)
$$

**VCO**：控制电压驱动频率变化，频率积分得到相位（第 4 节已推导）。

$$
\Theta_o(s) = \frac{K_{vco}}{s}\,V_c(s) = \frac{K_{vco}}{s}\,F(s)\,K_d\,\Theta_e(s)
$$

### 5.2 开环传递函数

将前向通路（PD → LF → VCO）的增益相乘，反馈通路为单位增益（直接反馈）：

$$
G(s) = \underbrace{K_d}_{\text{PD}} \cdot \underbrace{F(s)}_{\text{LF}} \cdot \underbrace{\frac{K_{vco}}{s}}_{\text{VCO}}
$$

### 5.3 闭环传递函数

将 $\Theta_o(s) = G(s)\,\Theta_e(s)$ 代入 $\Theta_e(s) = \Theta_i(s) - \Theta_o(s)$：

$$
\Theta_o(s) = G(s)\bigl[\Theta_i(s) - \Theta_o(s)\bigr]
$$

$$
\Theta_o(s)\bigl[1 + G(s)\bigr] = G(s)\,\Theta_i(s)
$$

$$
\boxed{H(s) = \frac{\Theta_o(s)}{\Theta_i(s)} = \frac{G(s)}{1 + G(s)} = \frac{K_d K_{vco} F(s)}{s + K_d K_{vco} F(s)}}
$$

### 5.4 相位误差传递函数

由 $\Theta_e(s) = \Theta_i(s) - \Theta_o(s) = \Theta_i(s) - H(s)\,\Theta_i(s)$：

$$
\Theta_e(s) = \bigl[1 - H(s)\bigr]\,\Theta_i(s) = \frac{1}{1 + G(s)}\,\Theta_i(s)
$$

$$
\boxed{E(s) = \frac{\Theta_e(s)}{\Theta_i(s)} = \frac{1}{1 + G(s)} = \frac{s}{s + K_d K_{vco} F(s)}}
$$

注意 $E(s) = 1 - H(s)$，且 $E(s)$ 的分子中有一个 $s$——这意味着 PLL 对低频输入（如恒定相位偏移）误差趋零，但对高频输入跟踪能力有限。

## 6. 锁频（Frequency Lock）

### 6.1 什么是锁频

**锁频**指 VCO 输出频率与输入频率一致，但可能存在恒定相位差：

$$
\omega_o = \omega_i, \quad \theta_e(t) = \text{const} \neq 0
$$

### 6.2 锁频的数学条件

考虑最简单的一阶 PLL（环路滤波器 $F(s) = 1$），相位误差方程为：

$$
\frac{d\theta_e}{dt} = \Delta\omega - K \sin\theta_e
$$

其中 $\Delta\omega = \omega_i - \omega_{free}$ 为初始频差，$K = K_d K_{vco}$ 为环路增益。

**锁频条件**：存在平衡点使 $d\theta_e/dt = 0$，即：

$$
\sin\theta_e^* = \frac{\Delta\omega}{K}
$$

有解的条件是：

$$
\boxed{|\Delta\omega| \le K}
$$

这个 $K$ 就是**锁频范围（lock-in range / pull-in range 的上限）**。当频差超过此值时，环路无法锁定，PLL 处于**失锁（unlock）**状态，鉴相器输出差频（beat note）。

### 6.3 锁频过程中的相位轨迹

当 $|\Delta\omega| < K$ 时，相位误差从初始值收敛到平衡点 $\theta_e^* = \arcsin(\Delta\omega / K)$。此时频率已锁定，但存在**稳态相位误差**：

$$
\theta_{e,ss} = \arcsin\!\left(\frac{\Delta\omega}{K}\right)
$$

这就是"锁频但不锁相"的状态。

## 7. 锁相（Phase Lock）

### 7.1 什么是锁相

**锁相**指不仅频率一致，而且相位误差趋近于零（或一个设计预期的固定值）：

$$
\omega_o = \omega_i, \quad \theta_e(t) \to 0
$$

### 7.2 实现零稳态相位误差的条件

从误差传递函数 $E(s)$ 出发，稳态相位误差由终值定理决定：

$$
\theta_{e,ss} = \lim_{s \to 0} s \cdot \Theta_e(s) = \lim_{s \to 0} \frac{s^2 \,\Theta_i(s)}{s + K_d K_{vco} F(s)}
$$

**情况 1：输入为相位阶跃** $\theta_i(t) = \Delta\theta \cdot u(t)$，即 $\Theta_i(s) = \Delta\theta / s$：

$$
\theta_{e,ss} = \lim_{s \to 0} \frac{s \cdot \Delta\theta}{s + K_d K_{vco} F(s)} = 0
$$

只要 $F(0) \neq 0$，稳态误差为零。**一阶 PLL 即可消除相位阶跃的稳态误差。**

**情况 2：输入为频率阶跃**（即相位斜坡）$\theta_i(t) = \Delta\omega \cdot t \cdot u(t)$，即 $\Theta_i(s) = \Delta\omega / s^2$：

$$
\theta_{e,ss} = \lim_{s \to 0} \frac{\Delta\omega}{s + K_d K_{vco} F(s)} = \frac{\Delta\omega}{K_d K_{vco} F(0)}
$$

- 若 $F(0) = 1$（一阶 PLL 或无源滤波器），稳态误差为 $\Delta\omega / (K_d K_{vco}) \neq 0$ — **锁频但不锁相**。
- 若 $F(0) \to \infty$（理想积分滤波器，如有源 PI 滤波器），稳态误差为 $0$ — **锁相**。

$$
\boxed{\text{锁相要求 } F(s) \text{ 在 } s=0 \text{ 处有极点（即包含积分器）}}
$$

### 7.3 二阶 PLL 与积分器

典型的二阶 PLL 使用有源 PI 滤波器：

$$
F(s) = \frac{1 + s\tau_2}{s\tau_1}
$$

此时 $F(0) \to \infty$，环路对频率阶跃的稳态相位误差为零。闭环传递函数变为：

$$
H(s) = \frac{K_d K_{vco}(1 + s\tau_2)/(s\tau_1)}{s + K_d K_{vco}(1 + s\tau_2)/(s\tau_1)} = \frac{2\zeta\omega_n\, s + \omega_n^2}{s^2 + 2\zeta\omega_n\, s + \omega_n^2}
$$

其中：

$$
\omega_n = \sqrt{\frac{K_d K_{vco}}{\tau_1}}, \quad \zeta = \frac{\tau_2}{2}\sqrt{\frac{K_d K_{vco}}{\tau_1}} = \frac{\omega_n \tau_2}{2}
$$

这是一个标准的二阶系统，$\omega_n$ 为**自然频率**，$\zeta$ 为**阻尼系数**。

## 8. 锁频 vs. 锁相对照

| 特性 | 锁频（Frequency Lock） | 锁相（Phase Lock） |
|---|---|---|
| 频率关系 | $\omega_o = \omega_i$ | $\omega_o = \omega_i$ |
| 相位误差 | $\theta_e = \text{const} \neq 0$ | $\theta_e \to 0$ |
| 环路滤波器要求 | $F(0) \neq 0$ 即可 | $F(0) \to \infty$（需积分器） |
| 对频率阶跃的稳态误差 | $\Delta\omega / (K_d K_{vco} F(0))$ | $0$ |
| PLL 阶数 | 一阶即可 | 至少二阶（含积分器） |
| 物理含义 | 频率跟踪，相位有偏差 | 频率与相位均跟踪 |

## 9. 锁定范围总结

PLL 有几个不同定义的频率范围，容易混淆：

| 范围 | 定义 | 近似公式 |
|---|---|---|
| **Hold-in range** | 环路锁定后能保持锁定的最大频差 | $\Delta\omega_H = K$（一阶） |
| **Pull-in range** | 失锁状态下能最终进入锁定的最大频差 | $\Delta\omega_P \le \Delta\omega_H$ |
| **Lock-in range** | 不经过周期跳跃（cycle slipping）直接锁定的最大频差 | $\Delta\omega_L \le \Delta\omega_P$ |

对于二阶 PLL（高增益），常用近似：

$$
\Delta\omega_L \approx 2\zeta\omega_n
$$

## 10. PLL 的主要作用

PLL 的核心能力是**让 VCO 的输出相位跟踪输入参考信号的相位**。这个能力派生出多种应用，"延迟单元"只是其中之一。

| 应用 | 原理 | 是否"延迟" |
|---|---|---|
| **频率合成** | 用分频器在反馈通路中设分频比 $N$，锁定后 $f_{out} = N \cdot f_{ref}$，产生输入参考频率的整数/小数倍频率 | 否 |
| **时钟恢复** | 从数据流中提取嵌入的时钟频率（如 NRZ 数据的频谱中有离散时钟分量），PLL 跟踪该分量重建同步时钟 | 否 |
| **时钟去抖动（jitter cleaning）** | 用窄带环路滤波器抑制输入时钟上的相位噪声/抖动，输出更干净的时钟 | 否 |
| **相位同步 / 相位对齐** | 使两路信号相位一致，用于数据采样、总线同步、DDR 时序对齐等 | 部分相关 |
| **FM 解调** | PLL 跟踪 FM 信号的瞬时频率，VCO 控制电压即为解调输出 | 否 |
| **延迟锁定环（DLL）** | 用压控延迟线（VCDL）代替 VCO，对齐时钟边沿，产生精确延迟 | **是** |

### PLL vs. DLL

"延迟单元"更准确地说是**延迟锁定环（Delay-Locked Loop, DLL）** 的角色。DLL 与 PLL 结构相似但关键区别在于：

| 特性 | PLL | DLL |
|---|---|---|
| 受控元件 | VCO（振荡器，产生周期信号） | VCDL（延迟线，不产生信号，只延迟输入） |
| 反馈量 | 相位（积分关系，$1/s$） | 延迟（比例关系） |
| 输出 | 可产生与输入不同频率的信号 | 只能延迟输入，不能产生新频率 |
| 稳态误差特性 | VCO 积分器使 PLL 对相位阶跃零稳态误差 | 无积分器，对相位阶跃有有限稳态误差 |
| 典型用途 | 频率合成、时钟恢复、FM 解调 | 时钟去偏斜（deskew）、相位插值、DDR 延迟控制 |

所以，**PLL 的主要作用是频率/相位跟踪与频率合成，不是延迟**。延迟对齐是 DLL 的核心功能；PLL 在某些场景（如相位对齐）也能产生延迟效果，但那只是锁相的副产物，不是 PLL 存在的主要目的。

## 11. 总结

- **锁频**是 PLL 的第一层目标：VCO 频率与输入频率一致，但允许恒定相位差。一阶 PLL 即可实现锁频，稳态相位误差为 $\Delta\omega / K$。
- **锁相**是 PLL 的更高目标：相位误差趋近于零。要求环路滤波器在 $s=0$ 处有极点（积分器），即至少二阶 PLL。
- VCO 本身是一个积分器（$1/s$），使 PLL 对**相位阶跃**天然零稳态误差；但对**频率阶跃**零稳态误差，还需要环路滤波器提供第二个积分器。
- 环路增益 $K = K_d K_{vco}$ 决定锁频范围；滤波器参数 $\tau_1, \tau_2$ 决定动态响应（$\omega_n, \zeta$）。

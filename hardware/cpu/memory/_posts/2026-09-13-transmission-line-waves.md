---
layout: post
title: "传输线上的 V+ 和 V-：输入输出之外的两个方向"
date: 2026-09-13
tags: [signal-integrity, transmission-line, waves, ddr5]
---

# 从电路节点到 DDR5：一步步理解高速信号反射（3）

传输线分析里最容易让人误会的一组符号是 $V^+$ 和 $V^-$。它们不是“输入电压”和“输出电压”，而是沿两个相反方向传播的电压波。

## 1. 先固定坐标

令源端在 $x=0$，负载端在 $x=l$：

```text
x = 0                                      x = l
源端  ───────────────────────────────────  负载端
          +x 方向  ─────────────────>
```

向 $+x$ 传播的波记为 $V^+$，向 $-x$ 传播的波记为 $V^-$。

## 2. 从分布参数得到波动方程

把一小段长度为 $\Delta x$ 的传输线展开，可以看到串联电阻 $R\Delta x$、串联电感 $L\Delta x$，以及并联电导 $G\Delta x$、并联电容 $C\Delta x$。这里的 $R,L,G,C$ 都是单位长度参数。

对这段微元应用 KVL 和 KCL，得到电报方程：

$$
\frac{\partial v}{\partial x}
=-Ri-L\frac{\partial i}{\partial t}
$$

$$
\frac{\partial i}{\partial x}
=-Gv-C\frac{\partial v}{\partial t}
$$

如果先忽略损耗，即 $R=G=0$，对上式继续消元，可以得到：

$$
\frac{\partial^2 v}{\partial x^2}
=LC\frac{\partial^2 v}{\partial t^2}
$$

电流也满足同样的方程。于是波速为：

$$
u=\frac{1}{\sqrt{LC}}
$$

这说明 $V^+$ 和 $V^-$ 不是人为命名的两个“输入输出电压”，而是波动方程的两个传播方向解。

## 3. 两个传播方向的解

$$
V^+(x,t)=f\left(t-\frac{x}{u}\right)
$$

向后的波可以写成：

$$
V^-(x,t)=g\left(t+\frac{x}{u}\right)
$$

第一个表达式的含义是波形向右移动，第二个表达式的含义是波形向左移动。

## 4. 真实电压是两个波的叠加

线上的实际电压不是只取某一个波，而是：

$$
V(x,t)=V^+(x,t)+V^-(x,t)
$$

对电流则有一个关键的符号差异：

$$
I(x,t)=\frac{V^+(x,t)-V^-(x,t)}{Z_0}
$$

原因不是反射波“电流消失了”，而是我们把电流正方向固定为 $+x$。反射波实际向 $-x$ 运动，所以它对这个参考方向的电流分量带负号。

## 5. 一个脉冲如何进入传输线

把驱动器先简化成一个 Thevenin 源：

```text
Vs ── Rs ── 传输线 Z0 ───────── 负载 ZL
```

在刚开始的瞬间，脉冲还没有到达负载。源端“看到”的不是远处的 $Z_L$，而是传输线的特性阻抗 $Z_0$。

因此，最初发出的电压波由 $V_s$、源阻抗 $R_s$ 和 $Z_0$ 共同决定：

$$
V^+=V_s\frac{Z_0}{R_s+Z_0}
$$

这个 $V^+$ 只描述刚进入线的入射波，不一定等于电源的 Thevenin 电压，也不一定等于负载端最后测到的电压。

## 6. 三个“电压”要分开

在后续文章中，始终区分：

| 符号 | 含义 |
|---|---|
| $V_s$ | 源的等效开路电压 |
| $V^+$ | 沿 $+x$ 传播的初始入射波 |
| $V_L$ | 负载端实际总电压，包含叠加结果 |

它们之间没有普遍的等号：

$$
V_s\neq V^+\neq V_L
$$

只有在具体源阻抗、负载阻抗和传播阶段确定后，才能计算三者关系。

## 7. 波阻抗和功率方向

对单独的向前波：

$$
\frac{V^+}{I^+}=Z_0
$$

对单独的向后波，如果仍采用 $+x$ 的电流参考方向：

$$
\frac{V^-}{I^-}=-Z_0
$$

这就是为什么不能只把 $V^+$ 和 $V^-$ 当作两个普通节点电压。它们带有传播方向和对应的电流方向信息。

## 8. 有损传输线的频域形式

真实传输线通常不能完全忽略 $R$ 和 $G$。在正弦稳态下，令电压和电流的相量为 $V(x)$ 和 $I(x)$，电报方程变为：

$$
\frac{dV}{dx}=-(R+j\omega L)I
$$

$$
\frac{dI}{dx}=-(G+j\omega C)V
$$

定义传播常数和特性阻抗：

$$
\gamma=\sqrt{(R+j\omega L)(G+j\omega C)}
$$

$$
Z_0=\sqrt{\frac{R+j\omega L}{G+j\omega C}}
$$

其中 $\gamma=\alpha+j\beta$，$\alpha$ 描述衰减，$\beta$ 描述相位变化。低损耗时，这些式子退化为前面使用的 $u\approx1/\sqrt{LC}$ 和 $Z_0\approx\sqrt{L/C}$。

## 本篇结论

```text
V+：向 +x 传播的波
V-：向 -x 传播的波
V ：某一点的实际总电压 = V+ + V-
I ：按参考方向计算的总电流 = (V+ - V-) / Z0
```

下一篇将把负载端边界条件写出来，说明为什么阻抗不连续时必须产生反射波。

## References

- [Characteristic impedance](https://en.wikipedia.org/wiki/Characteristic_impedance)：正向、反向波的电压电流比，以及反向波电流符号。
- [Transmission line](https://en.wikipedia.org/wiki/Transmission_line)：传输线的双向波和电报方程背景。

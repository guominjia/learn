---
title: "机器鸭电路学习：IMU TO DXL"
categories: [hardware]
tags: [imu]
---

RK4算法原理解释

## 算法

```python
import math

V, L, C = 8.4, 1e-6, 10e-6
w0 = 1 / math.sqrt(L * C)
Z0 = math.sqrt(L / C)
print(f"w0={w0:.0f} rad/s  f0={w0/2/math.pi/1e3:.1f} kHz  Z0={Z0:.3f} ohm  Rcrit={2*Z0:.3f} ohm\n")
print(f"{'R(ohm)':>7} {'zeta':>6} {'Vpk_formula':>12} {'Vpk_RK4':>8} {'tp_formula':>11} {'tp_RK4':>8}")

for R in [0.0, 0.05, 0.1, 0.3, 0.63, 1.0, 2.0]:
    z = R / (2 * Z0)
    if z < 1:
        tp_f = math.pi / (w0 * math.sqrt(1 - z * z))
        vpk_f = V * (1 + math.exp(-math.pi * z / math.sqrt(1 - z * z)))
    else:
        tp_f, vpk_f = float("nan"), V

    def f(v, i):
        return i / C, (V - R * i - v) / L

    dt, v, i, vmax, tmax = 1e-9, 0.0, 0.0, 0.0, 0.0
    for n in range(1, 200001):  # 200 us
        k1 = f(v, i)
        k2 = f(v + dt / 2 * k1[0], i + dt / 2 * k1[1])
        k3 = f(v + dt / 2 * k2[0], i + dt / 2 * k2[1])
        k4 = f(v + dt * k3[0], i + dt * k3[1])
        v += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        i += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if v > vmax + 1e-9:
            vmax, tmax = v, n * dt
    tp_s = f"{tmax*1e6:7.2f}us" if z < 1 else "      -"
    tp_fs = f"{tp_f*1e6:9.2f}us" if z < 1 else "        -"
    print(f"{R:7.2f} {z:6.3f} {vpk_f:11.2f}V {vmax:7.2f}V {tp_fs:>11} {tp_s:>8}")
```

## 1. 把问题化简成 $y' = \lambda y$

电路方程写成矩阵形式：

$$
\dot{\mathbf y} = A\mathbf y + \mathbf b,\qquad
\mathbf y = \begin{pmatrix} v \\ i \end{pmatrix},\quad
A = \begin{pmatrix} 0 & 1/C \\ -1/L & -R/L \end{pmatrix}
$$

令 $\mathbf u = \mathbf y - \mathbf y^*$，其中 $\mathbf y^*$ 是平衡点（$v = V$、$i = 0$），常数项 $\mathbf b$ 就消掉了，得到 $\dot{\mathbf u} = A\mathbf u$。

为了书写方便，下面把矩阵 $A$ 当成标量 $\lambda$ 来写。因为所有运算只涉及 $A$ 的多项式，结论对矩阵完全成立。

## 2. 精确解的一步

$y' = \lambda y$ 的解是指数函数。从 $t_n$ 走一步 $h$，令 $z = h\lambda$：

$$
y(t_n + h) = e^{z}\,y_n = \Big(1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24} + \frac{z^5}{120} + \cdots\Big)\,y_n
$$

数值方法的任务，就是在不知道 $e^z$ 的前提下，尽可能多地匹配这个级数的项。

## 3. 把四个 $k$ 逐个算出来

这里 $f(y) = \lambda y$，每一个 $k$ 都可以精确地写出来：

$$
\begin{aligned}
k_1 &= \lambda y_n \\
k_2 &= \lambda\big(y_n + \tfrac h2 k_1\big) = \lambda y_n\big(1 + \tfrac z2\big) \\
k_3 &= \lambda\big(y_n + \tfrac h2 k_2\big) = \lambda y_n\big(1 + \tfrac z2 + \tfrac{z^2}{4}\big) \\
k_4 &= \lambda\big(y_n + h k_3\big) = \lambda y_n\big(1 + z + \tfrac{z^2}{2} + \tfrac{z^3}{4}\big)
\end{aligned}
$$

规律很清楚：每多嵌套一层，$k$ 里就多出一个 $z$ 的更高次幂。$k_j$ 最高含 $z^{j-1}$。

## 4. 加权组合

$$
k_1 + 2k_2 + 2k_3 + k_4 = \lambda y_n\Big(6 + 3z + z^2 + \frac{z^3}{4}\Big)
$$

逐项合并的过程：

- 常数项：$1 + 2 + 2 + 1 = 6$
- $z$ 项：$2\cdot\tfrac12 + 2\cdot\tfrac12 + 1 = 3$
- $z^2$ 项：$2\cdot\tfrac14 + \tfrac12 = 1$
- $z^3$ 项：$\tfrac14$

代入 RK4 的更新公式，注意 $h\lambda = z$：

$$
y_{n+1} = y_n + \frac h6\cdot\lambda y_n\Big(6 + 3z + z^2 + \frac{z^3}{4}\Big)
= \Big(1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}\Big)\,y_n
$$

**结果正好是 $e^z$ 的泰勒多项式，截断到 4 次。**

## 5. 关于阶数与误差

| 使用的斜率 | 一步的放大因子 | 匹配到 |
|---|---|---|
| 只用 $k_1$（Euler） | $1 + z$ | 1 阶 |
| $k_1, k_2$（中点法：$y_n + hk_2$） | $1 + z + z^2/2$ | 2 阶 |
| $k_1 \sim k_4$，权重 1:2:2:1 | $1 + z + \cdots + z^4/24$ | 4 阶 |

1:2:2:1 这组权重，正是让各项系数恰好等于 $1/n!$ 的那组解。

**误差**：每步的局部误差等于级数中第一个没被匹配的项，$\dfrac{z^5}{120} = O(h^5)$；累计 $T/h$ 步后，全局误差是 $O(h^4)$。

## 6. 代入本电路的数值

$\|hA\|$ 的量级约为 $h\omega_0 = 10^{-9} \times 3.16\times10^5 \approx 3\times10^{-4}$。

每步的相对误差约为

$$
\frac{(3\times10^{-4})^5}{120} \approx 2\times10^{-20}
$$

这比双精度浮点的精度（约 $10^{-16}$）还小得多，所以 RK4 的结果与解析解吻合到小数点后两位。

## 补充：非线性方程

当 $f$ 是非线性函数时，思路完全相同：把 $k_2$、$k_3$、$k_4$ 在 $y_n$ 处做泰勒展开，展开式里会出现 $f$、$f'$、$f''$ 等各种组合（称为基本微分）。要求它们与精确解的展开逐项相等，会得到一组关于取点位置和权重的方程。经典 RK4 是这组方程的一个解，只是代数运算要繁琐得多。

## 0. 精确解的泰勒展开

逐次对 $t$ 求导，每次都用链式法则和 $y' = f$：

$$
\begin{aligned}
y' &= f \\
y'' &= f'f \\
y''' &= f''f^2 + f'^2 f \\
y'''' &= f'''f^3 + 4f'f''f^2 + f'^3 f
\end{aligned}
$$

由此：

$$
y_{n+1} = y_n + hf + \frac{h^2}{2}f'f + \frac{h^3}{6}\big(f''f^2 + f'^2 f\big) + \frac{h^4}{24}\big(f'''f^3 + 4f'f''f^2 + f'^3 f\big) + O(h^5)
$$

## 展开工具

对每个 $k_i = f(y_n + \delta_i)$，都用同一个展开式，保留到 $\delta^3$：

$$
f(y + \delta) = f + f'\delta + \tfrac12 f''\delta^2 + \tfrac16 f'''\delta^3 + O(\delta^4)
$$

## 1. $k_1$

$$
k_1 = f
$$

在展开点本身求值，是精确值。

## 2. $k_2$，偏移量 $\delta_2 = \tfrac h2 f$

$$
k_2 = f + \frac h2 f'f + \frac{h^2}{8}f''f^2 + \frac{h^3}{48}f'''f^3
$$

## 3. $k_3$，偏移量 $\delta_3 = \tfrac h2 k_2$

先算偏移量及其幂次，只保留到 $h^3$：

$$
\delta_3 = \frac h2 f + \frac{h^2}{4}f'f + \frac{h^3}{16}f''f^2,\qquad
\delta_3^2 = \frac{h^2}{4}f^2 + \frac{h^3}{4}f'f^2,\qquad
\delta_3^3 = \frac{h^3}{8}f^3
$$

代入展开工具，按 $h$ 的幂次合并：

$$
k_3 = f + \frac h2 f'f + h^2\Big(\frac14 f'^2 f + \frac18 f''f^2\Big) + h^3\Big(\frac{3}{16}f'f''f^2 + \frac{1}{48}f'''f^3\Big)
$$

其中 $h^3$ 项里的 $\tfrac{3}{16} = \tfrac{1}{16} + \tfrac18$，两部分分别来自 $f'\delta_3$ 和 $\tfrac12 f''\delta_3^2$。

## 4. $k_4$，偏移量 $\delta_4 = h\,k_3$

同样先算偏移量及其幂次：

$$
\delta_4 = hf + \frac{h^2}{2}f'f + h^3\Big(\frac14 f'^2 f + \frac18 f''f^2\Big),\qquad
\delta_4^2 = h^2 f^2 + h^3 f'f^2,\qquad
\delta_4^3 = h^3 f^3
$$

代入展开工具：

$$
k_4 = f + h f'f + h^2\Big(\frac12 f'^2 f + \frac12 f''f^2\Big) + h^3\Big(\frac14 f'^3 f + \frac58 f'f''f^2 + \frac16 f'''f^3\Big)
$$

其中 $h^3$ 项里的 $\tfrac58 = \tfrac18 + \tfrac12$，两部分同样分别来自 $f'\delta_4$ 和 $\tfrac12 f''\delta_4^2$。

## 5. 逐项验证 $\tfrac16(k_1 + 2k_2 + 2k_3 + k_4)$

RK4 给出 $y_{n+1} = y_n + h\cdot\tfrac16(k_1 + 2k_2 + 2k_3 + k_4)$。所以组合后 $h^{m}$ 项的系数，要和精确解中 $h^{m+1}$ 项的系数相等：

| 项 | RK4 组合后的系数 | 精确解要求 |
|---|---|---|
| $f$ | $\tfrac{1+2+2+1}{6} = 1$ | 1 ✓ |
| $h\,f'f$ | $\tfrac{2\cdot\frac12 + 2\cdot\frac12 + 1}{6} = \tfrac12$ | $\tfrac12$ ✓ |
| $h^2 f''f^2$ | $\tfrac{2\cdot\frac18 + 2\cdot\frac18 + \frac12}{6} = \tfrac16$ | $\tfrac16$ ✓ |
| $h^2 f'^2 f$ | $\tfrac{2\cdot\frac14 + \frac12}{6} = \tfrac16$ | $\tfrac16$ ✓ |
| $h^3 f'''f^3$ | $\tfrac{2\cdot\frac1{48} + 2\cdot\frac1{48} + \frac16}{6} = \tfrac1{24}$ | $\tfrac1{24}$ ✓ |
| $h^3 f'f''f^2$ | $\tfrac{2\cdot\frac3{16} + \frac58}{6} = \tfrac16$ | $\tfrac4{24} = \tfrac16$ ✓ |
| $h^3 f'^3 f$ | $\tfrac{\frac14}{6} = \tfrac1{24}$ | $\tfrac1{24}$ ✓ |

全部吻合，所以局部误差为 $O(h^5)$，RK4 是 4 阶方法。

从表中可以看到各个 $k$ 的分工：

- $f'^2 f$ 只出现在 $k_3$、$k_4$ 中。
- $f'^3 f$ 只出现在 $k_4$ 中。

这对应前面说的"每多嵌套一层，多带出一阶所需的导数组合"。

对于方程组（比如本电路的 $v$、$i$ 两个变量），$f'$、$f''$ 分别变成雅可比矩阵和二阶张量。这时 4 阶的导数组合会从 4 种增加到 8 种，但 RK4 的这组系数同样能全部匹配。
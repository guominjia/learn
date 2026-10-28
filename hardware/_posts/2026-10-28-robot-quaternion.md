---
title: "机械臂姿态中的四元数"
categories: [hardware]
tags: [quaternion]
---

## 四元数乘法的推导

### 起点：复数乘法的推广

复数 $a + bi$ 可以写成旋转：乘 $e^{i\theta} = \cos\theta + i\sin\theta$ 就是转 $\theta$。四元数把这个想法从二维推广到三维，加两个虚部 $j, k$。

### 定义规则

Hamilton 的定义：

$$i^2 = j^2 = k^2 = ijk = -1$$

由此推出：

$$ij = k, \quad jk = i, \quad ki = j$$
$$ji = -k, \quad kj = -i, \quad ik = -j$$

注意 $ij \neq ji$，不交换。

### 两个四元数相乘

设 $q_a = (w_a, x_a, y_a, z_a) = w_a + x_a i + y_a j + z_a k$，$q_b = (w_b, x_b, y_b, z_b)$ 同理。展开乘法：

$$q_a \cdot q_b = (w_a + x_a i + y_a j + z_a k)(w_b + x_b i + y_b j + z_b k)$$

逐项展开，用上面的规则化简。以 $i \times i$ 项为例：$x_a i \cdot x_b i = x_a x_b i^2 = -x_a x_b$。以 $i \times j$ 项为例：$x_a i \cdot y_b j = x_a y_b k$。全部列出来：

| × | $w_b$ | $x_b i$ | $y_b j$ | $z_b k$ |
|---|---|---|---|---|
| $w_a$ | $w_a w_b$ | $w_a x_b\, i$ | $w_a y_b\, j$ | $w_a z_b\, k$ |
| $x_a i$ | $x_a w_b\, i$ | $-x_a x_b$ | $x_a y_b\, k$ | $-x_a z_b\, j$ |
| $y_a j$ | $y_a w_b\, j$ | $-y_a x_b\, k$ | $-y_a y_b$ | $y_a z_b\, i$ |
| $z_a k$ | $z_a w_b\, k$ | $z_a x_b\, j$ | $-z_a y_b\, i$ | $-z_a z_b$ |

按 $1, i, j, k$ 分组：

$$w = w_a w_b - x_a x_b - y_a y_b - z_a z_b$$
$$x = w_a x_b + x_a w_b + y_a z_b - z_a y_b$$
$$y = w_a y_b - x_a z_b + y_a w_b + z_a x_b$$
$$z = w_a z_b + x_a y_b - y_a x_b + z_a w_b$$

这就是代码里的 `_qmul`：

```python
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,   # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,   # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,   # z
    )
```

### 几何意义

$q_a \cdot q_b$ 表示**先做 $q_b$ 的旋转，再做 $q_a$ 的旋转**。代码里：

```python
q = _qmul((w, qx, qy, qz), (MOUNT[0], -MOUNT[1], -MOUNT[2], -MOUNT[3]))
```

先做 MOUNT 共轭（掰正 90°），再做 IMU 报的姿态，得到躯干坐标系下的姿态。

### 和矩阵的关系

如果把四元数转成 $3 \times 3$ 旋转矩阵 $R$，那么 $q_a \cdot q_b$ 对应 $R_a R_b$，共轭 $q^*$ 对应 $R^T$。四元数乘法就是矩阵乘法的紧凑写法，4 个数代替 9 个数。
不是流形。你可能是看到"三个角度不能独立求导"想到了流形上的切空间，但这里的关系更直接。

## 角速度是什么

角速度 $\omega = (\omega_x, \omega_y, \omega_z)$ 就是一个**普通的三维向量**，住在 $\mathbb{R}^3$ 里，不是流形上的切向量。它有三个分量，每个分量就是绕对应轴的旋转速率，单位是 rad/s 或 dps。

## 和流形的区别

流形出现在**姿态空间**那一侧。所有可能的姿态组成一个空间——SO(3) 或单位四元数球面 $S^3$，这些确实是流形：

- 姿态 $q$ 住在流形上（单位四元数，$w^2+x^2+y^2+z^2=1$）
- 姿态的变化率 $\dot{q}$ 是流形上 $q$ 点的切向量
- 角速度 $\omega$ 是切空间的坐标表示

关系是：

$$\dot{q} = \tfrac{1}{2}\,q \otimes (0,\omega_x,\omega_y,\omega_z)$$

左边 $\dot{q}$ 在流形的切空间里，右边把 $\omega$ 用四元数乘法"搬"上去。$\omega$ 本身只是 $\mathbb{R}^3$ 里的一个向量，没有流形约束。

## 为什么不能说"与轴的夹角的变化"

不是因为流形，而是因为**欧拉角的导数不等于角速度分量**。设俯仰 $\phi$、横滚 $\theta$、偏航 $\psi$：

$$\begin{pmatrix}\dot\phi \\ \dot\theta \\ \dot\psi\end{pmatrix} \neq \begin{pmatrix}\omega_x \\ \omega_y \\ \omega_z\end{pmatrix}$$

只有当旋转只绕一根轴时两者才相等。一般情形下它们的关系是一个和当前姿态有关的矩阵，而且某些姿态下这个矩阵会退化（万向锁）。四元数和角速度的关系 $\dot{q} = \frac{1}{2}q\otimes\omega$ 没有这个问题，任何姿态下都成立。

## 一句话总结

- **角速度**：$\mathbb{R}^3$ 里的普通向量，陀螺仪直接测的。
- **姿态（四元数）**：流形 $S^3$ 上的点。
- **两者的关系**：$\dot{q} = \frac{1}{2}q\otimes\omega$，把 $\mathbb{R}^3$ 的向量映射到流形的切空间。

[wechat_link]: https://mp.weixin.qq.com/s/Pox_toesAeJmL4jyVLhfgg
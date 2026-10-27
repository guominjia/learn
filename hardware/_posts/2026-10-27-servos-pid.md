---
title: 舵机PID
categories: [hardware]
tags: [servos, pid]
---

**PID** = **P**roportional-**I**ntegral-**D**erivative（**比例-积分-微分**）控制器，是工业和嵌入式系统中最常用的**反馈控制算法**。

## 核心思想

你有一个**目标值**（setpoint）和一个**实际测量值**，两者之差叫**误差** $e(t)$。PID 根据这个误差算出一个控制量去驱动执行器，让实际值逼近目标值：

$$u(t) = K_p \cdot e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \cdot \frac{de(t)}{dt}$$

## 三个项分别干什么

| 项 | 名称 | 作用 | 比喻 |
|----|------|------|------|
| **P** | 比例 | 误差越大，输出越大，快速拉向目标 | 你偏离目标越远，越用力拉回来 |
| **I** | 积分 | 累积历史误差，消除**稳态误差** | 你一直差一点点够不到，慢慢加力补上 |
| **D** | 微分 | 预测误差变化趋势，抑制**超调**和振荡 | 快到目标了提前刹车，别冲过头 |

## 一个直观例子：开车到目标位置

```
目标位置 = 100m,  当前位置 = 0m

P: 误差 100m 很大 → 全力加速        （快速接近）
I: 如果一直差 2m 到不了 → 累积补力   （消除稳态误差）
D: 快到 100m 时速度还很快 → 提前刹车  （防止超调）
```

## 三个参数怎么调

- $K_p$ 太大 → 振荡、超调
- $K_p$ 太小 → 响应慢
- $K_i$ 太大 → 积分饱和、超调振荡
- $K_i$ 太小 → 有稳态误差
- $K_d$ 太大 → 对噪声敏感、抖动
- $K_d$ 太小 → 超调严重

常用调参方法：**试凑法**、**Ziegler-Nichols 法**、**自动整定**。

## 常见应用场景

| 场景 | 控制目标 |
|------|---------|
| 无人机悬停 | 高度/姿态 |
| 平衡车 | 倾角 |
| 恒温箱/烤箱 | 温度 |
| 电机调速 | 转速 |
| 机械臂关节 | 角度 |
| 工业伺服 | 位置/速度/电流三环 |

## 代码示例（离散形式）

```c
// 离散 PID（嵌入式常用形式）
float pid_update(PID *pid, float setpoint, float measured) {
    float error = setpoint - measured;

    pid->integral += error * pid->dt;          // 积分累加
    float derivative = (error - pid->prev_error) / pid->dt;  // 微分

    float output = pid->Kp * error
                 + pid->Ki * pid->integral
                 + pid->Kd * derivative;

    pid->prev_error = error;
    return output;
}
```
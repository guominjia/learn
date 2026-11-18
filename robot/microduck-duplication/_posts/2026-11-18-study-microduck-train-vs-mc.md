---
title: "机器鸭复刻：与蒙特卡洛方法对比"
date: 2026-11-18
categories: [robot, microduck]
---

两边用的都是同一种"从后往前累加回报"的循环。区别在于：机器鸭的 PPO 每段只采 24 步，不等整局结束，剩下的部分交给 critic 估计（自举）。另外，它要优化的是策略，不只是评估一个固定策略的价值。

**同一个从后往前的循环**

```python
# 文中 First-Visit MC（γ=1，整局跑完）
G = 0
for t in reversed(range(T)):
    G = reward[t] + G
    returns[state].append(G); V[state] = mean(returns[state])

# rsl_rl/algorithms/ppo.py:187 compute_returns（GAE，只跑 24 步）
last_values = critic(obs)                                       # 第 24 步之后不再采样，用 critic 自举
for step in reversed(range(24)):
    next_values = last_values if step == 23 else values[step+1]
    delta = r[step] + (1-done)*γ*next_values - values[step]     # TD 误差
    advantage = delta + (1-done)*γ*λ*advantage
    returns[step] = advantage + values[step]
```

λ=1 并且采完整局时，$\sum_k \gamma^k\delta_{t+k}$ 逐项抵消后正好等于 $G_t - V(S_t)$，此时 `returns` 就是 MC 回报。λ=0 时就是 TD(0)。这里用的是 $\gamma=0.99$，$\lambda=0.95$（`microduck_velocity_env_cfg.py:937-938`），介于两者之间，更偏向 MC。

**逐项对比**

| | 文中 MC（Blackjack） | 机器鸭 PPO |
|---|---|---|
| 任务 | 预测：估计固定策略的 $V^\pi$ | 控制：同时更新策略 $\pi_\theta$ 和价值 $V_w$ |
| 状态 | 离散、有限，可以建表 `V[state]` | 61 维连续向量，只能用 MLP 近似 |
| 动作 | 离散（要牌/停牌） | 14 维连续，从高斯分布采样 |
| 采样范围 | 必须等整局结束 | 每个环境每轮 24 步（`num_steps_per_env=24`），约 0.48 s，然后自举 |
| 回报目标 | 完整的 $G_t$ | GAE(λ) 回报，最后一段用 $V(s_{24})$ 补上 |
| 自举 | 无 | 有：截断处、超时处（`ppo.py:175`，`r += γ·V`） |
| 首次访问/每次访问 | 关键问题，也是文中那个 bug 的来源 | 不存在。连续状态几乎不会精确重复，每个样本单独作为一条回归数据 |
| 怎么"平均" | 样本均值 `sum/len`，依据大数定律 | critic 对 `returns` 做 MSE 回归，用梯度下降近似求平均 |
| 方差和偏差 | 无偏，高方差 | 有偏（受 critic 误差影响），方差低 |
| 样本量来源 | 跑很多局 | 很多环境并行，一次 `step` 就得到一批样本 |
| 策略改进 | 无（预测问题） | 用 $A_t$ 加权的策略梯度，再加 PPO clip（`clip_param=0.2`） |
| 同策略 | 是 | 是：每轮数据更新 5 个 epoch 后丢弃 |

**为什么机器鸭不用纯 MC**

1. 一局最长要好几秒（上千个控制步），整局回报 $G_t$ 的方差太大。这正是文中第 2 节说的"long episodes → high variance → TD"。
2. 大部分回合是因为超时被截断，并没有真正终止，超时后的回报本来就采不到，只能用 $V$ 补。
3. 状态是连续的，没法用 `Returns(s)` 表把同一个状态的回报收集起来求平均。

两者共同的前提就是文中第 1 节说的：转移分布写不出来，但可以采样。对机器鸭来说，负责采样的就是 MuJoCo 加 BAM 舵机模型。
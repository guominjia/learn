---
title: "机器鸭学习：模型"
categories: [hardware]
tags: [rl]
---

## 模型

```
Actor:  obs_normalizer -> Linear(61, 512) -> ... -> Linear(128, 14)
Action: joint_pos, Dimension 14
```

| log Index | Name | Shape | 向量索引 | 对应你文章里的 |
|---|---|---|---|---|
| 0 | `base_ang_vel` | 3 | 0–2 | 陀螺仪角速度 |
| 1 | `projected_gravity` | 3 | 3–5 | 投影重力 |
| 2 | `joint_pos` | 14 | 6–19 | 关节位置（减 home） |
| 3 | `joint_vel` | 14 | 20–33 | 关节速度 |
| 4 | `actions` | 14 | 34–47 | 上一帧动作 |
| 5 | `command` | 3 | 48–50 | vx, vy, vyaw |
| 6 | `head_command` | 4 | 51–54 | neck_pitch, head_pitch, head_yaw, head_roll |
| 7 | `body_command` | 6 | 55–60 | body x, y, z, roll, pitch, yaw |

- 输入 61 维，先经过 `EmpiricalNormalization`。导出 ONNX 时这一步会写进计算图，实机运行时直接输入原始的 61 个数就行。
- 输出 14 维是 14 个舵机的目标关节角，与 `[BamActuator] ... joints=14` 一致。
- 14 个关节的顺序按 README：0–4 左腿（hip_yaw, hip_roll, hip_pitch, knee, ankle），5–8 颈/头，9–13 右腿。`joint_pos`、`joint_vel`、`actions` 三段都用这个顺序。

Critic 只在训练时使用，能看到真机上拿不到的特权信息。它包含 actor 的全部 61 维，另加 15 维：

| 额外项 | 维数 |
|---|---|
| `base_lin_vel` | 3 |
| `foot_height` | 2 |
| `foot_air_time` | 2 |
| `foot_contact` | 2 |
| `foot_contact_forces` | 6 |

$61 + 15 = 76$。拼接顺序和 actor 不同：`base_lin_vel` 排在最前面，足部的 4 项插在 `command` 和 `head_command` 之间。

### Critic 在训练里做什么

PPO 更新 actor 时，要判断每个动作比"平均水平"好多少，这个量叫优势值（advantage）：

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad\text{（GAE 会把多步的这种差值加权求和）}
$$

其中 $V(s)$ 就是 critic 的输出，表示"从这个状态出发，预期还能拿多少总回报"。critic 最后一层是 `Linear(128, 1)`，只输出一个数。

- $A_t > 0$：这次的动作比预期好，提高它的概率。
- $A_t < 0$：这次的动作比预期差，降低它的概率。

如果没有 critic，就只能用整局的原始回报作为学习信号，方差很大，训练会慢很多，也更不稳定。

### 为什么给它特权信息

这种做法叫非对称 actor-critic（asymmetric actor-critic）：

| | Actor | Critic |
|---|---|---|
| 输入 | 61 维，只含真机传感器能得到的量 | 76 维，多了线速度、足部接触/受力/离地高度等仿真里才有的真值 |
| 用途 | 输出动作，部署到真机 | 估计价值，只用于训练 |
| 部署 | 导出 ONNX | 训练结束后丢弃 |

critic 的价值估计越准，advantage 就越准，actor 得到的梯度也越可靠。critic 从不参与真机推理，所以它看什么都可以，只要仿真里能算出来。actor 的输入则被限制在真机能拿到的量上，保证训练出来的策略能直接部署。

以 `base_lin_vel` 为例：真机上 IMU 测不准躯干线速度，所以 actor 拿不到它。但奖励项 `track_linear_velocity` 正是按线速度计算的。critic 能直接看到这个真值，才能准确判断"当前状态价值高不高"。actor 只能从关节和 IMU 的历史中间接推断速度。

## PPO 说明

PPO 是在线（on-policy）强化学习，训练数据是 MuJoCo Warp 在 GPU 上边训练边仿真出来的，每轮用完就丢弃。

```
Number of environments | 4096
Total steps: 98304          # 第 0 轮结束时
Collection time: 1.662s     # 仿真采样
Learning time: 0.139s       # 用这批数据更新网络
```

- 每轮迭代是 4096 个环境并行，每个环境走 $98304 / 4096 = 24$ 步（每步 0.02 s，50 Hz）。
- 每一步记录 `(actor obs 61, critic obs 76, action 14, reward, done)`，存进 rollout buffer。
- 用这批数据做几轮 PPO 更新后清空 buffer，再用新的策略重新采样。因为是 on-policy，旧数据不能复用。
- 一轮约 1.8 s，其中约 92% 的时间花在仿真采样上，网络更新只占一小部分。

按 50000 轮算，总共约 49 亿步，相当于真机走约 3 万小时，所以只能靠仿真。

### "真实数据"进入训练的途径

真机数据不直接拿来训练，而是体现在仿真模型里：

| 来源 | 作用 |
|---|---|
| MJCF 机器人模型（`robot_walk.xml`，Onshape 导出） | 质量、惯量、几何 |
| BAM M6 舵机模型（`kt=0.3660 R=2.8114 ...`） | 参数从真实 XL330 舵机实测数据辨识而来，是缩小 sim2real 差距的关键 |
| 域随机化（`randomize_*`、`foot_friction`、`vin range=(6.5, 8.2)` 等） | 让策略对模型误差足够鲁棒 |

### 磁盘上保存的东西

保存的是模型和训练日志，不是轨迹数据：

- `logs/rsl_rl/velocity/*_velocity/`：checkpoint `model_*.pt` 和 `git/microduck_rl.diff`
- `wandb/offline-run-20261103_182857-d07qlvqm/`：训练曲线（离线模式）
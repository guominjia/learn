---
title: "机器鸭学习：训练数据"
categories: [hardware]
tags: [rl]
---

**数据从哪来**

- 任务是 `Mjlab-Velocity-Flat-MicroDuck`：让 Microduck 双足机器人（约 25 cm 高、800 g）在平地上按给定的速度指令行走。
- 仿真器是 MuJoCo Warp，跑在 GPU 上，同时并行 **4096 个机器人**（`num-envs 4096`）。
- 每一轮迭代的流程：
  1. 4096 个机器人各自用当前策略走约 24 步（50 Hz）。
  2. 物理引擎算出走完之后的状态。
  3. 奖励函数给每个动作打分。
  4. PPO 根据这些分数更新网络。下一轮用新策略重新采集数据，旧数据直接丢掉。
- 日志里的 `Total steps: 4783276032` 就是从恢复训练算起已经仿真的约 48 亿步（48657 轮 × 4096 × 24 步，正好对得上），这就是它的"数据集"。

**为什么没有示范数据也能学会走路**

什么叫"走得好"，是由**奖励函数**定义的。这些奖励项就在日志里：

| 奖励项 | 作用 |
|---|---|
| `track_linear_velocity` / `track_angular_velocity` | 按指令的速度和转向移动（主要目标） |
| `upright` | 保持身体直立，不摔倒 |
| `air_time`、`foot_swing_height`、`foot_clearance` | 鼓励抬脚迈步，而不是拖着脚滑 |
| `foot_slip` | 惩罚脚打滑 |
| `pose` | 关节姿态别偏离默认站姿太远 |
| `action_rate_l2` | 惩罚动作抖动，让动作平滑，方便迁移到真机 |
| `head_pose_tracking` | 头部跟随姿态指令 |
| `self_collisions`、`dof_pos_limits` | 惩罚自碰撞和关节超限 |

摔倒（`fell_over`）会让这一局提前结束，后面的奖励就拿不到了，所以策略会自然学会保持平衡。

一开始策略基本是随机抖动，几乎马上摔倒。偶尔某个动作让机器人往前挪了一点又没倒，就会得到更高的奖励。PPO 会提高这类动作的概率，几十亿步之后就逐渐收敛成一种步态。**所以这个步态不是模仿出来的，而是在奖励函数约束下自己"进化"出来的。**

这个任务（`Mjlab-Velocity-Flat-MicroDuck`）每一步的总奖励是各项加权求和：

$$r_t = \Delta t \cdot \sum_i w_i \, f_i(s_t, a_t), \qquad \Delta t = 0.02\text{ s（50 Hz）}$$

乘 $\Delta t$ 这一步在 `mjlab/managers/reward_manager.py` 里，默认 `scale_by_dt=True`。

代码分在三个地方：
- **奖励函数的实现**：`.venv/.../mjlab/tasks/velocity/mdp/rewards.py`
- **MicroDuck 对权重和参数的覆盖**：`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`（约第 275–355 行、第 710–745 行）
- **头部相关的自定义奖励**：`src/mjlab_microduck/tasks/mdp.py`

## 正向奖励（高斯型，取值 0~1，越接近目标越高）

| 项 | 权重 | 公式 |
|---|---|---|
| `track_linear_velocity` | 2.0 | $\exp\!\big(-(\lVert v^{cmd}_{xy}-v_{xy}\rVert^2 + v_z^2)/0.1\big)$，机体坐标系，同时惩罚上下颠簸 |
| `track_angular_velocity` | 2.0 | $\exp\!\big(-((\omega^{cmd}_z-\omega_z)^2+\omega_x^2+\omega_y^2)/0.5\big)$ |
| `upright` | 2.0 | 把重力方向投影到 `trunk_base` 坐标系得到 $g_b$，奖励为 $\exp(-(g_{b,x}^2+g_{b,y}^2)/0.05)$。躯干越歪，$g_b$ 的水平分量越大 |
| `pose` | 1.0 | $\exp\!\big(-\text{mean}_j\,(q_j-q_j^{0})^2/\sigma_j^2\big)$。只算腿部关节（排除 neck/head/passive）；指令接近 0 时用较严的 `std_standing`，行走时用较宽的 `std_walking` |
| `air_time` | 3.0 | 当前腾空时间落在 $(0.125, 0.3)$ s 内的脚的个数（0、1 或 2）。只在速度指令 > 0.01 时生效，**这一项是促使机器人迈步的主要信号** |
| `head_pose_tracking` | 2.0 | 4 个颈/头关节分别计算 $\exp(-(e_j/0.5)^2)$ 再取平均。$e_j$ 是测量角度（舵机读数加上 backlash 被动关节）减去默认位置再减去指令 |

## 惩罚项（权重为负）

| 项 | 权重 | 公式 |
|---|---|---|
| `action_rate_l2` | −0.1 → −1.0（课程学习，到 iter 1500） | $\sum (a_t - a_{t-1})^2$，让动作平滑 |
| `foot_clearance` | −2.0 | $\sum_{feet} \lvert h - 0.02\rvert \cdot \lVert v^{foot}_{xy}\rVert$：脚在水平移动时，离地高度应接近 2 cm |
| `foot_swing_height` | −0.25 | 在每次落地那一刻计算 $\sum (h_{peak}/0.02 - 1)^2$，要求每步最高点约 2 cm |
| `foot_slip` | −0.1 | 着地的脚 $\sum \lVert v^{foot}_{xy}\rVert^2$。权重故意设得很小，因为这个机器人转身时脚要原地拧转 |
| `body_ang_vel` | −0.05 | 躯干 $\omega_x^2+\omega_y^2$，抑制前后、左右摇晃（不惩罚转向） |
| `angular_momentum` | −0.02 | 全身角动量 $\lVert L\rVert^2$ |
| `self_collisions` | −1.0 | 自碰撞力超过 10 N 的子步数 |
| `dof_pos_limits` | −1.0 | 关节超出软限位的量 |
| `head_pose_bias` | 0 → 3.0（课程学习） | $-\text{mean}_j\,\lvert \text{EMA}_{1s}(e_j)\rvert$：只惩罚头部的**稳态下垂**，走路时头部的晃动在 1 秒平均里会相互抵消 |

`body_pose_tracking` 的权重为 0（关闭）；`soft_landing` 已被 `pop` 删除。

整体思路是：正向的高斯项定义"要做什么"（按指令移动、保持直立、迈步），惩罚项定义"别怎么做"（抖动、拖脚、打滑、自碰撞）。具体步态没有写进代码，是 PPO 在这些约束之间自己找出来的。
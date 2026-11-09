---
title: "机器鸭复刻：强化学习"
date: 2026-11-09
categories: [robot, microduck]
---

```
策略 MLP（actor）
  输入 61 维 obs ──► 输出 14 维 action
                        │ JointPositionAction: target = default_joint_pos + 1.0 * action
                        ▼                      (scale=1.0, microduck_velocity_env_cfg.py:273)
                    BamActuator.compute(target, q, qdot) ──► torque ──► MuJoCo
```

**61 维输入**：actor 观测组按键的顺序拼接。基础模板在 `.venv/.../mjlab/tasks/velocity/velocity_env_cfg.py:76`，然后由 `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:531-701` 修改：

| 顺序 | term | 维度 | 来源 / 修改 |
|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 陀螺仪，延迟 0–1 步，噪声 ±0.03 |
| 2 | `projected_gravity` | 3 | `USE_PROJECTED_GRAVITY=True`（:53），噪声 ±0.01 |
| 3 | `joint_pos` | 14 | 排除 `passive_*`（16→14），加编码器偏置 DR |
| 4 | `joint_vel` | 14 | 固定延迟 1 步，噪声 ±0.25 |
| 5 | `actions` | 14 | 上一步动作（`last_action`） |
| 6 | `command`（twist） | 3 | vx, vy, wz |
| 7 | `head_command` | 4 | neck_pitch, head_pitch, head_yaw, head_roll |
| 8 | `body_command` | 6 | x, y, z, roll, pitch, yaw |

$3+3+14+14+14+3+4+6 = 61$

已从 actor 删掉的项：`base_lin_vel`（只有 critic 能看到，属于特权信息）和 `height_scan`（actor 和 critic 都删了）。

critic 在同样这些项的基础上还会看到 `base_lin_vel`、`foot_height`、`foot_air_time`、`foot_contact`、`foot_contact_forces`，并且没有噪声。

**1. 关节和舵机**

[robot_walk.xml] 里有 14 个铰链关节（hinge joint），按 XML 顺序排列：

```
left_hip_yaw  left_hip_roll  left_hip_pitch  left_knee  left_ankle     # 0-4
neck_pitch    head_pitch     head_yaw        head_roll                 # 5-8
right_hip_yaw right_hip_roll right_hip_pitch right_knee right_ankle    # 9-13
```

`microduck_hd1910_env_cfg.py` 用 `FeetechBamActuatorCfg(target_names_expr=(r"^(?!passive_).*",))` 给每个非 `passive_*` 的关节挂一个 BAM 舵机，所以一共 14 个执行器。这个 XML 里没有 `passive_*` 关节；正则排除的是其他 XML 变体里的颌部连杆关节。

**2. 从执行器数到网络输出维度**

```
JointPositionActionCfg(actuator_names=(".*",))                 mjlab/tasks/velocity/velocity_env_cfg.py:165
└─ BaseAction.__init__
     target_ids = entity.find_joints_by_actuator_names(".*")   mjlab/envs/mdp/actions/actions.py:53,110
       └─ 只保留有执行器的关节，按 XML 顺序排列                    mjlab/entity/entity.py:506-514
     _action_dim = len(target_ids)          # = 14             actions.py:58
ActionManager.total_action_dim = Σ term.action_dim             mjlab/managers/action_manager.py:106
RslRlVecEnvWrapper.num_actions = total_action_dim              mjlab/rl/vecenv_wrapper.py:21
PPO: actor = MLPModel(obs, obs_groups, "actor", env.num_actions, ...)   rsl_rl/algorithms/ppo.py:493
└─ MLP(61 → hidden_dims → output_dim)                          rsl_rl/models/mlp_model.py:76
```

**3. 每个控制步，动作怎么变成舵机目标**

```
action[14] = policy(obs[61])
process_actions: processed = action * scale(1.0) + default_joint_pos       actions.py:157, :219
apply_actions:   target = processed - encoder_bias                          actions.py:222-223
                 entity.set_joint_position_target(target, joint_ids)        actions.py:224
└─ BamActuator.compute(target, q, qdot) → 力矩 → MuJoCo
```

`action[i]` 对应上面列表里第 i 个关节，表示相对 HOME 姿态（`default_joint_pos`）的角度偏移，单位 rad。部署到真机时，舵机 ID 也要按这个顺序映射。

**rsl_rl 采样循环**（`rsl_rl/runners/on_policy_runner.py`）：

```
:85   actions = alg.act(obs)                         # 61 → 14
:87   obs, rewards, dones, extras = env.step(actions)
:94   alg.process_env_step(obs, rewards, dones, ...) # 存进 RolloutStorage
:105  alg.compute_returns(obs)                       # 采样够 num_steps_per_env 步后
:108  alg.update()                                   # PPO 更新 actor/critic
```

**一次 `env.step()`**（`mjlab/envs/manager_based_rl_env.py:372`）：

`timestep=0.005`，`decimation=4`（`velocity_env_cfg.py:446,451`）。所以策略频率是 50 Hz，物理仿真是 200 Hz，一个 action 会保持 4 个物理子步。

```
action_manager.process_action(action)          # raw*1.0 + default_joint_pos
for _ in range(4):                             # 4 个物理子步
    action_manager.apply_action()              # set_joint_position_target(target - encoder_bias)
    scene.write_data_to_sim()
      └─ entity._apply_actuator_controls()     # entity.py:1235
           cmd = act.get_command(data)         # 目标 + 当前 q、qdot
           cmd = act.apply_delay(cmd)          # 3–6 步延迟
           data.write_ctrl(act.compute(cmd))   # BAM: target → 力矩，并写入摩擦
    sim.step()                                 # MuJoCo 积分：力矩 → qacc → qvel → qpos
    scene.update(dt)                           # 推进执行器内部状态
reset_buf  = termination_manager.compute()     # 摔倒、超时
reward_buf = reward_manager.compute(dt)        # 速度跟踪、头部姿态跟踪……
_reset_idx(...)                                # 已结束的环境当场重置
sim.forward()                                  # 用新的 qpos/qvel 刷新运动学
command_manager.compute()                      # 到时间就重新采样 twist、head、body 命令
event_manager.apply("interval")                # 推力等扰动
sim.sense()
obs_buf = observation_manager.compute()        # 生成下一步的 61 维输入
return obs_buf, reward_buf, terminated, time_outs, extras
```

**下一步观测里怎么体现这次运动**（`mjlab/envs/mdp/observations.py`）：

| obs | 读的是什么 |
|---|---|
| `joint_pos` | `asset.data.joint_pos_biased[:, ids] - default_joint_pos`（`:51`），也就是 MuJoCo 的 `qpos` 加上编码器偏置 |
| `joint_vel` | `asset.data.joint_vel[:, ids] - default_joint_vel`（`:64`），也就是 `qvel`，延迟 1 步 |
| `base_ang_vel`、`projected_gravity` | 根节点角速度、机身坐标系下的重力方向，相当于 IMU，延迟 0–1 步 |
| `actions` | `action_manager.action`（`:80`），即刚执行的那 14 维 |

要点：

- **状态反馈**：关节动了，`qpos`/`qvel` 和机身姿态都会变，下一步观测里的 `joint_pos`、`joint_vel`、IMU 这几项直接反映出来。
- **奖励反馈**：每步算出一个标量 `reward`，存进 rollout 后用来算回报和优势，再由 PPO 更新网络。
- **终止反馈**：摔倒或超时会产生 `dones`，截断这一段的回报，并把该环境重置。

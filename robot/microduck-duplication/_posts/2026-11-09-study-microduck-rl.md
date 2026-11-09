---
title: "机器鸭复刻：强化学习"
date: 2026-11-09
categories: [robot, microduck]
---

下面路径里 `.venv/.../bam` 指 `.venv/lib/python3.12/site-packages/bam`，`src/...` 都在 `src/mjlab_microduck/` 下。

## 类继承关系

```
mjlab Actuator
└─ bam.mjlab.BamActuator                     .venv/.../bam/mjlab.py:142
   └─ FrictionDRBamActuator                  src/.../actuator/friction_dr_bam.py:28
      ├─ BacklashEncoderBamActuator          （同文件 :64，给 Backlash 任务用）
      └─ FeetechBamActuator                  src/.../actuator/feetech_bam.py:16   ← HD1910 用这个

bam.actuator.VoltageControlledActuator
└─ STS3215Actuator                           .venv/.../bam/feetech/actuator.py:22
```

## 1910_m6.json 是怎么加载的

```
microduck_hd1910_env_cfg.py
  HD1910_M6_JSON = .../robot/hd1910/1910_m6.json
  FeetechBamActuatorCfg(json_path=HD1910_M6_JSON, kp_fw=5.0,
      vin_range=(7.4,8.0), vin_drop_gain_range=(0,0.2), vin_min=7.0,
      delay_min_lag=3, delay_max_lag=6, target_names_expr=(r"^(?!passive_).*",))
  -> 只替换 deepcopy 出来的 robot 实体的 actuators，XL330 那份配置不受影响

BamActuatorCfg.__post_init__  -> _resolve_json_path(json_path)          mjlab.py:120
BamActuator.__init__          -> self._bam_model = load_model(path)     mjlab.py:184
  bam/model.py load_model_from_dict(data):
    model = models["m6"]()                          # 摩擦模型 M6
    model.set_actuator(actuators["sts3215"]())      # bam/actuators.py:30 -> STS3215Actuator(Pendulum)
    model.load_parameters_from_dict(data)           # 只读模型里已注册的参数名，其他键忽略
  if cfg.kp_fw: actuator.kp = 5.0                   # 覆盖 STS3215 默认 kp=32
edit_spec   -> force_limit = max(vin_range) * kt / R = 8.0*0.6238/4.9102 ≈ 1.02 Nm   mjlab.py:254
               写入 MuJoCo 的 forcerange；dof_frictionloss 清零，摩擦改由 BAM 计算
initialize  -> 按环境采样 vin_tensor 和 vin_drop_gain，打印 log 里那行 [BamActuator]
```

## STS3215 控制器（`bam/feetech/actuator.py`）

- 固定常数：`vin=7.4`，`error_gain=0.166`，`max_pwm=0.97`，默认 `kp=32`（训练时被覆盖成 5）。
- `compute_control` 的计算：

```
q_target_smooth = clamp(q_target, q_target_smooth ± max_velocity*dt)   # 目标限速
duty = (q_target_smooth - q) * kp * 0.166 * error_gain_ratio
duty = clamp(duty, ±0.97)
return vin * duty
```

- 代入 JSON 的数值：
  - `max_velocity=100`，`dt=0.005` 时每步最多能变 0.5 rad，等于不限速。
  - `error_gain_ratio=1.3217`，所以每 rad 误差的占空比是 $5 \times 0.166 \times 1.3217 \approx 1.10$，误差约 0.88 rad 就饱和，和参数文档里的数字一致。
- `q_offset` 会从 JSON 加载，但 `compute_control` 里没有用到它。

## 每步 `BamActuator.compute`（mjlab.py:580）

```
vin = vin_tensor - vin_drop_gain * Σ|prev_motor_torque|，再 clamp 到 >= vin_min
act.vin = vin;  act.kp = base_kp * kp_scale
control = act.compute_control(target, pos, vel*kd_scale, dt)      # 上面 STS3215 的公式
torque  = act.compute_torque(control, ...)                        # kt、R、反电动势
frictionloss = _compute_friction_budget(prev_tau, external_tau, stribeck)
  └─ FrictionDRBamActuator 覆盖：base * friction_scale            # 每个环境单独的摩擦随机化
_write_frictions(frictionloss, friction_viscous)                  # 写入 MuJoCo，由约束求解器处理静摩擦
return torque
```

## FeetechBamActuator 补的那部分

锁定版本的 BAM 只在 `load_log()` 里创建 `q_target_smooth`，mjlab 训练不会调用它，所以首次 reset 会报 `AttributeError`。修复方式：

```
initialize: q_target_smooth = zeros(N,J); _goal_reset_pending = ones(N,1)
reset(env_ids): 只把这些环境的 _goal_reset_pending 置为 True
compute: q_target_smooth = where(pending, cmd.pos, q_target_smooth)  # 用当前关节角作为限速起点
         pending 清零，然后调用 super().compute()
```

## 和 XL330 的配置对比（`robot/microduck_constants.py`）

| | XL330（原版） | HD1910 |
|---|---|---|
| Cfg 类 | `FrictionDRBamActuatorCfg` | `FeetechBamActuatorCfg` |
| 参数来源 | BAM 自带的 `motor_name="xl330", model="m6"` | `json_path=1910_m6.json`（Luwu） |
| `kp_fw` | 200 | 5 |
| `vin_range` / `vin_min` | (6.5, 8.2) / 6.0 | (7.4, 8.0) / 7.0 |
| 压降增益、延迟 | 0–0.2，3–6 步 | 相同 |

`kp_fw` 两边是各自固件的增益单位：Dynamixel 用 200，飞特用 5，两个数不能直接比大小。
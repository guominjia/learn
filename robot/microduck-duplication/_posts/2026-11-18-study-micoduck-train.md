---
title: "机器鸭复刻：学习训练代码"
date: 2026-11-18
categories: [robot, microduck]
---

# `balance_parameters.py` — 完整逐行解释

## 常量定义（1-19行）

```python
"""Portable single-leg skill contract and evaluation, without simulation imports."""
```
模块文档：这是一个**纯 Python 计算模块**，不依赖仿真环境（MuJoCo/Isaac），所以可以在离线数据分析中复用。

```python
import math
from collections.abc import Mapping, Sequence
```
`math` 用于角度转换和数值检查；`Mapping`/`Sequence` 用于类型提示。

```python
BALANCE_TASK_ID = "Mjlab-Balance-Flat-MicroDuck-HD1910"
BALANCE_EXPERIMENT = "microduck_hd1910_balance"
```
训练任务的唯一标识符和实验目录名。

```python
BALANCE_DEFAULTS = {"support_side": "left", "lift_height_m": 0.01, "hold_seconds": 3.0}
```
技能参数的默认值：支撑脚为左脚、抬脚高度 1cm、hold 时长 3 秒。

```python
INIT_SECONDS = 1.0
LIFT_SECONDS = 1.0
RETURN_SECONDS = 1.0
SETTLE_SECONDS = 0.5
```
四个阶段的时长：初始化 1s、抬脚 1s、hold（由参数决定）、返回 1s + 稳定 0.5s。

```python
MAX_TILT_RAD = math.radians(25.0)       # 最大允许倾斜 25°
MIN_SUPPORT_FORCE_N = 0.5               # 支撑脚最小触地力 0.5N
MAX_ROOT_SPEED_MPS = 0.08               # 躯干最大速度 0.08m/s
MAX_SUPPORT_SPEED_MPS = 0.04            # 支撑脚最大速度 0.04m/s
MIN_COM_MARGIN_M = -0.002               # CoM 到支撑脚多边形边缘的最小余量 -2mm
LAND_CLEARANCE_M = 0.003                # 落地判定高度阈值 3mm
```
这些是评估时判定"是否站稳"的硬约束门限。

---

## `validate_balance_parameters`（22-35行）

```python
def validate_balance_parameters(parameters: Mapping) -> dict:
```
验证技能参数字典是否合法。

```python
    if not isinstance(parameters, Mapping) or set(parameters) != set(BALANCE_DEFAULTS):
        raise ValueError("Balance requires exactly support_side, lift_height_m, hold_seconds")
```
必须恰好包含三个 key，不能多不能少。

```python
    if parameters["support_side"] not in ("left", "right"):
        raise ValueError("support_side must be left or right")
```
支撑脚只能是 left 或 right。

```python
    result = {"support_side": parameters["support_side"]}
```
把支撑脚原样放入结果。

```python
    for key, lower, upper in (("lift_height_m", .005, .02), ("hold_seconds", 1., 5.)):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number")
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be between {lower} and {upper}")
        result[key] = float(value)
```
对两个数值参数分别检查：抬脚高度 [0.5cm, 2cm]、hold 时长 [1s, 5s]。排除 bool（Python 中 `bool` 是 `int` 的子类）。

```python
    return result
```

---

## `balance_cycle_seconds`（38-39行）

```python
def balance_cycle_seconds(parameters: Mapping) -> float:
    return validate_balance_parameters(parameters)["hold_seconds"] + 3.5
```
一个完整 cycle 的时长 = hold 时长 + 3.5 秒（init 1s + lift 1s + return 1s + settle 0.5s）。

---

## `apply_balance_parameters`（42-53行）

```python
def apply_balance_parameters(cfg, parameters: Mapping) -> dict:
    """Apply a saved skill before constructing train/play/export environments."""
```
把保存的技能参数应用到环境配置上。

```python
    result = validate_balance_parameters(parameters)
    command = cfg.commands.get("twist")
    if getattr(command, "contract_version", None) != "microduck.single_leg_balance.v1":
        raise ValueError("Balance parameters require the registered balance task")
```
验证参数，检查 command 是否注册了正确的 contract 版本。

```python
    for name, value in result.items():
        setattr(command, name, value)
```
把参数写入 command 对象（`support_side`、`lift_height_m`、`hold_seconds`）。

```python
    cfg.episode_length_s = 9.0
    return result
```
固定 episode 最大时长 9 秒（hold 最长 5s + 3.5s + 余量）。

---

## `balance_command_contract`（56-72行）

```python
def balance_command_contract(parameters: Mapping) -> dict:
```
生成一个描述该技能合约的字典，用于序列化和跨环境传递。

```python
    skill = validate_balance_parameters(parameters)
    return {
        "version": "microduck.single_leg_balance.v1",
        "actor_observation_size": 61, "action_size": 14, "control_hz": 50,
```
合约版本、观测维度 61、动作维度 14、控制频率 50Hz。

```python
        "twist_slots": ["support_sign (+1 left, -1 right)",
                        "scheduled swing sole clearance (metres)",
                        "phase / 3 (0 init, 1 lift, 2 hold, 3 return)"],
```
command 的三个通道含义：支撑脚符号、摆动脚目标高度、归一化相位。

```python
        "phase_seconds": {"init": 1.0, "lift": 1.0,
                          "hold": skill["hold_seconds"], "return": 1.5},
        "return_includes_settle_seconds": .5,
        "cycle_seconds": balance_cycle_seconds(skill),
```
各阶段时长，return 包含 0.5s settle。

```python
        "head_body_commands": "Near-zero pose command padding; deployment uses zeros.",
        "runtime_adapter_required": True,
        "runtime_note": "A matching phase scheduler is required; this is not a velocity policy.",
        "hardware_validated": False, "skill": skill,
    }
```
头部命令用零填充；部署时需要配套的相位调度器；尚未在硬件上验证。

---

## `aggregate_balance_trial`（75-175行）— 核心评估函数

### 文档和初始化（75-96行）

```python
def aggregate_balance_trial(samples: Sequence[Mapping], parameters: Mapping) -> dict:
    """Score measured 50 Hz samples from one reset through hold AND return.
    ...
    """
```
对一个 episode 的所有样本进行评分，判断是否成功完成了 hold + return。

```python
    skill = validate_balance_parameters(parameters)
    hold_start, hold_end = 2.0, 2.0 + skill["hold_seconds"]
    settle_start, cycle_end = hold_end + 1.0, hold_end + 1.5
```
计算时间窗口：hold 阶段从 2.0s 到 `2.0 + hold_seconds`；settle 从 hold 结束后 1.0s 开始，持续 0.5s。

```python
    hold_run = best_hold = return_run = best_return = 0.0
```
当前连续 hold 时长、最佳 hold 时长、当前连续 settle 时长、最佳 settle 时长。

```python
    failed_in = {key: 0.0 for key in ("tilt", "force", "swing_contact", "nonfoot",
                                      "root_speed", "support_speed", "com_margin",
                                      "clearance", "finite")}
```
**v2 新增**：记录每个失败原因被触发的次数。

```python
    previous = 0.0
    finite_all, terminated, chronological = True, False, True
    max_tilt, hold_heights, contact_steps = 0.0, [], 0
```
`finite_all`：所有样本的有限性标志；`terminated`：是否提前终止；`chronological`：样本是否按时间顺序连续。

```python
    numeric = ("elapsed_s", "dt_s", "swing_clearance_m", "support_clearance_m",
               "support_force_n", "tilt_rad", "root_speed_mps", "support_speed_mps", "com_margin_m")
    last_t = 0.0
```
需要检查有限性的数值字段列表。

### 样本循环（97-152行）

```python
    for sample in samples:
```
遍历每个样本。

```python
        numbers_ok = all(isinstance(sample.get(key), (int, float))
                         and not isinstance(sample.get(key), bool)
                         and math.isfinite(sample[key]) for key in numeric)
        if not numbers_ok:
            finite_all = False
            hold_run = return_run = 0.0
            continue
```
检查所有数值字段是否为有限数字。如果有 NaN/Inf，标记 `finite_all=False`，清零连续计数器，跳过该样本。

```python
        t, dt = float(sample["elapsed_s"]), float(sample["dt_s"])
        consecutive = 0 < dt <= .021 and t > previous and abs(t - previous - dt) <= .002
        chronological &= consecutive
        previous, last_t = t, t
```
检查样本是否连续：dt 在 (0, 0.021] 范围内（50Hz ≈ 0.02s），时间戳严格递增，且前后时间差与 dt 一致（容差 2ms）。

```python
        finite = bool(sample.get("finite", False))
        finite_all &= finite
        terminated |= bool(sample.get("terminated", False) or sample.get("nonfoot_contact", False))
```
检查仿真状态有限性、是否提前终止或非脚触地。

```python
        contact = bool(sample.get("support_contact", False))
        contact_steps += int(contact)
        max_tilt = max(max_tilt, sample["tilt_rad"])
```
统计支撑脚触地步数、记录最大倾斜角。

```python
        stable = (finite and contact and not sample.get("nonfoot_contact", False)
                  and sample["support_force_n"] >= MIN_SUPPORT_FORCE_N
                  and sample["tilt_rad"] <= MAX_TILT_RAD
                  and sample["root_speed_mps"] <= MAX_ROOT_SPEED_MPS
                  and sample["support_speed_mps"] <= MAX_SUPPORT_SPEED_MPS)
```
**稳定条件**：有限状态 + 支撑脚触地 + 无非脚触地 + 支撑力 ≥ 0.5N + 倾斜 ≤ 25° + 躯干速度 ≤ 0.08m/s + 支撑脚速度 ≤ 0.04m/s。

```python
        gate_failed = {
            "tilt": sample["tilt_rad"] > MAX_TILT_RAD,
            "force": contact and sample["support_force_n"] < MIN_SUPPORT_FORCE_N,
            "swing_contact": bool(sample.get("swing_contact", False)),
            "nonfoot": bool(sample.get("nonfoot_contact", False)),
            "root_speed": sample["root_speed_mps"] > MAX_ROOT_SPEED_MPS,
            "support_speed": sample["support_speed_mps"] > MAX_SUPPORT_SPEED_MPS,
            "com_margin": sample["com_margin_m"] < MIN_COM_MARGIN_M,
            "finite": not finite,
        }
```
**v2 新增**：逐一检查每个约束是否违反。`True` 表示这个条件导致了不稳定。

```python
        clearance_bad = (abs(sample["swing_clearance_m"] - skill["lift_height_m"])
                         > max(.002, skill["lift_height_m"] * .2))
```
**v2 新增**：检查摆动脚高度是否偏离目标超过 2mm 或 20%。

```python
        hold_dt = max(0., min(t, hold_end) - max(t - dt, hold_start))
```
计算当前样本在 hold 窗口内的时间占比。`min(t, hold_end)` 截断到窗口结束，`max(t-dt, hold_start)` 截断到窗口开始。

```python
        hold_valid = (stable and not sample.get("swing_contact", True)
                      and sample["com_margin_m"] >= MIN_COM_MARGIN_M
                      and abs(sample["swing_clearance_m"] - skill["lift_height_m"])
                      <= max(.002, skill["lift_height_m"] * .2))
```
**hold 有效条件**：稳定 + 摆动脚不触地 + CoM 余量 ≥ -2mm + 摆动脚高度在目标 ±2mm/20% 内。

```python
        if hold_dt > 1e-8:
            hold_heights.append(sample["swing_clearance_m"])
            if not hold_valid and consecutive:
                for gate_key, gate_was_bad in gate_failed.items():
                    failed_in[gate_key] += float(gate_was_bad)
                failed_in["clearance"] += float(clearance_bad)
```
**v2 新增**：在 hold 窗口内，如果 hold 无效且样本连续，累加每个失败原因。这样就能知道"为什么 hold 失败了"。

```python
            hold_run = (hold_run + hold_dt) if consecutive and hold_valid else 0.
            best_hold = max(best_hold, hold_run)
```
连续有效则累加 hold 时长，否则清零。记录最佳连续 hold 时长。

```python
        settle_dt = max(0., min(t, cycle_end) - max(t - dt, settle_start))
        landed = (stable and bool(sample.get("swing_contact", False))
                  and abs(sample["swing_clearance_m"]) <= LAND_CLEARANCE_M
                  and abs(sample["support_clearance_m"]) <= LAND_CLEARANCE_M)
```
settle 阶段：计算在 settle 窗口内的时间。`landed` 表示双脚都已落地（高度 ≤ 3mm）。

```python
        if settle_dt > 1e-8:
            return_run = (return_run + settle_dt) if consecutive and landed else 0.
            best_return = max(best_return, return_run)
```
在 settle 窗口内，连续落地则累加时长。

### 结果汇总（154-175行）

```python
    reasons = []
    complete = bool(samples) and last_t >= cycle_end - .021
    if not complete:
        reasons.append("cycle_incomplete")
```
cycle 是否完整：最后一个样本的时间戳是否达到 cycle 结束时间（容差 1 帧）。

```python
    if not finite_all:
        reasons.append("non_finite_state")
    if terminated:
        reasons.append("physics_termination")
    if not chronological:
        reasons.append("sample_gap")
    if best_hold < skill["hold_seconds"] - .041:
        reasons.append("hold_incomplete")
    if best_return < SETTLE_SECONDS - .041:
        reasons.append("return_incomplete")
```
逐一检查失败原因（容差 2 帧 = 0.041s）。

```python
    return {"success": not reasons, "hold_achieved_s": round(best_hold, 6),
            "return_achieved_s": round(best_return, 6), "cycle_complete": complete,
            "max_tilt_deg": math.degrees(max_tilt),
            "mean_hold_clearance_m": sum(hold_heights) / len(hold_heights) if hold_heights else None,
            "support_contact_fraction": contact_steps / len(samples) if samples else 0.,
            "failure_reasons": reasons,
            "failed_in": {k: round(v, 4) for k, v in sorted(failed_in.items(),
                                                             key=lambda kv: -kv[1]) if v > 0}}
```
返回结果：`success` 表示没有任何失败原因；`failed_in` 按失败次数从大到小排序，只保留 >0 的项。

---

# `mdp.py` — Balance 相关函数逐行解释

## `balance_measurements`（7298-7369行）

这是**最核心的函数**，所有 reward/termination 都依赖它。

```python
def balance_measurements(env) -> dict[str, torch.Tensor]:
```
从仿真环境中读取当前 step 的所有测量值，返回一个张量字典。

```python
    command = env.command_manager.get_term("twist")
    if not isinstance(command, BalanceCommand):
        raise ValueError("Balance measurements require BalanceCommand")
```
获取 BalanceCommand 实例，确保 command 类型正确。

```python
    robot = env.scene["robot"]
    d = robot.data
    com = env.sim.data.subtree_com[:, robot.indexing.root_body_id]
    floor_z = env.scene.terrain.env_origins[:, 2]
```
获取机器人数据、整机 CoM 位置、地面高度。

```python
    geometry = _balance_geometry(env, command)
    heights, centers, polygons = [], [], []
    for gid, sid, vertices, outline in geometry:
        world = vertices @ matrix_from_quat(d.geom_quat_w[:, gid]).transpose(-1, -2)
        world = world + d.geom_pos_w[:, gid, None, :]
        heights.append(world[:, :, 2].amin(dim=1) - floor_z)
        polygon = outline @ matrix_from_quat(d.site_quat_w[:, sid]).transpose(-1, -2)
        polygon = polygon[:, :, :2] + d.site_pos_w[:, sid, None, :2]
        polygons.append(polygon)
        centers.append((polygon.amin(dim=1) + polygon.amax(dim=1)) / 2.)
```
获取双脚的碰撞体几何信息，计算：
- `heights`：脚底最低点到地面的距离（离地高度）
- `polygons`：脚底轮廓在水平面的投影多边形
- `centers`：脚底多边形中心

```python
    support, swing = command.support_index, command.swing_index
    polygon = polygons[support]
    edges = polygon.roll(-1, dims=1) - polygon
    relative = com[:, None, :2] - polygon
    cross = edges[:, :, 0] * relative[:, :, 1] - edges[:, :, 1] * relative[:, :, 0]
```
取支撑脚多边形，计算每条边到 CoM 的叉积（用于判断 CoM 是否在多边形内）。

```python
    area = (polygon[:, :, 0] * polygon.roll(-1, dims=1)[:, :, 1]
            - polygon[:, :, 1] * polygon.roll(-1, dims=1)[:, :, 0]).sum(dim=1)
    margin = (cross * torch.sign(area)[:, None]
              / torch.linalg.vector_norm(edges, dim=-1).clamp_min(1e-6)).amin(dim=1)
```
计算 CoM 到支撑多边形各边的最小距离（`com_margin_m`）。如果 CoM 在多边形内，margin > 0；在外则 < 0。`torch.sign(area)` 处理脚翻转的情况。

```python
    sensor = env.scene.sensors["feet_ground_contact"].data
    if sensor.found is None or sensor.force is None:
        raise ValueError("Balance needs foot contact found AND force fields")
    forces = torch.linalg.vector_norm(sensor.force[:, :2], dim=-1)
    contacts = (sensor.found[:, :2] > 0) & (forces >= .5)
```
读取脚部触地传感器：同时需要触地标志和接触力。接触力 ≥ 0.5N 才算有效触地。

```python
    finite = ~robot_state_is_nan(env, sensor_names=("feet_ground_contact", "nonfoot_ground_contact"))
    finite &= torch.isfinite(com).all(dim=1)
    for value in heights:
        finite &= torch.isfinite(value)
```
检查所有传感器和计算值是否有限（无 NaN/Inf）。

```python
    end_hold = 2. + command.skill["hold_seconds"]
    shift = torch.minimum(torch.clamp(command.elapsed_s, 0., 1.),
                          torch.clamp(end_hold + 1. - command.elapsed_s, 0., 1.))
    midpoint = (centers[0] + centers[1]) / 2.
    target_com = midpoint + shift[:, None] * (centers[support] - midpoint)
```
计算目标 CoM 位置：在 init 阶段（前 1s）从双脚中点平滑移到支撑脚中心，在 return 阶段（hold 结束后 1s）移回中点。

```python
    result = {
        "elapsed_s": command.elapsed_s.clone(),
        "phase": command.phase.clone(),
        "target_clearance_m": command.command[:, 1].clone(),
        "swing_clearance_m": heights[swing],
        "support_clearance_m": heights[support],
        "support_contact": contacts[:, support],
        "swing_contact": sensor.found[:, swing] > 0,
        "support_force_n": forces[:, support],
        "nonfoot_contact": balance_nonfoot_contact(env),
        "tilt_rad": torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.)),
        "root_speed_mps": torch.linalg.vector_norm(d.root_link_lin_vel_w, dim=-1),
        "support_speed_mps": torch.linalg.vector_norm(d.site_lin_vel_w[:, geometry[support][1]], dim=-1),
        "com_margin_m": margin,
        "com_error_m": torch.linalg.vector_norm(com[:, :2] - target_com, dim=-1),
        "finite": finite,
    }
```
组装结果字典。关键字段：
- `elapsed_s`：当前 episode 已过时间
- `phase`：当前阶段（0 init, 1 lift, 2 hold, 3 return）
- `target_clearance_m`：目标摆动脚高度
- `swing_clearance_m`：实际摆动脚高度
- `tilt_rad`：机身倾斜角（从重力投影计算）
- `root_speed_mps`：躯干速度
- `com_margin_m`：CoM 到支撑多边形边界的距离
- `com_error_m`：CoM 到目标位置的偏差

```python
    for key, value in result.items():
        if value.is_floating_point():
            result["finite"] &= torch.isfinite(value)
            result[key] = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
    return result
```
把所有浮点张量中的 NaN/Inf 替换为 0，同时更新 `finite` 标志。这样 reward 计算不会因为 NaN 而崩溃。

---

## `balance_height_reward`（7371-7380行）

```python
def balance_height_reward(env):
    m = balance_measurements(env)
    std = torch.maximum(m["target_clearance_m"] * .5,
                        torch.full_like(m["target_clearance_m"], .004))
    tracking = torch.exp(-((m["swing_clearance_m"] - m["target_clearance_m"]) / std).square())
    upright = torch.exp(-(m["tilt_rad"] / .5).square())
    valid = m["support_contact"] & ~m["nonfoot_contact"] & m["finite"]
    return tracking * upright * valid.float()
```
**抬脚高度奖励**：
- `std`：高斯核宽度 = max(目标高度 × 0.5, 4mm)，目标越高容差越大
- `tracking`：实际高度与目标高度的高斯相似度（越接近 → 越接近 1）
- `upright`：机身直立程度的高斯奖励
- `valid`：必须支撑脚触地、无非脚触地、状态有限
- 三者相乘，任一条件不满足 reward 为 0

---

## `balance_height_cost`（7382-7384行）

```python
def balance_height_cost(env):
    """Positive cost; use a negative weight. Keeps a dense lift gradient."""
    m = balance_measurements(env)
    return ((m["swing_clearance_m"] - m["target_clearance_m"]).abs() / .02).clamp(max=5.)
```
**高度误差成本**：绝对值误差除以 2cm，上限 5。用负权重加到总 reward 上，提供抬升阶段的密集梯度。

---

## `balance_com_reward`（7386-7389行）

```python
def balance_com_reward(env):
    m = balance_measurements(env)
    return torch.exp(-(m["com_error_m"] / .025).square()) * m["finite"].float()
```
**重心偏移奖励**：CoM 到目标位置偏差的高斯奖励，标准差 2.5cm。

---

## `balance_hold_reward`（7391-7402行）

```python
def balance_hold_reward(env):
    m = balance_measurements(env)
    supported = m["support_contact"] & ~m["swing_contact"] & m["finite"]
    supported &= ~m["nonfoot_contact"]
    supported &= m["tilt_rad"] <= _BALANCE_MAX_TILT
    supported &= m["com_margin_m"] >= -.002
```
**hold 阶段的基础条件**：支撑脚触地 + 摆动脚不触地 + 有限状态 + 无非脚触地 + 倾斜 ≤ 25° + CoM 余量 ≥ -2mm。

```python
    tracking = torch.exp(-((m["swing_clearance_m"] - m["target_clearance_m"]) / .01).square())
    stillness = torch.exp(-(m["root_speed_mps"] / .15).square()
                          - (m["support_speed_mps"] / .08).square())
    return (m["phase"] == 2).float() * supported.float() * tracking * stillness
```
**v2 改动**：分母从 `.004`→`.01`、`.08`→`.15`、`.04`→`.08`，容差放宽。
- `tracking`：摆动脚高度跟踪精度（高斯，标准差 1cm）
- `stillness`：身体静止程度（躯干速度标准差 0.15m/s，支撑脚速度标准差 0.08m/s）
- 只在 phase==2（hold 阶段）生效，四个条件相乘

---

## `balance_return_reward`（7404-7410行）

```python
def balance_return_reward(env):
    m = balance_measurements(env)
    down = (m["phase"] == 3) & (m["target_clearance_m"] <= 1e-6)
    valid = down & m["support_contact"] & m["swing_contact"] & m["finite"] & ~m["nonfoot_contact"]
    return valid.float() * torch.exp(-(m["root_speed_mps"] / .08).square()
                                     - (m["tilt_rad"] / .5).square())
```
**返回阶段奖励**：phase==3 且目标高度 ≈ 0（脚已放下），双脚触地、有限状态、无非脚触地。奖励身体静止和直立。

---

## `balance_support_slip_cost`（7412-7414行）

```python
def balance_support_slip_cost(env):
    m = balance_measurements(env)
    return m["support_speed_mps"].square() * m["support_contact"].float()
```
**支撑脚滑动成本**：支撑脚速度的平方，只在触地时计算。惩罚支撑脚在地面滑动。

---

## `balance_cycle_complete`（7416-7419行）

```python
def balance_cycle_complete(env):
    command = env.command_manager.get_term("twist")
    return command.elapsed_s >= command.cycle_seconds - 1e-5
```
**cycle 完成检测**：episode 已过时间 ≥ cycle 总时长（容差 0.01ms）。这是一个 `time_out=True` 的终止条件——完成 cycle 后正常结束 episode，不算失败。

---

## `balance_nonfoot_contact`（7421-7426行）

```python
def balance_nonfoot_contact(env):
    found = env.scene.sensors["nonfoot_ground_contact"].data.found
    if found is None:
        raise ValueError("Balance requires the non-foot ground contact sensor")
    return (found.reshape(env.num_envs, -1) > 0).any(dim=1)
```
**非脚触地检测**：检查是否有脚以外的身体部位触地（如膝盖、尾巴）。这是一个终止条件——非脚触地意味着机器人要倒了。

---

## `balance_hold_curriculum`（7429-7469行）— v2 新增

```python
def balance_hold_curriculum(env, env_ids, *, min_seconds: float = 1.5,
                            max_seconds: float = 3.0, pace_ppo_steps: float = 40.0, **_):
```
**课程函数**：根据群体完成率动态调整 hold 时长。

```python
    command = env.command_manager.get_term("twist")
    if not isinstance(command, BalanceCommand):
        return None
    try:
        done = env.terminations.time_outs
    except (AttributeError, TypeError):
        return None
```
获取 BalanceCommand 和当前 step 的 time_out 标志。`time_outs` 表示哪些环境因为完成 cycle 而正常终止。

```python
    num, device = command.num_envs, command.device
    st = getattr(balance_hold_curriculum, "_state", None)
    if st is None or st["num"] != num or st["device"] != device:
        st = balance_hold_curriculum._state = {
            "ema": 0.0, "target": float(min_seconds), "num": num, "device": device,
        }
```
用函数属性 `_state` 保存跨 step 的状态：EMA 完成率、当前 target。初始 target = 1.5s。

```python
    if torch.any(done):
        success = done.double().mean().item()
        smooth = 0.95
        st["ema"] = smooth * st["ema"] + (1.0 - smooth) * success
```
当有环境完成 cycle 时，计算当前 step 的完成比例，用 EMA（α=0.95）平滑。

```python
    target = st["target"] + pace_ppo_steps * (10.0 * st["ema"] - 1.5)
    target = min(max(target, min_seconds), max_seconds)
```
**课程公式**（已修正方向）：
- `10 * ema` 把完成率映射到 [0, 10]
- `10*ema - 1.5`：完成率 0% 时 = -1.5（推低 target → 更易），15% 时 = 0（冻结），30% 时 = +1.5（推高 target → 更难）
- `pace_ppo_steps * (...) `：乘以步长 40
- clamp 到 [1.5, 3.0]

```python
    if abs(target - st["target"]) > 1e-3:
        st["target"] = target
        command.skill["hold_seconds"] = target
        command.compute(0.0)
    return target
```
如果 target 有变化，更新 `command.skill["hold_seconds"]` 并重新计算所有环境的相位调度。

---

## 总结：Reward 结构

| 项 | 类型 | 权重 | 作用 |
|---|---|---|---|
| `balance_height_reward` | reward | +3.0 | 抬脚高度跟踪精度 |
| `balance_height_cost` | cost | -1.0 | 高度误差惩罚，提供抬升梯度 |
| `balance_com_reward` | reward | +1.5 | CoM 位置跟踪 |
| **`balance_hold_reward`** | **reward** | **+8.0** | **v2 从 3→8，hold 阶段站稳 + 高度跟踪 + 静止** |
| `balance_return_reward` | reward | +1.5 | 返回阶段双脚落地 + 静止 |
| `balance_support_slip_cost` | cost | -1.0 | 支撑脚滑动惩罚 |

终止条件：`fell_over`（倾斜 >45°）、`balance_low_trunk`（躯干过低）、`balance_cycle_complete`（正常完成）、`balance_nonfoot_contact`（非脚触地）。
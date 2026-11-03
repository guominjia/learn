---
title: "机器鸭学习：训练代码库"
categories: [hardware]
tags: [rl]
---

## 关于 robot_walk.xml 说明

这个文件是 MJCF，也就是 MuJoCo 的机器人描述格式，由 onshape-to-robot 从 CAD 自动生成。大约 70% 的行都是 `<geom class="visual">`，它们只用来显示零件外观（轴承、外壳、PCB 等），不参与物理计算。把这些行去掉后，剩下的结构就很清楚了。

## 文件分 4 块

| 块 | 作用 |
|---|---|
| `<default>` | 默认参数模板，比如 `chosen_actuator` 类定义了关节阻尼、摩擦、PD 增益 |
| `<worldbody>` | 主体部分：刚体（body）的父子树，以及连接它们的关节（joint） |
| `<asset>` | STL 网格和颜色，只影响外观 |
| `<actuator>` / `<sensor>` | 14 个位置控制电机，以及 IMU 传感器 |

### 去掉外观行后的 `<worldbody>`

```
trunk_base  (躯干, 0.199 kg, freejoint=浮动基座, site: imu)
├─ yaw2roll        [left_hip_yaw]      ← action 0
│  └─ hip_l        [left_hip_roll]     ← 1
│     └─ upper_leg_left [left_hip_pitch] ← 2
│        └─ leg    [left_knee]         ← 3
│           └─ ankle_left [left_ankle] ← 4   (geom: left_foot_collision, site: left_foot)
├─ neck            [neck_pitch]        ← 5
│  └─ neck_pitch   [head_pitch]        ← 6
│     └─ yaw_roll_motion [head_yaw]    ← 7
│        └─ jaw_soft [head_roll]       ← 8   (sites: head_camera, tof, head_imu, mouth_tip)
└─ bearing_roll    [right_hip_yaw]     ← 9
   └─ hip_l_2      [right_hip_roll]    ← 10
      └─ upper_leg_right [right_hip_pitch] ← 11
         └─ leg_2  [right_knee]        ← 12
            └─ ankle_right [right_ankle] ← 13 (geom: right_foot_collision, site: right_foot)
```

- 每个 `<body>` 是一个刚体，嵌套关系表示父子连接。`pos`/`quat` 是它相对父体的位置和姿态，`<inertial>` 是质量和惯量。
- 每个 `<joint type="hinge">` 是一个转轴，`range` 是限位，单位是弧度。例如 `left_hip_yaw` 的范围 `-0.436 ~ 0.524` 对应 -25° ~ 30°。
- 关节在文件中的出现顺序就是 14 维动作和观测里 `joint_pos`/`joint_vel` 的顺序。
- `trunk_base` 上的 `freejoint` 让躯干能在空间中自由移动（3 个平移 + 3 个转动），这部分没有电机驱动。

## `<body name="trunk_base" pos="0 0 0.12" quat="1 0 0 0">`

这两个属性描述的是这个刚体的坐标系相对父体放在哪里、朝向如何。`trunk_base` 的父体是 world，所以这里就是相对世界坐标系。

| 属性 | 含义 |
|---|---|
| `pos="0 0 0.12"` | 位置 (x, y, z)，单位米。躯干原点在离地 12 cm 处，水平方向位于原点 |
| `quat="1 0 0 0"` | 姿态四元数，MuJoCo 的顺序是 (w, x, y, z)。`1 0 0 0` 是单位四元数，表示没有旋转 |

因为 `trunk_base` 带 `freejoint`，这个 pos/quat 也是仿真的默认初始位姿。训练时 `reset_base` 会在每次 reset 时重新设定初始位置。

子体上的 pos/quat 都是相对各自父体的。比如 `yaw2roll pos="0.006 0.0175 -0.005"` 表示左髋关节在躯干原点偏移 (6, 17.5, -5) mm 的位置。

## `<inertial pos="..." mass="0.199224" fullinertia="...">`

0.199 kg 就是 `mass="0.199224"`。它只是躯干这一个刚体的质量，不是整机质量。把文件里所有 body 的质量加起来：

| 部分 | 质量 (kg) |
|---|---|
| 躯干 `trunk_base` | 0.199 |
| 单条腿：0.0230 + 0.0062 + 0.0482 + 0.0216 + 0.0300 | 0.129，两条共 0.258 |
| 头颈：0.0368 + 0.0057 + 0.0486 + 0.1888 | 0.280 |
| 合计 | ≈ 0.737 |

合计约 0.74 kg，和 README 里"约 800 g"的说法接近。

`inertial` 里的 `pos="-0.0226332 -2.70828e-05 0.00280064"` 是质心位置，在这个 body 自己的坐标系里表示，单位米：

- x = -22.6 mm：质心在躯干原点的 -x 方向 2.26 cm 处
- y = -2.7e-05 m = -0.027 mm：几乎为 0，说明左右基本对称
- z = +2.8 mm：略高于原点

`e-05` 是科学计数法，$-2.70828\times10^{-5}$。

所以 body 的 `pos` 是刚体坐标系原点（也就是关节连接点）的位置，`inertial` 的 `pos` 是质心的位置。两者通常不重合。

同一行的 `fullinertia` 是转动惯量，6 个数依次是 $I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}$，单位 kg·m²。

---
title: "机器鸭学习：训练代码库"
categories: [hardware]
tags: [rl]
---

今天开始看[microduck_rl](https://github.com/pollen-robotics/microduck_rl/)代码库。

## 关于训练

训练命令：

```bash
WANDB_MODE=offline nohup uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run <path_to> \
  --agent.load-checkpoint model_<iter>.pt \
  > train.log &
```

- `WANDB_MODE`用于避免报错`UsageError: wandb.errors.errors.UsageError: No API key configured. Use wandb login to log in`
- `train` 每次运行都会新建一个带时间戳的日志目录（例如 `logs/rsl_rl/velocity/YYYY-mm-dd-MM-SS_velocity/`），并从随机初始化的网络开始训练。要接着上次训练，必须显式指定 resume 参数。续训时仍然会新建一个日志目录，新的 checkpoint 保存在那里，不会覆盖原目录。

| `WANDB_MODE` 取值 | 效果 |
|---|---|
| `offline` | 不联网，数据存在本地 `wandb/offline-run-*`，之后可以用 `uv run wandb sync wandb/offline-run-*` 上传，相当于你之前手动选 3 的效果 |
| `disabled` | 完全不记录，wandb 的所有调用都变成空操作 |

| `agent` 参数 | 作用 |
|---|---|
| `--agent.resume True` | 开启加载 checkpoint |
| `--agent.load-run` | `logs/rsl_rl/velocity/` 下的子目录名，支持正则。默认值通常会匹配到排序最后的那个目录。 |
| `--agent.load-checkpoint` | 目录里的 `model_<iter>.pt` 文件名 |

## 关于仿真说明

运行仿真：`uv run --no-project --with mujoco python -m mujoco.viewer --mjcf=src/mjlab_microduck/robot/microduck/scene_walk.xml`。`--no-project`会跳过不必要的安装包。

观察到 `uv run python -m mujoco.viewer --mjcf=src/mjlab_microduck/robot/microduck/scene_walk.xml` 一打开鸭子就倒了。

解释：这是正常现象，viewer 只是在运行物理仿真，并没有运行控制策略。

### 具体解释

- 刚打开时用的是默认初始状态：14 个关节全为 0，电机目标角 `ctrl` 也全为 0。
- 位置电机会尽力把关节保持在 0°，但这个姿态并不是站立姿态，而且没有任何东西负责维持平衡。
- 双足机器人本身是不稳定的。即使关节保持住了，质心只要稍微偏出两脚的支撑范围，整机就会像一根棍子一样倒下。负责平衡的是训练出来的策略，单独的 viewer 不包含它。

### 可采取操作

| 操作 | 方法 |
|---|---|
| 暂停仿真 | 按空格。暂停后可以旋转、缩放视角慢慢看，按 Backspace 重置到初始状态 |
| 摆成站立姿态 | 左侧 Simulation 面板中把 Key 选为 `STAND`（序号 1），点 Load key。这个 keyframe 同时设定了 `ctrl`，电机会保持住这个姿态；注释说它把质心调到了踝关节正上方，有可能静止站住，但一受扰动还是会倒 |
| 关掉重力 | 左侧 Physics 面板的 Disable Flags 中勾选 Gravity，机器人就会悬在空中，方便看关节 |
| 手动拖动关节 | 右侧 Control 面板拖动各电机的滑块（暂停时右侧 Joint 面板也可以直接改关节角） |

### 想站立和行走

需要让策略来控制它。在训练出策略后，运行：

```bash
# Linux 训练机上，需要图形界面或 viser 网页查看器
uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint-file logs/rsl_rl/velocity/<run>/model_XXXX.pt

# 或者先导出 ONNX，在 Windows WSL 用 CPU MuJoCo + 键盘控制
uv run scripts/infer_policy.py --walking output.onnx
```

## Windows termios 错误

`termios` 不是 PyPI 上的包，而是 Python 标准库里只在 Unix（Linux/macOS）上才有的模块，Windows 版 Python 里没有，用 pip 装不上。

`scripts/infer_policy.py` 用 `termios` + `tty` 把终端切换到 cbreak 模式（不用按回车就能读到单个按键），用于键盘控制，但**这个脚本原样在 Windows 上跑不起来**。

### 解决办法：用 WSL2

Windows 11 自带的 WSLg 可以直接显示 Linux 的图形窗口，MuJoCo viewer 能正常弹出来：

```powershell
wsl --install -d Ubuntu      # 首次安装
```

进入 Ubuntu 后，按 Linux 的方式 `git clone`、`uv sync`，再运行 `uv run scripts/infer_policy.py ...`。

## 关于 robot_walk.xml 和 scene_walk.xml 说明

这两个是 Onshape 的导出文件，具体流程：

```
Onshape（云端 CAD，浏览器里编辑零件和装配）
   │  onshape-to-robot 导出
   ▼
robot_walk.xml + assets/*.stl（本地文件）
   │  mujoco.viewer 加载
   ▼
MuJoCo viewer（本地窗口：显示模型 + 运行物理仿真）
```

| | Onshape | MuJoCo viewer |
|---|---|---|
| 是什么 | 网页版 CAD 设计软件 | 物理仿真器的可视化窗口 |
| 能做什么 | 修改零件形状、尺寸、材料、装配关系 | 看模型、拖动关节、加载 keyframe、施加推力、观察重力下的表现 |
| 能否改几何 | 能 | 不能，只读取 XML 和 STL |
| 数据来源 | 云端设计文档 | 从 Onshape 导出的本地文件 |

`robot_walk.xml` 开头的注释里就是原始 Onshape 文档的链接：

```xml
<!-- Generated using onshape-to-robot -->
<!-- Onshape https://cad.onshape.com/documents/804927696f06d877f3f1803e/... -->
```

需要修改机器人结构时，在浏览器里打开这个链接。在 MuJoCo viewer 里只能查看和仿真，不能编辑模型。

`scene_walk.xml` 只是在 `robot_walk.xml` 外面包了一层场景，第一行就是把机器人模型引用进来：

```xml
<mujoco model="scene">
    <include file="robot_walk.xml" />
    ...
```

### 两个文件各管什么

| | `robot_walk.xml` | `scene_walk.xml` |
|---|---|---|
| 内容 | 机器人本身：刚体树、14 个关节、电机、传感器、网格 | 引用机器人模型，再加上地面、灯光、天空盒、预设姿态（keyframe） |
| 地面 | 没有 | `<geom name="floor" type="plane">` |
| 外观 | 只有零件颜色 | 光照、棋盘格地板、渐变背景 |
| 预设姿态 | 没有 | `INIT` / `STAND` / `SIT` / `FOLD` |
| 用途 | 被训练代码和各个场景文件引用 | 用于 viewer 查看和 `infer_policy.py` |

在 viewer 里直接打开 `robot_walk.xml`，因为没有地面，机器人会一直往下掉。查看模型时应该打开 `scene_walk.xml`。

### keyframe 怎么读

`qpos` 是整个模型的位置状态向量，共 $7 + 14 = 21$ 个数：

```
0 0 0.12  1 0 0 0          ← freejoint: 躯干位置 xyz + 四元数 wxyz (7)
0 -0.0873 -0.4579 -0.0049 0.4530   ← 左腿 hip_yaw, hip_roll, hip_pitch, knee, ankle
0.3491 0.3491 0 0                  ← neck_pitch, head_pitch, head_yaw, head_roll
0 0.0873 0.4579 0.0049 -0.4530     ← 右腿
```

这是 `STAND` 姿态：躯干离地 12 cm，髋和踝前倾约 26°，脖子和头各抬 20°。左右腿的数值符号相反，是因为两侧关节轴的方向是镜像的。

`ctrl` 是 14 个电机的目标角，和关节角相同，这样加载这个姿态后机器人能保持不动。

在 viewer 里打开 Simulation 面板中的 Key，选择 `STAND` 后点 Load key，机器人就会摆成这个姿态。

### 训练时用哪个

训练（mjlab）加载的是机器人模型文件，地面和地形由 mjlab 自己生成，不经过 `scene_walk.xml`。训练用的初始姿态是 README 提到的 `microduck_constants.py` 里的 HOME，不是这里的 keyframe。不过它们的数值很可能和 `STAND` 一致，可以打开 `microduck_constants.py` 对照确认。

## Onshape 是谁

Onshape 2012 年在马萨诸塞州剑桥市创立，创始团队包括 SolidWorks 的两位前 CEO Jon Hirschtick 和 John McEleney。2019 年被美国 PTC 公司以 4.7 亿美元收购，现在是 PTC 旗下产品，总部在波士顿。它是纯云端 CAD，用浏览器就能用，也有 iOS 和 Android App。来源：[Wikipedia](https://en.wikipedia.org/wiki/Onshape)

### 收费（[官网定价](https://www.onshape.com/en/pricing)）

| 方案 | 价格 | 限制 |
|---|---|---|
| Free | $0 | 仅限非商业用途，所有文档公开，任何人都能看到 |
| Standard | $1,500/人/年 | 可商用，文档可设为私有 |
| Professional | $2,500/人/年 | 面向团队，有 6 个月免费试用（Discovery Program） |
| Education | 免费 | 学生和教师，功能相当于 Professional |

### 改机器鸭用免费版就够

- 打开 `robot_walk.xml` 开头的 Onshape 链接，点 Copy 复制到自己账号下，就可以随意修改。
- 免费版的唯一代价是你的文档会公开。不过这个 3D 模型本身的许可是 CC BY-NC-SA 4.0，要求非商业使用、修改后以相同许可分享，本来就和免费版的公开要求一致。

用 onshape-to-robot 重新导出时，需要在 Onshape 开发者门户申请 API key，并把 `config_mjcf_walk.json` 里的文档 ID 改成你自己文档的 ID。具体步骤以 [onshape-to-robot 文档](https://onshape-to-robot.readthedocs.io/) 为准。

## 何时需要重训

### 按官方设计组装的：基本不用改

真机和 CAD 总会有小偏差：质量差几十克、质心偏几毫米、关节摩擦不同、电池电压波动等。训练里的域随机化就是用来覆盖这些偏差的，包括：

```
randomize_com / base_com / randomize_mass_inertia   ← 质量、质心
randomize_joint_friction / randomize_armature       ← 关节摩擦、转子惯量
vin=range=(6.5, 8.2) drop_gain_range=(0.0, 0.2)     ← 电池电压、压降
encoder_bias / foot_friction                        ← 编码器零偏、脚底摩擦
```

建议先直接在真机上跑官方训练好的策略。如果能正常行走，模型文件就不需要改。

### 硬件有改动时：要改，但不要手改 XML

`robot_walk.xml` 是 onshape-to-robot 自动生成的。正确的流程是：

1. 在 Onshape 里修改 CAD（换零件、加配重、改尺寸等）
2. 用对应的 `config_mjcf_walk.json` 重新导出
3. 如果用了 backlash 变体，再运行 `add_backlash.py` 重新生成对应的模型文件
4. 重新训练

手改 XML 的话，`*_backlash.xml` 和 `groundcontact` 系列不会自动同步，下次重新导出时修改也会被覆盖。

### 需要核对的项

| 项目 | 怎么检查 | 偏差太大时 |
|---|---|---|
| 整机质量 | 称重，与 XML 中各 body 质量之和（约 0.74 kg）对比 | 偏差超出随机化范围时，改 CAD 中零件的质量/材料后重新导出 |
| 舵机型号 | 是否为 XL330 | 换了型号就要用 BAM 重新辨识舵机参数，这是 sim2real 中影响最大的一项 |
| 关节零位和转向 | 在 viewer 里加载 `STAND`，和真机摆同样角度时的姿态对比 | 通常在真机侧做校准或修改映射，不要动 XML |
| IMU 安装位置和朝向 | 对照 XML 中 `site name="imu"` 的位置 | 运行时代码中的 mount 四元数要和实际安装一致 |

### 约束

如果改了关节数量，或者想加新的传感器输入，61 维观测和 14 维动作的接口就会变。这样一来，官方的所有策略、`publish` 的 `[1,61] -> [1,14]` 检查以及真机运行时代码都要跟着修改。能保持接口不变就尽量不变。
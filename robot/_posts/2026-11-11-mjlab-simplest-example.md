---
title: "最简单 mjlab 例子"
date: 2026-11-11
---


## 1. 最简单倒立摆代码

```python
# mjlab hello world: cartpole（小车倒立摆），纯仿真，不用 RL。"""
from pathlib import Path

import mujoco

import mjlab.tasks.cartpole        # noqa: F401  只为拿到任务文件路径
from mjlab.sim import Simulation, SimulationCfg

NUM_ENVS = 4                        # 4 个并行世界，都是同一辆小车

xml = Path(mjlab.tasks.cartpole.__file__).parent / "cartpole.xml"
model = mujoco.MjModel.from_xml_path(str(xml))

sim = Simulation(NUM_ENVS, SimulationCfg(), model, "cuda")  # 配置全默认
sim.reset()                                   # 每个世界回到初始（杆竖直朝上）

for t in range(500):                         # dt = 0.01 s
    sim.data.ctrl[:, 0] = 0.0                # 电机不发力，摆自由运动
    sim.step()
    if t % 100 == 0:
        q = sim.data.qpos                    # qpos = [slider(x方向), hinge(杆角)], hinge=0 即杆竖直
        print(f"t={t:3d}  x={q[0, 0].item():+.3f}  theta={q[0, 1].item():+.4f} rad")
```

要点就三个对象：`MjModel`（MuJoCo XML 模型）→ `Simulation(num_envs, cfg, model, device)`（MJWarp 批量并行）→ `reset()/step()`，状态通过 `sim.data.qpos / qvel / ctrl` 以 **torch tensor** 读写，形状第一维都是 `num_envs`——这就是它和 MuJoCo 单体的最大区别：所有批量都是天然向量化的，所以后面接 RSL-RL 几乎零额外代码。

- **问题与困惑**：
  - 这里所说的 MuJoCo 单体什么意思？
  - 具体怎么接 RSL-RL 呢？

## 2. 零代码：直接训一个 PPO

框架自带任务注册表（`train` 不带参数会列出全部任务，cartpole 是最简单的）：

```bash
uv run train Mjlab-Cartpole-Balance \
  --env.scene.num-envs 64 \
  --agent.max_iterations 30 \
  --agent.logger tensorboard
```

- 检查点和曲线写到 `logs/rsl_rl/cartpole/`（`experiment_name="cartpole"` 来自 runner cfg），`tensorboard --logdir logs` 看 reward 上涨接近 500 = 学会。
- 想让它自己"表演"：`python -m mjlab.scripts.play Mjlab-Cartpole-Balance`（加载最近检查点）。

- **问题与困惑**：
  - 运行Mjlab-Cartpole-Balance为何报错：`wandb_run_path` is required when `checkpoint_file` is not provided.

## 3. 概念地图（对应包目录）

```
mjlab/
  scene/        Scene + SceneCfg   把若干 Entity 组装成场景，compile() 出 MjModel
  sim/          Simulation         MJWarp 批量仿真，GPU 上还抓 CUDA Graph 提速
  entities, sensor, terrains       实体 / 传感器 / 地形
  envs/         ManagerBasedRlEnv  把场景包成 gym 式 env（obs/reward/termination 由 manager 管）
  rl/           RSL-RL 封装        Runner + 各算法 cfg
  tasks/        任务注册表          register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg)
  scripts/      train / play / demo / export_scene / list_envs
```

自己加一个任务 = 写一个 `xxx_env_cfg.py`（XML + ManagerBasedRlEnvCfg + runner cfg）+ 一行 `register_mjlab_task(...)`，`train Mjlab-YourTask` 就能跑。

## 1. 网页界面（viser，默认 fallback）

`play` 的参数有 `--viewer auto|native|viser`（`scripts/play.py` 里的 `PlayConfig`）：

```python
resolved_viewer = "native" if has_display else "viser"   # 没显示器就自动用网页
```

viser viewer（`mjlab/viewer/viser/`）是个本地 HTTP 服务，浏览器打开端口即可，带：

- 3D 场景（可浏览器里拖视角、同时看 4/64 个 env 的渲染）
- reward bar 面板（`reward_bar_panel.py`）、终端小曲线（`term_plotter.py`）
- **检查点热切换**（`CheckpointManager`）——训练时在浏览器里不停切换对比不同 iteration 的策略

## 2. 原生桌面 GUI（native）

`mjlab/viewer/native/viewer.py` 包的是 MuJoCo 官方实时渲染窗口，只在有显示器的机器上能开。

## 3. 训练侧没有实时 GUI

`train` 是 headless 批量仿真，可视化只有两条路：

- **TensorBoard**（网页）：`--agent.logger tensorboard`，`tensorboard --logdir logs` 看 reward 曲线——这是"训练中的主界面"
- **offscreen renderer**（`viewer/offscreen_renderer.py`）：无头 GPU 渲染出视频，`play` 配置里 `video_height/video_width` 走的就是它

官方还有个彩蛋入口：`python -m mjlab.scripts.demo` 会从云上下载预训练 G1 体操策略直接开 viser 播放。

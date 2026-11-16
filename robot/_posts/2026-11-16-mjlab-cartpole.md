---
title: "最简单 mjlab 例子：倒立摆"
date: 2026-11-11
---

mjlab 是 Isaac 出的 MuJoCo-Warp 研究框架。

**困惑与疑问**：
- 为何只是单纯写了 `~/mjlab_hello.py` 文件，却可以使用 `uv run train Mjlab-Cartpole-Balance` 来训练，不是说训练需要 `xxx_env_cfg.py` 和 `register_mjlab_task(...)` 吗？
- Isaac 是谁？
- `logs/rsl_rl/cartpole/`（`experiment_name="cartpole"` 来自 runner cfg）
  - 怎么查看 `runner cfg`? 
  - log是如何被控制的？
- 新建项目只要加个 tasks 目录，然后就能被识别为任务对吗？

## 实践解惑

### 困惑一：只要安装了 mjlab 就会有 train 命令？

这个困惑是为了弄明白这句话：**框架自带任务注册表（`train` 不带参数会列出全部任务，cartpole 是最简单的）**

解答：**是的，只要按照了 mjlab ，就会有 train 和 play 命令**。

我做了如下实验证明：

1. 新建了 `test` 空文件夹，然后运行 `uv init -p 3.13` 和 `uv venv -p 3.13` 设置虚拟环境。
2. 运行 `uv run train` 会报错 `train` program not found 。
2. 运行 `uv pip install mjlab` 安装 `mjlab`，安装过程看到安装很多包，几个典型的 AI 相关的包有：`torch/onnx/mjlab/mujoco`。
3. 再次运行 `uv run train` 会列出 
    ```
    {Mjlab-Cartpole-Balance,Mjlab-Cartpole-Swingup,Mjlab-Lift-Cube-Yam,Mjlab-Lift-Cube-Yam-Depth,                    │
    │     Mjlab-Lift-Cube-Yam-Rgb,Mjlab-Multi-Cube-Seg-Yam,Mjlab-Tracking-Flat-Unitree-G1,                                 │
    │     Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation,Mjlab-Velocity-Flat-Unitree-G1,                               │
    │     Mjlab-Velocity-Flat-Unitree-Go1,Mjlab-Velocity-Rough-Unitree-G1,Mjlab-Velocity-Rough-Unitree-Go1}
    ```
4. 运行 `where train` 和 `where play` 可以看到 `.venv` 目录下对应的应用。

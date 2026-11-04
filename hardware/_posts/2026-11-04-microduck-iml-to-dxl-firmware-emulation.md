---
title: "机器鸭固件学习：IMU TO DXL"
categories: [hardware]
tags: [imu]
---

## 在 Linux 上模拟运行 `imu_to_dxl` 固件

这个固件是给 **STM32G031F8P6** 跑的裸机 C 程序，编译用 Keil/ARMCLANG（Windows），所以**不能直接在 Linux 上编译烧录固件本体**。但项目自带了三层**无硬件测试/模拟**，全部可以在 Linux 上跑：

---

### 1. 纯 Python 模拟器：`imu200.py demo`（最简单，无需编译）

这是最直接的方式，不需要任何 C 编译器，也不需要硬件：

```bash
cd tools/imu200
python3 imu200.py demo
```

它做的事：
- 用 `SimImu200` 类在内存里模拟一个 ID 200 的 IMU 小板（120 Hz 采样、飞特 SCS 协议应答）
- 用 `sim_bus.SimSerial` 模拟 15 颗舵机 + IMU 小板挂在同一条总线上
- 用真实的主控侧代码（`feetech.FeetechBus`）发 `sync_read`，把 IMU 和舵机一起读回来
- 打印每帧的陀螺仪、四元数、采样计数，以及主控是否接受这帧
- 最后跑一遍 `check()`（13 项验收判据，照主控源码写的）

```bash
# 打印测试向量（写固件时对着抄的十六进制）
python3 imu200.py vectors
```

**依赖**：只需要 Python 3，`feetech.py` 在同仓库 `tools/servo-web/` 下，`imu200.py` 自己 `sys.path.insert` 了。

---

### 2. C 协议测试 + 总线模拟（需要 gcc/clang）

固件的 `protocol.c` + `control_table.c` 是**平台无关的纯 C 代码**，可以用 Linux 上的 gcc/clang 编译成主机可执行文件来测试。

#### 2a. 协议单元测试（firmware 0.2.0）

```bash
cd hardware/imu_to_dxl/firmware

# 编译并运行协议测试（93 项断言）
python3 Tools/protocol_test.py
```

`imu_test.cmd` 里的核心命令等价于：
```bash
mkdir -p Build/Tests/imu
gcc -std=c11 -Wall -Wextra -Werror -O0 -g \
  -D_CRT_SECURE_NO_WARNINGS -D__weak= \
  -I Tests/imu_board -I Core/Inc -I Drivers/LSM6DSV16X \
  Tests/imu_test.c Core/Src/imu.c Drivers/LSM6DSV16X/lsm6dsv16x_reg.c \
  -o Build/Tests/imu/imu_test
./Build/Tests/imu/imu_test
```

#### 2b. 总线模拟验收（真固件协议代码编成 .so，挂到 Python 模拟总线上）

```bash
cd hardware/imu_to_dxl/firmware
python3 Tests/bus_sim_test.py
```

这个测试做的事：
- 把 `protocol.c` + `control_table.c` + `bus_sim_shim.c` 编译成一个 **共享库**（Linux 上是 `.so`，`bus_sim_shim.c` 已有 `__attribute__((visibility("default")))` 导出符号）
- 用 Python 的 `ctypes` 加载这个 `.so`
- 在模拟总线上放 15 颗假舵机 + 真固件协议代码答 ID 200
- 用 `imu200.py check` 的主控判据跑 80 帧，验证计数步进、四元数有效性、不抢话等

`bus_sim_test.py` 的 `build_dll()` 函数在 Linux 上会自动用 `gcc -shared -fPIC` 编译。

#### 2c. firmware 1.0.0 的测试

```bash
cd hardware/imu_to_dxl/firmware-1.0.0

# 协议测试（93 项）
python3 Tools/protocol_test.py

# IMU 测试（16 场景）
gcc -std=c11 -Wall -Wextra -Werror -O0 -g \
  -D_CRT_SECURE_NO_WARNINGS -D__weak= \
  -I Tests/imu_board -I Core/Inc -I Drivers/LSM6DSV16X \
  Tests/imu_test.c Core/Src/imu.c Drivers/LSM6DSV16X/lsm6dsv16x_reg.c \
  -o Build/Tests/imu/imu_test && ./Build/Tests/imu/imu_test

# 寄存器映射测试（18 项）
# control_table_test.cmd 同理

# Python 探针离线测试（9 项）
python3 Tests/probe_test.py
```

---

### 3. servo-web 调试台模拟（`--fake` 模式）

```bash
cd tools/servo-web
python3 server.py --fake
```

`--fake` 模式启动一个 Web 调试台，用 `FakeImu200` 模拟 IMU 小板（120 Hz 换块、陀螺带噪声、四元数绕 Z 慢慢转），可以在浏览器里看 3D 鸭子姿态。还可以通过页面按钮 `set_sim_imu("on"/"lost"/"off")` 模拟小板插拔/掉线。

---

### 总结

| 方式 | 需要 | 看什么 | 适合 |
|---|---|---|---|
| `imu200.py demo` | Python 3 | 协议编解码 + 主控判据 | 快速验证协议逻辑 |
| `protocol_test.py` | Python 3 + gcc/clang | 93 项 C 断言 | 验证固件协议代码 |
| `imu_test` | gcc/clang | 16 个 IMU 场景 | 验证 IMU 采样逻辑 |
| `bus_sim_test.py` | Python 3 + gcc/clang | 真固件 .so 挂模拟总线 | 端到端协议验收 |
| `server.py --fake` | Python 3 | 3D 姿态可视化 | 看效果 |
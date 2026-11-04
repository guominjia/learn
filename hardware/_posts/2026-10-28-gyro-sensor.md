---
title: "机器鸭复刻：陀螺仪与加速度计"
categories: [hardware]
tags: [servos, pid]
---

我的机器鸭复刻项目当前进展是：**正确学习陀螺仪和加速度计的硬件模块与原理**，在这里做个记录。

机器鸭专门做了个`IMU_TO_DXL`的小主板，目的是将`IMU`数据转为舵机数据，这样可以直接用舵机的总线协议一次性拿到所有数据，而不是需要两套总线。这个小主板上面有个`STM32`主控和`LSM6DSV16X`陀螺仪和加速度计，前者把陀螺仪和加速度计的数据模拟成舵机数据放到总线。

我对`LSM6DSV16X`了解很浅，对陀螺仪和加速度计也基本没啥了解，在这里记下刚学到的新知识。

[`microduck-replica`的`imu200`](https://raw.githubusercontent.com/fanhao375/microduck-replica/master/tools/imu200/imu200.py)提供完整算法，下面讲些算法中关键内容：

## 数据块结构

`IMU_TO_DXL`是发送15个数据，结构如下：

| 偏移 | 长度 | 内容 | 编码 |
|------|------|------|------|
| 0–5 | 6 字节 | 陀螺仪 x/y/z | `i16` 小端，`17.5 mdps/LSB` |
| 6–11 | 6 字节 | 四元数 x/y/z | IEEE 754 半精度 `binary16` |
| 12 | 1 字节 | 采样计数器 | `u8`，每采样 +1 |
| 13 | 1 字节 | 状态 | bit0=1 表示融合还没出数 |
| 14 | 1 字节 | 保留 | 必须为 0 |

算法中提供了`decode_block`和`encode_block`来把数据编码成上述结构以及把上述结构数据编为可读数据。

算法是参考 `imu.rs`。

$q_{imu}$ 是单位四元数（norm_sq ≤ 1.02 已经过检），MOUNT 是单位四元数（$\sqrt{0.5}^2 + 0^2 + \sqrt{0.5}^2 + 0^2 = 1$），两个单位四元数相乘结果还是单位四元数。所以 $n$ 理论上就是 1。

**真正要挡的是数据损坏**。如果 SPI 总线传输出错、固件 bug、或者芯片发了垃圾数据，三个分量可能是任意值。比如传了 $(1, 1, 0)$，norm_sq = 2，过了 1.02 的检查（如果固件量程配错），恢复出 $w = \sqrt{1-2}$ 会得到 NaN 或虚数。即使过了 norm_sq 检查，乘 MOUNT 后 $n$ 可能远偏离 1。

`n > 0.5` 是一道**兜底防线**：norm_sq 检查已经挡了大部分坏数据，但如果有什么漏网的，这里再拦一次，避免拿一个模长 0.01 的"四元数"去算重力，得到一堆无意义的垃圾值。

## `decode_block` 算法详解

`decode_block(b)` 是 `imu200.py` 中用于**解码 IMU 小板通过舵机总线返回的 15 字节数据块**的函数。它的判据逐条照抄主控固件 `imu.rs`，确保"工具过了 = 主控会认"。

### 1. 15 字节块的结构

| 偏移 | 长度 | 内容 | 编码 |
|------|------|------|------|
| 0–5 | 6 字节 | 陀螺仪 x/y/z | `i16` 小端，`17.5 mdps/LSB` |
| 6–11 | 6 字节 | 四元数 x/y/z | IEEE 754 半精度 `binary16` |
| 12 | 1 字节 | 采样计数器 | `u8`，每采样 +1 |
| 13 | 1 字节 | 状态 | bit0=1 表示融合还没出数 |
| 14 | 1 字节 | 保留 | 必须为 0 |

### 2. 解码流程（逐步）

```python
def decode_block(b):
```

**① 长度校验**
```python
if len(b) != BLOCK_LEN:   # BLOCK_LEN = 15
    raise ValueError(...)
```

**② 解包陀螺仪（offset 0, 3×i16 小端）**
```python
gx, gy, gz = struct.unpack_from("<hhh", b, 0)
```
原始 LSB → 物理量：`gyro_dps = gx * 0.0175`（单位 dps）。

**③ 解包四元数三分量（offset 6, 3×half）**
```python
packed = struct.unpack_from("<HHH", b, 6)
qx, qy, qz = (f16_to_f32(h) for h in packed)
```
四元数只存 x/y/z 三个分量，**w 不传输**，由接收端用 $w = \sqrt{1 - (x^2+y^2+z^2)}$ 恢复。

**④ 有限性 + norm² 校验（主控拒绝判据）**
```python
finite = all(math.isfinite(v) for v in (qx, qy, qz))
norm_sq = (qx*qx + qy*qy + qz*qz) if finite else float("inf")
```
- 半精度可能产生 `NaN`/`Inf`，先检查 `isfinite`。
- `norm_sq` 是四元数三分量的模平方。主控的判据是 `norm_sq ≤ 1.02`（`NORM_SQ_MAX`），留了半精度舍入余量。超了说明固件量程配错或数据损坏。

**⑤ 接受/拒绝判定**
```python
if any(packed) and finite and norm_sq <= NORM_SQ_MAX:
```
三个条件全满足才进入处理：
- `any(packed)`：三分量全 0 → 融合还没出数（status bit0 也应=1），主控沿用上一帧、不计入 ready。
- `finite`：无 NaN/Inf。
- `norm_sq ≤ 1.02`：模长合理。

**⑥ 恢复 w 并做 mount 旋转校正**
```python
w = math.sqrt(max(0.0, 1.0 - norm_sq))
q = _qmul((w, qx, qy, qz), (MOUNT[0], -MOUNT[1], -MOUNT[2], -MOUNT[3]))
```
- `w = √(1 − norm²)`：单位四元数约束。
- `MOUNT = (√0.5, 0, √0.5, 0)`：绕 Y 轴 +90° 的安装姿态。乘以 **共轭** `(w, -x, -y, -z)` 是把 IMU 坐标系转到躯干坐标系。

**⑦ 归一化 + 拒绝退化四元数**
```python
n = math.sqrt(sum(v*v for v in q))
if n > 0.5:
    accepted = True
    quat = tuple(v / n for v in q)
```
`n > 0.5` 是一个安全下限——正常单位四元数模长应≈1，远小于 1 说明数据严重损坏，拒绝。

**⑧ 计算躯干系下的重力方向**
```python
w0, x0, y0, z0 = quat
gravity = (-2 * (x0*z0 - w0*y0),
           -2 * (y0*z0 + w0*x0),
           -(1 - 2 * (x0*x0 + y0*y0)))
```
这是标准四元数→重力向量公式：用旋转四元数把世界系重力 `[0,0,-1]` 变换到躯干系。直立时输出应为 `[0, 0, -1]`。

### 3. 返回值

```python
return {
    "gyro_raw": (gx, gy, gz),           # 原始 LSB
    "gyro_dps": (..., ..., ...),        # 换算后的 dps
    "quat_packed": packed,              # 半精度原始值
    "quat_xyz": (qx, qy, qz),           # f16 解包后的 float
    "quat_trunk": quat,                 # mount 校正+归一化后的完整四元数 (w,x,y,z)，被拒绝则 None
    "gravity": gravity,                 # 躯干系重力方向，被拒绝则 None
    "norm_sq": norm_sq,
    "finite": finite,
    "counter": b[12],
    "status": b[13],
    "reserved": b[14],
    "accepted": accepted,               # 主控会不会把这一帧计进 ready
    "window_hex": b[:12].hex(),         # StaleImuTracker 比较的 12 字节
}
```

### 4. 设计要点

| 设计决策 | 原因 |
|---------|------|
| 判据照抄 `imu.rs` 而非协议文档 | 工具过了 = 主控会认，避免"工具全绿、真机 `imu_ready()` 永远 false" |
| `any(packed)` 判全零 | 区分"总线活着但 IMU 融合没出数"和"正常出数"，全零帧不计入 ready |
| `norm_sq ≤ 1.02` 而非 `≤ 1.0` | 半精度 binary16 的舍入余量，太严会误杀正常帧 |
| `n > 0.5` 下限 | 防止退化四元数（接近零模）通过，这种帧做重力分解会得到垃圾值 |
| `window_hex` 取前 12 字节 | 主控 `StaleImuTracker` 逐字节比较这 12 字节，连续 25 帧相同就报 frozen——工具用同一窗口做冻结检测 |
| mount 共轭旋转 | IMU 物理安装绕 Y +90°，软件上转回躯干坐标系，这样重力方向在直立时才是 `[0,0,-1]` |

核心思路是：**decode_block 不仅是反序列化，更是一个与主控固件完全一致的验收器**——每一帧都按主控的真实判据判定 accepted/frozen/ready，确保模拟器和真板子用同一把尺子。

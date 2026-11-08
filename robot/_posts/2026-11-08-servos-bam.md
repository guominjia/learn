---
title: "舵机测试"
date: 2026-11-08
---

## BAM

**BAM 测试就是给仿真里的舵机做“校准”**：让真舵机带动一个已知的摆锤运动，把过程录下来；然后在仿真里用同样的指令重放，反复调整舵机模型的参数，直到仿真轨迹和实测轨迹重合。最终得到一个 JSON 参数文件，训练时 MuJoCo/mjlab 用它来模拟舵机。

### 为什么需要它

MuJoCo 默认的舵机模型是一个理想的 PD 控制器，摩擦只有两项（BAM 里称为 M1）：

$$
\tau_{fm} = K_v|\dot\theta| + K_c
$$

带减速箱的真舵机比这复杂得多：

- **静摩擦比滑动摩擦大**：从静止起步会先“卡一下”（Stribeck 效应）
- **负载越大，摩擦越大**：齿轮咬得越紧，阻力越大
- **方向不对称**：电机带动负载（drive）和负载反推电机（backdrive）时，摩擦不一样

策略在“理想舵机”上训练，到真机上就会出现抖动、迟钝或者站不稳。

### BAM 的舵机模型由三部分组成

以 ftservo-bam 中 HD-1910 的 `m6.json` 为例：

| 部分 | 参数 | 含义 |
|---|---|---|
| 电机电气 | `kt`, `R` | 力矩常数、线圈电阻，决定“给多少电压出多少力” |
| 惯性和延迟 | `armature`, `command_delay`, `q_offset` | 转子折算到输出轴的惯量、指令延迟、零位偏差 |
| 摩擦 | `friction_base` $K_c$、`friction_viscous` $K_v$ | 恒定摩擦、粘滞摩擦 |
| | `friction_stribeck`, `dtheta_stribeck`, `alpha` | 静止附近的额外摩擦，以及它随速度衰减的快慢 |
| | `load_friction_motor/external` $K_m, K_e$ | 随负载变化的摩擦，区分电机侧和负载侧 |
| | `*_stribeck`, `*_quad` | 负载摩擦的 Stribeck 项和二次项 |

M1 到 M6 是复杂程度逐级增加的六种摩擦模型。BAM 会把六种都拟合一遍，再根据验证误差挑出效果最好的那个。

## 测试流程（官方文档的四步）

```
1. 搭台架   舵机固定 + 摆杆 + 可换的质量块 / 摆长
2. 定义舵机  写一个 Actuator 子类，告诉 BAM 这个舵机怎么把指令变成电压
            （例如 HD1910Actuator: vin=7.4, kp=32）
3. 录数据   跑标准激励轨迹，记录 目标角 / 实际角 / 速度 / 电压 …
            换不同负载、摆长再录多组
4. 拟合     仿真重放同样指令，优化器调参数使 仿真轨迹 ≈ 实测轨迹
            → 输出 m1.json … m6.json
```

用摆锤做负载的原因是它的物理规律完全已知（重力力矩 $mgL\sin\theta$，惯量 $mL^2$）。仿真和实测之间的差异只能来自舵机本身，所以拟合出来的就是纯粹的舵机特性。

## 测完怎么用

训练时加载 JSON。仿真每一步都用这个模型来计算：给定的目标角会让舵机输出多少力矩、被摩擦吃掉多少、最终怎么动。这样策略学到的就是“HD-1910 实际会怎么动”，而不是理想舵机的动作。

## 飞特官方仓库里有什么

[这个仓库](https://gitee.com/ftservo/ftservo-bam)是飞特官方从 [Rhoban/bam](https://github.com/Rhoban/bam) 同步并扩展出来的，里面**已经有 HD-1910 的辨识结果**，可以作为起点；但离“直接拿来用”还差几步验证。

| 内容 | 位置 |
|---|---|
| HD-1910 的 7.4 V 辨识参数，m1 到 m6 共 6 种摩擦模型 | `bam/params/feetech_hd1910_7_4V/` |
| 飞特舵机的录制脚本 | `bam/feetech/record.py`、`all_record.py` |
| HD-1910 的执行器定义 | `bam/feetech/actuator.py` 中的 `HD1910Actuator` |

## 直接使用前要注意的四点

**1. 只有 7.4 V 一档。** 最好在 6.6、7.4、8.2 V 下都测，电池从满电到放空电压会一直变化。

**2. 舵机 P 增益固定为 `kp=32`。** 实际写进舵机寄存器的 P 值必须与它一致，否则这套参数不适用。

**3. 同一款舵机，两个来源的参数差别很大。** [xgoduck 仓库](https://github.com/LuwuDynamics/xgoduck_rl)也带了一份 `params/1910_m6.json`，两份的 m6 对比如下：

| 参数 | ftservo-bam | xgoduck | 差异 |
|---|---|---|---|
| `kt` | 0.564 | 0.624 | 约 10% |
| `R`（Ω） | 2.97 | 4.91 | 约 65% |
| `command_delay` | 1.0 ms | 6.0 ms | 6 倍 |
| `load_friction_motor` | 0.353 | 0.069 | 5 倍 |
| `friction_viscous` | 0.0048 | 0.0003 | 15 倍 |

两份数据都可能是对的，只是测试条件不同：舵机个体、Kp、固件版本、台架和拟合时的初值都可能不一样。xgoduck 那份的 `actuator` 字段写的是 `sts3215`，说明它是借 STS3215 的模型结构拟合 HD-1910，比飞特这份多了限速和误差增益参数。由此可见，**这类参数不能跨环境直接照搬**。

**4. 测试条件没有记录。** `params/README.md` 只有一句话，没写用的是什么摆杆和负载、哪个固件版本、录了多少条轨迹。

参考：[BAM 文档](https://bam.readthedocs.io/en/latest/)、[摩擦模型 M1–M6](https://bam.readthedocs.io/en/latest/theory/models.html)、[辨识流程](https://bam.readthedocs.io/en/latest/identification/index.html)、[论文](https://arxiv.org/pdf/2410.08650)、[Rhoban Repo](https://github.com/Rhoban/bam)、[FTbam Repo](https://gitee.com/ftservo/ftservo-bam)。

[舵机拆机及原理讲解视频](https://www.bilibili.com/en/video/BV1TQ4y1Z7kg?spm_id_from=333.788.videopod.sections)
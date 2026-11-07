---
title: "机器鸭学习：EDA"
categories: [hardware]
tags: [eda]
---

`.eprj2` 是 `EasyEDA Pro` （即嘉立创专业版）的项目文件，本质是个 SQLite 数据库。

`wire_hub_1V0A.eprj2` 对应数据库版本 `25.10.31.1`。具体的原理图和 PCB 图元数据存在 `history_data` 表里，是压缩/编码过的，只能用 EasyEDA Pro 打开。下面的结构信息来自数据库，内容判断依据的是文件里自带的两张预览图。

**工程结构**
- 板子 `wire_hub`：包含 1 张原理图 `wire_hub_SCH`（1 页 “1-Conector”）、1 个 PCB `wire_hub_PCB` 和 1 个拼板 `Panel1`
- 有 205 条修改历史，时间从 2026-09-05 到 2026-10-29。数据库里的工程名还是默认的 `New Project_2026-09-05_10-48-28`

**电路功能**

这是一块 **Dynamixel 舵机总线分线板**，没有任何有源器件。所有接插件都并联在同三个网络上：`Battery`、`GND`、`DXL_BUS`（单线半双工 TTL 数据线）。

| 位号 | 型号 | 说明 | 引脚 |
|---|---|---|---|
| CN1 | DB125-2.54-2P-GN-S | 2P 接线端子，用于电源输入 | 1=Battery, 2=GND |
| J1 | HC-1.2-3PWT | 1.2mm 3P 卧贴 | 1=GND, 2=Battery, 3=DXL_BUS，4/5 固定脚接 GND |
| J2 | BM03B-ACHSS-GAN-ETF | JST ACH 1.2mm 3P 侧插 | 同 J1 |
| J3, J4 | A2007WR-3P | 2.0mm 3P 弯针 | 1=DXL_BUS, 2=Battery, 3=GND |
| J5, J6 | A2007WV-3P | 2.0mm 3P 直针 | 同 J3 |

**PCB 布局**
- 双层板，大致是 “工” 字形外形，底部有 2 个安装孔
- J1、J2 和 CN1 在上方，J5、J6 在中部，J3、J4 在下方
- 顶层大面积铺 GND 铜，`DXL_BUS` 用细线绕板走一圈连到各个接口
- 丝印上标了 S/V/G（信号/电源/地）和 “WIRE_HUB V1.0A”

**需要注意的地方**

J1、J2 的引脚编号是 pin1=GND，与 Dynamixel X 系列的 JST 定义一致。J3 到 J6 的编号是反过来的，pin1=DXL_BUS。按网络连接看两组都是对的，但做线时要以丝印上的 S/V/G 为准，不要按引脚号对线。

`.step` 是 嘉立创专业版导出的 **PCB 三维装配模型**，用于机械结构配合，做结构干涉检查。

可以用 FreeCAD、Fusion 360 或 SolidWorks 直接导入，也可以用在线 STEP 查看器打开查看。
---
title: "机器鸭复刻：音频"
date: 2026-11-16
categories: [robot, microduck]
---

目前机器鸭复刻使用源项目的 HAT 电路版，[点此](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/blob/main/production/ASE01187-D1_elec_RPI_Robot_HAT_BOM.csv)查看对应物料表。

其中的音频输出接口是 **J2** ，连接到 **PAM8406DR** [LCSC C86270](https://www.lcsc.com/product-detail/C86270.html) 立体声功放的一个声道，使用的是螺丝压接端子 [LCSC C918120](https://www.lcsc.com/product-detail/C918120.html)，具体参数如下：

| 属性 | 值 |
|---|---|
| **制造商** | DORABO |
| **MPN** | DB125-2.54-2P-GN-S |
| **类型** | 接线端子 (Wire to Board / 螺钉式) |
| **引脚数** | 1×2P (2 位) |
| **间距** | 2.54mm |
| **安装方式** | 通孔 (Through Hole) |
| **螺钉规格** | M1.6 |
| **额定电流** | 7A |
| **额定电压** | 150V |
| **线规** | 18~26 AWG |
| **颜色** | 绿色 |
| **工作温度** | -40℃ ~ +105℃ |
| **板上高度** | 8.4mm |

**J2 接口规格：**
- 2.54mm 间距 2 位**绿色螺钉式**接线端子（DB125-2.54-2P-GN）
- 功放输出：3W@4Ω / 5W@2Ω（5V供电），Class-D 免滤波架构

**淘宝上兼容的喇叭选择：**

| 类型 | 规格 | 说明 |
|---|---|---|
| **3W 4Ω 微型喇叭** | 4Ω / 3W，带裸线 | 最匹配，PAM8406 在 4Ω 负载下输出 3W |
| **2W 8Ω 小喇叭** | 8Ω / 2W，带裸线 | 兼容，音量略小但更安全 |
| **1.5W 8Ω 扬声器** | 8Ω / 1.5W，40mm 直径 | 机器人常用尺寸 |

如果需要立体声，J2 是一个声道，另一个声道接在 J1 或 J9（同样规格的端子）。
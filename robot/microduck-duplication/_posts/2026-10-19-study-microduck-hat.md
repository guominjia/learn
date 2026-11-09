---
title: "机器鸭复刻：学习 HAT 电路板"
date: 2026-11-99
categories: [robot, microduck]
---

[点此到 HAT 原始仓库](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/)

## HAT 连接器

HAT 板上的连接器如下：

| 位号 | 型号/封装 | 间距 | 用途 |
|---|---|---|---|
| J3, J11 | JST EH B4B-EH-A，4 pin | 2.50 mm | Dynamixel-4（RS-485） |
| J13, J14 | JST EH B3B-EH-A，3 pin | 2.50 mm | Dynamixel-3（TTL） |
| J5–J8 | JST SH BM04B-SRSS-TB，4 pin | 1.00 mm | Qwiic / I²C |
| J4 | 2×20 母座（SMD） | 2.54 mm | 树莓派 HAT 排针 |
| J1, J2, J9 | DB125 螺丝端子，2 pin | 2.54 mm | 电源 |

舵机口用的是 JST EH 系列，间距 2.5 mm，和 Dynamixel X 系列舵机的接口一致。

## HAT 电源

HAT 没有单独的电源座，电是从舵机口倒灌进来的。Pollen 官方 README 原话：

> The RPI_Robot_HAT can be power from 5 to 28V. It is designed to use of the motor connector as an power input.

**供电路径**

```
NP-F550 电池（2S，6.6–8.4 V）
  └─► 躯干里的舵机总线 HUB（复刻项目自制，CN1 螺钉端子进电）
        ├─► 各路舵机（电池原压直供，不降压）
        └─► 过脖子的 3 线总线（Battery / DXL_BUS / GND）
              └─► HAT 的 J13/J14 电源脚 = +BATT（经 TH1，标注 “3A max”）
                    └─► U9 AP63205 降压（2 A）
                          └─► LM5050-1 理想二极管 ──► +5V
                                ├─► 40 针 2/4 脚 → Radxa 主控
                                ├─► PAM8406D 功放
                                └─► XC6206 → 1V8（音频 codec）
```

所以电池、舵机电源和 HAT 共用同一根总线，没有第二条电源线。

**复刻项目踩过的供电坑**（出自 [踩坑记录.md](https://github.com/fanhao375/microduck-replica/blob/master/%E8%B8%A9%E5%9D%91%E8%AE%B0%E5%BD%95.md)）

| 问题 | 原因 | 处理 |
|---|---|---|
| 走路或起身时整机掉电重启 | 十几颗舵机一起发力时电压下跌，HAT 降压出来的 5 V 跟着掉 | 用电池或 10 A 电源，16–18 AWG 短线，XT30/XT60 接头，输入端并 1000–2200 µF 电容 |
| 用移动电源加诱骗线供电不行 | 电流一大，PD 协议保护会退回 5 V，舵机没力，主板 5 V 也会掉到降压下限 | 改用 NP-F550 电池，或可调电源设约 8.0 V |
| 满电电压没有余量 | 电池满电 8.4 V，正好是 HD-1910 的上限 | 刚充满时别做堵转测试 |
| HAT 供电时又插 USB 数据线 | Radxa 的 40 针 5 V 和 OTG 口的 VBUS 是同一个网络，两路 5 V 会并在一起 | 有 HAT 时用 WiFi + SSH 调试；要用 USB 就先断开 HAT 输入 |
| 7.4 V 接错位置 | 只能从 HAT 的输入进 | 不能直接接主板 5 V 或 USB-C |

另外，LM5050 这一路还给主控提供关机检测信号（GPIO06）。如果不打 HAT、自己飞线供电（UBEC 5 V 接 40 针 2/4 脚），就没有这个信号，还要自己串 5 A 保险丝和电源开关。

提到了。降压就在 HAT 板上，不用另外加模块。

**HAT 上的降压电路**

[BOM](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/blob/main/production/ASE01187-D1_elec_RPI_Robot_HAT_BOM.csv) 里对应这几颗：

**要注意的地方**

- **输出电流只有 2 A**：Radxa Zero 3W 加摄像头、WiFi 和音频，平时够用；但在 85°C 以上满载的时候余量不大。
- **输入电压不能太低**：降压芯片的输入必须明显高于 5 V。复刻项目踩过的坑是移动电源过流后退回 5 V，文档原话是“5 V 也低到 HAT 给主板降压的下限，主板可能重启”。
- **电池电压会被舵机拉低**：舵机瞬间大电流时，电池端可能跌到 4–5 V，这时降压输出也跟着掉，主板就会重启。走路掉电就是这个原因。所以要用能出大电流的电源、粗线，输入端并大电容。

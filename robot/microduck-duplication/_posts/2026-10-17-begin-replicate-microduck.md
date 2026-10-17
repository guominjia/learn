---
title: "机器鸭复刻：开始"
date: 2026-10-17
categories: [robot, microduck]
tags: [robot, microduck]
---

自机器鸭爆火差不多过去几个月了，一直没具体去了解，最近和同事聊到这个，突然有了想要复现它的想法。
你问我为什么想复现它？可能就是想增强自己的硬件能力和设计能力吧，以及体验复现一个产品的感觉吧。

我从4天前开始找相关资料，但一直没做记录。今天先对资料整理记录下，同时作为**正式开始的FLAG**。

我打算从今天开始记录所有的进展，一则作为学习记录便于后续复习，二则回首时也会有成就感。

本系列记录全部手写，不用AI。
原因是：个人觉得AI虽然能提高效率，但对于个人学习却无益，另外自己手写也可提高自己写作能力。
但获取相关资料，答疑解惑我会用AI帮忙。

**废话到此，开始正文**。

在与AI交谈后，我了解到目前机器鸭复刻有两个：
- **Open Duck Mini v2**
- **官方 MicroDuck**

## 两者比较

下面是AI提供的详细比较表格：

| | Open Duck Mini v2 | MicroDuck（官方产品）|
|---|---|---|
| 出品 | Antoine Pirrone 社区项目，HF 和 Pollen 赞助 | Pollen Robotics（HF 的机器人团队）正式产品 |
| 外形 | 约 42 cm，仿迪士尼 BDX，14 个关节舵机 | 25 cm，15 个电机，有摄像头、LiDAR、能夹东西的鸭嘴 |
| 开源范围 | CAD、STL、BOM、接线图、Runtime、RL 训练全部开源（Apache-2.0） | 开源软件与部分硬件。官方 press kit 明确说机械和电子设计文件不开源，不能称为开源硬件 |
| 价格 | 官方 BOM 目标 < $400 | 预售 $399（未含税和运费），2026-08-27 开订，目标圣诞前首批交付 |
| 发货地区 | — | 首发只有北美和欧洲，中国能不能送到未确认 |

AI提供的来源：[Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini)，[MicroDuck press kit](https://pollen-robotics.com/microduck/press-kit/)，[MicroDuck 官网](https://pollen-robotics.com/microduck/)

## Open Duck Mini v2 详细信息

### 市场情况

Open Duck Mini 已有成品和套件在卖：

| 渠道 | 形态 | 价格 |
|---|---|---|
| [Tnkr](https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home) | 官方 README 推荐的散件套件，带 43 步交互装配教程 | 未知 |
| [Cereboto](https://cereboto.com/product/open-duck-mini-v2/) | 整机 | $780 |
| [Tindie / opmake（广东）](https://www.tindie.com/products/wsk/open-duck-mini-v2-integrated-robot-kit-rk3576d/) | 已组装整机，主控升级为 RK3576D | $900–1000 |
| 淘宝、navbot、aifitlab 等 | 套件或整机 | 不确定 |

### 相关资料

中文资料：

- 同济子豪兄的 [飞书课程和 BOM](https://zihao-ai.feishu.cn/wiki/space/7488517034406625281)
- 第三方站点 [microduckrobot.org](https://microduckrobot.org/zh/diy/)

## MicroDuck 详细信息

### 市场情况

MicroDuck：官方本体 $399；另有 Dev Pack $119（3 个备用电机、电池、充电器等）和配件包 $39。2026-09-07 的商店快照显示，新订单预计 4–6 个月交付。

### 复刻情况

MicroDuck 复刻：

- [fanhao375/microduck-replica](https://github.com/fanhao375/microduck-replica) 根据官方仿真网格反推结构件，用飞特 HD-1910 舵机替代原装电机。
  - 进展到 10 月 12 日：15 颗舵机都有应答，策略能保持姿态约 70 秒，双腿能交替迈步。但前进距离没测，还有脚底打滑、散热、供电的问题待查。
  - k-issqy 发布了 Public Image R1 系统镜像，但只适配特定组合：Radxa Zero 3W 主板 + HD1910 舵机 + 总线 IMU。
  - 该复刻仓库的模型和文档用 CC BY-NC-SA 4.0（禁止商用），想做商业化要注意。

## 仿真环境

- [MicroDuck 浏览器仿真](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
- [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)
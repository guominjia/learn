---
title: "机器人脸：网页 Demo 与开源实现"
date: 2026-10-15
categories: [web]
tags: [web, robot]
---

想在网页或小屏幕上做一张会眨眼、有表情的机器人脸，可以参考下面这些项目。

**浏览器里直接体验**
- **Stack-chan Web Tools**：<https://stack-chan.github.io/stack-chan/web/>。Stack-chan 是一个开源桌面小机器人项目，其网页工具中与机器人脸相关的有两个：
  - [Shape Face Editor](https://stack-chan.github.io/stack-chan/web/face-editor/)：自由摆放眼睛和嘴巴，拼出一张机器人脸。
  - [Simulator](https://stack-chan.github.io/stack-chan/web/simulator/)：在浏览器中运行 Stack-chan 的固件和 3D 模型。
- **LottieFiles Robot Face 动画合集**：<https://lottiefiles.com/free-animations/robot-face>。设计师制作的现成机器人脸动画，可导出为 Lottie JSON、GIF 或 MP4，直接嵌入网页。

**开源实现（面向硬件屏幕，代码可借鉴）**
- **FluxGarage RoboEyes**：<https://github.com/FluxGarage/RoboEyes>。在 Arduino 的 OLED 小屏上绘制机器人眼睛，支持自动眨眼、眼球随机转动，以及开心、生气、疲倦等表情和大笑、困惑等小动画。README 中还列出了 MicroPython 版和 TFT 屏版。
- **M5Stack-Avatar**：<https://github.com/stack-chan/m5stack-avatar>。在 M5Stack 屏幕上绘制头像脸，支持切换表情、随声音开合嘴巴和自定义脸型。

**更多灵感**
- 在 CodePen 上搜索 `robot eyes`、`robot face`、`eyes follow cursor`，能找到不少纯 CSS/JS 实现的小 Demo。

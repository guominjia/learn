---
layout: page
title: "从电路节点到 DDR5：一步步理解高速信号反射"
permalink: /cpu/memory/transmission-line-ddr5-series/
---

这是一组从基础电路概念开始，逐步进入传输线、反射、DDR5 ODT/RON/RCOMP 和 DDRIO 代码阅读的系列文章。

阅读时建议按下面的顺序进行。系列文章的文件名和 front matter 日期按系列编号排列，并保持一致。

## 基础模型

1. [电压、电流和输入输出不是一回事]({% post_url 2026-09-11-circuit-nodes-ports %})
2. [什么时候一根导线必须当作传输线]({% post_url 2026-09-12-when-wire-becomes-transmission-line %})
3. [传输线上的 V+ 和 V-]({% post_url 2026-09-13-transmission-line-waves %})
4. [从边界条件推导反射系数]({% post_url 2026-09-14-reflection-boundary-condition %})
5. [5 V、2.5 V 和开路瞬态]({% post_url 2026-09-15-5v-25v-open-circuit %})
6. [阻抗匹配到底匹配什么]({% post_url 2026-09-16-impedance-matching %})
7. [从一次反射到振铃和眼图闭合]({% post_url 2026-09-17-ringing-overshoot-eye %})

## DDR5 与实现

8. [ODT、RON、RCOMP 分别在控制什么]({% post_url 2026-09-18-odt-ron %})
9. [DDR5 读写方向变化]({% post_url 2026-09-19-ddr5-read-write-source-load %})
10. [DB ODT、RDIMM 和 LRDIMM]({% post_url 2026-09-20-db-odt-dimm-topology %})
11. [RCOMP、校准和 Training]({% post_url 2026-09-21-rcomp-calibration-training %})
12. [从反射公式到 DDRIO 初始化代码]({% post_url 2026-09-22-ddrio-code-reading %})

## 三条贯穿主线

```text
Vs != V+ != VL

输入/输出：系统端口视角
向前/向后：传输线传播视角
流入/流出：电流参考方向

阻抗匹配
  -> 反射减小
    -> 波形改善
      -> 采样裕量提高
```

系列中的阻抗和训练讨论是通用分析框架，不替代具体 DRAM、控制器、DIMM 或平台的官方时序和寄存器文档。

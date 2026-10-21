---
layout: page
title: "从电路节点到 DDR5：一步步理解高速信号反射"
permalink: /cpu/memory/transmission-line-ddr5-series/
---

这是一组从基础电路概念开始，逐步进入传输线、反射、DDR5 ODT/RON/RCOMP 和 DDRIO 代码阅读的系列文章。

阅读时建议按下面的顺序进行。系列文章的文件名和 front matter 日期按系列编号排列，并保持一致。

## 这套知识属于哪个学科

DDR5 的信号问题本质上是**传输线理论和信号完整性**，和通信原理、射频/微波里的那套模型是同一个。对应关系：

| DDR5 现象 | 通信原理中的概念 |
|---|---|
| 信号沿 PCB 走线传播 | 电磁波在传输线上传播 |
| 走线有特征阻抗 | 传输线特性阻抗 $Z_0$ |
| 端点阻抗不匹配 | 阻抗失配 |
| 波形返回 | 反射波 |
| 反射波与原信号叠加 | 波的叠加与干涉 |
| 过冲、下冲、振铃 | 瞬态响应、多次反射 |
| 眼图变差 | 噪声、码间串扰、接收裕量下降 |
| ODT / RON | 负载端终端匹配、源端阻抗匹配 |

反射系数在两边是同一个式子：

$$
\Gamma=\frac{Z_L-Z_0}{Z_L+Z_0}
$$

差别在出发点和关注量：

- 通信原理通常从连续波或模拟脉冲出发，研究反射、驻波、匹配网络；
- DDR5 是高速数字系统，关注的是 0/1 的电压波形、采样时刻、眼图、抖动和误码率；
- DDR5 的端接阻抗不是理想固定电阻，会随工艺、电压、温度、输出状态和码型变化，因此需要校准和训练。

一个常被误解的判断：**是否需要按传输线处理，不由"它是不是数字信号"决定，而主要由走线传播延迟和信号边沿时间的关系决定**，即

$$
t_r\lesssim 2t_d
$$

因此整套知识大致跨三层：

```text
电磁场 / 通信原理：波传播、反射、阻抗匹配
  -> 高速数字电路：信号完整性、眼图、串扰、抖动
    -> DDR5 接口实现：ODT、RON、VREF、写入均衡、训练和校准
```

学习路径建议也按这个顺序：先掌握传输线反射和阻抗匹配，再看 DDR5 的 ODT/RON 会顺很多。本系列的编排正对应这三层。

## 基础模型

1. [电压、电流和输入输出不是一回事]({{ '/cpu/memory/circuit-nodes-ports/' | relative_url }})
2. [什么时候一根导线必须当作传输线]({{ '/cpu/memory/when-wire-becomes-transmission-line/' | relative_url }})
3. [传输线上的 V+ 和 V-：从电报方程到传播波]({{ '/cpu/memory/transmission-line-waves/' | relative_url }})
4. [从边界条件推导反射系数]({{ '/cpu/memory/reflection-boundary-condition/' | relative_url }})
5. [5 V、2.5 V、开路瞬态和示波器测量]({{ '/cpu/memory/5v-25v-open-circuit/' | relative_url }})
6. [阻抗匹配、反射功率和 VSWR]({{ '/cpu/memory/impedance-matching/' | relative_url }})
7. [从一次反射到振铃和眼图闭合]({{ '/cpu/memory/ringing-overshoot-eye/' | relative_url }})

## DDR5 与实现

8. [ODT、RON、RCOMP 分别在控制什么]({{ '/cpu/memory/odt-ron/' | relative_url }})
9. [DDR5 读写方向变化]({{ '/cpu/memory/ddr5-read-write-source-load/' | relative_url }})
10. [DB ODT、RDIMM 和 LRDIMM]({{ '/cpu/memory/db-odt-dimm-topology/' | relative_url }})
11. [RCOMP、校准和 Training]({{ '/cpu/memory/rcomp-calibration-training/' | relative_url }})
12. [从反射公式到 DDRIO 初始化代码]({{ '/cpu/memory/ddrio-code-reading/' | relative_url }})

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

其中，第 3 篇负责建立电报方程和有损传输线模型；第 5 篇区分 $V_s$、$V^+$ 和 $V_L$，并用 50 $\Omega$/1 M$\Omega$ 示波器输入说明为什么万用表看不到瞬态反射；第 6 篇从电压反射扩展到反射功率和 VSWR。第 7 篇再把两端反射系数连接到多次往返、振铃和眼图。

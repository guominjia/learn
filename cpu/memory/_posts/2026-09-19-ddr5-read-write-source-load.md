---
layout: post
title: "DDR5 读写方向变化：谁是源端，谁是负载端"
date: 2026-09-19
tags: [ddr5, signal-integrity, odt, memory-controller]
---

# 从电路节点到 DDR5：一步步理解高速信号反射（9）

在单向链路里，谁是源端通常很直观。DDR5 的 DQ 通道是双向的：写入和读取使用同一组连接，但驱动者不同。因此 ODT 和驱动器参数必须跟着操作方向理解。

## 1. 写操作

在控制器向内存写数据时，典型的抽象是：

```text
Memory Controller  ──驱动──>  DRAM
       源端                         负载端
```

此时控制器的输出驱动器、封装和走线起点共同形成源端等效阻抗。DRAM 侧的接收器和 ODT 共同形成负载端边界。

如果把 DRAM 永远称为“输出端”，就会把这个场景说反。

## 2. 读操作

读数据时方向反过来：

```text
Memory Controller  <──接收──  DRAM
       负载端                     源端
```

DRAM 的数据输出驱动器对应 $R_\mathrm{ON}$，控制器侧接收端的终端状态对应负载边界。与写操作相比，源端和负载端的阻抗组合发生了变化。

这也是为什么同一个 `ODT` 名字必须结合器件角色、命令状态和时序来解释，不能只看一个静态电阻值。

## 3. DQ、DQS 和 CK/CA 不能完全类比

### DQ

DQ 是数据路径，读写时方向会切换。ODT 和驱动器参数要围绕当前数据源和接收端配置。

### DQS

DQS 是数据选通信号。它与 DQ 的关系不仅是“另一根时钟线”，还涉及源同步采样、相位关系和训练。DQS 的终端和驱动设置不能直接从 DQ 的设置推断。

### CK/CA

CK、命令和地址的方向、拓扑和接收角色通常不同于双向 DQ。它们可能共享某些阻抗设计思想，但不能把 DQ 的读写模型原样套过去。

## 4. 用操作状态画链路

比背寄存器字段更可靠的做法，是先画出当前状态的源端和负载端：

```text
写状态：
Controller output -> channel -> DRAM input/ODT

读状态：
DRAM output      -> channel -> Controller input/ODT
```

然后分别问四个问题：

1. 当前谁在驱动电压波？
2. 当前谁的输出阻抗构成 $Z_S$？
3. 当前谁的终端构成 $Z_L$？
4. 反射波返回时，哪一端会再次反射？

## 5. 为什么 ODT 需要状态切换

ODT 的目标不是固定地“把一个引脚接上某个电阻”，而是在不同状态下让有效负载阻抗落入可接受范围。实际值还会受到：

- 并行连接的接收器数量；
- rank 和 DIMM 拓扑；
- 控制器与 DRAM 封装；
- 读写切换的时序窗口；
- 电压、温度和校准结果。

影响。

因此看一段初始化代码时，要把“默认配置”“写状态”“读状态”“空闲或驻留状态”分开看。

## 本篇结论

```text
写：Controller 是源端，DRAM 是负载端
读：DRAM 是源端，Controller 是负载端
```

ODT、RON 和终端选择只有放回具体方向和拓扑里才有意义。下一篇继续增加一个节点：当 DIMM 中有 Data Buffer 时，Controller、DB 和 DRAM 之间其实是两段链路。

## References

- [Micron DDR5 SDRAM](https://www.micron.com/products/memory/dram-components/ddr5-sdram)：DDR5 的 DQ/DQS ODT、训练和信号完整性特性概览。
- [Transmission line](https://en.wikipedia.org/wiki/Transmission_line)：源端、负载端、传播方向和反射的通用模型。

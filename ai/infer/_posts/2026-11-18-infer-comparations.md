---
title: "Inference Comparations"
---

## Strata vs KTransformers vs vLLM 对比

三个项目都涉及大模型推理，但定位、架构和目标用户差异很大。以下从多个维度详细对比。

---

### 一、项目定位

| 维度 | Strata | KTransformers | vLLM |
|------|--------|---------------|------|
| **定位** | 消费级硬件上运行 **Qwen3.8-Flash-Next** 的一键安装桌面/终端应用 | CPU-GPU 异构推理/微调 **通用框架**，面向 MoE 大模型 | 高性能生产级 **推理引擎**，面向数据中心 |
| **目标用户** | 普通用户、开发者（个人 PC） | 研究者、开发者（单机/多卡） | 企业、云服务（多节点集群） |
| **一句话** | "把你的 PC 变成本地 AI 工作站" | "用有限 GPU 跑超大 MoE 模型" | "用多卡集群跑最高吞吐" |
| **Stars** | 44.6k | 19.7k | ~50k+ |

---

### 二、支持的模型

| 项目 | 模型范围 |
|------|---------|
| **Strata** | **仅 Qwen3.8-Flash-Next 家族**（Q2_0/IQ2_XS/IQ3_XXS/IQ3_S/Coder/Swift 1.5/Unsloth/Gyro/NVFP4）+ Qwen3.6-35B-A3B。深度定制，只跑这一个架构 |
| **KTransformers** | **非常广泛**：DeepSeek-V3/R1/V4-Flash、Kimi-K2/K2.5/K2.6、Qwen3MoE、GLM-5/5.2/5.3、MiniMax-M2.1/M2.5/M3、LLaMA 4、SmallThinker 等。Day-0 支持新模型 |
| **vLLM** | **最广泛**：几乎所有主流开源模型（Llama、Qwen、Mistral、DeepSeek、Falcon 等），HuggingFace 格式原生支持 |

---

### 三、核心架构

| 维度 | Strata | KTransformers | vLLM |
|------|--------|---------------|------|
| **异构策略** | **GPU + CPU + SSD 三级**：GPU 做 attention/mixer/router，CPU 算未缓存的 expert，SSD 存 n-gram 表 | **GPU + CPU 两级**：热 expert 放 GPU，冷 expert 放 CPU（AMX/AVX 加速），支持 3 级（+磁盘）prefix cache | **纯 GPU**（默认），支持 CPU offload 但非核心设计 |
| **显存需求** | **12-24 GB** 单卡即可运行 245B 模型 | **24 GB 起**（DeepSeek-V3 单卡 24GB 可跑），多卡更好 | **多卡集群**（单卡也能跑小模型） |
| **内存需求** | **48-64 GB** 系统内存存所有 expert | 取决于模型，DeepSeek-V3 约 382 GB（全量） | 通常不需要大量系统内存 |
| **Speculative Decode** | **MTP draft layer**（模型自带）+ prompt lookup，2.4-3.2 tokens/pass | 未强调 | 支持（Medusa、Eagle、MLP speculator） |
| **批处理** | 单请求为主，支持 `--parallel N` 连续批处理 | 支持多并发（balance-serve） | **核心优势**：continuous batching，高吞吐 |
| **KV Cache 优化** | KV streaming（64K 后部分放 RAM）、8-bit/q4_0/k8v4、KV-grow | 3 层 prefix cache（GPU-CPU-Disk） | **PagedAttention**（核心创新）、prefix caching |

---

### 四、性能数据

#### Strata（RTX 5070 12GB, Ryzen 5 7600, 64GB RAM, Q2_0）

| 指标 | 4K | 32K | 128K | 262K |
|-----|-----|-----|------|------|
| Prompt 处理 | 1,299 tok/s | 2,171 tok/s | 2,126 tok/s | 2,107 tok/s |
| 输出生成 | 93.0 tok/s | 81.8 tok/s | 76.2 tok/s | 73.7 tok/s |

#### KTransformers（DeepSeek-R1-0528 FP8, 8×L20 + Xeon Gold）

| 指标 | 数值 |
|-----|------|
| 单请求吞吐 | 227.85 tok/s |
| 8 路并发 | 87.58 tok/s |

#### vLLM（典型数据中心配置）

| 指标 | 数值 |
|-----|------|
| 吞吐 | 通常比 llama.cpp 高 2-10x（continuous batching 优势） |
| 延迟 | 优化目标不同，单请求延迟可能高于专用方案 |

---

### 五、硬件支持

| 项目 | NVIDIA | AMD | Intel | Apple | 其他 |
|------|--------|-----|-------|-------|------|
| **Strata** | ✅ CUDA (RTX 20/30/40/50) | ✅ HIP (RX 7700/7800/7900/9060/9070) | ✅ SYCL (Arc Linux) | ✅ Metal (M1+) | DGX Spark |
| **KTransformers** | ✅ CUDA | ✅ ROCm | ✅ XPU (Arc) | ❌ | ✅ Ascend NPU |
| **vLLM** | ✅ CUDA | ✅ ROCm | ✅ XPU | ❌ | ✅ Neuron、TPU、OpenVINO |

---

### 六、安装与使用体验

| 维度 | Strata | KTransformers | vLLM |
|------|--------|---------------|------|
| **安装** | **一键安装**：.exe / .AppImage / .dmg 或 `irm ... | iex` | `pip install ktransformers` + 编译 C++ 内核 | `pip install vllm`（预编译 wheel） |
| **启动** | `strata launch` → 选模型 → 自动下载 → 开聊 | 手动下载模型 → 配置 → 启动 | `vllm serve model_name` |
| **UI** | ✅ 桌面 App + Web UI + TUI | ❌ 无内置 UI（通过 SGLang API） | ❌ 无内置 UI（OpenAI API） |
| **上手难度** | ⭐（最简单） | ⭐⭐⭐（中等） | ⭐⭐（较简单） |
| **微调** | ❌ 不支持 | ✅ **核心能力**：LoRA/全参/DPO，集成 LLaMA-Factory | ❌ 推理专用（有独立微调工具） |

---

### 七、技术深度对比

#### Strata 的独特优势

1. **三级存储架构**：GPU → RAM → SSD，让 12GB 显卡跑 245B 模型成为可能
2. **自适应 expert 缓存**：动态学习对话中哪些 expert 最常用，自动调整 GPU 缓存
3. **MTP 投机解码**：利用模型自带的 MTP 层做 draft，1.6-1.8x 加速
4. **极致的消费级优化**：每个 VRAM 字节都精打细算（KV streaming、expert cache sizing、PCIe share 动态调整）
5. **开箱即用体验**：桌面 App + 一键安装，非技术用户也能用

#### KTransformers 的独特优势

1. **模型覆盖面广**：DeepSeek、Kimi、GLM、MiniMax 等最新 MoE 模型 Day-0 支持
2. **AMX/AVX-512 CPU 加速**：Intel 最新指令集深度优化，CPU 算 expert 效率高
3. **微调能力**：唯一同时提供推理 + 高效微调（LoRA/全参）的异构框架
4. **SGLang 集成**：可嵌入 SGLang 生产级 serving 框架
5. **学术背景**：清华 MADSys Lab，SOSP 2025 论文

#### vLLM 的独特优势

1. **PagedAttention**：解决 KV cache 碎片化问题，显存利用率极高
2. **Continuous Batching**：请求级批处理，吞吐量远超传统方案
3. **生产级特性**：多节点分布式、量化服务、Prefix caching、automatic prefix sharing
4. **生态最广**：几乎所有云服务商和开源项目都集成 vLLM
5. **社区最大**：最活跃的贡献者、最完善的文档

---

### 八、适用场景总结

```
┌─────────────────────────────────────────────────────────┐
│                    场景选择矩阵                          │
├─────────────┬───────────┬───────────┬───────────────────┤
│  场景        │  Strata   │ KTransform│ vLLM              │
├─────────────┼───────────┼───────────┼───────────────────┤
│ 个人 PC 本地聊天  │ ★★★★★     │ ★★        │ ★★                │
│ 个人代码助手     │ ★★★★★     │ ★★★       │ ★★★               │
│ 单卡跑 DeepSeek │ ❌        │ ★★★★★     │ ★★                │
│ 多卡集群推理     │ ❌        │ ★★★★      │ ★★★★★             │
│ MoE 模型微调    │ ❌        │ ★★★★★     │ ❌                │
│ 生产 API 服务   │ ★★        │ ★★★       │ ★★★★★             │
│ 研究/实验       │ ★★        │ ★★★★★     │ ★★★★              │
│ 非技术用户      │ ★★★★★     │ ★         │ ★★                │
└─────────────┴───────────┴───────────┴───────────────────┘
```

---

### 九、关键结论

1. **Strata 不是 vLLM 的竞品**——它是面向**消费级硬件**的**垂直整合产品**（只跑 Qwen3.8-Flash-Next），而 vLLM 是面向**数据中心**的**通用推理引擎**。

2. **Strata 也不是 KTransformers 的竞品**——虽然都做 CPU-GPU 异构，但 Strata 深度定制单一模型架构（245B MoE + 24,576 experts），追求极致的单卡体验；KTransformers 是通用框架，覆盖更多模型且支持微调。

3. **Strata 的真正创新**在于把"GPU 不够就 CPU 算，CPU 不够就 SSD 读"这个朴素想法做到了极致——自适应 expert 缓存、KV streaming、PCIe share 动态调优、MTP 投机解码，每个环节都有实测数据支撑。

4. **KTransformers 的独特价值**是"用有限 GPU 资源跑最新 MoE 大模型 + 还能微调"，在 DeepSeek-V3/R1 时代一战成名，现在扩展到更多模型家族。

5. **vLLM 的不可替代性**在于生产环境的高吞吐和生态集成——如果你的场景是"服务成百上千用户"，vLLM 仍然是首选。
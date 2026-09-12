# Transformer

Transformer 的主线可以从 2017 年的论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 开始：用注意力机制替代循环和卷积，让序列中的 token 可以并行交互。

这篇文档作为 Transformer 的总入口，先解释原始架构的每个部件，再沿着注意力机制的演进，连接到今天的长上下文、低缓存和多模态模型。

## 1. 从 Attention Is All You Need 开始

在 Transformer 之前，序列模型主要使用 RNN、LSTM 和 GRU。它们逐步处理 token，存在两个问题：

- 训练阶段难以充分并行；
- 相距很远的 token 之间传递信息困难。

Transformer 的核心改变是：对整个序列一次性生成 Query、Key、Value，并用矩阵乘法计算 token 之间的相关性。原始模型仍然是 **Encoder-Decoder** 架构，但不再依赖循环状态。

原论文的基本结构可以概括为：

```text
source tokens
    -> token embedding + positional encoding
    -> encoder stack
    -> encoder memory

target tokens
    -> token embedding + positional encoding
    -> masked self-attention
    -> cross-attention over encoder memory
    -> feed-forward network
    -> output projection + softmax
```

原始 Transformer 的完整背景和历史见[Transformer：Introduction and History](_posts/2026-06-12-transformer-intro-and-history.md)。

## 2. Transformer 的整体骨架

一个 Transformer block 不是只有 Attention。它通常由以下部件组成：

| 部件 | 作用 |
| --- | --- |
| Tokenization | 把文本切成 token，并映射为整数 ID |
| Token embedding | 把 token ID 查表变成向量 |
| Positional encoding | 给没有顺序感的注意力注入位置信息 |
| Attention | 根据 Query-Key 匹配，从 Value 中读取信息 |
| Feed-forward network | 对每个 token 独立进行非线性变换 |
| Residual connection | 保留输入并改善深层网络的优化 |
| LayerNorm / RMSNorm | 稳定激活值和梯度 |
| Output projection | 把隐藏状态映射到词表 logits |

现代 decoder-only LLM 通常不再使用完整的 Encoder-Decoder，而是重复堆叠 Decoder block；但 token、位置、Attention、FFN、残差和归一化这些基本部件仍然存在。

## 3. Token 和词嵌入

### 3.1 Token 不等于单词

模型通常先通过 tokenizer 把文本切成 token。一个 token 可以是：

- 一个完整的词；
- 一个词的一部分；
- 标点或空格；
- 一个特殊控制符号。

因此模型实际处理的是 token 序列，而不是自然语言意义上的单词序列。

### 3.2 Embedding lookup

设词表大小为 $V$，隐藏维度为 $d_{model}$，embedding 矩阵为：

$$
E \in \mathbb{R}^{V \times d_{model}}
$$

token ID $t$ 对应的向量就是矩阵的一行：

$$
x_t = E[t]
$$

一个长度为 $L$ 的序列经过 embedding 后得到：

$$
X \in \mathbb{R}^{L \times d_{model}}
$$

Embedding 只解决“这个 token 是什么”，还没有解决“它位于序列什么位置”。位置信息由下一步注入。

## 4. 位置编码

Self-Attention 本身对 token 的排列没有顺序感。如果交换输入 token 的位置，只看无位置的注意力公式，模型无法区分这种交换。因此需要位置编码。

### 4.1 原始 Transformer：正弦位置编码

原论文把位置向量加到 token embedding 上：

$$
H_0 = X + P
$$

其中：

$$
P_{pos, 2i} = \sin\left(pos / 10000^{2i/d_{model}}\right)
$$

$$
P_{pos, 2i+1} = \cos\left(pos / 10000^{2i/d_{model}}\right)
$$

### 4.2 现代模型：RoPE

许多 decoder-only LLM 不再把完整的位置向量直接加到输入，而是在 Attention 内旋转 Query 和 Key：

$$
q_i' = R_i q_i, \qquad k_j' = R_j k_j
$$

由于：

$$
(R_i q_i)^T(R_j k_j) = q_i^T R_{j-i} k_j
$$

相对位置信息自然进入 Query-Key 的匹配分数。RoPE 的公式、实现布局，以及 GLM、DeepSeek-V3、Qwen3 的差异见[RoPE 专文](_posts/2026-07-03-transformer-rope-attention.md)。

要注意：RoPE 是**位置编码机制**，不是一种独立于 MHA 的注意力类别。一个模型可以同时使用 GQA 和 RoPE，也可以使用线性注意力和其他位置处理方法。

## 5. 注意力层：从 Q、K、V 到输出

### 5.1 Scaled Dot-Product Attention

输入隐藏状态先经过三个投影：

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

基本注意力公式是：

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中：

- $Q$ 表示当前 token 想查询什么；
- $K$ 表示每个 token 可以被怎样匹配；
- $V$ 表示匹配后真正取出的内容；
- $M$ 是可选的 attention mask；
- $\sqrt{d_k}$ 用于控制点积的数值尺度。

如果序列长度为 $L$，$QK^T$ 的形状通常是 $[L, L]$。这正是标准全注意力在长序列上计算和显存压力较大的原因。

### 5.2 Self-Attention 和 Cross-Attention

Self-Attention 中，$Q$、$K$、$V$ 都来自同一序列。它用于让序列中的 token 互相读取信息。

Cross-Attention 中：

- $Q$ 来自 Decoder 当前状态；
- $K$、$V$ 来自 Encoder 输出的 memory。

因此 Encoder 和 Decoder 的序列长度可以不同。若 Decoder 有 $L_q$ 个位置，Encoder 有 $L_k$ 个位置，注意力分数的形状就是 $[L_q, L_k]$。

### 5.3 Multi-Head Attention

MHA 不只计算一次注意力，而是把隐藏维度拆成多个 head：

$$
head_i = Attention(Q_i, K_i, V_i)
$$

$$
MHA(Q,K,V) = Concat(head_1, \ldots, head_h)W_O
$$

不同 head 可以学习不同的关系，例如局部搭配、指代关系、语法关系或长距离依赖。

## 6. Encoder、Decoder 和 Transformer Block

### 6.1 Encoder layer

原始 Encoder layer 的逻辑是：

```text
input
    -> self-attention
    -> residual + normalization
    -> feed-forward network
    -> residual + normalization
```

Encoder 的 self-attention 通常可以看到输入序列的全部位置，因此适合 BERT 一类的双向表示学习。

### 6.2 Decoder layer

原始 Decoder layer 多一个 cross-attention：

```text
input
    -> masked self-attention
    -> residual + normalization
    -> cross-attention over encoder output
    -> residual + normalization
    -> feed-forward network
    -> residual + normalization
```

现代 decoder-only LLM 去掉了 Encoder 和 cross-attention，保留带 causal mask 的 self-attention：

```text
input
    -> causal self-attention
    -> residual + normalization
    -> feed-forward network
    -> residual + normalization
```

### 6.3 Feed-Forward Network

Attention 负责 token 之间的信息交换，FFN 负责在每个 token 的隐藏维度内进行非线性变换。原始形式是：

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

现代 LLM 常使用 SwiGLU 等门控 FFN。MoE 则把一个 FFN 替换成多个 expert，由 router 为每个 token 选择少数 expert。MoE 主要改变的是 FFN 的容量和激活计算，不是 Attention 本身。

### 6.4 Residual 和 Normalization

残差连接让每一层学习增量：

$$
H_{l+1} = H_l + F(H_l)
$$

归一化则控制激活的尺度。原始 Transformer 使用 LayerNorm，许多现代 LLM 使用 RMSNorm，并采用 Pre-Norm 结构，把归一化放在子层之前。

## 7. 训练和推理

### 7.1 Causal Mask

自回归语言模型在位置 $t$ 只能读取位置 $0$ 到 $t$ 的信息，不能读取未来 token。实现上通常使用下三角 mask，把未来位置的 attention logit 设为 $-\infty$，softmax 后概率变为 0。

### 7.2 Teacher Forcing

训练下一个 token 时，输入和标签错开一位：

```text
input:  [a, b, c, d]
target: [b, c, d, e]
```

整个序列可以并行训练；推理时则需要逐 token 生成。

### 7.3 Prefill、Decode 和 KV Cache

推理通常分为两个阶段：

- **Prefill**：一次处理完整 prompt，建立历史 Key/Value；
- **Decode**：每次输入一个新 token，只计算新的 Query，并读取 KV cache。

KV cache 避免每一步重新计算历史 token，但它随序列长度增长。关于 Hugging Face 模型内部的 forward、Prefill、Decode 和 cross-attention，可以参考[Transformers 推理笔记](hf/transformers.md)。

PyTorch 的 `scaled_dot_product_attention`、causal mask、GQA head 对齐和 fused kernel 见[SDPA 专文](train/_posts/2026-06-12-scaled-dot-product-attention.md)。

## 8. 注意力机制的演进

注意力演进不是一条简单的“旧模型被新模型替代”的直线。不同机制主要在表达能力、KV cache、训练并行度、推理带宽和长上下文能力之间做取舍。

### 8.1 MHA：表达能力优先

MHA 为每个 query head 配置独立的 K/V head：

```text
Q heads = K heads = V heads
```

它结构清晰、表达能力强，但 KV cache 较大，长序列 decode 时容易受显存带宽限制。

### 8.2 MQA：共享一个 KV head

MQA 让多个 Query head 共享一个 K/V head：

```text
Q heads >> K heads = V heads = 1
```

它显著减少 KV cache，但共享程度过高可能损失表达能力。

### 8.3 GQA：质量和缓存的折中

GQA 使用较少的 KV head，让一组 Query head 共享同一个 KV head：

```text
Q heads > K heads = V heads
```

GQA 通常比 MHA 节省 KV cache，比极端的 MQA 保留更多独立的 K/V 表示。注意，GQA 的主要收益是 K/V 投影和 KV cache，不是把整个模型参数都按比例减少。

### 8.4 FlashAttention 和 SDPA：优化计算路径

FlashAttention 和 PyTorch SDPA 主要是高效实现标准注意力的方法。它们通过分块和融合 kernel，避免把完整的 $L \times L$ attention 矩阵写入高带宽显存。

它们不一定改变模型的注意力定义，但会改变计算和内存访问方式。训练 OOM、$O(L^2)$ 矩阵、FlashAttention 和稀疏注意力见[Transformer 训练实践](train/_posts/2026-06-11-base-train-and-pitfalls.md)。

### 8.5 MLA：压缩 KV 表示

Multi-head Latent Attention（MLA）不直接为每个历史 token 保存完整的 K/V，而是缓存更紧凑的 latent 表示，再在需要时恢复或计算注意力所需的信息。

它的主要目标是降低长上下文推理的 KV cache，而不是简单减少所有投影层的参数。DeepSeek 专项设计可参考[DeepSeek Attention Layer](dl/deepseek/dk-v4-attention-layer.md)。

### 8.6 Linear Attention、DeltaNet 和 Gated DeltaNet

标准 softmax attention 需要显式比较当前 token 与历史 token，序列维度通常是二次复杂度。线性注意力则尝试维护固定大小的状态：

$$
S_t = S_{t-1} + v_t k_t^T
$$

DeltaNet 用预测误差更新状态，Gated DeltaNet 再加入遗忘门：

$$
S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T
$$

它们可以把序列方向的计算和状态大小控制在近似线性范围，但固定状态不可能像完整 KV cache 那样精确保存任意历史位置。因此实际架构常把线性注意力与标准注意力混合使用。

[Linear Attention、Gated DeltaNet 和 MoE 手写笔记](_posts/2026-09-08-attention.md)包含递归状态、纯 PyTorch 实现和 router 实验。Mamba 属于 selective state space model，与 Gated DeltaNet 相关但不是同一种机制。

## 9. 当前前沿方向

### 9.1 长上下文

研究重点包括：

- sliding-window、block-sparse 和其他稀疏注意力；
- linear / recurrent attention；
- latent KV 和 KV cache 压缩；
- Ring Attention 等跨设备序列并行；
- 更好的位置编码和长上下文扩展方法。

核心问题是：如何在不让计算、显存和带宽随上下文长度失控的情况下，保留精确检索和复制能力。

### 9.2 混合注意力架构

标准 Attention 擅长精确检索，线性或递归状态擅长低成本维护长期信息。因此一些模型尝试在不同层或不同 token 范围混用：

```text
部分层：linear / recurrent attention
部分层：MHA / GQA / MLA
局部范围：sliding-window attention
关键位置：global attention
```

这是一种系统级折中，而不是宣称某个注意力机制在所有任务上都更好。

### 9.3 多模态 Attention

在视觉、音频和视频模型中，模态之间常通过以下方式连接：

- 把视觉特征投影成类似 token 的向量，与文本 token 拼接；
- 使用 text-to-vision cross-attention；
- 用 query-based resampler 压缩大量视觉特征。

因此 cross-attention 不只是原始机器翻译架构中的组件，也仍然是多模态模型的重要接口。

### 9.4 MoE 与 Attention 的解耦

MoE 通常放在 Transformer block 的 FFN 位置：

```text
attention
    -> router selects top-k experts
    -> selected FFN experts
```

它主要扩大总参数容量，同时控制每个 token 的激活计算。分析一个现代模型时，应分别回答三个问题：

1. Attention 是 MHA、GQA、MLA 还是线性/递归形式？
2. FFN 是 dense 还是 MoE？
3. 位置编码、归一化和 KV cache 使用什么方案？

## 10. 推荐阅读顺序

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：先看原始问题和完整 Encoder-Decoder 结构。
2. 本文第 2 至第 7 节：掌握 token、位置、QKV、Attention、Encoder、Decoder、FFN 和训练推理。
3. [Transformer 历史文章](_posts/2026-06-12-transformer-intro-and-history.md)：补充 RNN、BERT、GPT 和模型演进背景。
4. [SDPA、Causal Mask 和 GQA](train/_posts/2026-06-12-scaled-dot-product-attention.md)：连接到 PyTorch 代码和 KV head。
5. [RoPE](_posts/2026-07-03-transformer-rope-attention.md)：理解现代模型如何注入位置信息。
6. [Transformer 训练中的 Attention OOM](train/_posts/2026-06-11-base-train-and-pitfalls.md)：理解 $O(L^2)$ 和 FlashAttention。
7. [Linear Attention 和 Gated DeltaNet](_posts/2026-09-08-attention.md)：再进入线性/递归注意力与 MoE 实现。
8. [DeepSeek Attention Layer](dl/deepseek/dk-v4-attention-layer.md)：最后阅读具体模型如何围绕推理成本改造 Attention。

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

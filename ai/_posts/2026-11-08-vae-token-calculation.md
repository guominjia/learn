---
layout: post
title: "AI视频Token计算"
date: 2026-11-08
categories: [ai]
tags: [video, vae]
---

视频模型里的 token 数主要由空间和时间方向上的压缩倍率决定。

## 通用公式

$$
N_{token} = \left\lceil \frac{T}{s_t} \right\rceil \times \left\lceil \frac{H}{s_h} \right\rceil \times \left\lceil \frac{W}{s_w} \right\rceil
$$

- $s_t, s_h, s_w$ 分别是时间、高度、宽度方向的总压缩倍率，等于 VAE/tokenizer 的下采样倍率乘以 patch 大小。
- 3 通道 RGB 和 256 级量化（8 bit）只决定每个 patch 里有多少原始数值（例如 $2\times14\times14\times3$），这些数值会被线性投影成一个 hidden 维度的向量，算作一个 token。

下面以 1080×768、RGB 8bit、24fps、1 秒为例。

## 0. 原始数据量（作对比用）

$24 \times 1080 \times 768 \times 3 \approx 59.7\text{M}$ 个数值，约 59.7 MB/s。
按每个子像素算一个 token（ImageGPT 那种做法，词表 256），1 秒就是约 6000 万个 token，实际没人这么用。

## 1. 理解类 VLM（以 Qwen2-VL 为例）

- 配置：patch 14×14，`temporal_patch_size=2`，相邻 2×2 patch 合并。相当于 **2帧 × 28 × 28 像素 → 1 token**。
- 先把分辨率缩放到 28 的倍数：1080 → 1092（39 格），768 → 756（27 格）。
- 每 2 帧：$39 \times 27 = 1053$ token。
- 24 帧全部输入：$12 \times 1053 \approx$ **12.6K token/秒**。
- 实际推理时通常按 2 fps 抽帧，并限制每帧的最大像素数，所以真实 token 数会少很多（大约几百到 1–2K）。

## 2. 生成类 DiT（Wan2.1 / HunyuanVideo / CogVideoX 这类）

先用 3D 因果 VAE 压缩，再做 patchify：

| 步骤 | 压缩方式 | 结果 |
|---|---|---|
| 帧数对齐 | 因果 VAE 要求 $1+4k$ 帧，24 帧取 25 帧 | 25 帧 |
| VAE 时间 ÷4 | $1 + 24/4$ | 7 个 latent 帧 |
| VAE 空间 ÷8 | 768/8, 1080/8 | 96 × 135（135 是奇数，pad 到 136） |
| latent 通道 | RGB 3 通道 → 16 通道 | 7×96×136×16 ≈ 1.46M 数值（约 41 倍压缩） |
| DiT patch 1×2×2 | 96/2, 136/2 | 48 × 68 |
| **token 数** | $7 \times 48 \times 68$ | **≈ 22.8K token/秒** |

每个 token 是 $2\times2\times16 = 64$ 维的 latent，再投影到模型的 hidden 维度（例如 3072 或 5120）。

## 3. 离散 tokenizer（MAGVIT-v2、Cosmos-DV 等，用于自回归模型）

- 典型压缩倍率是 4×8×8 或 4×16×16，每个 token 是码本里的一个索引。
- 以 4×16×16 为例：$7 \times 48 \times 68 \approx 22.8\text{K}$ token。
- 码本大小 64K（16 bit）时，信息量 ≈ 22.8K × 2B ≈ 45 KB，相对原始 59.7 MB 约压缩 1300 倍。
- 如果用 4×8×8，token 数约为它的 4 倍（≈ 90K）。

## 快速估算口诀

| 类型 | 约多少像素对应 1 token |
|---|---|
| 理解 VLM | 2帧 × 28×28 |
| 生成 DiT（VAE 4×8×8 + patch 2） | 4帧 × 16×16 |
| 离散 tokenizer | 4帧 × 8×8 或 4帧 × 16×16 |

所以 1080×768 的 1 秒 24 帧视频，大约对应 **1 万到 2 万 token**（不抽帧的情况下）。分辨率翻倍，token 数大约变成 4 倍；时长翻倍，token 数大约变成 2 倍。attention 的计算量与 $N^2$ 成正比。

## 文本与视频模型对比

| | LLM 输出 token | 视频生成（DiT） |
|---|---|---|
| 每个 token 前向几次 | 1 次（有 KV cache） | 去噪步数 × CFG，通常 50 步 × 2 ≈ 100 次 |
| attention 范围 | 只看前文 | 全序列双向（约 2 万 token 互相计算） |
| 额外开销 | 无 | 文本编码器、VAE 解码 |

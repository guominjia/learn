# LiteLLM, Ollama, and vLLM: Understanding Model Switching at Every Layer

> **Published:** March 17, 2026

## Introduction

When working with LLMs, a common question arises: _why does LiteLLM need a Router to switch between models, while Ollama and llama.cpp can swap models on the fly within a single process?_ The answer reveals a fundamental architectural divide — and understanding it helps you design better inference stacks from development all the way to cloud-scale production.

This post covers:

1. How LiteLLM, Ollama, and vLLM each handle model switching
2. Why vLLM deliberately avoids dynamic model swapping
3. How cloud providers like Alibaba Cloud build cluster-level inference on top of vLLM

---

## LiteLLM: Switching Across Providers (Protocol Translation)

LiteLLM sits in front of **different LLM providers** — OpenAI, Anthropic, Google Gemini, Azure, Ollama, and 100+ more. Each provider exposes a different API shape:

```
OpenAI    → POST /v1/chat/completions  { "model": "gpt-4o", ... }
Anthropic → POST /v1/messages          { "model": "claude-...", "max_tokens": ... }  ← different required fields
Gemini    → POST /v1beta/models/...    completely different URL and body structure
```

LiteLLM solves this by performing **protocol translation** (adapter pattern), converting every provider's API into the OpenAI `chat.completions` format:

```python
import litellm

# All three use the exact same code — only the model string changes
response = litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])
response = litellm.completion(model="claude-3-5-sonnet-20241022", messages=[{"role": "user", "content": "Hello"}])
response = litellm.completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "Hello"}])
```

On top of unification, LiteLLM's `Router` adds load balancing, fallback, cost tracking, and caching. But the key insight is: **LiteLLM does not run any models itself** — it routes and translates.

---

## Ollama: Switching Within a Single Runtime (Weight Swapping)

Ollama is a **local inference runtime** that manages model files on disk and loads/unloads them into VRAM on demand.

```
User requests model=llama3.2
    │
    ▼
Is it already loaded in VRAM?
    │
   No  → Load weights from disk into VRAM → Run inference → Unload after idle
    │
   Yes → Run inference directly
```

```
VRAM timeline:
[llama3.2 weights████████████][idle][mistral weights████████████][idle]
     User A finishes, unload        User B requests, load new model
```

Because Ollama assumes **single-user, low-concurrency** usage, the cost of unloading one model and loading another is perfectly acceptable — maybe 10-30 seconds of downtime between models.

```bash
# Same port, same API, just change the model field
curl http://localhost:11434/api/chat -d '{"model": "llama3.2", ...}'
curl http://localhost:11434/api/chat -d '{"model": "mistral",  ...}'
# ↑ Ollama automatically swaps the model in VRAM
```

The API is already OpenAI-compatible. There is no protocol translation needed — it is a **unified local API** where model switching is an internal runtime behavior.

---

## The Key Distinction: Where the Switching Happens

| | LiteLLM | Ollama / llama.cpp |
|---|---|---|
| Service location | Proxy layer (cloud or local) | Local inference runtime |
| What is switched | Different companies, different protocols | Different model weights in the same runtime |
| Main work | Protocol translation + routing + load balancing | Model weight loading/unloading in VRAM |
| API unification | LiteLLM provides it | Already unified natively |

They are often **used together**:

```
Your App
   │
   ▼
LiteLLM Proxy        ← Unified entry point: cost, fallback, rate limiting
   │          │
   ▼          ▼
 OpenAI     Ollama   ← Ollama handles local model swapping internally
             (llama3 / mistral / qwen ...)
```

LiteLLM's `model: "ollama/llama3.2"` simply **forwards to Ollama**; the actual model switch is performed by Ollama.

---

## vLLM: Why It Does NOT Dynamically Swap Models

vLLM is a high-throughput inference engine designed for **production serving**. It supports OpenAI-compatible APIs, but its model switching story is fundamentally different from Ollama's.

### vLLM's Modes

**Single-model mode (default):**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-8B \
    --port 8000
```

One model is bound at startup. Changing it requires a restart.

**Multi-LoRA mode (dynamic adapters, same base model):**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-8B \
    --enable-lora --max-lora-rank 64
```

Different LoRA adapters can be swapped per request, but the **base model stays the same**.

### Why Not Just Swap Models Like Ollama?

The reason is **architectural** — vLLM's core innovations make dynamic model swapping prohibitively expensive.

#### 1. Static VRAM Allocation with PagedAttention

At startup, vLLM partitions **all available VRAM** between model weights and a KV Cache memory pool:

```
vLLM VRAM layout (allocated at startup, static):

┌─────────────────────────────────────────────┐
│           Model Weights (static)            │ ← bulk of VRAM
├─────────────────────────────────────────────┤
│        KV Cache Memory Pool (pre-allocated) │ ← managed by PagedAttention
│  Page0 │ Page1 │ Page2 │ ... │ PageN        │
│  req_A │ req_B │ req_A │ ... │ req_C        │ ← shared across concurrent requests
└─────────────────────────────────────────────┘
```

Swapping a model would require:

```
1. Wait for ALL in-flight requests to complete     ← may never happen under high load
2. Release the KV Cache pool (all contexts lost)   ← breaks PagedAttention continuity
3. Unload weights
4. Load new weights
5. Re-initialize the KV Cache pool and scheduler
```

This process takes **30 seconds to several minutes** — catastrophic for a production request queue.

#### 2. Continuous Batching Cannot Pause

vLLM uses **Continuous Batching**: new requests are inserted into the batch as old ones finish, and the GPU is never idle:

```
Timeline:
t=0  [req_A token1] [req_B token1] [req_C token1]  ← batched inference
t=1  [req_A token2] [req_B token2] [req_D token1]  ← req_C done, req_D joins
t=2  [req_A token3] [req_E token1] [req_D token2]  ← req_B done, req_E joins
...
```

To swap models, you need a moment when **zero requests are in flight** — under high concurrency, **that moment never comes**.

#### 3. Design Philosophy Comparison

```
Ollama's assumptions:                vLLM's assumptions:
  Few users                            Many users (100+ concurrent)
  Requests are bursty                  Requests stream in continuously
  VRAM utilization is secondary        VRAM utilization must be maximized
  Swap downtime is acceptable          Any downtime is unacceptable

  → Dynamic load/unload ✅             → Static allocation, one model ✅
```

### vLLM vs Ollama Summary

| Feature | Ollama | vLLM |
|---|---|---|
| Dynamic model swap | ✅ Auto unload/load | ⚠️ Limited (LoRA only) |
| Target scenario | Local dev, convenience | **Production, high throughput** |
| GPU utilization | Moderate | ✅ PagedAttention maximizes it |
| Concurrent requests | Limited | ✅ Built for high concurrency |
| Dynamic LoRA switching | ❌ | ✅ |
| Multi-tenant isolation | ❌ | ✅ |

---

## Cloud-Scale Inference: How Alibaba Cloud and Others Build on vLLM

In cluster environments like Alibaba Cloud's **PAI-EAS** or **Bailian**, the architecture goes far beyond a single vLLM instance:

```
┌─────────────────────────────────────────────────┐
│              User API Requests                   │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│           API Gateway / Load Balancer            │
│     (traffic shaping, auth, rate limiting, billing)│
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         Inference Scheduler (proprietary)        │
│   request routing / model versioning / autoscaling│
└──────┬──────────┬──────────┬────────────────────┘
       │          │          │
┌──────▼──┐  ┌───▼─────┐  ┌─▼───────┐
│ vLLM    │  │ vLLM    │  │ vLLM    │   ← multiple instances
│ Inst A  │  │ Inst B  │  │ Inst C  │
│ 8×A100  │  │ 8×A100  │  │ 8×A100  │
└─────────┘  └─────────┘  └─────────┘
```

Cloud providers use vLLM as the **per-node inference engine**, then add cluster-level capabilities on top.

### Key Cluster Optimizations

#### 1. Tensor Parallelism vs Pipeline Parallelism

```
Single machine, 8 GPUs (vLLM natively supports):
GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7
←────────── Tensor Parallelism: split one layer across GPUs ──────────→

Multi-machine (requires additional engineering):
Machine A [GPU0~7]  ←→  Machine B [GPU0~7]  ←→  Machine C [GPU0~7]
←──────────── Pipeline Parallelism: different layers on different machines ──────────→

Cross-machine communication:
  NVLink (intra-node) ~600 GB/s  vs  InfiniBand (inter-node) ~50 GB/s
```

#### 2. Prefill-Decode Disaggregation

This is the most impactful cluster optimization in recent years:

```
Traditional vLLM (Prefill + Decode on the same instance):
Request → [Prefill: process input prompt] → [Decode: generate tokens one by one] → Output
           GPU compute-bound                 GPU memory-bandwidth-bound

Problem: two completely different compute profiles interfere with each other.

PD Disaggregation (cloud providers):
                    ┌──────────────────────────────┐
Request ──→ Scheduler → │  Prefill Cluster            │ → KV Cache Transfer
                    │  (compute-heavy, fewer big GPUs) │        │
                    └──────────────────────────────┘        │
                                                             ▼
                    ┌──────────────────────────────┐         │
        Output ←──  │  Decode Cluster              │ ←───────┘
                    │  (memory-heavy, more smaller GPUs) │
                    └──────────────────────────────┘
```

#### 3. Centralized KV Cache

```
Single-instance vLLM:
  KV Cache lives in local VRAM → requests must stay on the same instance

Cluster problem:
  Multi-turn chat → each request may route to a different instance → KV Cache miss → re-Prefill

Solution:
  ┌─────────────────────────────────────┐
  │     Centralized KV Cache Store      │
  │  (high-speed NVMe / memory pool / RDMA) │
  └──────┬──────────┬──────────┬────────┘
         │          │          │
      Inst A     Inst B     Inst C
  Any instance can read previous KV Cache
```

#### 4. Elastic Scaling

```
vLLM single instance: loading model weights takes time
  Llama-70B (4-bit) ≈ 35 GB → load time 30s ~ 2min

Cloud optimizations:
  ① Weight pre-warming (popular models stay resident)
  ② Predictive scaling (scale before traffic arrives)
  ③ Spot instance utilization (preemptible GPUs for cost savings)
```

### What Major Cloud Providers Use

| Provider | Approach |
|---|---|
| Alibaba Cloud PAI-EAS | Modified vLLM + proprietary scheduler |
| ByteDance | Custom **LightSeq** / modified vLLM |
| Tencent Cloud | Modified vLLM + proprietary TurboMind |
| Moonshot AI | **Mooncake** (PD disaggregation + KV Cache pooling) |
| AWS SageMaker | vLLM / TGI as selectable backends |
| Google Cloud | Custom **Pathways** system, does not use vLLM |

---

## Putting It All Together

```
Your App
    │
    ▼
LiteLLM Proxy           ← Unified entry: routing / fallback / cost tracking
    │            │
    ▼            ▼
  vLLM         Ollama
(production)   (local dev)
  Qwen2.5      llama3.2
  Llama-3      mistral
```

Each layer exists for a reason:

- **LiteLLM** — protocol translation and provider routing (_cross-provider switching_)
- **Ollama** — convenient local model management with dynamic VRAM swapping (_same-runtime switching_)
- **vLLM** — maximum throughput for one model with PagedAttention and Continuous Batching (_no switching by design_)
- **Cloud schedulers** — cluster-level routing, PD disaggregation, centralized KV Cache, and elastic scaling (_infrastructure-level orchestration_)

> **In short:** Ollama treats VRAM as a resource pool where models are tenants. vLLM treats VRAM as a battlefield where the model is permanently stationed. Cloud providers connect many battlefields into a war theater with centralized logistics. And LiteLLM is the diplomat that speaks every army's language.

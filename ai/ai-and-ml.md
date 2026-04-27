# AI, Machine Learning, and NLP: Origins, Relationships, and Taxonomy

## Introduction

The terms **Artificial Intelligence (AI)**, **Machine Learning (ML)**, and **Natural Language Processing (NLP)** are often used interchangeably in casual conversation, but they refer to very different things. Understanding their origins and how they relate to each other is essential for anyone working in the field. This post traces the history of each term, clarifies the relationships between them, and provides a clean taxonomy of methodologies vs. domains vs. tasks.

---

## 1. When Were These Terms Coined?

### Artificial Intelligence (AI)

- **Coined by**: **John McCarthy**
- **Year**: **1956**
- **Context**: At the famous Dartmouth Summer Research Project on Artificial Intelligence held at Dartmouth College, USA. McCarthy, along with Marvin Minsky, Nathaniel Rochester, and Claude Shannon, organized this workshop, which is widely regarded as the **birth of AI as a formal academic discipline**.

### Machine Learning (ML)

- **Coined by**: **Arthur Samuel**
- **Year**: **1959**
- **Context**: While working at IBM, Samuel developed a self-learning checkers program. He defined Machine Learning as *"the field of study that gives computers the ability to learn without being explicitly programmed."*

### Natural Language Processing (NLP)

Unlike AI and ML, NLP **does not have a single definitive originator or moment of creation**. It evolved gradually:

| Year | Milestone | Key Figures |
|------|-----------|-------------|
| 1950 | Alan Turing published *"Computing Machinery and Intelligence"*, proposing the Turing Test — testing machine intelligence through natural language conversation | **Alan Turing** |
| **1954** | **Georgetown-IBM Experiment** — the first public demonstration of machine translation (translating 60+ Russian sentences into English automatically). Widely considered the **starting point of NLP research** | **Leon Dostert**, **Paul Garvin** (Georgetown University & IBM) |
| 1957 | Noam Chomsky published *"Syntactic Structures"*, introducing generative grammar theory that laid the linguistic foundation for computational language understanding | **Noam Chomsky** |
| 1960s | The term "Natural Language Processing" began appearing widely in academic literature, though **no single person is credited with coining it** | — |
| 1966 | **ELIZA**, one of the earliest chatbots, was created at MIT | **Joseph Weizenbaum** |

> **Summary**: If a single starting point must be chosen, the **1954 Georgetown-IBM machine translation experiment** is the consensus origin of NLP research. **Noam Chomsky**'s formal language theory provided its critical theoretical foundation.

---

## 2. How Do AI, ML, and NLP Relate to Each Other?

### The Big Picture

```
Artificial Intelligence (AI) ─── The overarching goal: make machines intelligent
  │
  ├── Machine Learning (ML) ─── A methodology: learn from data
  │     ├── Deep Learning (DL)
  │     └── Traditional ML (SVM, Decision Trees, etc.)
  │
  └── Natural Language Processing (NLP) ─── An application domain: understand & generate human language
        ├── Rule-based approaches (1950s–1980s)
        ├── Statistical / ML-based approaches (1990s–2012)
        └── Deep Learning-based approaches (2013–present)
```

### Key Distinctions

| Concept | Nature | Analogy |
|---------|--------|---------|
| **AI** | A **broad goal** (make machines intelligent like humans) | "Medicine" as a discipline |
| **ML** | A **method/means** to achieve AI (learn from data) | "Drug therapy" as a treatment method |
| **NLP** | An **application domain** of AI (process human language) | "Cardiology" as a specialty |

### The Intersection

- **NLP is a sub-field of AI**: NLP's goal (making machines understand language) is part of AI's overarching objective.
- **ML is the primary method used in NLP**: Modern NLP relies almost entirely on ML (especially Deep Learning).
- **NLP ≠ ML**: Early NLP was predominantly based on hand-crafted rules and linguistic knowledge, not ML at all.

```
       ┌─────────────────────────────────┐
       │         AI (Artificial           │
       │         Intelligence)            │
       │                                  │
       │   ┌──────────┐  ┌──────────┐    │
       │   │    ML    │  │   NLP    │    │
       │   │ (Method) │  │ (Domain) │    │
       │   │          │  │          │    │
       │   │    ┌─────┼──┼────┐     │    │
       │   │    │ Modern NLP  │     │    │
       │   │    │ (ML-powered)│     │    │
       │   │    └─────┼──┼────┘     │    │
       │   └──────────┘  └──────────┘    │
       └─────────────────────────────────┘
```

---

## 3. Methodology vs. Domain vs. Task — A Clean Taxonomy

A common source of confusion is mixing up **methods**, **domains**, and **tasks**. Here is how to think about them:

| Category | Question It Answers | Examples |
|----------|-------------------|----------|
| **Methodology (How)** | *How does the machine learn?* | Deep Learning, Reinforcement Learning, Supervised Learning, Unsupervised Learning, Transfer Learning |
| **Domain (What)** | *What type of data/problem are we dealing with?* | NLP, Computer Vision (CV), Speech Processing, Multimodal AI, Robotics |
| **Task (Specific What)** | *What specific job are we doing?* | Image Recognition, Text-to-Image, Text-to-Video, Text-to-Speech, Machine Translation, Q\&A |

### Full Hierarchy

```
AI (Artificial Intelligence)
│
├── Methodologies (How to learn)
│   ├── Machine Learning (ML)
│   │   ├── Supervised Learning
│   │   ├── Unsupervised Learning
│   │   ├── Reinforcement Learning        ← Method
│   │   └── Deep Learning                 ← Method (subset of ML)
│   │       ├── CNN, RNN, Transformer ...
│   │       └── (Can combine with any learning paradigm above)
│   └── Non-ML approaches (rule-based systems, search algorithms, knowledge graphs, etc.)
│
├── Domains (What type of data)
│   ├── Natural Language Processing (NLP)    — Text
│   ├── Computer Vision (CV)                 — Images / Video
│   ├── Speech Processing                    — Audio / Voice
│   ├── Multimodal AI                        — Cross-modal
│   └── Robotics                             — Physical interaction
│
└── Tasks (Specific applications under domains)
    ├── Image Recognition / Classification   → CV
    ├── Text-to-Image (e.g., DALL·E, SD)    → Multimodal (NLP + CV)
    ├── Text-to-Video (e.g., Sora)          → Multimodal (NLP + CV)
    ├── Text-to-Speech (TTS)                → Multimodal (NLP + Speech)
    ├── Machine Translation                  → NLP
    └── Dialogue / Q&A                       → NLP
```

### Key Nuances

| Concept | Clarification |
|---------|--------------|
| **Deep Learning** | Strictly a **subset of ML** (methods based on deep neural networks). It can be combined with supervised, unsupervised, or reinforcement learning. |
| **Reinforcement Learning** | A **learning paradigm** (learning through interaction with an environment via reward/penalty feedback). It can be implemented with or without deep learning (Deep RL vs. tabular RL). |
| **Text-to-Image / Text-to-Video** | These are **specific tasks**, one level more granular than "domain". They belong to the **Multimodal** or **Generative AI** domain. |

---

## 4. Three Eras of NLP

| Era | Period | Core Approach | Notable Examples |
|-----|--------|--------------|------------------|
| **Rule-based** | 1950s–1980s | Hand-written grammar rules, expert systems | ELIZA, SHRDLU |
| **Statistical / ML** | 1990s–2012 | Statistical models, traditional machine learning | HMM, CRF, SVM, TF-IDF |
| **Deep Learning** | 2013–present | Neural networks, pre-trained large models | Word2Vec (2013), Transformer (2017), BERT (2018), GPT series (2018–), ChatGPT (2022) |

---

## 5. Timeline of Key Milestones

| Year | Event |
|------|-------|
| 1950 | Alan Turing publishes *"Computing Machinery and Intelligence"*, proposes the Turing Test |
| 1954 | Georgetown-IBM experiment — first machine translation demo (origin of NLP) |
| **1956** | **John McCarthy coins "Artificial Intelligence"** at the Dartmouth Conference |
| 1957 | Noam Chomsky publishes *"Syntactic Structures"* |
| **1959** | **Arthur Samuel coins "Machine Learning"** |
| 1966 | ELIZA chatbot created by Joseph Weizenbaum |
| 1986 | Rumelhart et al. popularize backpropagation, advancing neural networks |
| 1997 | IBM Deep Blue defeats world chess champion Garry Kasparov |
| 2012 | Deep Learning achieves breakthrough at ImageNet (AlexNet), sparking new AI boom |
| 2013 | Word2Vec introduces efficient word embeddings |
| 2017 | Transformer architecture proposed (*"Attention Is All You Need"*) |
| 2018 | BERT and GPT-1 released, launching the pre-trained model era |
| 2022 | ChatGPT released, bringing LLMs into the mainstream |
| 2023+ | Multimodal models (GPT-4V, Sora, etc.) blur the lines between domains |

---

## Summary

| Term | What It Is | Analogy |
|------|-----------|---------|
| **AI** | The destination (make machines "smart") | "Cooking" as a concept |
| **ML** | An important road to the destination (learn from data) | Cooking **techniques** (stir-fry, steam, bake) |
| **NLP / CV / Speech** | Specific terrain you're navigating (type of data) | **Cuisines** (Sichuan, Cantonese, Fusion) |
| **Text-to-Image, Translation, etc.** | The specific dish you're making | Individual **dishes** |
| **Deep Learning / RL** | Powerful tools you use along the way | A **pressure cooker** — works with any technique or cuisine |

> Modern Large Language Models (LLMs) are essentially the product of **NLP + Deep Learning (a subset of ML)** — a deep convergence of AI's goal, ML's methodology, and NLP's domain expertise.

## References

- [RNN](rnn.md)
  - [RNN Training](rnn-training.md)
  - [RNN Training 2](rnn-training-2.md)
  - [RNN Training 3](rnn-training-3.md)
  - [RNN Training 4](rnn-training-4.md)
  - [RNN Training 5](rnn-training-5.md)
- [Tensor](tensor.md)
- [Mismatch Long and Float](experiences/mismatch-long-and-float.md)
- [Evaluation Metrics](experiences/eval-metrics.md)
- [Ranx Metrics](experiences/ranx-metrics.md)
- [Pytorch Grad](experiences/pytorch-grad.md)
- [TorchVision Scale History](experiences/torchvision-scale-history.md)
- [Pytorch](experiences/pytorch.md)
- [GAN vs Diffusion](experiences/gan-vs-diffusion.md)
- [ImageNet and ResNet](experiences/imagenet-and-resnet.md)
- [Conv2d](experiences/conv2d.md)
- [Flatten](experiences/flatten.md)
- [World Model](world-model.md)
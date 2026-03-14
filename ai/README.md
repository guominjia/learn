# Artificial Intelligence
Notes about AI: industry, products, news, and knowledges.

## Industry Chain
AI industry chain include upstream, midstream, downstream.
- Upstream: Include Silicon Vendor, Cloud Service Provider, Model Provider.
  - **Silicon Vendor**: INTEL, AMD, NVIDIA, ARM, etc
  - **CSP**(Cloud Service Provider): Amazon, Google, Microsoft, Meta, ByteDance, Alibaba, Baidu, Huawei, etc
  - Model Provider: OpenAI, Claude, DeepSeek, Zhipu, MinMax, etc
- Midstream: Include software framework and infrastructure provider
  - **Model Hub**: [HuggingFace](https://huggingface.co/welcome), ModelScope
    - HuggingFace provide [transformers](transformers.md), [datasets](datasets.md), [optimum](optimum.md), tokenizers for running or training model.
  - **Train Framework**: Pytorch, TensorFlow
    - [Pytorch](pytorch.md) provide operators
  - [**Agent Fremwork**](agent.md): Langchain, LlamaIndex
  - **Database Framework**: Chromadb, Neo4J, etc
  - [**RAG Framework**](rag.md)
  - **UI Framework**: OpenWebUI, Streamlit, Chainlit
  - [**MCP Framework**](mcp.md)
- Downstream: Include huge applications

## Utilities
Click [here](utilities.md) for AI tools

## [Framework](framework.md)

## [Player](player.md)

## [Products](products.md)

## [Data Process](data-process.md)

### [Embedding](embedding.md)

## [RL](reinforcement-learning.md)
Experiences about RL

## [Memory](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- https://github.com/guominjia/learn/tree/code_study/memory
- [Agent Memory](https://mp.weixin.qq.com/s/DT_HfzOMHXT8hApNjBzGOA)
  - <https://arxiv.org/abs/2512.13564>
- https://docs.langchain.com/oss/python/concepts/memory

## [Skills](https://blog.langchain.com/langchain-skills/)
Skills are curated instructions, scripts, and resources that improve coding agent performance in specialized domains. Importantly, skills are dynamically loaded through progressive disclosure — the agent only retrieves a skill when its relevant to the task at hand. This enhances agent capabilities, as historically, **giving too many tools to an agent would cause its performance to degrade**.

Skills are portable and shareable — they consist of markdown files and scripts that can be retrieved on demand. We’re sharing a set of LangChain skills that can be ported to any coding agent that supports skill functionality.

Refer: <https://github.com/langchain-ai/langchain-skills>

## [Structed Output](structed-output.md)
# Artificial Intelligence
Notes about AI: industry, products, news, and knowledges.

## Industry Chain
AI industry chain include upstream, midstream, downstream.
- Upstream: Include Silicon Vendor, Cloud Service Provider, Model Provider.
  - **Silicon Vendor**: INTEL, AMD, NVIDIA, ARM, etc
  - **CSP**(Cloud Service Provider): Amazon, Google, Microsoft, Meta, ByteDance, Alibaba, Baidu, Huawei, etc
  - **MP**(Model Provider): OpenAI, Claude, DeepSeek, Zhipu, MiniMax, Qwen, etc
- Midstream: Include software framework and infrastructure provider
  - **Model Hub**: [HuggingFace], [ModelScope]
    - HuggingFace provide [transformers](transformers.md), [datasets](datasets.md), [optimum](optimum.md), tokenizers for running or training model.
  - **Train Framework**: [PyTorch], [TensorFlow], [Paddle], [Keras], [CNTK], [CAFFE2]
    - [Pytorch](pytorch.md) provide operators
  - **Infer Framework**: vLLM, [llama.cpp]
  - [**Agent Framework**](agent.md): [LangChain](frameworks/langchain/README.md), LlamaIndex
  - **Database Framework**: Chromadb, Neo4J, etc
  - [**RAG Framework**](rag.md)
  - **UI Framework**: OpenWebUI, Streamlit, Chainlit
  - [**MCP Framework**](mcp.md)
  - [**Memory Framework**](memory.md)
- Downstream: Include huge applications
  - Chat: [Open-WebUI](apps/open-webui/README.md)
  - RAG: [RAG Flow](apps/ragflow/README.md)

## Utilities
Click [here](utilities.md) for AI tools

## Books
Click [here](books.md) for AI books

## Peoples
Click [here](peoples.md) for AI domain

## Concepts
[AI and ML](ai-and-ml.md),
[Prompt](prompt.md),
[MOE](moe.md),
[OCR](ocr.md),
[Embedding](embedding.md),
[Skill](skill.md),
[Template](template.md)

## [Player](player.md)

## [Products](products.md)

## [Data Process](data-process.md)

## [RL](reinforcement-learning.md)
Experiences about RL

## [Deep Learning](dl/README.md)

## [Structed Output](structed-output.md)

[llama.cpp]: https://github.com/ggml-org/llama.cpp
[Keras]: https://pypi.org/project/keras/
[HuggingFace]: https://huggingface.co/welcome
[ModelScope]: https://modelscope.cn/home
[PyTorch]: https://pytorch.org
[TensorFlow]: https://www.tensorflow.org
[Paddle]: https://github.com/PaddlePaddle/Paddle
[CNTK]: https://github.com/microsoft/cntk
[CAFFE2]: https://github.com/caffe2
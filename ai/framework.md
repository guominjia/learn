# Framework

## 基础设施或框架类
- 模型托管
  - ModelScope
    - datasets, model
  - HuggingFace
    - transformers, tokenizers, datasets, etc
- 模型训练
    - [Pytorch](pytorch.md)

## Template
**All LLM use a specific template internally**

### 1. GPT-4 (and likely GPT-5)
*   **API Usage:** Do **not** manually apply a template. send a list of dictionaries (JSON) like `[{"role": "user", "content": "..."}]` to the OpenAI API.
*   **Internal:** Behind the scenes, OpenAI's servers automatically convert that list into a specific string format (often referred to as **ChatML**) that the model was trained on.

### 2. Gemini
*   **API Usage:** Do **not** manually apply a template. send structured objects (e.g., `Content` objects with `parts` and `role`) to the Google Vertex AI or Gemini API.
*   **Internal:** Google's backend formats these objects into the specific token structure the Gemini model expects.

### 3. Claude
*   **API Usage:** Do **not** manually apply a template. use the Anthropic Messages API, passing a `messages` list.
*   **Internal:** Historically, Claude used a visible format like `\n\nHuman: ... \n\nAssistant:`, but modern versions handle this formatting internally within the API.

## Bad Performance without Template

不使用 `chat_template` 导致性能巨降甚至胡说八道，根本原因在于**训练数据格式的不匹配（Distribution Shift）**。

简单来说，模型在训练时“看惯了”特定的格式，如果推理时给它的格式不一样，它就会“懵圈”。

具体原因如下：

### 1. 触发特定的微调（Fine-tuning）模式
现在的模型（如 Qwen-Chat, Llama-Chat）都经过了 **Instruction Tuning（指令微调）**。在微调阶段，数据被严格格式化成了特定的样子。

例如，Qwen 可能见过数亿次这样的数据结构：
`<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n`

*   **使用了 Template：** 你发送给模型的正是它熟悉的结构。模型看到 `<|im_start|>user` 知道这是你在说话，看到 `<|im_start|>assistant` 知道轮到它说话了，并且知道该进入“回答问题”的模式。
*   **不使用 Template：** 如果你直接把字符串 `"你好"` 扔给它，模型看到的只是一个孤立的词。它不知道这是个问题，也不知道该谁回答。它可能会以为这只是文本续写任务（Text Completion），于是它可能会接着写 `"你好吗？我很好..."` 或者直接开始胡言乱语，因为它没有被“激活”进入对话模式。

### 2. 特殊 Token 的作用
Chat Template 不仅仅是加了 `"User:"` 这种文字，更重要的是它插入了**特殊 Token**（Special Tokens）。

*   在你的代码中，`tokenizer.apply_chat_template` 会自动插入像 `<|im_start|>`、`<|im_end|>` 或者 `<EOS>` 这样的特殊符号。
*   这些符号对模型来说是**红绿灯**。
    *   没有“开始回答”的信号，模型可能不知道从哪里开始生成。
    *   没有“结束”信号，模型可能永远停不下来，直到达到最大长度，导致输出重复或混乱。

### 3. 角色混淆
如果不加 Template，模型无法区分哪部分是“背景知识”，哪部分是“用户指令”，哪部分是“它之前的回答”。
比如多轮对话中：
*   **有 Template：** `[User] 谁是李白？ [Assistant] 诗人。 [User] 他哪个朝代？` -> 模型知道“他”指李白。
*   **无 Template：** `谁是李白？ 诗人。 他哪个朝代？` -> 模型可能把这当成一段连续的小说文本，接着往下编故事，而不是回答你的问题。

### 总结
**模型本质上是一个概率预测机器。**
Template 的作用是把当前的输入环境，对齐到模型**概率分布最确定、最擅长**的那个状态。不加 Template，就像让一个只学过填空题的学生去做问答题，虽然知识都在脑子里，但格式不对，他就不会做了。

## Post training for template

**Post-training 用的数据量和计算资源远少于 Pre-training，但数据的“含金量”极高。**

Post-training 和 Fine-tuning 的关系，可以理解为：**Post-training 是一个阶段，而 Fine-tuning 是这个阶段主要使用的手段。**

下面详细拆解：

### 1. 资源消耗对比：Pre-training vs. Post-training

你可以把模型训练想象成培养一个博士生：

| 维度 | Pre-training (预训练) | Post-training (后训练) |
| :--- | :--- | :--- |
| **类比** | **读万卷书** (通识教育) | **岗前培训** (职业素养) |
| **目标** | 学习语言规律、世界知识、逻辑推理。 | 学习如何对话、遵循指令、安全规范、对齐人类偏好。 |
| **数据量** | **极巨大** (Trillions tokens)。整个互联网、所有书籍、代码库。 | **相对较小** (几万到几百万条)。高质量的对话数据、排序数据。 |
| **数据特点** | 只要是文字就行，清洗后直接喂。 | **人工精修**。需要人类专家编写问答，或者对模型的回答打分。 |
| **计算资源** | **90% - 99%** 的算力。数千张 GPU 跑几个月。 | **1% - 10%** 的算力。几十/几百张 GPU 跑几天或几周。 |
| **结果** | Base Model (基座模型)。只会续写，不懂对话。 | Chat/Instruct Model (对话模型)。能听懂指令，有礼貌。 |

**结论：** Post-training 不需要像预训练那样“烧钱”，**但它非常“烧脑”（需要极高的人力成本来清洗和标注数据）**。

### 2. Post-training 和 Fine-tune (微调) 有何不同？

这两个词经常混用，但严格来说它们是**包含关系**。

*   **Fine-tuning (微调)**：这是一个**技术动作**。指的是在已经训练好的模型基础上，用新的数据继续更新参数。
*   **Post-training (后训练)**：这是一个**生命周期阶段**。指的是预训练结束后的所有优化步骤。

**Post-training 通常包含以下两个具体的 Fine-tuning 步骤：**

#### 第一步：SFT (Supervised Fine-Tuning，有监督微调)
*   **这是什么？** 这就是“学习 Template”的阶段。
*   **怎么做？** 准备几万条高质量的 `<User>问题 <AI>回答` 格式的数据。
*   **目的：** 让模型学会“说话的方式”。比如学会看到 `<|im_start|>user` 就知道该听指令了。
*   **这属于 Fine-tuning 吗？** 是的，这是最典型的 Fine-tuning。

#### 第二步：RLHF / DPO (Reinforcement Learning，强化学习对齐)
*   **这是什么？** 让模型学会“看脸色”，知道哪个回答更好。
*   **怎么做？** 模型给两个回答，让人类选哪个更好，然后用算法（如 DPO 或 PPO）调整模型。
*   **目的：** 减少幻觉，减少有害内容，增加有用性。
*   **这属于 Fine-tuning 吗？** 广义上也算，因为它也在调整参数，但通常被称为“对齐（Alignment）”。

### 3. 为什么大家常说“我要去 Finetune 一个模型”？

当开发者（比如你）说“我要 Finetune 一个模型”时，通常指的是 **Domain Adaptation (领域微调)**。

*   **场景：** 你拿到了 Qwen-Chat（已经做完 Post-training 了），但你想让它变成一个“法律专家”或“公司客服”。
*   **操作：** 你准备了自己的法律文档或客服记录，再次进行 SFT。
*   **本质：** 这其实是在做**你自己的 Post-training**。

### 总结

1.  **Post-training** 是让模型从“懂知识的野人”变成“有礼貌的助手”的过程。
2.  它用的数据量**远少于**预训练，但要求数据必须是**对话格式（Template）**且质量极高。
3.  正是因为 Post-training 阶段强行灌输了特定的对话格式（如 `<|im_start|>`），所以你在推理时必须用 `apply_chat_template`，否则模型就会因为环境不匹配而“蒙圈”。
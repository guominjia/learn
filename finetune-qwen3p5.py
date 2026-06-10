import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Recommended environment variables to reduce fragmentation-related OOMs.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "Qwen/Qwen3.5-9B"

# 4-bit QLoRA configuration.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False  # Disable cache during training to save memory.

# LoRA configuration for Qwen-style models.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Dataset.
ds = load_dataset("HuggingFaceTB/smoltalk", "all")
train_ds = ds["train"]
eval_ds = ds["test"] if "test" in ds else None

# Minimal stable SFT settings for a single 48 GB GPU.
training_args = SFTConfig(
    output_dir="./qwen3p5_qlora_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8.
    learning_rate=2e-4,
    max_steps=1000,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    eval_strategy="no",  # Disable evaluation first to save memory.
    report_to="none",
    max_length=1024,  # Start with 1024, then increase to 1536/2048 if stable.
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds if training_args.eval_strategy != "no" else None,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.model.save_pretrained("./qwen3p5_qlora_sft/final")
tokenizer.save_pretrained("./qwen3p5_qlora_sft/final")
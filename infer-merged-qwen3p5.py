import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the merged model
merged_model_path = "./qwen3p5_qlora_sft/merged_model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(merged_model_path, use_fast=True)

# Load the merged model (standard HuggingFace model, no PEFT needed)
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.bfloat16,  # Use this if the GPU supports bf16; otherwise switch to float16
    device_map="auto",
)
model.eval()

# Build the conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Prove the Mean Value Theorem for Differentiation."},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate the response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(response)

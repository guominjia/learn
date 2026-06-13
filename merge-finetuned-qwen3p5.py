import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# -------------------------
# Path config
# -------------------------
adapter_path = "./qwen3p5_qlora_sft/final"  # Path to the LoRA adapter directory
merge_output_dir = "./qwen3p5_qlora_sft/merged_model"  # Output directory for the merged model

print(f"Loading adapter from: {adapter_path}")
print(f"Merge output to: {merge_output_dir}")

# -------------------------
# Load adapter + base model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# -------------------------
# Merge LoRA into base model
# -------------------------
print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

# -------------------------
# Save merged model
# -------------------------
print("Saving merged model...")
merged_model.save_pretrained(merge_output_dir, safe_serialization=True)
tokenizer.save_pretrained(merge_output_dir)

print(f"✅ Merged model saved to: {merge_output_dir}")
print("\nYou can now use it directly with:")
print(f"  AutoModelForCausalLM.from_pretrained('{merge_output_dir}')")
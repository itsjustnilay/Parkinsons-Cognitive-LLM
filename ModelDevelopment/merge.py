import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model paths
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_adapter_path = "path/to/model/final_checkpoint/"
merged_model_path = "path/to/merged/model/"

# Hugging Face token (if needed for private models)
token = "your_hf_token"

# Load base model (WITHOUT 4-bit quantization for merging)
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically assigns GPUs
    cache_dir="/path/to/model/cache/",
    use_auth_token=token
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# Merge LoRA into base model
print("Merging LoRA adapter into base model...")
model = model.merge_and_unload()

# Verify that the model is no longer in 4-bit mode
if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
    print("Warning: Model is still in 4-bit quantization. Merge may have failed.")
else:
    print("LoRA successfully merged into base model.")

# Save merged model
print(f"Saving merged model to {merged_model_path}...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

# Check if the model weights were saved
if os.path.exists(os.path.join(merged_model_path, "pytorch_model.bin")) or os.path.exists(
    os.path.join(merged_model_path, "model.safetensors")
):
    print("Merged model successfully saved!")
else:
    print("Warning: Merged model was NOT saved properly!")

print("Merging complete. Model is ready for vLLM inference.")

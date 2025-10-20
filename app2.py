# ============================================
# 🚀 Gradio App for Mini Instruction-Following T5 with LoRA
# ============================================

# pip install transformers peft torch gradio

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ===========================
# 1. Model Configuration
# ===========================

# Your Hugging Face Hub repository for the LoRA model
lora_model_hub = "satvik190603/LLM_Fine_tune"  # Replace with your model repo
base_model_name = "t5-small"  # Base model used for LoRA

print("⚡ Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

print("⚡ Loading LoRA weights from Hugging Face Hub...")
model = PeftModel.from_pretrained(base_model, lora_model_hub)

tokenizer = AutoTokenizer.from_pretrained(lora_model_hub)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Model loaded and ready!")

# ===========================
# 2. Inference Function
# ===========================
def generate_answer(instruction, question):
    prompt = instruction + ": " + question
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===========================
# 3. Gradio Interface
# ===========================
demo = gr.Interface(
    fn=generate_answer,
    inputs=["text", "text"],
    outputs="text",
    title="Mini Instruction-Following T5 with LoRA",
    description="Ask a question like: 'Who discovered gravity?'"
)

# Launch on Hugging Face Space
demo.launch()

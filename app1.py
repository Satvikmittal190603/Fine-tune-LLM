# ============================================
# 🚀 Mini Instruction-Following T5 with LoRA (Inference Only)
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# ============================================
# 1. Load Fine-Tuned Model
# ============================================
model_name = "fine_tuned_t5"  # folder with save_pretrained
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ============================================
# 2. Inference Function
# ============================================
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

# Test output (optional)
print("🔥 Sample Output:", generate_answer("Answer the question", "Who discovered gravity?"))

# ============================================
# 3. Gradio Web Interface
# ============================================
demo = gr.Interface(
    fn=generate_answer,
    inputs=["text", "text"],
    outputs="text",
    title="Mini Instruction-Following T5 with LoRA",
    description="Ask a question like: 'Who discovered gravity?'"
)

# Important: use server_name and server_port for deployment
demo.launch(server_name="0.0.0.0", server_port=7861)

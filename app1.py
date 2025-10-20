# ============================================
# 🚀 Instruction Tuning with LoRA on T5-small
# ============================================

# pip install transformers datasets peft accelerate gradio torch

import random
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import gradio as gr

# ============================================
# 1. Create small instruction dataset
# ============================================

dataset_list = []

qa_pairs = [
    ("Who was the first person to walk on the moon?", "Neil Armstrong"),
    ("What is the capital of France?", "Paris"),
    ("Who discovered gravity?", "Isaac Newton"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the tallest mountain in the world?", "Mount Everest"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("What is the capital of Japan?", "Tokyo")
]

# Generate 200 Q&A examples
for i in range(200):
    inp, out = random.choice(qa_pairs)
    dataset_list.append({
        "instruction": "Answer the question",
        "input": inp,
        "output": out
    })

# Shuffle dataset
random.shuffle(dataset_list)

# Optional: Save to JSONL
with open("instruction_dataset.jsonl", "w") as f:
    for item in dataset_list:
        f.write(json.dumps(item) + "\n")

print(f"✅ Instruction dataset created with {len(dataset_list)} examples!")

# ============================================
# 2. Convert to HF Dataset & Split
# ============================================
dataset = Dataset.from_list(dataset_list)
dataset = dataset.train_test_split(test_size=0.2)

# ============================================
# 3. Load Tokenizer & Model
# ============================================
model_name = "t5-small"  # use t5-large if GPU is strong
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ============================================
# 4. LoRA Config for T5
# ============================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "k", "v", "o"],  # ✅ correct for T5
    lora_dropout=0.1,
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================
# 5. Preprocess Function
# ============================================
def preprocess(example):
    # Consistent prompt format
    prompt = example["instruction"] + ": " + example["input"]

    # Tokenize input
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)

    # Tokenize output
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["output"], truncation=True, padding="max_length", max_length=64)

    # Replace padding token id's of the labels by -100
    labels_ids = labels["input_ids"]
    labels_ids = [(-100 if token == tokenizer.pad_token_id else token) for token in labels_ids]
    inputs["labels"] = labels_ids

    return inputs

tokenized_dataset = dataset.map(preprocess, batched=False)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ============================================
# 6. Training Arguments
# ============================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./capstone_lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,                      # ⬆️ more epochs for better learning
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    predict_with_generate=True,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

# ============================================
# 7. Trainer
# ============================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

# ============================================
# 8. Train the Model
# ============================================
trainer.train()

# ============================================
# 9. Test Inference
# ============================================
def generate_answer(instruction, question):
    prompt = instruction + ": " + question
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test on a sample
print("🔥 Sample Output:", generate_answer("Answer the question", "Who discovered gravity?"))

# ============================================
# 10. Deploy with Gradio (optional)
# ============================================
demo = gr.Interface(
    fn=generate_answer,
    inputs=["text", "text"],
    outputs="text",
    title="Mini Instruction-Following T5 with LoRA",
    description="Ask a question like: 'Who discovered gravity?'"
)
demo.launch()

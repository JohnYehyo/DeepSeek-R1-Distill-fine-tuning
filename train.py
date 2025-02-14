from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import torch
# import logging
# logging.basicConfig(level=logging.DEBUG)

# **加载模型**
local_model_path = "./model/"
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# print('----------------')
# for name, module in model.named_modules():
#     if 'self_attn' in name:
#         print(f"{name}: {module}")

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# **设置 LoRA 参数**
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
)

# **添加 LoRA**
model = get_peft_model(model, lora_config)

# **打印可训练参数**
model.print_trainable_parameters()

# **加载数据**
dataset = load_dataset("json", data_files="./dataset/train.json")
print(dataset)

# **数据预处理**
def format_conversation(messages):
    """格式化 Qwen 对话数据"""
    formatted = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted += f"[系统] {msg['content']}\n"
        elif msg["role"] == "user":
            formatted += f"[用户] {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"[助手] {msg['content']}\n"
    return formatted

def preprocess_function(examples):
    """Tokenize 对话数据"""
    texts = [format_conversation(msgs) for msgs in examples["messages"]]
    print(f"Formatted texts: {texts[:5]}")
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# **数据 Collator**
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 关闭 MLM，启用自回归训练
)

# **训练参数**
training_args = TrainingArguments(
    output_dir="./deepseek_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=10,
    save_steps=50,
    save_total_limit=50,
    logging_steps=1,
    fp16=False,
    optim="adamw_torch",
    lr_scheduler_type="linear"
)

# **Trainer 训练**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")


# **保存 LoRA 适配器**
model.save_pretrained("deepseek-lora-adapter")
tokenizer.save_pretrained("deepseek-lora-adapter")

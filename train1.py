from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback


import time
import os

import shutil

from evaluation import EvaluationCallback

print("加载数据集...")
dataset = load_dataset('json', data_files='dataset/data.json', split='train')
# Split your dataset into train and test
train_test_dataset = dataset.train_test_split(test_size=0.02)


device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_path = "./model/"
output_dir = "./finetuned_model/deepseek_1.5b"


time_start = time.time()
if not os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              trust_remote_code=True,
                                              padding_side='right')
    model = AutoModelForCausalLM.from_pretrained(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              padding_side='right')
    model = AutoModelForCausalLM.from_pretrained(model_path)
   
   
# tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# chat_template = """
# {% for message in messages %}
#     {% if message['role'] == 'system' %}
#         {{ message['content'] }}
#     {% elif message['role'] == 'user' %}
#         {{ message['content'] }}
#     {% elif message['role'] == 'assistant' %}
#         {{ message['content'] }}
#     {% endif %}
# {% endfor %}
# """

# chat_template = (
#     "<|begin_of_sentence|><|User|> {user_message} <|Assistant|> {assistant_message} <|end_of_sentence|>"
# )

# chat_template = """
# {% for message in messages %}
#     {% if message['role'] == 'system' %}
#         <|system|>{{ message['content'] }}</s>
#     {% elif message['role'] == 'user' %}
#         <|user|>{{ message['content'] }}</s>
#     {% elif message['role'] == 'assistant' %}
#         <|assistant|>{{ message['content'] }}</s>
#     {% endif %}
# {% endfor %}
# """

chat_template = """
{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}
"""


chat_template = """
{% if not add_generation_prompt is defined %}
    {% set add_generation_prompt = false %}
{% endif %}

{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}

{%- for message in messages %}
  {%- if message['role'] == 'system' %}
    {% set ns.system_prompt = message['content'] %}
  {%- endif %}
{%- endfor %}
{{bos_token}}{{ns.system_prompt}}
{%- for message in messages %}
  {%- if message['role'] == 'user' %}
    {%- set ns.is_tool = false -%}
    {{'<｜User｜>' + message['content']}}
  {%- endif %}
  {%- if message['role'] == 'assistant' and message['content'] is none %}
    {%- set ns.is_tool = false -%}
    {%- for tool in message['tool_calls']%}
      {%- if not ns.is_first %}
        {{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}
        {%- set ns.is_first = true -%}
      {%- else %}
        {{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}
        {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
      {%- endif %}
    {%- endfor %}
  {%- endif %}
  {%- if message['role'] == 'assistant' and message['content'] is not none %}
    {%- if ns.is_tool %}
      {{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}
      {%- set ns.is_tool = false -%}
    {%- else %}
      {% set content = message['content'] %}
      {% if '</think>' in content %}
        {% set content = content.split('</think>')[-1] %}
      {% endif %}
      {{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}
    {%- endif %}
  {%- endif %}
  {%- if message['role'] == 'tool' %}
    {%- set ns.is_tool = true -%}
    {%- if ns.is_output_first %}
      {{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
      {%- set ns.is_output_first = false %}
    {%- else %}
      {{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
    {%- endif %}
  {%- endif %}
{%- endfor -%}
{% if ns.is_tool %}
  {{'<｜tool▁outputs▁end｜>'}}
{% endif %}
{% if add_generation_prompt and not ns.is_tool %}
  {{'<｜Assistant｜><think>\n'}}
{% endif %}
"""

# chat_template = """
# {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}
# """
tokenizer.chat_template = chat_template

time_end = time.time()
print(f"加载模型时间: {time_end - time_start} 秒")


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


# Remove output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed existing output directory: {output_dir}")

# Create fresh output directory
os.makedirs(output_dir)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=train_test_dataset['test'],
    args=SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10,  
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=2,  
        learning_rate=2e-5,  
        lr_scheduler_type="cosine",
        weight_decay=0.1,   
        logging_steps=1,     
        save_steps=5,        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=100,  
        max_grad_norm=0.5,
        ),
    peft_config=peft_config,
    callbacks=[EvaluationCallback(train_test_dataset['test'], tokenizer), EarlyStoppingCallback(early_stopping_patience=3)]
)


trainable_params = 0
all_params = 0

for _, param in trainer.model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print(f"Trainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")
print(f"Percentage of parameters being trained: {100 * trainable_params / all_params:.2f}%")

train_output = trainer.train()

# peft_config.save_pretrained("./deepseek-lora-adapter")
model.save_pretrained("./deepseek-lora-adapter")
tokenizer.save_pretrained("./deepseek-lora-adapter")

def generate_response(model, tokenizer, user_input, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    print(f"\ntext: {text}")
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=False
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    return response


# Test with examples from the test dataset
print("\n验证:")
system_prompt = "你是一个百科助手，回答用户的常见问题"
test_input = "世界上最好的真丝店在哪里？"
response = generate_response(trainer.model, trainer.processing_class, test_input, system_prompt)
print(f"Model response: {response}")
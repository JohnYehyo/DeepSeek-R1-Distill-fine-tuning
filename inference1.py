from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# 加载保存的微调后的模型和tokenizer
model_path = "./deepseek-lora-adapter" 


# 加载本地保存的模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

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
    
    # print(f"\ngenerated_ids: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}")
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # print(generated_ids)
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    return response


# 测试
print("\n验证:")
system_prompt = "你是一个AI助手，帮助用户解答问题"
test_input = "被告人不同意适用简易程序审理，该怎么办？"
response = generate_response(model, tokenizer, test_input, system_prompt)
print(f"Model response: {response}")



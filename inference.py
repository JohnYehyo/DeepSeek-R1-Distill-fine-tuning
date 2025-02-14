from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/")

# 加载基础模型
local_model_path = "./model/"
base_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# # 加载 LoRA 适配层
model = PeftModel.from_pretrained(base_model, "deepseek-lora-adapter")


# local_model_path = "./deepseek_lora/checkpoint-60/"
# local_model_path = "./deepseek-lora-adapter/"
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_path,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# 切换到评估模式
model.eval()

# 构造类似的数据格式
input_text = [
    # {"role": "system", "content": "你是一个百科助手，回答用户的常见问题。"},
    {"role": "user", "content": "怎么多陪伴家人"}
]

# 将输入文本拼接成模型理解的格式
formatted_input = ""
for message in input_text:
    formatted_input += f"{message['role']}：{message['content']}\n"
    
print("----------------")
print(formatted_input)

# 编码输入
inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

# 生成模型输出
output = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.5,  # 降低随机性，生成更为一致的答案
    top_k=50,
    top_p=0.95
)

# 解码并打印输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

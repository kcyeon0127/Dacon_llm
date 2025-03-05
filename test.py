import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "DBCMLAB/Llama-3-instruction-constructionsafety-layertuning"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer.chat_template)
print(type(tokenizer.chat_template))

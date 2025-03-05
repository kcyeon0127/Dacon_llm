import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
import json
from tqdm import tqdm

model_id = "OLAIR/ko-r1-14b-v2.0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(tokenizer.pad_token)
print(tokenizer.eos_token)
tokenizer.pad_token = tokenizer.eos_token

print()
print(tokenizer.pad_token)
print(tokenizer.eos_token)

print(tokenizer.bos_token)
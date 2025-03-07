import sys
sys.path.append('/home/elicer/DaconAcc/')

import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
import json
from tqdm import tqdm
from lora.search_system import SearchEngine


# model_id = "DBCMLAB/Llama-3-instruction-constructionsafety-layertuning"
# lora_path = "/home/elicer/DaconAcc/finetuned_DBCMLAB"
    
model_id = "juungwon/Llama-3-instruction-constructionsafety"
lora_path = "/home/elicer/DaconAcc/finetuned_juungwon_WO_Quant_similarity"
    
    
torch_dtype = torch.float16

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")

# model = AutoModelForCausalLM.from_pretrained(lora_path)

model.eval()

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

stop_word = "<|eot_id|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


model = PeftModel.from_pretrained(model, lora_path)
batch_size = 1
pipe = pipeline("text-generation", model=model, tokenizer = tokenizer, torch_dtype=torch.bfloat16, device_map="auto", batch_size=batch_size)


# txt_save_path = "DBCMLAB_8bit_finetuned_result_valid_fewshot0.txt"
# test_path = "/home/elicer/DaconAcc/dataset/valid_prompt.csv"
txt_save_path = "finetuned_juungwon_WO_Quant_Sim_8bit.txt"

test_path = "/home/elicer/DaconAcc/dataset/test_prompt.csv"
dataset = pd.read_csv(test_path)["question"].tolist()

if os.path.exists(txt_save_path):
    with open(txt_save_path) as f:
        texts = f.read().strip()
        if texts == '':
            number_of_saved = 0
        else:
            number_of_saved = len(texts.strip().split('\n'))
    print("number_of_saved:", number_of_saved)
else:
    number_of_saved = 0
    print("File not exits")



outputs = []
system_token = "<|start_header_id|>system<|end_header_id|>\n\n친절한 건설안전전문가로서 상대방의 요청에 최대한 '자세하고' 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.<|eot_id|>"
se = SearchEngine(faiss_index_path = '/home/elicer/DaconAcc/faiss_index/train_drop_prompt')
prompts = []
k = 0
for line in tqdm(dataset[number_of_saved:]):
    if k != 0:
        few_shots = se.search(line, k = k)
    else:
        few_shots = []
    example_prompt = []
    for few_shot in few_shots:
        example_prompt += few_shot
    
    messages = [
        {"role": "user", "content": line}
    ]
    
    messages = example_prompt + messages
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt[:len(tokenizer.bos_token)].strip() == tokenizer.bos_token:
        prompt = prompt[len(tokenizer.bos_token):]
    prompt = system_token + prompt
    if prompt[:len(tokenizer.bos_token)].strip() != tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt
    prompts.append(prompt)

print()
print(prompts[0])
print()


generate_token = "<|im_start|>assistant" if "r1" in model_id else "<|start_header_id|>assistant<|end_header_id|>"

for i in tqdm(range(0, len(prompts), batch_size)):
    batch = prompts[i:i+batch_size]
    outputs = pipe(batch, max_new_tokens=128, do_sample=False, stop_sequence=stop_word)
    outputs = [o[0]["generated_text"].strip() for o in outputs]

    # output = output[0]["generated_text"].strip()
    
    for output in outputs:
        # generated_text = output.split(generate_token)[1].split("<|im_end|>")[0].strip()
        generated_text = output.split(generate_token)[1].strip()
        # prompt = output[:-len(generated_text)]
        prompt = output.split(generate_token)[0].strip()
        
        with open(txt_save_path,'a',encoding='utf-8') as f:
            json.dump({
                "prompt":prompt,
                "output": generated_text
            }, f, ensure_ascii=
                    False)
            f.write("\n")    
    


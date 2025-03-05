import transformers
import torch
from tqdm import tqdm

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

save_path = "/home/elicer/DaconAcc/lora/results/DBCMLAB_8bit_finetuned_result_valid_fewshot0_llm.txt"
output_path = "/home/elicer/DaconAcc/DBCMLAB_8bit_finetuned_result_valid_fewshot0.txt"
with open(output_path) as f:
    outputs = [line.strip() for line in f]


PROMPT = '''You are an AI assistant. Please remove the duplicate parts in the following sentence without changing any of the characters. 당신은 AI 어시스턴트입니다. 다음 문장에서 글자 변경없이 중복된 부분을 제거해 주세요.'''
for instruction in tqdm(outputs):
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
        # temperature=0.6,
        # top_p=0.9
    )

    clean_text = outputs[0]["generated_text"][len(prompt):]
    with open(save_path, 'a') as f:
        f.write(clean_text.replace('\n', ' ') + '\n')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "DBCMLAB/Llama-3-instruction-constructionsafety-layertuning"
tuned_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline("text-generation", model=tuned_model, tokenizer = tokenizer, torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": "친절한 건설안전전문가로서 상대방의 요청에 최대한 '자세하고' 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.",
    },
    {"role": "user", "content": "인적사고 '끼임'이고, 물적사고'없음' 입니다. 공사종류 대분류 '토목', 중분류 '댐' 공사 중 공종 대분류 '토목', 중분류 '지반개량공사' 작업에서 사고객체 '건설공구'(중분류: '몰탈혼합기')와 관련된 사고가 발생했습니다. 작업 프로세스는 '설치작업'이며, 장소는 '기타 / 내부', 부위는 '몰탈혼합기 / 하부(아래)' 입니다. 사고 원인은 '모터 교체 작업 시 모터를 내려놓을 곳에 미리 받침대를 비치 하지 않음'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=128, do_sample=False)
print(outputs[0]["generated_text"])



"모터 교체 작업 시 모터를 내려놓을 곳에 미리 받침대를 설치하여 모터가 땅에 떨어지더라도 손이 직접 땅에 부딪치지 않도록 방지하고 피해자 치료를 시행하는 조치."
import os, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer
import gc
from datasets import Dataset
print(os.getcwd())

ROOT_DIR = '/home/elicer/DaconAcc/dataset'
RANDOM_STATE = 42

# 데이터 불러오기
train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"), encoding = 'utf-8-sig')
test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"), encoding = 'utf-8-sig')


valid = train.sample(n=1000, random_state=42)

train_drop = train.drop(valid.index)

# 수정된 데이터를 새로운 파일로 저장
train_drop.to_csv(os.path.join(ROOT_DIR, 'train_drop.csv'), index=False, encoding='utf-8-sig')
valid.to_csv(os.path.join(ROOT_DIR, 'valid.csv'), index=False, encoding='utf-8-sig')
print(" 파일 저장됨 ")

# 데이터 전처리
train_drop['공사종류(대분류)'] = train_drop['공사종류'].str.split(' / ').str[0]
train_drop['공사종류(중분류)'] = train_drop['공사종류'].str.split(' / ').str[1]
train_drop['공종(대분류)'] = train_drop['공종'].str.split(' > ').str[0]
train_drop['공종(중분류)'] = train_drop['공종'].str.split(' > ').str[1]
train_drop['사고객체(대분류)'] = train_drop['사고객체'].str.split(' > ').str[0]
train_drop['사고객체(중분류)'] = train_drop['사고객체'].str.split(' > ').str[1]


valid['공사종류(대분류)'] = valid['공사종류'].str.split(' / ').str[0]
valid['공사종류(중분류)'] = valid['공사종류'].str.split(' / ').str[1]
valid['공종(대분류)'] = valid['공종'].str.split(' > ').str[0]
valid['공종(중분류)'] = valid['공종'].str.split(' > ').str[1]
valid['사고객체(대분류)'] = valid['사고객체'].str.split(' > ').str[0]
valid['사고객체(중분류)'] = valid['사고객체'].str.split(' > ').str[1]


# 대분류별 중분류 분리된 데이터 생성
category_columns = ["공사종류", "공종", "사고객체"]
split_symbols = ["/", ">", ">"]

# 새로운 데이터프레임 생성
train_separated = pd.DataFrame()

# 각 카테고리에 대해 분리된 데이터를 생성
for col, symbol in zip(category_columns, split_symbols):
    train_separated[f"{col}(대분류)"] = train_drop[col].str.split(f" {symbol} ").str[0]
    train_separated[f"{col}(중분류)"] = train_drop[col].str.split(f" {symbol} ").str[1]


print("훈련 데이터 통합 생성")
combined_training_data = train_drop.apply(
    lambda row: {
        "question": (
            f"사고인지 시간 '{row['사고인지 시간']}'이고, 날씨 '{row['날씨']}' 일 때 "
            f"기온 '{row['기온']}', '{row['층 정보']}'입니다. "
            f"인적사고 '{row['인적사고']}'이고, 물적사고'{row['물적사고']}' 입니다. "
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 장소는 '{row['장소']}', 부위는 '{row['부위']}' 입니다. "
            f"사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)

# DataFrame으로 변환
combined_training_data = pd.DataFrame(list(combined_training_data))

# 훈련 데이터 통합 생성
combined_valid_data = valid.apply(
    lambda row: {
        "question": (
            f"사고인지 시간 '{row['사고인지 시간']}'이고, 날씨 '{row['날씨']}' 일 때 "
            f"기온 '{row['기온']}', '{row['층 정보']}'입니다. "
            f"인적사고 '{row['인적사고']}'이고, 물적사고'{row['물적사고']}' 입니다. "
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 장소는 '{row['장소']}', 부위는 '{row['부위']}' 입니다. "
            f"사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)

# DataFrame으로 변환
combined_valid_data = pd.DataFrame(list(combined_valid_data))

print("데이터 준비 중")
# Train 데이터 준비
train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()

# valid 데이터 준비
valid_questions_prevention = combined_valid_data['question'].tolist()
valid_answers_prevention = combined_valid_data['answer'].tolist()


train_documents = [
    {
        "messages": [
            {"role": "user", "content": q1}, 
            {"role": "assistant", "content": a1}
        ]
    }
    for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
]

valid_documents = [
    {
        "messages": [
            {"role": "user", "content": q1}, 
            {"role": "assistant", "content": a1}
        ]
    }
    for q1, a1 in zip(valid_questions_prevention, valid_answers_prevention)
]


# Q_LoRA
torch_dtype = torch.float16

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

model_id = "juungwon/Llama-3-instruction-constructionsafety"
# 모델 로드
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
model.eval()
for param in model.parameters():
    param.requires_grad = False
# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# PEFT (LoRA)
peft_params = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=3,
    bias="none",
    task_type="CAUSAL_LM",
)

print("학습 파라미터 설정")
from transformers import TrainingArguments

training_params = TrainingArguments(
    output_dir="/home/elicer/DaconAcc/dataset/res",
    num_train_epochs = 20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    fp16=True,
    push_to_hub=False,
    load_best_model_at_end=True,
    save_strategy = "best",
    # eval_steps=40,
    evaluation_strategy = 'epoch',
    save_total_limit = 2,
    report_to="none",
)

train_documents = Dataset.from_list(train_documents)
valid_documents = Dataset.from_list(valid_documents)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_documents,
    eval_dataset=valid_documents,
    peft_config=peft_params,
    # dataset_text_field="text",
    # max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_params,
    # packing=False,
)

gc.collect()
torch.cuda.empty_cache()

trainer.train()
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
train_drop = pd.read_csv(os.path.join(ROOT_DIR, "train_drop.csv"), encoding = 'utf-8-sig')
valid = pd.read_csv(os.path.join(ROOT_DIR, "valid.csv"), encoding = 'utf-8-sig')
test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"), encoding = 'utf-8-sig')


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

test['공사종류(대분류)'] = test['공사종류'].str.split(' / ').str[0]
test['공사종류(중분류)'] = test['공사종류'].str.split(' / ').str[1]
test['공종(대분류)'] = test['공종'].str.split(' > ').str[0]
test['공종(중분류)'] = test['공종'].str.split(' > ').str[1]
test['사고객체(대분류)'] = test['사고객체'].str.split(' > ').str[0]
test['사고객체(중분류)'] = test['사고객체'].str.split(' > ').str[1]


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

# 테스트 데이터 통합 생성
combined_test_data = test.apply(
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
        )
    },
    axis=1
)

# DataFrame으로 변환
combined_test_data = pd.DataFrame(list(combined_test_data))



# 수정된 데이터를 새로운 파일로 저장
combined_training_data.to_csv(os.path.join(ROOT_DIR, 'train_drop_prompt.csv'), index=False, encoding='utf-8-sig')
combined_valid_data.to_csv(os.path.join(ROOT_DIR, 'valid_prompt.csv'), index=False, encoding='utf-8-sig')
combined_test_data.to_csv(os.path.join(ROOT_DIR, 'test_prompt.csv'), index=False, encoding='utf-8-sig')
print(" 파일 저장됨 ")


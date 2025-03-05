import os
import pandas as pd
from typing import Tuple

def get_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ROOT_DIR = os.environ["dataset_path"]

    # 데이터 불러오기
    train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"), encoding = 'utf-8-sig')
    test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"), encoding = 'utf-8-sig')

    # 데이터 전처리
    train['공사종류(대분류)'] = train['공사종류'].str.split(' / ').str[0]
    train['공사종류(중분류)'] = train['공사종류'].str.split(' / ').str[1]
    train['공종(대분류)'] = train['공종'].str.split(' > ').str[0]
    train['공종(중분류)'] = train['공종'].str.split(' > ').str[1]
    train['사고객체(대분류)'] = train['사고객체'].str.split(' > ').str[0]
    train['사고객체(중분류)'] = train['사고객체'].str.split(' > ').str[1]

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
        train_separated[f"{col}(대분류)"] = train[col].str.split(f" {symbol} ").str[0]
        train_separated[f"{col}(중분류)"] = train[col].str.split(f" {symbol} ").str[1]

    # 분리된 데이터 출력

    train_separated.info()


    # 훈련 데이터 통합 생성
    combined_training_data = train.apply(
        lambda row: {
            "question": (
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
                f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            ),
            "answer": row["재발방지대책 및 향후조치계획"]
        },
        axis=1
    )

    # DataFrame으로 변환
    combined_training_data = pd.DataFrame(list(combined_training_data))


    # 테스트 데이터 통합 생성
    combined_test_data = test.apply(
        lambda row: {
            "question": (
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
                f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            )
        },
        axis=1
    )

    # DataFrame으로 변환
    combined_test_data = pd.DataFrame(list(combined_test_data))
    return combined_training_data, combined_test_data
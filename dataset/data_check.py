import pandas as pd

# 데이터프레임 전체 출력 설정
pd.set_option('display.max_rows', None)  # 모든 행 출력
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', None)  # 한 줄에 모든 열 출력
pd.set_option('display.max_colwidth', None)  # 열의 최대 너비 설정


train = pd.read_csv("/home/elicer/DaconAcc/dataset/train.csv", encoding = 'utf-8-sig')

df = pd.DataFrame(train)

# 확인할 키워드 리스트
keywords_to_check = ["안전교육", "안전 교육", "교육"]


mask = ~df["재발방지대책 및 향후조치계획"].astype(str).apply(lambda x: any(keyword in x for keyword in keywords_to_check))
filtered_df = df[mask]

for i, line in enumerate(filtered_df["재발방지대책 및 향후조치계획"], start=1):
    print(i, line)


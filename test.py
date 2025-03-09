import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def remove_near_duplicates_from_df(df, text_column='text', threshold=0.9):
    """
    pandas DataFrame의 텍스트 컬럼에서 Sentence-BERT를 이용해 유사 중복을 제거합니다.
    
    매개변수:
        df (pd.DataFrame): 원본 DataFrame
        text_column (str): 텍스트가 있는 컬럼명 (기본값: 'text')
        threshold (float): 코사인 유사도 임계치 (0~1 사이, 높을수록 엄격하게 중복 판정)
    
    반환:
        pd.DataFrame: 중복이 제거된 DataFrame
    """
    # Sentence-BERT 모델 로드 (여기서는 경량 모델 사용)
    model = SentenceTransformer("jhgan/ko-sbert-sts")
    
    # DataFrame에서 텍스트 리스트 추출  
    texts = df[text_column].tolist()
    
    # 모든 텍스트에 대해 embedding을 한 번에 계산
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    unique_indices = []       # 중복이 아닌 row의 index 저장
    unique_embeddings = []    # 선택된 unique 텍스트의 embedding 저장

    for idx, (text, emb) in enumerate(zip(tqdm(texts), embeddings)):
        is_duplicate = False
        if unique_embeddings:
            # 리스트에 저장된 텐서들을 하나의 배치 텐서로 변환
            unique_tensor = torch.stack(unique_embeddings)
            # 현재 텍스트 embedding과 기존에 선택된 embedding 간 코사인 유사도 계산
            cosine_scores = util.cos_sim(emb, unique_tensor)
            max_score = cosine_scores.max().item()
            if max_score >= threshold:
                is_duplicate = True
        if not is_duplicate:
            unique_indices.append(idx)
            unique_embeddings.append(emb)
    
    # unique한 row만 남긴 DataFrame 반환 (index 재설정)
    return df.iloc[unique_indices].reset_index(drop=True)



df = pd.read_csv("/home/elicer/DaconAcc/dataset/train_drop_prompt.csv")

# near duplicate 제거 (코사인 유사도 임계치 0.9)
df_unique = remove_near_duplicates_from_df(df, text_column='answer', threshold=0.85)

print(df_unique)

df_unique.to_csv("/home/elicer/DaconAcc/dataset/unique_train_drop_prompt_0.85.csv", index=False)


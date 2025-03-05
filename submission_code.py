import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

ROOT_DIR = "/home/elicer/DaconAcc/results"
txt_file_path = '/home/elicer/DaconAcc/DBCMLAB_8bit_fewshot0_output_ngram1.txt'
csv_file_name = 'DBCMLAB_8bit_fewshot0_ngram1.csv'

with open(txt_file_path) as f:
    generated_texts = f.read().strip().split('\n')


sub = []
for text in generated_texts:
    sub.append(text.strip())
    
print(len(sub))
print(sub[-1])
print(sub[-2])

if len(sub) == 965:
    sub = sub[:-1]
    
embedding_model_name = "jhgan/ko-sbert-sts"
embedding = SentenceTransformer(embedding_model_name)

# 문장 리스트를 입력하여 임베딩 생성
pred_embeddings = embedding.encode(sub)
print(pred_embeddings.shape)  # (샘플 개수, 768)

submission = pd.read_csv(os.path.join(ROOT_DIR,'sample_submission.csv'), encoding = 'utf-8-sig')

# 최종 결과 저장
submission.iloc[:,1] = sub
submission.iloc[:,2:] = pred_embeddings
submission.head()

# 최종 결과를 CSV로 저장
submission.to_csv(os.path.join(ROOT_DIR,csv_file_name), index=False, encoding='utf-8-sig')
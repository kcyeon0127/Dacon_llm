result_path = '/home/elicer/DaconAcc/results/test.txt'
with open(result_path) as f:
    generated_texts = f.read().split('[[SEP]]')
    
    
sub = []
for text in generated_texts:
    sub.append(text.strip().split('\n')[0])
sub = sub[:-1]
    
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

embedding_model_name = "jhgan/ko-sbert-sts"
embedding = SentenceTransformer(embedding_model_name)

# 문장 리스트를 입력하여 임베딩 생성
pred_embeddings = embedding.encode(sub)
print(pred_embeddings.shape)  # (샘플 개수, 768)

ROOT_DIR = "/home/elicer/DaconAcc/results"


submission = pd.read_csv(os.path.join(ROOT_DIR,'sample_submission.csv'), encoding = 'utf-8-sig')

print("submission rows:", submission.shape[0])
print("sub length:", len(sub))
print(sub[-1])
print(sub[-2])


# 최종 결과 저장
submission.iloc[:,1] = sub
submission.iloc[:,2:] = pred_embeddings
submission.head()

# # 최종 결과를 CSV로 저장
submission.to_csv(os.path.join(ROOT_DIR,'NCsoft_submission.csv'), index=False, encoding='utf-8-sig')
print(" 저장됨 ")
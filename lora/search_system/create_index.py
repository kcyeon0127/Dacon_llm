import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':

   train_path = "/home/elicer/DaconAcc/dataset/train_drop_prompt.csv"
   train_pd = pd.read_csv(train_path)

   question = train_pd["question"]
   answer = train_pd["answer"]

   embedding_model_name = "jhgan/ko-sbert-sts"
   model = SentenceTransformer(embedding_model_name)
   encoded_data = model.encode(question)
   print('임베딩 된 벡터 수 :', len(encoded_data))


   index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
   index.add_with_ids(encoded_data, np.array(range(0, len(question))))

   faiss_index_path = "/home/elicer/DaconAcc/faiss_index/"
   faiss.write_index(index, faiss_index_path + "train_drop_prompt")

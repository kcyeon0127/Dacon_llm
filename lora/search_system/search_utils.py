from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

class SearchEngine:
    def __init__(
        self, 
        faiss_index_path,
        model = None, 
        model_name = "jhgan/ko-sbert-sts",
        train_path = None
    ):
        
        self.train_path = "/home/elicer/DaconAcc/dataset/train_drop_prompt.csv" if train_path == None else train_path
        self.train_pd = pd.read_csv(self.train_path)
        self.question = self.train_pd["question"].tolist()
        self.answer = self.train_pd["answer"].tolist()
        
        if model is not None:
            self.model = model
        else:
            self.model_name = model_name
            self.model = SentenceTransformer(self.model_name)
        
        self.index = faiss.read_index(faiss_index_path)

    def search(self, query, k = 5):
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, k)
        return [
            [{"role": "user", "content": self.question[_id]},
            {"role": "assistant", "content": self.answer[_id]}]
        for _id in top_k[1].tolist()[0]]
    
    
if __name__ == '__main__':
    se = SearchEngine(faiss_index_path = '/home/elicer/DaconAcc/faiss_index/train_drop_prompt')
    print(se.search('공사현장', 2))
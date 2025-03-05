import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

class Evaluater:
    def __init__(
        self,
        embedding_model = "jhgan/ko-sbert-sts"
    ):
        self.model = SentenceTransformer(embedding_model, use_auth_token=False)
        
    def vectorize(self, query: str):
        return self.model.encode([query])
        
    def cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0


    def jaccard_similarity(self, text1, text2):
        """자카드 유사도 계산"""
        set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
        intersection = len(set1.intersection(set2))  # 교집합 크기
        union = len(set1.union(set2))  # 합집합 크기
        return intersection / union if union != 0 else 0
    
    def evaluate(self, texts1: List[str], texts2: List[str]) -> float:
        assert len(texts1) == len(texts2)

        results = []
        for text1, text2 in zip(tqdm(texts1), texts2):
            vec1, vec2 = self.vectorize(text1)[0], self.vectorize(text2)[0]
            value = 0.7 * self.cosine_similarity(vec1, vec2) + 0.3 * self.jaccard_similarity(text1, text2)
            results.append(value)
        
        return sum(results)/len(texts1), results
    
    
if __name__ == '__main__':
    evaluater = Evaluater()
    
    texts1 = [
        "한글",
        "바보"
    ]
    
    texts2 = [
        "영어",
        "멍청이"
    ]

    print(evaluater.evaluate(texts1, texts2))
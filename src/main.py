import os
os.environ["dataset_path"] = "dataset/"

from transformers import BitsAndBytesConfig
from src.data_utils import get_df
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


combined_training_data, combined_test_data = get_df()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model_id = "NCSOFT/Llama-VARCO-8B-Instruct"
model_id = "juungwon/Llama-3-instruction-constructionsafety"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Train 데이터 준비
train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()

train_documents = [
    f"""<|start_header_id|>user<|end_header_id|>
{q1}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>\n
{a1}<|eot_id|>

""" 
    for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
]

print("임베딩 생성")
embedding_model_name = "jhgan/ko-sbert-nli"  # 임베딩 모델 선택
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

print("벡터 스토어에 문서 추가")
vector_store = FAISS.from_texts(train_documents, embedding)

print("Retriever 정의")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# retriever.search_type
# "similarity": 벡터 유사도 기반 검색
# "mmr": 다중 다양성 검색 (Maximum Marginal Relevance)
# "similarity_score_threshold": 특정 점수 이상만 반환


print("text_generation_pipeline")
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,  # sampling 활성화
    temperature=0.1, # 랜덤 샘플링 시 단어 선택의 다양성을 조절(값이 낮을수록 더 보수적인 응답, 값이 높을수록 더 창의적이고 다양한 응답)
    return_full_text=False, # 입력 프롬프트까지 포함하여 출력을 반환할지 여부
    max_new_tokens=64, 
)
# top_k=50(다음 단어를 선택할 때, 확률이 높은 k개 후보 중에서 선택)
# top_p=1.0(확률이 높은 토큰만 선택 (상위 p 확률을 합한 후보 중에서 샘플링))
# repetition_penalty=1.0(같은 단어 반복을 방지하는 패널티)

prompt_template = """
<|start_header_id|>system<|end_header_id|>당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.<|eot_id|>

{context}
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

"""

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

print("커스텀 프롬프트 생성")
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


print("RAG 체인 생성") # 질의응답 타입
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, # gpt 같은거 사용 가능
    chain_type="stuff",  # 단순 컨텍스트 결합 방식 사용 
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # 커스텀 프롬프트 적용
)
# chain_type
# "stuff": 모든 문서를 하나의 프롬프트에 삽입하여 LLM에게 전달
# "map_reduce": 각 개별 문서가 먼저 자체적으로 언어 모델로 전송되어 원래의 답변을 얻고, 그런 다음 해당 답변은 언어 모델에 대한 최종 호출을 통해 최종 답변으로 구성
# "refine": 첫 번째 문서에서 초기 응답을 생성한 후, 나머지 문서들을 하나씩 추가하며 답변을 점진적으로 개선
# "map_rerank": 검색된 문서들 각각에 대해 개별적으로 응답을 생성한 후, LLM이 가장 관련성 높은 응답을 선택


# 테스트 실행 및 결과 저장
test_results = []

print("테스트 실행 시작... 총 테스트 샘플 수:", len(combined_test_data))

for idx, row in tqdm(combined_test_data.iterrows(), total=len(combined_test_data)):
    # 50개당 한 번 진행 상황 출력
    # if (idx + 1) % 50 == 0 or idx == 0:
    #     print(f"\n[샘플 {idx + 1}/{len(combined_test_data)}] 진행 중...")

    # RAG 체인 호출 및 결과 생성
    prevention_result = qa_chain.invoke(row['question'])

    # 결과 저장
    result_text = prevention_result['result']
    with open('/home/elicer/DaconAcc/results/test.txt', 'a') as f:
        f.write(result_text + '[[SEP]]\n\n\n')
    print(result_text)
    test_results.append(result_text)

print("\n테스트 실행 완료! 총 결과 수:", len(test_results))
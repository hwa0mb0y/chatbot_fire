# preprocess.py

import PyPDF2
import openai
import faiss
import numpy as np
import pickle
import os

# OpenAI API 키 설정
openai.api_key = 'YOUR_OPENAI_API_KEY'

# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# 텍스트 전처리 함수
def preprocess_text(text):
    # 필요에 따라 전처리 로직 추가
    return text

# 임베딩 생성 함수
def get_text_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

def main():
    # PDF 파일이 저장된 디렉토리 경로
    pdf_dir = './pdf_files'

    # 모든 문서의 텍스트와 임베딩을 저장할 리스트
    all_texts = []
    all_embeddings = []

    # PDF 디렉토리 내의 모든 파일 처리
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"{filename} 처리 중...")
            text = extract_text_from_pdf(pdf_path)
            text = preprocess_text(text)
            embeddings = get_text_embedding(text)

            all_texts.append(text)
            all_embeddings.append(embeddings)

    # 임베딩을 numpy 배열로 변환
    all_embeddings = np.array(all_embeddings).astype('float32')

    # 벡터 스토어 구축
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    # 임베딩, 텍스트, 인덱스를 저장
    with open('texts.pkl', 'wb') as f:
        pickle.dump(all_texts, f)

    faiss.write_index(index, 'vector_index.index')

    print("사전 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()

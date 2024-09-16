import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # PDF 파일이 있는 폴더 경로
    pdf_folder = "./pdfs"

    # 모든 PDF 파일에서 텍스트 추출
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            print(f"Processing file: {file_path}")
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Successfully loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    if not documents:
        print("No documents were successfully loaded. Please check your PDF files and permissions.")
        return

    print(f"Total documents loaded: {len(documents)}")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    print(f"Total text chunks after splitting: {len(texts)}")

    if not texts:
        print("No text chunks were created. Please check if the PDFs contain extractable text.")
        return

    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

    # FAISS 벡터 저장소 생성
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
        print("Successfully created FAISS index")
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")
        return

    # 벡터 저장소 저장
    vector_store.save_local("faiss_index")

    print("전처리 완료: FAISS 인덱스가 생성되었습니다.")

if __name__ == "__main__":
    main()
    
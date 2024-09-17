import streamlit as st
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 페이지 설정
st.set_page_config(page_title="화학 물질 화재 사고 전문 챗봇", page_icon="🤖")
st.title("화학 물질 화재 사고 전문 챗봇")

# OpenAI API 키 설정
openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요", type="password")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 벡터 저장소 로드
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_vector_store()

# 챗봇 초기화
@st.cache_resource
def initialize_chatbot(_openai_api_key):
    llm = ChatOpenAI(
        temperature=0.1,
        max_tokens=500,
        openai_api_key=_openai_api_key
    )
    
    prompt_template = """
    당신은 전문 분야에 대한 지식이 풍부한 도우미입니다. 주어진 맥락을 바탕으로 다음 질문에 대해 상세하고 포괄적인 답변을 제공해 주세요. 
    가능한 한 많은 관련 정보를 포함하고, 예시나 설명을 추가하여 답변을 풍부하게 만들어 주세요.

    맥락: {context}
    
    질문: {question}
    
    답변:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return chain

# 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not openai_api_key:
        st.info("OpenAI API 키를 입력해주세요.")
        st.stop()

    # 챗봇 응답 생성
    chain = initialize_chatbot(openai_api_key)
    response = chain({"question": prompt, "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages]})
    
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# 사이드바에 참고 문서 표시
with st.sidebar:
    if st.session_state.messages and "answer" in response:
        st.subheader("참고 문서")
        for doc in response["source_documents"]:
            st.write(doc.page_content[:200] + "...")

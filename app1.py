import streamlit as st
import fitz
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드
load_dotenv()

# Streamlit UI 설정
st.set_page_config(page_title="보험 약관 분석기", page_icon="📋")
st.title("📋 보험 약관 보험료 분석기")

# 1. PDF 로드 및 인덱싱 (캐싱하여 성능 최적화)
@st.cache_resource
def get_vectorstore():
    pdf_path = "9회주는 암보험Plus_해약환급금 미지급형.pdf"

    if not os.path.exists(pdf_path):
        st.error(f"파일을 찾을 수 없습니다: {pdf_path}")
        return None

    pdf_doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in pdf_doc])

    text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitters.split_text(text)

    vectorstore = FAISS.from_texts(chunks, OpenAIEmbeddings(model="text-embedding-3-large"))
    return vectorstore

# 2. 벡터스토어 초기화
vectorstore = get_vectorstore()

# 3. 질문 처리 로직
if vectorstore:
    query = st.text_input("질문:", "")

    if st.button("분석 요청"):
        with st.spinner("답변을 생성 중입니다..."):
            # 검색 및 답변
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([i.page_content for i in docs])

            # gpt-5.5 대신 gpt-4o 사용
            llm = ChatOpenAI(temperature=0, model='gpt-5.5')

            prompt_template = ChatPromptTemplate.from_template("""
            다음 배경 지식을 사용해서 질문에 대답해 주세요.

            배경지식:
            {context}
            ============
            질문:
            {question}
            """)

            chain = prompt_template | llm | StrOutputParser()
            inputs = {f"context": context, "question": query}

            response = chain.invoke(inputs)
            st.info(response)

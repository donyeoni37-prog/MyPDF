import streamlit as st
import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="PDF 보험 분석기", page_icon="📄")
st.title("📄 PDF 보험 약관 분석기")

# 1. PDF 업로드 기능
uploaded_file = st.file_uploader("분석할 PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    # 2. PDF 분석 및 벡터스토어 생성 (세션 상태에 저장하여 중복 분석 방지)
    if "vectorstore" not in st.session_state:
        with st.spinner("PDF를 분석 중입니다..."):
            # 파일 읽기
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n\n".join([page.get_text() for page in pdf_doc])

            # 텍스트 분할
            text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitters.split_text(text)

            # 벡터 저장소 생성
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.success("분석 완료!")

    # 3. 질문 처리
    query = st.text_input("질문:", "")

    if st.button("분석 요청"):
        if "vectorstore" in st.session_state:
            with st.spinner("답변 생성 중..."):
                # 벡터스토어에서 검색
                docs = st.session_state.vectorstore.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

                # LLM 호출
                llm = ChatOpenAI(temperature=0, model='gpt-5.5')

                prompt = ChatPromptTemplate.from_template("""
                다음 배경 지식을 사용해서 질문에 대답해 주세요.

                배경지식:
                {context}
                ============
                질문:
                {question}
                """)

                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": query})

                st.info(response)
else:
    st.info("왼쪽 파일 업로드 영역에 PDF를 올려주세요.")

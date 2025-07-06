#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in your .env file.")
    st.stop()

st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Type your question here")

    if user_question:
        matched_docs = vector_store.similarity_search(user_question)
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matched_docs, question=user_question)
        st.write(response)


import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG PDF Chatbot")
st.markdown("Upload a PDF and ask questions from it")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embeddings
        embedding_model = MistralAIEmbeddings(
            model="mistral-embed"
        )

        # Vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.success("PDF processed successfully")

        # LLM
        llm = ChatMistralAI(
            model="mistral-small-2506"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI assistant.

                    Use only the provided context
                    to answer the question.

                    If the answer is not present
                    in context, say:

                    "I could not find the answer in the document."
                    """
                ),
                (
                    "human",
                    """
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        user_question = st.text_input(
            "Ask a question from the PDF"
        )

        if user_question:

            with st.spinner("Generating answer..."):

                docs = retriever.invoke(user_question)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": user_question
                })

                response = llm.invoke(final_prompt)

                st.subheader("Answer")
                st.write(response.content)

                with st.expander("Retrieved Context"):

                    for i, doc in enumerate(docs):

                        st.markdown(f"### Chunk {i+1}")
                        st.write(doc.page_content)
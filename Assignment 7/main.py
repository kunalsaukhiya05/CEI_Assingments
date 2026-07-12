"""
main.py
Streamlit front-end for the RAG-based Document Q&A tool.
Upload a PDF, and ask questions grounded strictly in its content.
"""

import os
import tempfile
import streamlit as st

from doc_loader import read_pdf, chunk_documents
from index_builder import load_embedder
from langchain_community.vectorstores import FAISS
from qa_engine import get_relevant_chunks, ask_llm

st.set_page_config(page_title="Ask My Docs", page_icon="📚")

st.title("📚 Ask My Docs — RAG Q&A")
st.caption("Upload a PDF (notes, resume, research paper) and ask questions grounded in its content.")

@st.cache_resource
def get_embedder():
    return load_embedder()

embedder = get_embedder()

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if "index" not in st.session_state:
    st.session_state.index = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

if uploaded_pdf is not None:
    if uploaded_pdf.name != st.session_state.current_file:
        with st.spinner(f"Reading and indexing '{uploaded_pdf.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            pages = read_pdf(tmp_path)
            chunks = chunk_documents(pages)
            st.session_state.index = FAISS.from_documents(chunks, embedder)
            st.session_state.current_file = uploaded_pdf.name

            os.remove(tmp_path)

        st.success(f"'{uploaded_pdf.name}' indexed with {len(chunks)} chunks!")

if st.session_state.index is not None:
    user_question = st.text_input("Ask something about the document:")

    if user_question:
        with st.spinner("Searching document..."):
            context = get_relevant_chunks(st.session_state.index, user_question)

        with st.spinner("Thinking..."):
            answer = ask_llm(user_question, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            st.write(context)
else:
    st.info("Upload a PDF above to get started.")

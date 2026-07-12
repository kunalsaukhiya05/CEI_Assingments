"""
doc_loader.py
Handles loading a PDF and breaking it into smaller text chunks
so the embedding model can work with manageable pieces.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_pdf(path):
    """Loads a PDF file and returns a list of page-wise documents."""
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages


def chunk_documents(pages, chunk_size=600, chunk_overlap=150):
    """
    Breaks pages into overlapping chunks.
    chunk_size -> max chars per chunk
    chunk_overlap -> repeated chars between chunks so context isn't lost
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(pages)


if __name__ == "__main__":
    pdf_path = "data/my_notes.pdf"

    pages = read_pdf(pdf_path)
    print(f"Total pages loaded: {len(pages)}")

    chunks = chunk_documents(pages)
    print(f"Total chunks created: {len(chunks)}")

    print("\n--- Sample chunk 1 ---")
    print(chunks[0].page_content)

"""
index_builder.py
Creates embeddings for text chunks and stores them in a FAISS index
so we don't need to re-embed the document every single run.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from doc_loader import read_pdf, chunk_documents

INDEX_PATH = "kunal_faiss_index"


def load_embedder():
    """Loads a lightweight sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_index(chunks, embedder):
    """Builds a FAISS index from chunks and saves it to disk."""
    index = FAISS.from_documents(chunks, embedder)
    index.save_local(INDEX_PATH)
    return index


def load_index(embedder):
    """Loads a previously saved FAISS index from disk."""
    return FAISS.load_local(
        INDEX_PATH,
        embedder,
        allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    pages = read_pdf("data/my_notes.pdf")
    chunks = chunk_documents(pages)
    print(f"Chunks ready for embedding: {len(chunks)}")

    embedder = load_embedder()
    index = create_index(chunks, embedder)
    print(f"FAISS index saved at '{INDEX_PATH}'")

    test_query = "Summarize the key skills mentioned"
    hits = index.similarity_search(test_query, k=2)

    print("\n--- Top matches ---")
    for i, h in enumerate(hits, 1):
        print(f"\nMatch {i}:\n{h.page_content}")

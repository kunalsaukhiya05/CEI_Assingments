"""
qa_engine.py
Core question-answering logic: fetch relevant chunks from the FAISS
index, then ask the LLM to answer strictly from that retrieved context.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from index_builder import load_embedder, load_index

load_dotenv()
llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ANSWER_PROMPT = """You are a document assistant. Use ONLY the context given below to
answer the user's question. If the context doesn't contain the answer, reply:
"I couldn't find that in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""


def get_relevant_chunks(index, question, top_k=3):
    """Runs a similarity search and joins the matched chunks into one context string."""
    matches = index.similarity_search(question, k=top_k)
    return "\n\n".join(m.page_content for m in matches)


def ask_llm(question, context):
    """Sends the question + retrieved context to the Groq LLM and returns the answer."""
    filled_prompt = ANSWER_PROMPT.format(context=context, question=question)

    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": filled_prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    embedder = load_embedder()
    index = load_index(embedder)

    question = "What internships has this person done?"
    context = get_relevant_chunks(index, question)

    print("--- Retrieved context ---")
    print(context)

    answer = ask_llm(question, context)
    print("\n--- Answer ---")
    print(answer)

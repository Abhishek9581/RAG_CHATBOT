# """
# query.py
# --------
# Retrieves relevant chunks from the Chroma vector DB for a user question,
# then asks Claude to answer using only that retrieved context.

# Usage:
#     python query.py
#     (interactive loop — type 'exit' to quit)
# """

# import os
# import sys

# import chromadb
# from chromadb.utils import embedding_functions
# from anthropic import Anthropic
# from dotenv import load_dotenv

# load_dotenv()

# DB_DIR = "chroma_db"
# COLLECTION_NAME = "documents"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# TOP_K = 4  # number of chunks to retrieve per question
# MODEL = "claude-sonnet-4-5"  # current default Sonnet model as of this writing

# SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
# provided context. If the context does not contain enough information to answer
# the question, say so clearly instead of guessing or using outside knowledge.
# Always mention which source(s) you used, if the context includes source names."""


# def get_collection():
#     client = chromadb.PersistentClient(path=DB_DIR)
#     embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=EMBEDDING_MODEL
#     )
#     try:
#         return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
#     except Exception:
#         print("No document collection found. Run `python ingest.py` first.")
#         sys.exit(1)


# def retrieve(collection, question: str, top_k: int = TOP_K):
#     results = collection.query(query_texts=[question], n_results=top_k)
#     chunks = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]
#     return list(zip(chunks, metadatas, distances))


# def build_context(retrieved: list) -> str:
#     parts = []
#     for i, (chunk, meta, dist) in enumerate(retrieved):
#         source = meta.get("source", "unknown")
#         parts.append(f"[Source: {source}]\n{chunk}")
#     return "\n\n---\n\n".join(parts)


# def ask_claude(client: Anthropic, question: str, context: str) -> str:
#     user_message = f"""Context:
# {context}

# Question: {question}"""

#     response = client.messages.create(
#         model=MODEL,
#         max_tokens=1000,
#         system=SYSTEM_PROMPT,
#         messages=[{"role": "user", "content": user_message}],
#     )
#     return response.content[0].text


# def main():
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         print("Set the ANTHROPIC_API_KEY environment variable before running.")
#         print("  export ANTHROPIC_API_KEY='your-key-here'   (Mac/Linux)")
#         print("  $env:ANTHROPIC_API_KEY='your-key-here'      (Windows PowerShell)")
#         sys.exit(1)

#     collection = get_collection()
#     client = Anthropic(api_key=api_key)

#     print("RAG chatbot ready. Ask a question about your documents (type 'exit' to quit).\n")

#     while True:
#         question = input("You: ").strip()
#         if question.lower() in {"exit", "quit"}:
#             break
#         if not question:
#             continue

#         retrieved = retrieve(collection, question)
#         if not retrieved:
#             print("Bot: I couldn't find any relevant content in the documents.\n")
#             continue

#         context = build_context(retrieved)
#         answer = ask_claude(client, question, context)

#         print(f"\nBot: {answer}\n")

#         # Show what was retrieved -- useful while you're debugging retrieval quality
#         print("(Retrieved from: " + ", ".join(sorted({m.get("source", "?") for _, m, _ in retrieved})) + ")\n")


# if __name__ == "__main__":
#     main()
"""
query.py
--------
Retrieves relevant chunks from the Chroma vector DB,
then asks a local Ollama model to answer using only that context.

Usage:
    python query.py
"""

import sys

import chromadb
from chromadb.utils import embedding_functions
from ollama import chat
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

# Local Ollama model
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context. If the context does not contain enough information to answer
the question, say so clearly instead of guessing or using outside knowledge.
Always mention which source(s) you used, if the context includes source names."""


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception:
        print("No document collection found. Run `python ingest.py` first.")
        sys.exit(1)


def retrieve(collection, question: str, top_k: int = TOP_K):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return list(zip(chunks, metadatas, distances))


def build_context(retrieved: list) -> str:
    parts = []

    for chunk, meta, dist in retrieved:
        source = meta.get("source", "unknown")

        parts.append(
            f"[Source: {source}]\n{chunk}"
        )

    return "\n\n---\n\n".join(parts)


def ask_ollama(question: str, context: str) -> str:

    user_message = f"""Context:
{context}

Question: {question}
"""

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    return response.message.content


def main():

    collection = get_collection()

    print(
        "RAG chatbot ready. Ask a question about your documents "
        "(type 'exit' to quit).\n"
    )

    while True:

        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        retrieved = retrieve(collection, question)

        if not retrieved:
            print(
                "Bot: I couldn't find any relevant content "
                "in the documents.\n"
            )
            continue

        context = build_context(retrieved)

        answer = ask_ollama(question, context)

        print(f"\nBot: {answer}\n")

        sources = sorted({
            m.get("source", "?")
            for _, m, _ in retrieved
        })

        print(
            "(Retrieved from: "
            + ", ".join(sources)
            + ")\n"
        )


if __name__ == "__main__":
    main()
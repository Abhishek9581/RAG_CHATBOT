"""
ingest.py
---------
Loads documents from the docs/ folder, chunks them, embeds them with
a local sentence-transformer model, and stores them in a persistent
Chroma vector database.

Usage:
    python ingest.py
"""

import os
import glob
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from tqdm import tqdm

DOCS_DIR = "docs"
DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough to start


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def load_documents(docs_dir: str) -> list[dict]:
    """Returns a list of {"text": ..., "source": ...} dicts, one per file."""
    documents = []
    patterns = ["*.txt", "*.md", "*.pdf"]
    filepaths = []
    for pattern in patterns:
        filepaths.extend(glob.glob(os.path.join(docs_dir, "**", pattern), recursive=True))

    if not filepaths:
        print(f"No documents found in '{docs_dir}/'. Add .txt, .md, or .pdf files there.")
        return documents

    for path in filepaths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                text = read_pdf_file(path)
            else:
                text = read_text_file(path)
            if text.strip():
                documents.append({"text": text, "source": os.path.basename(path)})
        except Exception as e:
            print(f"Skipping {path}: {e}")

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple fixed-size character chunking with overlap.

    This is the naive baseline. Once the pipeline works end to end, a good
    next experiment is swapping this for sentence-aware or semantic chunking
    and comparing retrieval quality.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    print("Loading documents...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        return

    print(f"Loaded {len(documents)} document(s). Chunking...")
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc["text"])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": chunk_idx})
            all_ids.append(f"doc{doc_idx}_chunk{chunk_idx}")

    print(f"Created {len(all_chunks)} chunks. Embedding and storing (this may take a moment)...")

    client = chromadb.PersistentClient(path=DB_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Fresh collection each run — simplest mental model while you're iterating.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]
        collection.add(documents=batch_chunks, ids=batch_ids, metadatas=batch_metadatas)

    print(f"\nDone. {len(all_chunks)} chunks stored in '{DB_DIR}/'.")
    print("Run `python query.py` to start asking questions.")


if __name__ == "__main__":
    main()

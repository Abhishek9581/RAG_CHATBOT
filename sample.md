# About This Sample Document

This is a placeholder document so you can test the pipeline immediately
after setup, before adding your own files.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique where a language model's
answer is grounded in specific retrieved documents rather than relying purely
on what the model memorized during training. The typical pipeline is:

1. Split documents into chunks
2. Convert each chunk into a vector embedding
3. Store embeddings in a vector database
4. At query time, embed the user's question and find the most similar chunks
5. Pass those chunks to the language model as context
6. The model answers using that context

## Why RAG matters

RAG lets a chatbot answer questions about information it was never trained
on -- your company's internal docs, a book you're reading, recent research
papers -- without needing to fine-tune the underlying model.

## Next steps for this project

Replace this file with your own .txt, .md, or .pdf documents in the docs/
folder, then re-run ingest.py.

# Simple RAG Chatbot

Ask questions over your own documents (.txt, .md, .pdf) using local embeddings
for retrieval and Claude for answer generation.

## How it works

```
docs/*.txt,*.md,*.pdf
        |
        v
   [ingest.py]  --chunk--> chunks --embed (local model)--> Chroma vector DB
        |
        v
   [query.py]  --your question--> embed question --> retrieve top-k chunks
        |                                                     |
        v                                                     v
   send question + retrieved chunks as context ------> Claude API --> answer
```

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate    # Mac/Linux
   venv\Scripts\activate       # Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Anthropic API key:
   ```
   cp .env.example .env
   # then edit .env and paste your key in
   ```
   Get a key at https://console.anthropic.com if you don't have one.

4. Add your own documents to the `docs/` folder (a sample one is included
   so you can test the pipeline immediately without adding anything).

## Usage

```
python ingest.py     # chunks + embeds your documents, builds the vector DB
python query.py       # interactive Q&A loop
```

## What to look at / tweak

- **`CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest.py`** — smaller chunks give
  more precise retrieval but less context per chunk; larger chunks give
  more context but noisier retrieval. Worth experimenting with your own
  documents.
- **`TOP_K` in `query.py`** — how many chunks get retrieved per question.
  Too low and the answer misses relevant info; too high and you dilute the
  context with irrelevant chunks (and pay for more tokens).
- **Chunking strategy** — this starter uses naive fixed-size chunking. A
  good next experiment: try splitting on paragraphs/sentences instead, and
  compare answer quality.

## Extension ideas, roughly in order of difficulty

1. **Show retrieval scores** — print the similarity distance next to each
   retrieved chunk so you can see how confident the retrieval step is.
2. **Add a simple eval set** — write 15-20 question/answer pairs you know
   the ground truth for, run them through the pipeline, and manually score
   whether the retrieved chunks actually contained the answer (retrieval
   quality) and whether Claude's answer was correct (generation quality).
   This is the single most valuable thing you can add — it's what turns a
   toy demo into something you can reason about and improve.
3. **Hybrid search** — combine keyword search (BM25) with embedding
   similarity; often outperforms embeddings alone, especially for exact
   terms/names/numbers.
4. **Re-ranking** — after retrieving top-k with embeddings, re-rank with a
   cross-encoder model for better precision.
5. **Conversation memory** — let follow-up questions reference earlier
   turns ("what about the second one?").
6. **Web UI** — wrap `query.py`'s logic in a Streamlit app for a proper
   demo-able interface.

## Notes

- Everything except the Claude API call runs locally — no cost for
  embeddings or vector storage.
- The vector DB persists in `chroma_db/` between runs. Delete that folder
  (or just re-run `ingest.py`, which rebuilds it) if you change your
  documents or chunking strategy.

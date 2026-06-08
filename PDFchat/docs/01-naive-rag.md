# 1. Naive RAG — the baseline

> **TL;DR:** Split the PDF into chunks, turn each chunk into a vector, find vectors close to the question's vector, hand those chunks + question to the LLM.

## The problem

LLMs only "know" what was in their training data. They have never read your PDF.
If you ask the model directly, it will either say it doesn't know or hallucinate.

## The fix in one picture

```
   ┌──────────┐
   │   PDF    │
   └────┬─────┘
        │ extract text
        ▼
   ┌──────────┐
   │  chunks  │   (overlapping word windows, e.g. 800 words with 150 overlap)
   └────┬─────┘
        │ embedder
        ▼
   ┌──────────┐
   │ vectors  │   one fixed-size vector per chunk
   └────┬─────┘
        │ stored in
        ▼
   ┌──────────┐
   │  FAISS   │   vector database
   └──────────┘

       (per question)

   user question ─► embed ─► nearest-neighbor search in FAISS
                                    │
                                    ▼
                            top-k chunks
                                    │
                                    ▼
                   ┌─── prompt = system + chunks + question
                   ▼
                  LLM ─► grounded answer
```

## How it actually works

### 1. Chunking
- Split text into overlapping windows.
- **Why overlap?** A sentence that falls on a boundary would be cut in half and miss retrieval. Overlap means it appears intact in at least one chunk.
- Sweet spot: 500–1000 words. Too small loses context; too large makes embeddings mushy.

### 2. Embedding (the bi-encoder)
- An embedding is a fixed-length list of numbers that captures **meaning**.
- Similar meanings → vectors close together (cosine similarity).
- We use `all-MiniLM-L6-v2` (HuggingFace, free, runs on CPU).

### 3. Vector storage
- FAISS holds all chunk vectors.
- At query time: embed the question, ask FAISS for the `k` nearest vectors.

### 4. The cosine-via-inner-product trick
- We **normalize** every vector to length 1.
- Then **dot product == cosine similarity** — faster.
- FAISS index type: `IndexFlatIP`.

### 5. Grounded prompting
- System prompt: "Use ONLY the provided context. If unsure, say so. Cite page numbers."
- This is the single most important anti-hallucination lever.

## Code (the minimum viable version)

```python
# 1. Read + chunk
reader = PdfReader(path)
chunks = []
for i, page in enumerate(reader.pages, start=1):
    words = (page.extract_text() or "").split()
    step = chunk_size - overlap
    for s in range(0, len(words), step):
        chunks.append(Chunk(text=" ".join(words[s:s+chunk_size]), page=i))

# 2. Embed + build index
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vecs = embedder.encode([c.text for c in chunks], normalize_embeddings=True)
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs.astype("float32"))

# 3. Retrieve
q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
_, idx = index.search(q_vec, k=4)
hits = [chunks[i] for i in idx[0]]

# 4. Generate
context = "\n\n".join(f"[p. {c.page}]\n{c.text}" for c in hits)
resp = groq.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Use ONLY the context. Cite pages as (p. N)."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ],
    temperature=0.2,
)
```

**In this repo:** `pdf_loader.py`, `embeddings.py`, `retriever.py`, `llm.py`.

## When to use it

- Single document. Direct, well-phrased questions. Small demo or prototype.
- Internal "ask my docs" tools where users phrase questions clearly.
- Personal knowledge bases.

## When it falls apart

| Failure | Why |
|---|---|
| Follow-up questions retrieve garbage | Embedding of "what about its limitations?" has no signal. **Fix → note #2** |
| Top-4 has noise; quality plateaus | Bi-encoder is fast but rough. **Fix → note #3** |
| Exact names/IDs/acronyms are missed | Embeddings smooth over rare tokens. **Fix → note #4** |
| Scanned PDFs return nothing | pypdf can't read images of text. Needs OCR (pytesseract / unstructured). |
| The model still occasionally invents stuff | Grounding prompt helps but isn't bulletproof. Use re-ranking + better prompts to push further. |

## Key concepts this teaches

- Chunking + overlap tradeoff
- What an embedding actually is
- Cosine similarity & the normalize-then-dot-product trick
- Vector databases as nearest-neighbor search
- Grounding prompts (the anti-hallucination lever)
- Page citations as a verifiability trick

→ Next: [Conversational Memory](02-conversational-memory.md)

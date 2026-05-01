# Generic Notes about Retriever-Reranker Systems, including Indexing

## How FAISS works

FAISS (Facebook AI Similarity Search) - library for dense vector nearest neighbor search. Given query `q` of dim `d`, and a set of `N` vectors of dim `d`, find nearest `k` vectors to `q`.

INPUT: 2D numpy array of vectors of shape (`N`, `d`); and query vector of shape (1, `d`)
OUTPUT: (1) **Scores** of shape (1, `k`); (2) **Indices** of shape (1, `k`) -- for the `k` nearest vectors to `q`

Embeddings are stored as `np.ascontiguousarray`, required by FAISS for efficiency

### Types of FAISS Indexes

1. IndexFlatIP (< 50K): Exact index, based on inner product (dot product) similarity. Since vectors are normalised, this is equivalent to cosine similarity. `cos(a, b) = (a . b) / (||a|| * ||b||)`. If `a` and `b` are normalised, then `cos(a, b) = a . b`
2. IndexIVFFlat (50K-500K): Inverted file index. Partitions the vector space into `nlist` Voronoi cells using k-means clustering. During search, only `nprobe` cells are searched
3. IndexIVPQ (500K-5M): Product Quantisation index. Same as IVF but vectors in each cell are quantised using product quantisation
4. IndexHNSWFlat (500K-5M): Hierarchical Navigable Small World graph index. Vectors are nodes and edges are connected in Heirarchical strcuture. Search is greedy graph traversal. Very fast + very high recall, but memory intensive (stores graph in memory), slow to build. Cannot remove vectors after building, (append only)
5. IndexIVFPQ + OPQ rotation(Optimised Product Quantisation) (5M+): Aggressive compression
6. FAISS.index_cpu_to_gpu(): Single GPU can search billions of vectors in milliseconds

### Chunking (field-level)

- For short structured docs (e.g. tables, JSON), even if each doc is a chunk, each chunk can produce multiple vectors (per row, per title, description, summary etc.)
- YAML files are structured, need not overlap chunks. Sliding window works for longer information
- Max sequence length of 512 tokens, but effective quality drops before. Short, focused passages best

> Long 500-token passage about "loan requirements and closing cost estimates" blurs chunking quality. Including keywords abd topics in embedding adds noise (for smaller models like TAS-B). For larger models, prepending topics may help. Include metadata when passing to LLM reranker.

**1:N Vector mapping:**

- Typical chunk produces 2-4 vectors in TAS-B index + N vectors in MPNet index (one per FAQ query). `FaissIndex.build()` tracks which FAISS row belongs to which chunk via `row_to_chunk`
- At search time, FAISS rows 20, 21, 22 may all belong to chunk 5. `max-sim` aggregation collapses these back to a single score per chunk

> While metadata is passed to LLM, we avoid passing the FAQs. FAQs are retrievel time signals used by MPNet, not really useful for relevance judegement by LLM

- Can also add retrieval signals/scores to reranker - LLM can use this as a weak prior. But in general, LLMs aren't great at interpreting numeric scores, they either anchor to them heavily or ignore them.

- If system evolved to split long docs into chunks,then include parent context into the chunk

- Pass in highlighted snippet showing which part of the chunk matched the query. Reduces LLM's work

> Adding full chunk to reranker is not great as chunks become large -- lost in the middle problem. More info = diminishing returns too. Can passed a compressed representation of the chunk (matched sentence + extractive summary)

As the system grows

- Medium Scale: Semantic chunking- Use sentence embeddings to detect context shift and chunk if similarity drops below theshold.
- Large Scale: Hierarchical Indexing - Level1 index on summary, Level2 index on sections or topics, Level3 index on paragraphs
- Metadata-based filtering. Time/location/staleness/source-based filtering before or after search.

Improvements

- Replace BM25 with Learnt Sparse model like SPLADE (learns term importance weights)
- Query rewriting: "How long do I have to pay it" --> "How long do I pay MIP on a FHA loan" (better recall, but risks query drift)
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer to query using LLM, then embed IT and search. Drawbacks: adds LLM latency, can hallucinate, doeesn't help when query is well-formed already
- ColBERT: Token-level late interaction. Store one vector per token. Compute MaxSim between query and document tokens. Much higher recall, but storage stonks 100x
- Caching: Embedding cache for history+query pairs. But not useful in banking (freedom w/o context vs while selecting buttons!)
- Better embeddings: E5-large-v2 or BGE-large (larger but better) - Chinese!
- Cohere embeddings / OpenAI text embeddings (costly, slower, but better)
- Weighted RRF
- Feedback loops somehow?

## Good System

- `to chunk texts -> list[str]`: **1:N mapping** - title, description, summary, sub topic etc. Each vector encoded independently. Best matching score is kept via **max-sim**. Useful for msmarco-distilbert-tasb (ASYMMETRIC)
  - 66M params
  - Trained on MS MARCO with Topic Aware Sampling (**TAS**) with Balanced training (**B**)
  - TAS-B embeddings are not normalised during training, so raw dot product preserves signal.

- `to faq texts -> list[str]`: **1:N mapping** - one vector per FAQ query. Best matching score is kept via **max-sim**. Useful for all-mpnet-base-v2 (SYMMETRIC)
  - Very reliable in industry
  - Trained on 1B+ **sentence pairs**
  - Embeddings are L2 normalised during training. Dot product equals cosine similarity.

- `to sparse -> str`: Concatenates title, keywords, topic name into single string. Description adds noise, covered by dense indexes

> Create a special `row_to_chunk` mapping from each FAISS row back to parent chunk index. max-sim only keeps the best score per chunk (across its vectors)

- BM25 matches exact words, helps because dense models may have not seen those terms in their training
- RRF used for aggregation. **RRF IS SCORE AGNOSTIC, only uses rank positions**. We use `k=60` as default. RRF score = `1 / (k + rank)`
  - Robust to outlier retrievers. If one retriever fails/is bad, RRF allows others to contribute too
- Build method for now only checks for empty directory and recreates if directory empty. Does not account for stale content. **Better approach: hash the contents**

## Pointwise vs Listwise Reranking

- Pointwise: Each chunk scored independently. Drawbacks: Cannot reliably compare chunks against each other. Two might get same score even if one is better. Also, N calls for N chunks
- Listwise: All chunks scored together. LLM can compare and give reliable ranking. Drawbacks: Lost in the middle, position bias, output JSON parsing, hallucinate article IDs
- Can also do Step1 retriever, Step2 Pointwise cross encoder, Step3 Listwise cross encoder on top 5-10 chunks --> INVERTED FUNNEL

### Why not a traditional reranker (like cross-encoder/ms-marco-MiniLM-L12-v2)?

- Finetuning needed always. Difficult to finetune

### Query rewriting

- Improve recall of retriever by rewriting vague queries into specific ones

## Other useful things

- Embedding models warmup avoids cold start latency
- CPU bound operations wrapped to `asyncio.to_thread()` to avoid blocking event loop`
- Compute encodings parallely
- ONNX helps with CPU inference speed by 2-4x, but not GPU. Aggressively quantisation

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stage 4 (Optional): Reranking for Precision
#
# In Stage 3, we combined dense vector search with BM25 keyword matching using
# **Reciprocal Rank Fusion (RRF)** -- a simple formula that merges two ranked
# lists by their positions. RRF is fast and surprisingly effective, but it
# treats all retrieval signals equally and ignores the actual *content* of the
# candidates.
#
# In this notebook, we add a **reranking** step. The idea is a two-pass system:
#
# 1. **Fast retrieval** -- Pull a broad candidate set using cheap, quantized
#    embeddings and BM25.
# 2. **Precision reranking** -- Re-score those candidates with a
#    higher-dimensional embedding that captures finer-grained semantics.
#
# This is like the way a search engine works: a fast first pass narrows
# billions of documents to thousands, then a more expensive model picks the
# best results.

# %% [markdown]
# ## 1. Setup + Load Data + BM25Vectorizer

# %%
import os, json, math, time
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import models

load_dotenv("../.env")
client = OpenAI()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

with open("../data/pubmed_dataset.json") as f:
    dataset = json.load(f)

papers = [p for p in dataset["papers"] if p.get("abstract")][:500]
print(f"Loaded {len(papers)} papers with abstracts")

# %%
import re


class BM25Vectorizer:
    """Minimal BM25 vectorizer that produces sparse vectors for Qdrant.

    Tokenizes text, builds a vocabulary with IDF weights, and encodes
    documents/queries into sparse (index, value) pairs using BM25 term
    frequencies. This is the same class used in Stage 3 (Hybrid Search).
    """

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.avg_dl = 0.0
        self.k1 = 1.5
        self.b = 0.75

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\w+", text.lower())

    def fit(self, documents):
        """Build vocabulary and compute IDF from a list of documents."""
        doc_freq = Counter()
        total_len = 0
        n_docs = len(documents)

        for doc in documents:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        self.avg_dl = total_len / n_docs if n_docs else 1.0

        # Build vocab: assign an integer index to each token
        for idx, token in enumerate(sorted(doc_freq.keys())):
            self.vocab[token] = idx
            df = doc_freq[token]
            self.idf[token] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        print(f"BM25 vocabulary: {len(self.vocab)} terms, avg doc length: {self.avg_dl:.1f}")
        return self

    def encode(self, text):
        """Encode text into sparse vector (indices, values) using BM25 scoring."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        doc_len = len(tokens)

        indices = []
        values = []
        for token, freq in tf.items():
            if token not in self.vocab:
                continue
            idx = self.vocab[token]
            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            )
            score = self.idf.get(token, 0) * tf_norm
            if score > 0:
                indices.append(idx)
                values.append(score)

        return indices, values


# Fit BM25 on all abstracts
bm25 = BM25Vectorizer()
bm25.fit([p["abstract"] for p in papers])

# %% [markdown]
# ## 2. Matryoshka Representation Learning (MRL)
#
# OpenAI's embedding models support **Matryoshka Representation Learning** --
# you can truncate embeddings to fewer dimensions and they still work. The most
# important information is packed into the first dimensions, like nested
# Russian dolls.
#
# This lets us use **short (1536-dim) vectors for fast retrieval** and
# **full-length (3072-dim) vectors for precision reranking**, from the **SAME
# model call**. We embed once with `text-embedding-3-large` at 3072 dims, then
# simply slice the first 1536 dimensions for the retriever. No extra API call
# needed.
#
# This is strictly better than using two different models: same latency, half
# the cost, and the vectors are guaranteed to be compatible.

# %% [markdown]
# ## 3. Create Qdrant Collection

# %%
COLLECTION = "pubmed_rerank"
RETRIEVER_DIM = 1536
RERANKER_DIM = 3072

if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "dense": models.VectorParams(
            size=RETRIEVER_DIM,
            distance=models.Distance.COSINE,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            ),
        ),
        "reranker": models.VectorParams(
            size=RERANKER_DIM,
            distance=models.Distance.COSINE,
            on_disk=True,
            hnsw_config=models.HnswConfigDiff(m=0),  # No HNSW index -- reranking only
        ),
    },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
    },
)
print(f"Collection '{COLLECTION}' created with 3 vector spaces")

# %% [markdown]
# ## 4. Three Vector Spaces
#
# We now have:
#
# - **Dense** (1536-dim, quantized): Fast approximate retrieval. Quantized to
#   INT8 and kept in RAM for maximum speed. This is the "first pass" that
#   narrows the candidate set.
# - **Reranker** (3072-dim, on disk): Precise rescoring. No HNSW index needed
#   (`m=0`) since we only compute cosine similarity against a small candidate
#   set -- no graph traversal required. Stored on disk to save memory.
# - **BM25** (sparse): Keyword matching. Catches exact terms that dense
#   embeddings might blur together.
#
# The key insight: the reranker vector **doesn't need an index**. We never
# search over all 500 papers with it -- we only use it to re-score the ~15
# candidates that the dense + BM25 retrievers surfaced.

# %% [markdown]
# ## 5. Ingest Papers

# %%
BATCH_SIZE = 100
start = time.time()

for i in range(0, len(papers), BATCH_SIZE):
    batch = papers[i : i + BATCH_SIZE]
    texts = [p["abstract"] for p in batch]

    # Get full 3072-dim embedding, then truncate for retriever
    response = client.embeddings.create(
        model="text-embedding-3-large", input=texts, dimensions=RERANKER_DIM
    )

    points = []
    for j, emb_data in enumerate(response.data):
        full_vector = emb_data.embedding
        retriever_vector = full_vector[:RETRIEVER_DIM]  # MRL truncation
        reranker_vector = full_vector  # Full precision

        bm25_idx, bm25_val = bm25.encode(batch[j]["abstract"])

        points.append(
            models.PointStruct(
                id=int(batch[j]["pmid"]),
                vector={
                    "dense": retriever_vector,
                    "reranker": reranker_vector,
                    "bm25": models.SparseVector(indices=bm25_idx, values=bm25_val),
                },
                payload={
                    "pmid": batch[j]["pmid"],
                    "title": batch[j]["title"],
                    "abstract": batch[j]["abstract"],
                    "authors": [a["name"] for a in batch[j].get("authors", [])],
                    "journal": batch[j].get("journal", ""),
                    "mesh_terms": [m["term"] for m in batch[j].get("mesh_terms", [])],
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"  Ingested {min(i + BATCH_SIZE, len(papers))}/{len(papers)} papers")

elapsed = time.time() - start
print(f"\nDone! {len(papers)} papers ingested in {elapsed:.1f}s")

# %% [markdown]
# ## 6. Reranking Search Function
#
# This is where the two-pass architecture comes together. We use Qdrant's
# `prefetch` mechanism:
#
# 1. **Prefetch** retrieves candidates from both dense and BM25 (fast, broad)
# 2. **Query** rescores those candidates with the full reranker vector (slow,
#    precise)
#
# Qdrant handles the fusion internally -- no manual RRF formula needed.

# %%
def reranking_search(question, top_k=5):
    """Hybrid search with reranker fusion (instead of RRF)."""
    q_full = client.embeddings.create(
        model="text-embedding-3-large", input=question, dimensions=RERANKER_DIM
    ).data[0].embedding
    q_retriever = q_full[:RETRIEVER_DIM]
    q_reranker = q_full

    bm25_idx, bm25_val = bm25.encode(question)

    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=q_retriever, using="dense", limit=top_k * 3),
            models.Prefetch(
                query=models.SparseVector(indices=bm25_idx, values=bm25_val),
                using="bm25",
                limit=top_k * 3,
            ),
        ],
        query=q_reranker,
        using="reranker",
        limit=top_k,
        with_payload=True,
    )
    return results.points


# Quick test
question = "What are the latest advances in using CRISPR-Cas9 for treating sickle cell disease?"
results = reranking_search(question)

print(f"Query: {question}\n")
for i, point in enumerate(results, 1):
    print(f"{i}. [Score: {point.score:.4f}] {point.payload['title'][:90]}")
    print(f"   PMID: {point.payload['pmid']} | Journal: {point.payload['journal']}")
    print()

# %% [markdown]
# ## 7. Compare RRF vs Reranker
#
# Let's define the RRF-based hybrid search (from Stage 3) and compare it
# head-to-head with our reranking approach on the same query.

# %%
def rrf_hybrid_search(question, top_k=5):
    """Hybrid search with Reciprocal Rank Fusion (Stage 3 approach)."""
    q_retriever = client.embeddings.create(
        model="text-embedding-3-large", input=question, dimensions=RERANKER_DIM
    ).data[0].embedding[:RETRIEVER_DIM]

    bm25_idx, bm25_val = bm25.encode(question)

    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=q_retriever, using="dense", limit=top_k * 3),
            models.Prefetch(
                query=models.SparseVector(indices=bm25_idx, values=bm25_val),
                using="bm25",
                limit=top_k * 3,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points


# %%
comparison_question = "What papers discuss p53 tumor suppressor gene mutations specifically in glioblastoma multiforme?"

rrf_results = rrf_hybrid_search(comparison_question)
rerank_results = reranking_search(comparison_question)

print(f"Query: {comparison_question}\n")
print("=" * 90)
print("RRF HYBRID SEARCH (rank fusion)")
print("=" * 90)
for i, point in enumerate(rrf_results, 1):
    print(f"  {i}. [Score: {point.score:.4f}] {point.payload['title'][:80]}")
    mesh = point.payload.get("mesh_terms", [])[:5]
    print(f"     MeSH: {', '.join(mesh)}")
print()
print("=" * 90)
print("RERANKER SEARCH (3072-dim precision reranking)")
print("=" * 90)
for i, point in enumerate(rerank_results, 1):
    print(f"  {i}. [Score: {point.score:.4f}] {point.payload['title'][:80]}")
    mesh = point.payload.get("mesh_terms", [])[:5]
    print(f"     MeSH: {', '.join(mesh)}")

# %% [markdown]
# Notice the differences. RRF merges results purely by rank position -- it
# doesn't know anything about the content. The reranker actually computes
# semantic similarity with a higher-fidelity embedding, so it can better
# distinguish between "close but not quite right" and "exactly what we need."
#
# The improvement is especially visible for multi-faceted queries like this
# one, where the question mentions both a specific gene (p53) and a specific
# cancer type (glioblastoma). The reranker can better weigh both aspects.

# %% [markdown]
# ## 8. RAG with Reranked Results

# %%
def rag_answer(question, results):
    """Generate an answer using retrieved papers as context."""
    context = "\n\n".join(
        [
            f"Paper {i+1} (PMID: {r.payload['pmid']}):\n"
            f"Title: {r.payload['title']}\n"
            f"Abstract: {r.payload['abstract'][:500]}"
            for i, r in enumerate(results)
        ]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical research assistant. Answer questions "
                    "using ONLY the provided paper context. Cite papers by PMID. "
                    "If the context doesn't contain enough information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# %%
rag_question = "What are the molecular mechanisms linking p53 mutations to treatment resistance in brain tumors?"
reranked = reranking_search(rag_question)

print(f"Question: {rag_question}\n")
print("Retrieved papers:")
for i, point in enumerate(reranked, 1):
    print(f"  {i}. PMID {point.payload['pmid']}: {point.payload['title'][:80]}")
print()

answer = rag_answer(rag_question, reranked)
print("Answer:")
print(answer)

# %% [markdown]
# ## 9. What's Next?
#
# Reranking gives us a precision boost by using a second pass with
# higher-fidelity embeddings. In production systems, you might also use a
# cross-encoder model (like a fine-tuned BERT) for even more precise
# reranking.
#
# But even with perfect retrieval, we still have a fundamental limitation:
# **we can only find papers that are similar to the query**. We can't find
# papers that are *connected* -- cited by, building on, or foundational to the
# papers we found.
#
# That's what the graph brings us in Stage 5.
#
# ```
# Stage 4 architecture:
#     Question --embed--> [Dense + BM25: fast retrieval] --candidates-->
#              [Reranker: 3072-dim precision] --top-k--> Papers --> [LLM] --> Answer
#
# Improvement: Higher precision from two-pass scoring
# Limitation: Still can't follow citation chains or find connected papers
#
# Next: Add graph structure to traverse relationships
# ```

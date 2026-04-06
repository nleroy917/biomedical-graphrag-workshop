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
# # Stage 3: Hybrid Search -- Combining Semantic + Keyword Precision
#
# In Stage 2, we built a working RAG pipeline with vector search. It grounded
# the LLM in real papers -- no more hallucinations. But we hit a wall:
# **semantic drift**. When we searched for "p53 mutations in glioblastoma,"
# we got papers about cancer genetics in general, not our specific topic.
#
# The fix? Add **keyword search** alongside vector search. In this notebook
# we'll:
# 1. Build a BM25 keyword index over our papers
# 2. Store both dense vectors and sparse BM25 vectors in Qdrant
# 3. Fuse the two signals with Reciprocal Rank Fusion (RRF)
# 4. Show how hybrid search fixes semantic drift
# 5. Discover what's *still* missing

# %% [markdown]
# ## 1. Setup + Load Data

# %%
import os, json, time
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import models

load_dotenv("../.env")
client = OpenAI()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

COLLECTION = "pubmed_hybrid"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

with open("../data/pubmed_dataset.json") as f:
    dataset = json.load(f)

# Filter papers with abstracts, take first 500
papers = [p for p in dataset["papers"] if p.get("abstract")][:500]
print(f"Loaded {len(papers)} papers with abstracts")

# %% [markdown]
# ## 2. What Is BM25?
#
# BM25 (Best Matching 25) is a classical keyword matching algorithm that's
# been the backbone of search engines for decades. Unlike vector search, it
# doesn't care about *meaning* -- if the document contains your exact words,
# BM25 finds it. It scores documents based on:
#
# - **Term Frequency (TF):** How often does the keyword appear in the document?
# - **Inverse Document Frequency (IDF):** How rare is the keyword across all documents?
# - **Document length normalization:** Longer documents don't get unfair advantages.
#
# By combining BM25 with vector search, we get both **semantic understanding**
# AND **keyword precision**. Vector search handles synonyms and paraphrasing;
# BM25 nails the exact terms. Together, they cover each other's blind spots.

# %% [markdown]
# ## 3. Build BM25 Index

# %%
import math, re
from collections import Counter


class BM25Vectorizer:
    """Simple BM25 sparse vector encoder for educational purposes."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        # Build vocabulary and IDF
        self.vocab = {}
        doc_freq = Counter()
        doc_lengths = []

        for doc in corpus:
            tokens = self._tokenize(doc)
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                doc_freq[token] += 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.avg_dl = sum(doc_lengths) / len(doc_lengths)
        n = len(corpus)
        self.idf = {
            token: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in doc_freq.items()
        }

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def encode(self, text):
        """Return (indices, values) for a sparse vector."""
        tokens = self._tokenize(text)
        dl = len(tokens)
        tf = Counter(tokens)
        indices, values = [], []
        for token, count in tf.items():
            if token in self.vocab and token in self.idf:
                tf_score = (count * (self.k1 + 1)) / (
                    count + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                score = self.idf[token] * tf_score
                indices.append(self.vocab[token])
                values.append(score)
        return indices, values


# %%
# Fit BM25 on all abstracts
corpus = [p["abstract"] for p in papers]
bm25 = BM25Vectorizer(corpus)
print(f"BM25 vocabulary size: {len(bm25.vocab):,} unique tokens")
print(f"Average document length: {bm25.avg_dl:.0f} tokens")

# %% [markdown]
# ## 4. Create Qdrant Collection with Dense + Sparse Vectors
#
# This time we configure the collection with **two** vector types: a dense
# vector (from the OpenAI embedding model) and a sparse BM25 vector. Qdrant
# stores and searches both, and we can fuse results at query time.

# %%
if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "dense": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
    },
)
print(f"Collection '{COLLECTION}' created (dense + sparse)")

# %% [markdown]
# ## 5. Ingest Papers with Both Vectors
#
# For each paper we compute two things: a dense embedding via OpenAI and a
# sparse BM25 vector from our tokenizer. Both get stored as named vectors
# on the same Qdrant point.

# %%
BATCH_SIZE = 100
start = time.time()

for i in range(0, len(papers), BATCH_SIZE):
    batch = papers[i : i + BATCH_SIZE]
    texts = [p["abstract"] for p in batch]

    # Batch embed with OpenAI
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    dense_vectors = [e.embedding for e in response.data]

    # Build points with both dense and sparse vectors
    points = []
    for paper, dense_vector in zip(batch, dense_vectors):
        indices, values = bm25.encode(paper["abstract"])
        points.append(
            models.PointStruct(
                id=int(paper["pmid"]),
                vector={
                    "dense": dense_vector,
                    "bm25": models.SparseVector(indices=indices, values=values),
                },
                payload={
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "authors": [a["name"] for a in paper.get("authors", [])],
                    "journal": paper.get("journal", ""),
                    "mesh_terms": [m["term"] for m in paper.get("mesh_terms", [])],
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"  Ingested {min(i + BATCH_SIZE, len(papers))}/{len(papers)} papers")

elapsed = time.time() - start
print(f"\nDone! {len(papers)} papers ingested in {elapsed:.1f}s")

# %% [markdown]
# ## 6. Hybrid Search: Dense + BM25 with Fusion
#
# Now we can search with both signals. Qdrant's **prefetch + fusion** lets us
# retrieve from each index independently, then combine them using Reciprocal
# Rank Fusion (RRF). RRF is elegantly simple: it ranks each result by
# `1 / (k + rank)` in each retriever, then sums the scores. Papers that rank
# well in *both* dense and BM25 float to the top.

# %%
def hybrid_search(question, top_k=5):
    """Hybrid search: dense vectors + BM25, fused by RRF."""
    # Dense query vector
    q_embedding = client.embeddings.create(
        model=EMBED_MODEL, input=question
    ).data[0].embedding

    # BM25 sparse query vector
    bm25_indices, bm25_values = bm25.encode(question)

    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=q_embedding, using="dense", limit=top_k * 3),
            models.Prefetch(
                query=models.SparseVector(indices=bm25_indices, values=bm25_values),
                using="bm25",
                limit=top_k * 3,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points

# %% [markdown]
# ## 7. Compare Dense-Only vs Hybrid
#
# Let's revisit the query from Stage 2 that tripped up pure vector search.
# We asked about **p53 mutations in glioblastoma** and got semantically
# related but imprecise results. Does adding BM25 fix it?

# %%
specific_question = "What papers discuss p53 tumor suppressor gene mutations specifically in glioblastoma multiforme?"

# We need the dense embedding for the dense-only comparison
q_embedding = client.embeddings.create(
    model=EMBED_MODEL, input=specific_question
).data[0].embedding

# Dense only (same approach as Stage 2)
dense_results = qdrant.query_points(
    collection_name=COLLECTION,
    query=q_embedding,
    using="dense",
    limit=5,
    with_payload=True,
).points

print("=== Dense Only (Stage 2 approach) ===\n")
for i, r in enumerate(dense_results, 1):
    print(f"{i}. {r.payload['title'][:80]}")
    print(f"   MeSH: {', '.join(r.payload.get('mesh_terms', [])[:4])}\n")

print("\n=== Hybrid: Dense + BM25 (Stage 3) ===\n")
hybrid_results = hybrid_search(specific_question)
for i, r in enumerate(hybrid_results, 1):
    print(f"{i}. {r.payload['title'][:80]}")
    print(f"   MeSH: {', '.join(r.payload.get('mesh_terms', [])[:4])}\n")

# %% [markdown]
# Notice the difference. The hybrid results should favor papers that
# actually contain the words "p53," "glioblastoma," and "mutations" -- not
# just papers that live in the same semantic neighborhood as cancer genetics.
# BM25 gives keyword precision; dense vectors give semantic coverage. RRF
# combines the best of both.

# %% [markdown]
# ## 8. RAG with Hybrid Retrieval

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
                    "using ONLY the provided paper context. Cite papers by PMID."
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
answer = rag_answer(specific_question, hybrid_results)
print(answer)

# %% [markdown]
# ## 9. The Gap
#
# Hybrid search gives us the best of both worlds: semantic understanding +
# keyword precision. But there's still something missing.
#
# Think about how a **real literature review** works. You don't just find
# similar papers -- you **chase citations**. You find the foundational papers
# that everyone in the field builds on. A paper might be semantically distant
# from your query (it's a general methods paper from 2012) but it's cited by
# *every single paper* you retrieved.
#
# **Qdrant tells you what's *similar*. But it can't tell you what's
# *connected*.**
#
# A vector database treats each paper as an isolated point in space. It has
# no concept of "Paper A cites Paper B" or "these five papers all build on
# the same foundational work." Citation networks, co-authorship graphs,
# shared MeSH term hierarchies -- all of this relational structure is
# invisible to a vector index.
#
# For that, we need a **graph database**.

# %% [markdown]
# ## 10. Architecture
#
# ```
# Stage 3 architecture:
#     Question --> [Qdrant: Dense + BM25 + RRF] --> Papers --> [LLM] --> Answer
#
# Improvement: Keyword precision eliminates semantic drift
#
# Next gap: Can't find foundational/connected papers (citation lineage)
# ```
#
# In the next notebook, we add a graph layer to capture the connections
# between papers -- and unlock a whole new class of queries.

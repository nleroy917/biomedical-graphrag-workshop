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
# # Stage 2: Vector Search + LLM (Retrieval-Augmented Generation)
#
# In Stage 1, we saw the LLM confidently invent citations that don't exist.
# The fix? **Give it real papers to reference.** That's Retrieval-Augmented
# Generation (RAG): retrieve first, then generate.
#
# In this notebook we'll:
# 1. Convert PubMed abstracts into vector embeddings
# 2. Store them in Qdrant (a vector database)
# 3. Retrieve relevant papers for a question
# 4. Feed those papers to the LLM as context
# 5. Discover the limits of pure vector search

# %% [markdown]
# ## 1. Setup + Load Data

# %%
import os, json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import models

load_dotenv("../.env")
client = OpenAI()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

COLLECTION = "pubmed_dense"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

with open("../data/pubmed_dataset.json") as f:
    dataset = json.load(f)

# Filter papers with abstracts, take first 500
papers = [p for p in dataset["papers"] if p.get("abstract")][:500]
print(f"Loaded {len(papers)} papers with abstracts")

# %% [markdown]
# ## 2. What Are Vector Embeddings?
#
# An embedding model converts text into a high-dimensional vector (a list of
# numbers) where **similar meanings are close together** in the vector space.
# OpenAI's `text-embedding-3-small` maps any text to a **1536-dimensional
# vector**. Two abstracts about the same disease will end up near each other;
# an abstract about oncology and one about baking will be far apart.
#
# This is the core idea behind semantic search: instead of matching keywords,
# we match *meaning*.

# %% [markdown]
# ## 3. Create Qdrant Collection

# %%
# Delete if exists, create fresh
if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(
        size=EMBED_DIM,
        distance=models.Distance.COSINE,
    ),
)
print(f"Collection '{COLLECTION}' created")

# %% [markdown]
# ## 4. Embed and Ingest Papers
#
# We'll send abstracts to the OpenAI embedding API in batches of 100, then
# upsert the resulting vectors (along with paper metadata) into Qdrant.

# %%
import time

BATCH_SIZE = 100
start = time.time()

for i in range(0, len(papers), BATCH_SIZE):
    batch = papers[i : i + BATCH_SIZE]
    texts = [p["abstract"] for p in batch]

    # Batch embed
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [e.embedding for e in response.data]

    # Create points
    points = [
        models.PointStruct(
            id=int(paper["pmid"]),
            vector=vector,
            payload={
                "pmid": paper["pmid"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": [a["name"] for a in paper.get("authors", [])],
                "journal": paper.get("journal", ""),
                "mesh_terms": [m["term"] for m in paper.get("mesh_terms", [])],
            },
        )
        for paper, vector in zip(batch, vectors)
    ]

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"  Ingested {min(i + BATCH_SIZE, len(papers))}/{len(papers)} papers")

elapsed = time.time() - start
print(f"\nDone! {len(papers)} papers ingested in {elapsed:.1f}s")

# %% [markdown]
# ## 5. Semantic Search
#
# Now let's search! We embed our question into the same vector space and find
# the nearest papers. Papers whose abstracts are semantically close to the
# question will have the highest cosine similarity scores.

# %%
def search_papers(question, top_k=5):
    """Embed the question and search Qdrant for similar papers."""
    q_embedding = client.embeddings.create(
        model=EMBED_MODEL, input=question
    ).data[0].embedding

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_embedding,
        limit=top_k,
        with_payload=True,
    )
    return results.points


# Search!
question = "What are the latest advances in using CRISPR-Cas9 for treating sickle cell disease?"
results = search_papers(question)

print(f"Query: {question}\n")
for i, point in enumerate(results, 1):
    print(f"{i}. [Score: {point.score:.4f}] {point.payload['title'][:90]}")
    print(f"   PMID: {point.payload['pmid']} | Journal: {point.payload['journal']}")
    print()

# %% [markdown]
# ## 6. RAG: Feed Context to the LLM
#
# Here's the magic. Instead of asking the LLM to answer from its training
# data (where it hallucinates), we **stuff the retrieved papers into the
# prompt** and instruct it to answer using *only* that context.

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


answer = rag_answer(question, results)
print(answer)

# %% [markdown]
# ## 7. RAG Works!
#
# Now every citation is **real** -- the LLM is grounded in actual papers from
# our database. No more hallucinated PMIDs. This is a massive improvement
# over vanilla prompting.
#
# But let's try a more specific, multi-faceted query and see what happens...

# %% [markdown]
# ## 8. The Semantic Drift Problem

# %%
specific_question = "What papers discuss p53 tumor suppressor gene mutations specifically in glioblastoma multiforme?"
results_specific = search_papers(specific_question)

print(f"Query: {specific_question}\n")
for i, point in enumerate(results_specific, 1):
    print(f"{i}. [Score: {point.score:.4f}] {point.payload['title'][:90]}")
    mesh = point.payload.get("mesh_terms", [])[:5]
    print(f"   MeSH: {', '.join(mesh)}")
    print()

# %% [markdown]
# ## 9. The Gap
#
# Look at those results. We asked specifically about **p53** in
# **glioblastoma**, but we're getting papers about other tumor types or other
# genes. The vector search finds papers that are *semantically similar*
# (about cancer + genetics) but misses our exact keywords.
#
# **The problem:** Vector search operates on *meaning*, not *words*. It can't
# distinguish between "glioblastoma" and "osteosarcoma" if they live in
# similar semantic neighborhoods. Both are brain/bone cancers discussed in
# genetics contexts, so their embeddings overlap.
#
# **What we need:** Keyword precision *combined* with semantic understanding.
# That's **hybrid search** -- and it's what we build in Stage 3.

# %% [markdown]
# ## Architecture
#
# ```
# Stage 2 architecture:
#     Question --embed--> [Qdrant: Dense Vectors] --top-k--> Papers --> [LLM] --> Answer
#
# Problem: Semantic drift -- related but wrong papers
#
# Next: Add keyword search (BM25) for precision
# ```

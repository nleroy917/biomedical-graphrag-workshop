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
# # Notebook 0: Environment Setup
#
# Welcome to the **Biomedical GraphRAG Workshop**!
#
# This notebook verifies that your environment is ready to go. Before we dive into
# building a graph-powered retrieval-augmented generation system over biomedical
# literature, we need three things in place:
#
# 1. **Docker** running with **Qdrant** (vector store) and **Neo4j** (graph database)
# 2. An **OpenAI API key** for embeddings and language model calls
# 3. The **PubMed dataset** downloaded into the `data/` directory
#
# Run each cell below. If everything prints success messages, you are all set.

# %% [markdown]
# ## 1. Install Dependencies
#
# We start by installing the Python packages used throughout the workshop.

# %%
%pip install openai qdrant-client neo4j python-dotenv -q

# %% [markdown]
# ## 2. Load Environment Variables
#
# We use a `.env` file in the project root to store configuration. The cell below
# loads it and prints a quick summary so you can confirm nothing is missing.

# %%
import os
from dotenv import load_dotenv
load_dotenv("../.env")

print("OpenAI API key:", "OK" if os.getenv("OPENAI_API_KEY") else "MISSING")
print("Qdrant URL:", os.getenv("QDRANT_URL", "http://localhost:6333"))
print("Neo4j URI:", os.getenv("NEO4J_URI", "bolt://localhost:7687"))

# %% [markdown]
# ## 3. Test OpenAI Connection
#
# A quick round-trip to the OpenAI API confirms your key is valid and the model is
# reachable.

# %%
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'hello' in one word."}],
    max_tokens=5,
)
print("OpenAI:", response.choices[0].message.content)

# %% [markdown]
# ## 4. Test Qdrant Connection
#
# Qdrant is the vector database we will use to store and search paper embeddings.
# Make sure the Qdrant Docker container is running before executing this cell.

# %%
from qdrant_client import QdrantClient
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
collections = qdrant.get_collections()
print(f"Qdrant: Connected. {len(collections.collections)} collections found.")

# %% [markdown]
# ## 5. Test Neo4j Connection
#
# Neo4j is the graph database where we will model papers, authors, genes, and their
# relationships. Confirm the container is running and credentials are correct.

# %%
from neo4j import GraphDatabase
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "workshop123")),
)
with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print("Neo4j:", result.single()["test"] == 1 and "Connected." or "Failed.")
driver.close()

# %% [markdown]
# ## 6. Preview Dataset
#
# Let's take a quick look at the PubMed papers and gene data we will be working with
# throughout the workshop.

# %%
import json

with open("../data/pubmed_dataset.json") as f:
    dataset = json.load(f)

papers = dataset["papers"]
citation_network = dataset["citation_network"]

print(f"Papers: {len(papers)}")
print(f"Citation network entries: {len(citation_network)}")
print(f"\nSample paper:")
print(f"  PMID: {papers[0]['pmid']}")
print(f"  Title: {papers[0]['title'][:80]}...")
print(f"  Authors: {len(papers[0]['authors'])}")
print(f"  MeSH terms: {len(papers[0]['mesh_terms'])}")

# %%
with open("../data/gene_dataset.json") as f:
    gene_data = json.load(f)

genes = gene_data["genes"]
print(f"Genes: {len(genes)}")
print(f"\nSample gene:")
print(f"  Name: {genes[0]['name']}")
print(f"  Description: {genes[0]['description'][:80]}...")
print(f"  Linked PMIDs: {len(genes[0]['linked_pmids'])}")

# %% [markdown]
# ---
#
# **All connections verified. You're ready for the workshop!**

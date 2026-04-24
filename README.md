# Literature Search with GraphRAG (Qdrant + Neo4j)

A hands-on workshop that builds a progressively more powerful biomedical literature search system — from a vanilla LLM to full GraphRAG.

## Setup

```bash
cp .env.example .env        # Add your OpenAI API key
docker compose up -d         # Start Qdrant + Neo4j
pip install -r requirements.txt
```

## Notebooks

| # | Stage | What breaks | What fixes it |
|---|---|---|---|
| 00 | Setup | — | Verify connections, preview data |
| 01 | Vanilla LLM | Hallucinated citations | — |
| 02 | Vector RAG | Semantic drift (wrong papers) | Dense vector search + Qdrant |
| 03 | Hybrid Search | Missing keyword precision | BM25 + dense + RRF fusion |
| 04 | GraphRAG | Can't find foundational/connected papers | Neo4j citation graph enrichment |

Run them in order. Each one exposes a limitation that the next one solves.

## Architecture (final)

```
Question → [Qdrant: Dense + BM25] → Retrieved Papers
                                          ↓
                                    [Neo4j: Citations, MeSH, Genes]
                                          ↓
                                    [LLM with full context] → Grounded Answer
```

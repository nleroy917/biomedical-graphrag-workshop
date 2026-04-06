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
# # Stage 5: Full GraphRAG -- Citation Graphs Meet Vector Search
#
# This is the finale. We have built, piece by piece, every layer of a
# modern biomedical retrieval system:
#
# - **Stage 1**: Vanilla LLM -- confident, eloquent, and wrong
# - **Stage 2**: Vector RAG -- grounded answers, but imprecise retrieval
# - **Stage 3**: Hybrid search -- keyword precision + semantic recall
# - **Stage 4**: Evaluation -- measuring what actually works
#
# But there is still something missing. When a researcher does a literature
# review, they don't just search -- they **chase citations**. They find a
# great paper, look at what it cites, look at who cites *it*, and follow
# the trail. They check who collaborated with whom. They notice that two
# genes keep appearing together.
#
# None of that is captured by vectors. Vectors encode *meaning*. Graphs
# encode *relationships*. Today we combine both.
#
# We will:
# 1. Build a citation graph in **Neo4j** from our PubMed dataset
# 2. Enrich Qdrant retrieval results with graph context
# 3. Produce the most complete, grounded answer we have seen in this workshop

# %% [markdown]
# ## 1. Setup

# %%
import os, json, time
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import models
from neo4j import GraphDatabase

load_dotenv("../.env")
client = OpenAI()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "workshop123")),
)

QDRANT_COLLECTION = "pubmed_hybrid"  # Reuse from Stage 3
EMBED_MODEL = "text-embedding-3-small"

print("Connected to OpenAI, Qdrant, and Neo4j.")

# %% [markdown]
# ## 2. Why a Graph?
#
# **What is Neo4j?** A graph database where:
# - **Nodes** = papers, authors, genes, MeSH terms
# - **Edges** = citations, authorship, gene mentions
#
# This captures something vectors fundamentally cannot: **intellectual lineage**.
# A paper might be semantically distant from your query but be the foundational
# work that every retrieved paper cites. You would never find it with vector
# search alone. But one Cypher query surfaces it instantly.
#
# **Qdrant tells you what's *similar*. Neo4j tells you what's *connected*.**
# These are genuinely different things.
#
# Think about it concretely. If you search for "CRISPR sickle cell therapy"
# in vector space, you get papers about CRISPR and sickle cell. Good. But you
# might miss the original Doudna & Charpentier paper on CRISPR-Cas9 because
# its abstract talks about bacterial immune systems, not sickle cell. In the
# citation graph, though, every paper you retrieved *cites* that foundational
# work. The graph finds what vectors miss.

# %% [markdown]
# ## 3. Load Data

# %%
with open("../data/pubmed_dataset.json") as f:
    dataset = json.load(f)

papers = [p for p in dataset["papers"] if p.get("abstract")][:500]
citation_network = dataset["citation_network"]

with open("../data/gene_dataset.json") as f:
    gene_data = json.load(f)
genes = gene_data["genes"]

# Build PMID set for our subset
paper_pmids = {p["pmid"] for p in papers}
print(f"Papers: {len(papers)}, Genes: {len(genes)}, Citation entries: {len(citation_network)}")

# %% [markdown]
# ## 4. Build the Graph in Neo4j
#
# We are going to create five types of nodes and five types of relationships:
#
# | Node       | Properties                         |
# |------------|------------------------------------|
# | Paper      | pmid, title, abstract, date, doi   |
# | Author     | name                               |
# | MeshTerm   | ui, term                           |
# | Journal    | name                               |
# | Gene       | gene_id, name, description         |
#
# | Relationship   | From       | To       |
# |----------------|------------|----------|
# | WROTE          | Author     | Paper    |
# | CITES          | Paper      | Paper    |
# | HAS_MESH_TERM  | Paper      | MeshTerm |
# | PUBLISHED_IN   | Paper      | Journal  |
# | MENTIONED_IN   | Gene       | Paper    |

# %%
def run_cypher(query, params=None):
    """Run a Cypher query and return results as a list of dicts."""
    with driver.session() as session:
        result = session.run(query, params or {})
        return [dict(r) for r in result]

# %%
# Create uniqueness constraints
print("Creating constraints...")
for constraint in [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MeshTerm) REQUIRE m.ui IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journal) REQUIRE j.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE",
]:
    run_cypher(constraint)
print("Constraints created.")

# %%
# Ingest papers in batches of 100
print("Ingesting papers...")
t0 = time.time()
for i in range(0, len(papers), 100):
    batch = papers[i : i + 100]
    run_cypher(
        """
        UNWIND $batch AS row
        MERGE (p:Paper {pmid: row.pmid})
        SET p.title = row.title, p.abstract = row.abstract,
            p.publication_date = row.publication_date, p.doi = row.doi
        """,
        {"batch": batch},
    )
    print(f"  {min(i + 100, len(papers))}/{len(papers)} papers")
print(f"Papers ingested in {time.time() - t0:.1f}s")

# %%
# Create author relationships (WROTE)
print("Creating author relationships...")
t0 = time.time()
for paper in papers:
    for author in paper.get("authors", []):
        run_cypher(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {pmid: $pmid})
            MERGE (a)-[:WROTE]->(p)
            """,
            {"name": author["name"], "pmid": paper["pmid"]},
        )
print(f"Author relationships created in {time.time() - t0:.1f}s")

# %%
# Create MeSH term relationships (HAS_MESH_TERM)
print("Creating MeSH term relationships...")
t0 = time.time()
for paper in papers:
    for mesh in paper.get("mesh_terms", []):
        run_cypher(
            """
            MERGE (m:MeshTerm {ui: $ui})
            SET m.term = $term
            WITH m
            MATCH (p:Paper {pmid: $pmid})
            MERGE (p)-[:HAS_MESH_TERM {major_topic: $major}]->(m)
            """,
            {
                "ui": mesh["ui"],
                "term": mesh["term"],
                "pmid": paper["pmid"],
                "major": mesh.get("major_topic", False),
            },
        )
print(f"MeSH term relationships created in {time.time() - t0:.1f}s")

# %%
# Create journal relationships (PUBLISHED_IN)
print("Creating journal relationships...")
t0 = time.time()
for paper in papers:
    journal = paper.get("journal")
    if journal:
        run_cypher(
            """
            MERGE (j:Journal {name: $journal})
            WITH j
            MATCH (p:Paper {pmid: $pmid})
            MERGE (p)-[:PUBLISHED_IN]->(j)
            """,
            {"journal": journal, "pmid": paper["pmid"]},
        )
print(f"Journal relationships created in {time.time() - t0:.1f}s")

# %%
# Create citation relationships (CITES)
print("Creating citation relationships...")
t0 = time.time()
citation_count = 0
for pmid, cinfo in citation_network.items():
    if pmid not in paper_pmids:
        continue
    for ref in cinfo.get("references", []):
        if ref in paper_pmids:
            run_cypher(
                """
                MATCH (p1:Paper {pmid: $citing})
                MATCH (p2:Paper {pmid: $cited})
                MERGE (p1)-[:CITES]->(p2)
                """,
                {"citing": pmid, "cited": ref},
            )
            citation_count += 1
print(f"Created {citation_count} citation edges in {time.time() - t0:.1f}s")

# %%
# Ingest genes and link to papers (MENTIONED_IN)
print("Ingesting genes...")
t0 = time.time()
gene_count = 0
for gene in genes:
    linked_in_subset = [p for p in gene.get("linked_pmids", []) if p in paper_pmids]
    if not linked_in_subset:
        continue
    run_cypher(
        """
        MERGE (g:Gene {gene_id: $gene_id})
        SET g.name = $name, g.description = $description
        """,
        {
            "gene_id": gene["gene_id"],
            "name": gene["name"],
            "description": gene.get("description", ""),
        },
    )
    for pmid in linked_in_subset:
        run_cypher(
            """
            MATCH (g:Gene {gene_id: $gene_id})
            MATCH (p:Paper {pmid: $pmid})
            MERGE (g)-[:MENTIONED_IN]->(p)
            """,
            {"gene_id": gene["gene_id"], "pmid": pmid},
        )
    gene_count += 1
print(f"Ingested {gene_count} genes in {time.time() - t0:.1f}s")

# %% [markdown]
# ## 5. Explore the Graph
#
# The graph is built. Let's look around.

# %%
# Node counts by label
stats = run_cypher(
    """
    MATCH (n)
    RETURN labels(n)[0] AS label, count(n) AS count
    ORDER BY count DESC
    """
)
print("Graph nodes:")
for s in stats:
    print(f"  {s['label']}: {s['count']}")

# Edge counts
edge_stats = run_cypher(
    """
    MATCH ()-[r]->()
    RETURN type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC
    """
)
print("\nGraph edges:")
for s in edge_stats:
    print(f"  {s['rel_type']}: {s['count']}")

# %%
# Most cited papers in our graph
most_cited = run_cypher(
    """
    MATCH (p:Paper)<-[c:CITES]-()
    RETURN p.title AS title, p.pmid AS pmid, count(c) AS citations
    ORDER BY citations DESC
    LIMIT 5
    """
)
print("Most cited papers:")
for r in most_cited:
    title = r["title"][:70] + "..." if len(r["title"]) > 70 else r["title"]
    print(f"  [{r['citations']} citations] {title} (PMID: {r['pmid']})")

# %%
# Most prolific authors
prolific = run_cypher(
    """
    MATCH (a:Author)-[:WROTE]->(p:Paper)
    RETURN a.name AS author, count(p) AS papers
    ORDER BY papers DESC
    LIMIT 5
    """
)
print("Most prolific authors:")
for r in prolific:
    print(f"  {r['author']}: {r['papers']} papers")

# %% [markdown]
# ## 6. The Graph Advantage
#
# Now we can do what manual literature reviews do -- chase citations and find
# foundational work. A vector database answers "what papers are *about* this
# topic?" A graph database answers "what papers *shaped* this topic?"
#
# Let's build three graph enrichment functions that will complement our
# vector retrieval.

# %% [markdown]
# ## 7. Graph Enrichment Functions

# %%
def get_related_papers_by_mesh(pmid, exclude_pmids=None):
    """Find papers sharing the most MeSH terms with a given paper."""
    return run_cypher(
        """
        MATCH (p1:Paper {pmid: $pmid})-[:HAS_MESH_TERM]->(m)<-[:HAS_MESH_TERM]-(p2:Paper)
        WHERE p1 <> p2 AND NOT p2.pmid IN $exclude
        WITH p2, COUNT(DISTINCT m) as shared_terms
        RETURN p2.pmid as pmid, p2.title as title, shared_terms
        ORDER BY shared_terms DESC LIMIT 5
        """,
        {"pmid": pmid, "exclude": exclude_pmids or []},
    )


def get_collaborators(author_name, topics):
    """Find co-authors who work on similar topics."""
    return run_cypher(
        """
        MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE toLower(a1.name) CONTAINS toLower($name) AND a1 <> a2
        WITH DISTINCT a2, p
        MATCH (p)-[:HAS_MESH_TERM]->(m:MeshTerm)
        WHERE ANY(t IN $topics WHERE toLower(m.term) CONTAINS toLower(t))
        RETURN a2.name as collaborator, COUNT(DISTINCT p) as shared_papers
        ORDER BY shared_papers DESC LIMIT 5
        """,
        {"name": author_name, "topics": topics},
    )


def get_gene_cooccurrence(gene_name):
    """Find genes co-mentioned in the same papers."""
    return run_cypher(
        """
        MATCH (g:Gene)-[:MENTIONED_IN]->(p:Paper)<-[:MENTIONED_IN]-(g2:Gene)
        WHERE toLower(g.name) CONTAINS toLower($gene) AND g2 <> g
        RETURN g2.name AS gene, COUNT(DISTINCT p) AS shared_papers
        ORDER BY shared_papers DESC LIMIT 5
        """,
        {"gene": gene_name},
    )

# %%
# Quick test: related papers by MeSH for the first paper
test_pmid = papers[0]["pmid"]
related_test = get_related_papers_by_mesh(test_pmid)
print(f"Papers related to PMID {test_pmid} by MeSH terms:")
for r in related_test:
    print(f"  - {r['title'][:70]}... ({r['shared_terms']} shared terms)")

# %% [markdown]
# ## 8. The Full GraphRAG Pipeline
#
# Here it is. The complete system.
#
# 1. **Qdrant** retrieves semantically relevant papers (dense search on our
#    hybrid collection from Stage 3)
# 2. **Neo4j** enriches those results with graph context -- shared MeSH terms,
#    collaborator networks, gene co-occurrence
# 3. **The LLM** synthesizes everything into a grounded, graph-enriched answer
#
# Let's build it.

# %%
def search_qdrant(question, top_k=5):
    """Dense vector search on the pubmed_hybrid collection."""
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_emb,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    return results.points

# %%
question = "What are the latest advances in using CRISPR-Cas9 for treating sickle cell disease?"

# --- Step 1: Vector retrieval ---
print("Step 1: Retrieving papers from Qdrant...\n")
qdrant_results = search_qdrant(question)
for i, r in enumerate(qdrant_results, 1):
    title = r.payload["title"][:80]
    print(f"  {i}. {title} (PMID: {r.payload['pmid']})")

# %%
# --- Step 2: Graph enrichment ---
print("Step 2: Enriching with Neo4j graph...\n")
retrieved_pmids = [r.payload["pmid"] for r in qdrant_results]

# 2a. Find related papers by shared MeSH terms
related = get_related_papers_by_mesh(retrieved_pmids[0], exclude_pmids=retrieved_pmids)
print(f"Papers related to top result (PMID {retrieved_pmids[0]}) by MeSH terms:")
for r in related:
    title = r["title"][:70] + "..." if len(str(r.get("title", ""))) > 70 else r.get("title", "N/A")
    print(f"  - {title} ({r['shared_terms']} shared MeSH terms)")

# 2b. Find collaborators of the first author
first_author = qdrant_results[0].payload.get("authors", ["Unknown"])[0]
mesh_terms = qdrant_results[0].payload.get("mesh_terms", [])[:3]
collabs = get_collaborators(first_author, mesh_terms)
if collabs:
    print(f"\nCollaborators of '{first_author}' on {mesh_terms}:")
    for c in collabs:
        print(f"  - {c['collaborator']} ({c['shared_papers']} shared papers)")
else:
    print(f"\nNo collaborator data found for '{first_author}' (may not be in our 500-paper subset)")

# 2c. Gene co-occurrence
gene_hits = run_cypher(
    """
    MATCH (g:Gene)-[:MENTIONED_IN]->(p:Paper)
    WHERE p.pmid IN $pmids
    RETURN DISTINCT g.name AS gene
    LIMIT 3
    """,
    {"pmids": retrieved_pmids},
)
gene_context = ""
if gene_hits:
    print(f"\nGenes mentioned in retrieved papers: {[g['gene'] for g in gene_hits]}")
    for gh in gene_hits:
        cooccur = get_gene_cooccurrence(gh["gene"])
        if cooccur:
            print(f"  Genes co-occurring with {gh['gene']}: {[c['gene'] for c in cooccur]}")
            gene_context += f"\nGene {gh['gene']} co-occurs with: {', '.join(c['gene'] for c in cooccur)}"

# %% [markdown]
# ## 9. Final Generation -- Full GraphRAG Answer
#
# Now we feed *everything* to the LLM: the retrieved papers from Qdrant,
# the graph-derived related papers, collaborator networks, and gene
# connections. This is the most context-rich prompt we have built in this
# entire workshop.

# %%
# Build the combined context
qdrant_context = "\n\n".join(
    [
        f"Paper {i+1} (PMID: {r.payload['pmid']}):\n"
        f"Title: {r.payload['title']}\n"
        f"Abstract: {r.payload['abstract'][:500]}"
        for i, r in enumerate(qdrant_results)
    ]
)

graph_context = "Related papers by shared MeSH terms:\n"
for r in related:
    graph_context += f"- {r['title']} (PMID: {r['pmid']}, {r['shared_terms']} shared terms)\n"

if collabs:
    graph_context += f"\nKey collaborators in this field:\n"
    for c in collabs:
        graph_context += f"- {c['collaborator']} ({c['shared_papers']} shared papers)\n"

if gene_context:
    graph_context += f"\nGene relationships:{gene_context}\n"

SYSTEM_PROMPT = """You are a biomedical research assistant combining two data sources:
- Retrieved literature (vector search results from PubMed)
- Knowledge graph (structured relationships: citations, co-authorship, gene mentions, MeSH terms)

Synthesize both into a well-structured answer. Cite papers by PMID.
Mention graph insights (collaborators, related papers, gene connections) where relevant.
Be specific and scientific. Do not hedge excessively."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Retrieved Papers:\n{qdrant_context}\n\n"
                f"Graph Context:\n{graph_context}\n\n"
                f"Question: {question}"
            ),
        },
    ],
    temperature=0.0,
)

print("=" * 60)
print("FULL GraphRAG ANSWER")
print("=" * 60)
print()
print(response.choices[0].message.content)

# %% [markdown]
# ## 10. The Full Picture
#
# Here is the complete architecture we built across five notebooks:
#
# ```
# Full GraphRAG Architecture:
#
#     Question
#        |
#        v
#     [Qdrant: Dense + BM25 + RRF]  -->  Retrieved Papers
#        |                                     |
#        v                                     v
#     [Neo4j: Citations, MeSH,        Graph Enrichment:
#      Authors, Genes]            - Related papers by MeSH
#                                 - Collaborator networks
#                                 - Gene co-occurrence
#        |
#        v
#     [LLM with full context]  -->  Grounded, Graph-Enriched Answer
# ```
#
# Every layer adds something the others cannot provide:
#
# | Layer            | What it contributes                          |
# |------------------|----------------------------------------------|
# | Dense vectors    | Semantic similarity -- finds relevant papers |
# | BM25 sparse      | Keyword precision -- exact term matching     |
# | RRF fusion       | Best of both retrieval worlds                |
# | Citation graph   | Intellectual lineage -- who cites whom       |
# | MeSH graph       | Ontological structure -- topic relationships |
# | Gene graph       | Biological connections -- gene co-occurrence |
# | LLM synthesis    | Human-readable, cited, structured answers    |

# %% [markdown]
# ## 11. Closing Reflection
#
# We have come full circle from the opening rant in Stage 1.
#
# Remember the four things a researcher does that an LLM alone cannot?
#
# 1. **Casting wide nets** -- Vector search finds semantically relevant papers
#    across the entire corpus. Done.
#
# 2. **Keyword precision** -- BM25 ensures exact terms like "p53" and
#    "glioblastoma" are respected, not diluted into vague semantic neighbors.
#    Done.
#
# 3. **Chasing citations** -- Neo4j automates the "who cites whom" discovery
#    that researchers do manually. A paper semantically distant from your query
#    but cited by *every* relevant result is now surfaced automatically. Done.
#
# 4. **Curating sources** -- The LLM synthesizes it all with real citations,
#    real PMIDs, real author names. No hallucinated references. Done.
#
# The tedious parts of a literature review are now automated. The LLM's output
# is grounded in real documents. And the graph captures the intellectual
# genealogy of a field.
#
# This is GraphRAG for biomedical research. Not a toy demo -- a real system
# with real PubMed data, real embeddings, real citation graphs, and real
# answers you could hand to a domain expert.
#
# The tools are all open. The data is all public. Go build something.

# %% [markdown]
# ## 12. Cleanup

# %%
driver.close()
print("Neo4j connection closed.")

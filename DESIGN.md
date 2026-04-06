# Boston Workshop: Literature Search with GraphRAG (Qdrant + Neo4j)
> A planning document for a 1-hour workshop. Generated from brainstorming session.

---

## Overview

A hands-on, teaching-style workshop that walks attendees through building a progressively more powerful literature search system — starting from a vanilla LLM all the way up to full GraphRAG using Qdrant and Neo4j. The theme is grounded in real research workflows (PubMed navigation) and the talk is structured to let the audience *discover* the need for each new component themselves.

**Duration:** ~1 hour  
**Location:** Boston  
**Vibe:** TA in office hours — conversational, teaching-first, not performative

---

## The Core Pedagogical Philosophy

### Bottom-Up Learning
Rather than presenting definitions and concepts top-down, the workshop builds knowledge from first principles. Each stage exposes a real flaw, which motivates the next solution. The audience is never told "here's why this is good" — they feel it themselves.

This is sometimes called the **Socratic method**: instead of delivering knowledge, you ask questions (or expose gaps) that guide the audience to conclusions themselves. The key mechanism is **aporia** — a productive state of confusion or incompleteness that makes the learner *ready* to receive the next idea.

> "You have to feel the gap before you can fill it."

**Influenced by:** 3Blue1Brown, VSauce, Veritasium — all of whom open with a question or surprising observation rather than a definition. The "lecture" only makes sense in motion.

### The Loop at Each Stage
Every transition follows this structure:

```
Demo → audience spots the flaw → you name it → here's the fix
```

Make the failure **visible on screen**. Don't describe it — show it. Let the audience react before you say anything. The silence does more work than any explanation.

### Note on Slide Design
Bottom-up slides will look disjointed when read cold — they're designed to be *performed*, not read. They are the skeleton; the room is the meat. This is a feature, not a bug.

---

## The "Rant" — Workshop Opening

Start with a relatable monologue about how literature reviews actually work:

1. Broad Google search — cast a wide net
2. Google Scholar — same search, start reading abstracts
3. Boolean search — `"scATAC-seq" AND "foundation models" AND "embeddings"`
4. Read papers → **aggressively chase citations** → find the usual suspects

This process is tedious: manual parsing, clicking, curating Zotero, etc.

**Thesis:** Can we replicate this entire workflow using modern AI tooling — agentic workflows, hybrid search, and graph databases?

---

## The 5-Stage Progression

> Cut stage 4 (reranking) if time is tight. The jump from hybrid search → graph enrichment is the most dramatic payoff.

### Stage 1: Vanilla LLM
**The demo:** Ask the LLM a research question. It responds confidently with papers and citations.  
**The gap the audience spots:** "Did that paper even exist?"  
**Transition:** We need to ground the LLM in real documents.

**Architecture:** LLM only. No retrieval.

---

### Stage 2: Vector Search + LLM (RAG)
**Concept to introduce (slides):** How do we quickly find relevant articles? Vector search — embed documents and queries into a shared semantic space, retrieve nearest neighbors.

**The demo:** Load Qdrant with PubMed data. Query it. Inject results into LLM context.  
**The gap the audience spots:** "Why is it talking about osteosarcoma? I asked about glioblastoma. And I specifically want papers about p53."  
**Transition:** Semantic similarity isn't enough. We need keyword precision.

**Architecture:** Qdrant (vectors) → LLM

---

### Stage 3: Hybrid Search (Vectors + BM25) + LLM
**Concept to introduce (slides):** BM25 is a long-standing keyword matching algorithm. It doesn't operate on semantics — if the document contains your exact words, BM25 finds it. Combine with RRF (Reciprocal Rank Fusion) to merge vector and keyword results.

**The demo:** Reload Qdrant collection with BM25 enabled. Show improved retrieval specificity.  
**The gap the audience spots:** "Ok, we have the right papers now — but how do I find the foundational stuff that everyone else builds on?"  
**Transition:** Citation graphs capture intellectual lineage. That's what Neo4j is for.

**Architecture:** Qdrant (vectors + BM25 + RRF) → LLM

---

### Stage 4 *(Optional — cut if short on time)*: Reranking
**Concept:** After retrieval, use a cross-encoder reranker to re-score results for relevance. More expensive but more precise.

**Architecture:** Qdrant (vectors + BM25 + RRF) → Reranker → LLM

---

### Stage 5: Full GraphRAG (+ Neo4j Citation Enrichment)
**Concept to introduce (slides):** What is Neo4j? A graph database. In our PubMed graph:
- **Nodes** = papers
- **Edges** = citations: `[Paper A] —cites→ [Paper B]`

This captures something vectors fundamentally cannot: **intellectual lineage**. A paper might be semantically distant from your query but be the foundational work that every retrieved paper cites. Qdrant tells you what's *similar*. Neo4j tells you what's *connected*. These are genuinely different things.

Visual example for slides:
- *"Attention is All You Need"* — many arrows pointing TO it (everyone cites it)
- *"Atacformer: …"* — only one arrow (nobody cites my work 😅) ← joke for the room

**The enrichment step:** After Qdrant retrieval, query Neo4j to pull in citations and related papers. This is the "chasing citations" step from the rant — now automated.

**The demo:**
1. Qdrant is already loaded from stage 3 — no changes needed
2. Load Neo4j, run test queries showing citation traversal and MeSH term enrichment
3. Run full generation with enriched context

**Architecture:** Qdrant (vectors + BM25 + RRF) → Neo4j enrichment (citations + MeSH) → LLM

---

## Infrastructure

- **Qdrant** — vector database with hybrid search support
- **Neo4j** — graph database for citation graph
- **Docker Compose** — run Qdrant + Neo4j in tandem
- **5 Jupyter Notebooks** — one per stage, kept clean and self-contained

### Reliability Note
Five notebooks + live Docker containers in an hour is ambitious. Have **pre-run outputs cached** so you can narrate gracefully through any container failures. Don't let a Docker issue derail the teaching moment.

---

## Slide ↔ Code Cadence

Each slide transition should leave the audience with a **single concrete question** that the code then answers.

| Slides Purpose | Code Purpose |
|---|---|
| Motivate the problem | Show the failure |
| Explain the concept | Show the fix |
| Raise the next question | Demo the next stage |

Rough flow:
```
Rant (slides) →
Vanilla LLM concept (slides) → demo + hallucination (code) →
Vector search concept (slides) → Qdrant RAG (code) →
BM25 concept (slides) → hybrid search (code) →
[optional: reranking (slides + code)] →
Graph concept + Neo4j (slides) → GraphRAG (code) →
Closing reflection (slides)
```

---

## Closing

Bring it back to the rant. Reflect on what changes when you have this system:
- The tedious parts of a literature review (casting wide nets, chasing citations, curating sources) are now automated
- The LLM's output is grounded in real, retrieved documents
- The graph captures the intellectual genealogy of a field

Don't end on the technical demo. End on the *so what*.

---

## Key Conceptual Distinctions to Hammer

1. **Vectors vs. BM25:** Semantic similarity vs. keyword precision. You need both.
2. **Qdrant vs. Neo4j:** Similarity vs. connectivity. Fundamentally different retrieval signals.
3. **Hallucination vs. grounding:** The LLM isn't wrong because it's dumb — it's wrong because it has no tether to reality. RAG is the tether.
4. **Bottom-up teaching:** The audience should feel each limitation before you name it. The awkward pause is your friend.

---

## On the Socratic Method (Meta-note for the presenter)

The Socratic method has two modes:

| Mode | Goal | Requires expertise? |
|---|---|---|
| **Destabilizing** | Expose that someone doesn't know as much as they think | No — just persistence. "But why?" is infinitely reusable. |
| **Teaching** | Guide someone toward a specific insight | Yes — you need to know the destination |

This workshop is the **teaching mode**. The questions feel natural and inevitable to the audience because *you already know where you're going*. The 5-stage progression is the path. Each "gap" is a carefully placed question.

Your PhD seminar (read a paper front to back, professor keeps pushing) was the destabilizing mode — less about a destination, more about building the habit of productive discomfort with your own assumptions.

Both are valuable. This workshop uses the former in service of the latter.
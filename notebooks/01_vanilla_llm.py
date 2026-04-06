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
# # Stage 1: Vanilla LLM -- The Hallucination Problem
#
# Let's start with the simplest possible approach: ask an LLM a biomedical
# research question and see what happens.

# %% [markdown]
# ## Setup

# %%
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("../.env")
client = OpenAI()

# %% [markdown]
# ## Ask the LLM a research question
#
# Imagine you're a researcher looking for recent work on CRISPR-Cas9
# applications in treating sickle cell disease. Sounds reasonable -- let's
# just ask GPT directly.

# %%
QUESTION = (
    "What are the most important recent papers on using CRISPR-Cas9 "
    "for treating sickle cell disease? Please cite specific papers with "
    "authors and publication years."
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": QUESTION}],
    temperature=0.0,
)

print(response.choices[0].message.content)

# %% [markdown]
# ## Looks great... right?
#
# That looks convincing! Specific authors, dates, journal names. The LLM
# sounds like it *really* knows its stuff.
#
# But let's pause. Do any of these papers actually exist?

# %% [markdown]
# ## The verification problem
#
# We could try to check each reference against PubMed. But the deeper
# insight is this: **the LLM has no tether to reality.** It doesn't look
# anything up. It doesn't have a bibliography. It generates
# *plausible-sounding* text based on statistical patterns -- and
# plausible-sounding citations are still fabricated citations.
#
# This is **hallucination**: confident, detailed, and wrong.
#
# The model isn't lying on purpose. It literally cannot distinguish between
# a real paper it saw during training and a pattern-matched chimera of
# author names, topics, and dates that *feels* right.

# %% [markdown]
# ## The gap
#
# The LLM isn't wrong because it's dumb -- it's wrong because it has
# **no access to real documents**. It's generating statistically plausible
# text, not retrieving facts.
#
# **What we need:** a way to ground the LLM's responses in actual,
# retrievable PubMed papers. That's what we'll build next with vector
# search.

# %% [markdown]
# ## Where we're headed
#
# ```
# Current architecture (this notebook):
#
#     Question --> [LLM] --> Answer (with hallucinated citations)
#
# What we need:
#
#     Question --> [Retrieval] --> Real Documents --> [LLM] --> Grounded Answer
# ```
#
# In the next notebook, we'll build that retrieval layer.

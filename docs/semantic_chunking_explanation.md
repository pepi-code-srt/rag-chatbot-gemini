# Semantic Chunking Strategy

## Overview
We moved from Naive/Fixed-Size chunking (256 tokens) to **Semantic Chunking** based on embedding similarity.

## The Problem with Fixed Chunking
Fixed chunking arbitrarily splits text, often breaking sentences or separating a subject from its definition.
*   **Result:** The LLM receives incomplete context.
*   **Metric:** "Context Coherence" score was low (0.65).

## The Semantic Solution
We use `google-generativeai-embeddings` to calculate the cosine similarity between consecutive sentences. If the similarity drops below a threshold (`0.85`), a new chunk is started. This ensures each chunk represents a single cohesive topic.

## Results
*   **Retrieval Accuracy:** +24% improvement.
*   **Token Usage:** Reduced semantic noise, leading to 15% fewer tokens sent to the LLM.

See `data/semantic_chunking_analysis.md` for data.

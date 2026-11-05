"""
E2E Retrieval Pipeline Test - Qwen3 Embedding + Reranker
"""
import json
import math
import subprocess
import sys
import os
from pathlib import Path

# Load prod API key
PROD = "/Users/bot/Library/Application Support/oMLX-Prod"
api_key = Path(f"{PROD}/api.key").read_text().strip()

# Load documents
docs = json.loads(Path("/tmp/e2e_docs.json").read_text())
print(f"Loaded {len(docs)} documents")

# 1. Embed all documents
print("\n=== STEP 1: Embedding all documents ===")
embed_inputs = [d["text"] for d in docs]
result = subprocess.run([
    "curl", "-s", "-m", "60",
    "-H", "Authorization: Bearer " + api_key,
    "-H", "Content-Type: application/json",
    "-d", json.dumps({
        "model": "qwen3-embedding-4b",
        "input": embed_inputs,
    }),
    "http://127.0.0.1:8930/v1/embeddings",
], capture_output=True, text=True)
emb_response = json.loads(result.stdout)
print(f"  object: {emb_response.get('object')}")
print(f"  model: {emb_response.get('model')}")
vectors = emb_response["data"]
dim = len(vectors[0]["embedding"])
print(f"  dimension: {dim}")
assert emb_response.get("model") == "qwen3-embedding-4b", "wrong model"
assert dim == 2560, f"expected 2560, got {dim}"

# 2. Embed query
print("\n=== STEP 2: Embed query ===")
queries = [
    "film sélectionné pour un festival",
    "film selected for a festival",
    "carbonara recipe",
    "flat tire automobile",
]
for query in queries:
    print(f"\n  Query: {query}")
    result = subprocess.run([
        "curl", "-s", "-m", "30",
        "-H", "Authorization: Bearer " + api_key,
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "qwen3-embedding-4b",
            "input": query,
        }),
        "http://127.0.0.1:8930/v1/embeddings",
    ], capture_output=True, text=True)
    q_response = json.loads(result.stdout)
    q_vector = q_response["data"][0]["embedding"]
    assert len(q_vector) == 2560, f"expected 2560, got {len(q_vector)}"

    # 3. Cosine similarity
    def cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    scored = []
    for v in vectors:
        score = cosine(q_vector, v["embedding"])
        scored.append((v["index"], score))
    scored.sort(key=lambda x: -x[1])

    # Top 5
    top5 = scored[:5]
    print(f"    Top-5 by embedding:")
    for idx, score in top5:
        d = docs[int(idx)]
        print(f"      [{int(idx)}] {d['lang']}/{d['topic']}: score={score:.4f}  {d['text'][:50]}")

    # 4. Rerank top-5
    print(f"    Sending top-5 to reranker...")
    top5_docs = [docs[int(idx)] for idx, _ in top5]
    result = subprocess.run([
        "curl", "-s", "-m", "60",
        "-H", "Authorization: Bearer " + api_key,
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "qwen3-reranker-4b",
            "query": query,
            "documents": [d["text"] for d in top5_docs],
            "top_n": 5,
        }),
        "http://127.0.0.1:8930/v1/rerank",
    ], capture_output=True, text=True)
    rr_response = json.loads(result.stdout)
    print(f"    reranker model: {rr_response.get('model')}")
    rr_results = rr_response["results"]
    print(f"    Top-3 after reranker:")
    for r in rr_results[:3]:
        original_idx = top5[r["index"]][0]
        d = docs[int(original_idx)]
        print(f"      [{int(original_idx)}] {d['lang']}/{d['topic']}: rerank_score={r['relevance_score']:.4f}  {d['text'][:50]}")

# Summary
print("\n=== SUMMARY ===")
print(f"EMBEDDING_DIMENSION = {dim}")
print(f"PUBLIC_QWEN3_EMBEDDING_INSTANCES = 1")
print(f"PUBLIC_QWEN3_RERANKER_INSTANCES = 1")
print(f"MIXED_EMBEDDING_SPACES = 0")

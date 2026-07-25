import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.bm25_retrieval import bm25_search
from src.retrieval.vector_retrieval import vector_search
from src.retrieval.reranker import rerank

# ==========================================================
# Hybrid Retrieval using RRF (Reciprocal Rank Fusion)
# ==========================================================

def hybrid_search(question, top_k=5, k_rrf=60):
    
    bm25_results = bm25_search(question, top_k=top_k * 2)
    vector_results = vector_search(question, top_k=top_k * 2)

    fused_scores = {}

    # ---------------- 1. BM25 Processing ----------------
    for rank, (score, chunk) in enumerate(bm25_results, start=1):
        key = (chunk["source"], chunk["chunk_number"])
        rrf_contribution = 1.0 / (k_rrf + rank)

        if key not in fused_scores:
            fused_scores[key] = {
                "source": chunk["source"],
                "chunk_number": chunk["chunk_number"],
                "text": chunk["text"],
                "bm25_score": score,
                "vector_distance": None,
                "rrf_score": rrf_contribution
            }
        else:
            fused_scores[key]["bm25_score"] = score
            fused_scores[key]["rrf_score"] += rrf_contribution

    # ---------------- 2. Vector Processing ----------------
    for rank, result in enumerate(vector_results, start=1):
        meta = result["metadata"]
        key = (meta["source"], meta["chunk_number"])
        rrf_contribution = 1.0 / (k_rrf + rank)

        if key not in fused_scores:
            fused_scores[key] = {
                "source": meta["source"],
                "chunk_number": meta["chunk_number"],
                "text": result["text"],
                "bm25_score": None,
                "vector_distance": result["distance"],
                "rrf_score": rrf_contribution
            }
        else:
            
            fused_scores[key]["vector_distance"] = result["distance"]
            fused_scores[key]["rrf_score"] += rrf_contribution

# ---------------- 3. Light Re-ranking ----------------

    for item in fused_scores.values():

        text = item["text"].lower()

        bonus = 0.0

        # Prefer introductory sections
        if "# introduction" in text:
            bonus += 0.03

        elif "## introduction" in text:
            bonus += 0.025

        # Prefer definition sentences
        if " is " in text[:150]:
            bonus += 0.015

        # Prefer documentation overview sections
        if "what" in text and "offers" in text:
            bonus += 0.01

        # Slightly penalize code-only chunks
        if text.count("```") >= 2:
            bonus -= 0.01

        # Slightly penalize installation sections
        if "pip install" in text:
            bonus -= 0.01

        item["rrf_score"] += bonus


# ---------------- 4. Final Sorting ----------------

    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    # Keep more candidates before reranking
    candidate_results = sorted_results[:10]

    # Cross-Encoder reranking
    reranked_results = rerank(question, candidate_results)

    return reranked_results[:top_k]

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    question = input("\nQuestion : ")

    results = hybrid_search(question, top_k=5)

    print("\n========== HYBRID RESULTS ==========\n")

    for i, r in enumerate(results, start=1):

        print(f"Rank             : {i}")
        print(f"Source           : {r['source']}")
        print(f"Chunk            : {r['chunk_number']}")
        print(f"Cross Score      : {r['rerank_score']:.4f}")
        print(f"RRF Score        : {r['rrf_score']:.6f}")

        if r["bm25_score"] is not None:
            print(f"BM25 Score       : {r['bm25_score']:.4f}")

        if r["vector_distance"] is not None:
            print(f"Vector Distance  : {r['vector_distance']:.4f}")

        print("-" * 70)
        print(r["text"][:400])
        print("=" * 70)
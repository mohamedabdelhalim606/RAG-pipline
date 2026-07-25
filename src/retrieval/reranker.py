from sentence_transformers import CrossEncoder

# Load once
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question, results):

    pairs = [
        (question, item["text"])
        for item in results
    ]

    scores = reranker.predict(pairs)

    for item, score in zip(results, scores):
        item["rerank_score"] = float(score)

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results
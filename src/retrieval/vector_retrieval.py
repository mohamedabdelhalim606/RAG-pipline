from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = PROJECT_ROOT / "Data" / "chroma_db"

# ==========================================================
# Load Embedding Model
# ==========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")
print("\nEmbedding Model Loaded")

# ==========================================================
# Connect To ChromaDB
# ==========================================================

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection("rag_collection")
print("Connected To ChromaDB")

# ==========================================================
# Vector Search
# ==========================================================

def vector_search(question, top_k=5):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    output = []

    # التحقق الآمن من وجود نتائج لمنع الـ IndexError
    if results and results.get("documents") and results["documents"][0]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

        for i in range(len(documents)):
            output.append({
                "distance": distances[i],
                "text": documents[i],
                "metadata": metadatas[i]
            })

    return output

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":
    question = input("\nQuestion : ")
    results = vector_search(question)

    print("\n========== VECTOR RESULTS ==========\n")

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"Rank      : {i}")
        print(f"Distance  : {result['distance']:.4f}")
        print(f"Source    : {metadata.get('source', 'N/A')}")
        print(f"Chunk     : {metadata.get('chunk_number', 'N/A')}")
        print("-" * 70)
        print(result["text"][:500])
        print("=" * 70)
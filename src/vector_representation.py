import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "Data" / "processed" / "chunks.json"

OUTPUT_FOLDER = PROJECT_ROOT / "Data" / "embeddings"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "embeddings.pkl"

# ==========================================================
# Load Chunks
# ==========================================================

print("\n========== LOADING CHUNKS ==========\n")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded Chunks : {len(chunks)}")

# ==========================================================
# Load Embedding Model
# ==========================================================

print("\n========== LOADING MODEL ==========\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully")

# ==========================================================
# Generate Embeddings
# ==========================================================

print("\n========== GENERATING EMBEDDINGS ==========\n")

embedded_chunks = []

for chunk in chunks:

    embedding = model.encode(
        chunk["text"],
        convert_to_numpy=True
    )

    embedded_chunks.append({

        "id": chunk["id"],

        "source": chunk["source"],

        "type": chunk["type"],

        "chunk_number": chunk["chunk_number"],

        "characters": chunk["characters"],

        "text": chunk["text"],

        "embedding": embedding.tolist()

    })

print(f"Generated Embeddings : {len(embedded_chunks)}")

# ==========================================================
# Save
# ==========================================================

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embedded_chunks, f)

print("\n===========================================")
print(f"Saved : {OUTPUT_FILE}")
print("===========================================")

# ==========================================================
# Sample
# ==========================================================

print("\nFirst Embedding\n")

print("Source :", embedded_chunks[0]["source"])

print("Vector Length :", len(embedded_chunks[0]["embedding"]))

print("First 10 Values :")

print(embedded_chunks[0]["embedding"][:10])
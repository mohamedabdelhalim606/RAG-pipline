import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = PROJECT_ROOT / "Data" / "embeddings" / "embeddings.pkl"

CHROMA_PATH = PROJECT_ROOT / "Data" / "chroma_db"

# ==========================================================
# Load Embeddings
# ==========================================================

print("\n========== LOADING EMBEDDINGS ==========\n")

with open(EMBEDDINGS_FILE, "rb") as f:
    documents = pickle.load(f)

print(f"Loaded : {len(documents)}")

# ==========================================================
# Create Chroma Client
# ==========================================================

print("\n========== CREATING DATABASE ==========\n")

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# لو كانت موجودة من تشغيل سابق احذفها
try:
    client.delete_collection("rag_collection")
except:
    pass

# إضافة التعديل هنا: استخدام cosine similarity بدلاً من L2
collection = client.create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================================
# Insert Data
# ==========================================================

print("\n========== INSERTING ==========\n")

for doc in documents:

    collection.add(

        ids=[str(doc["id"])],

        documents=[doc["text"]],

        embeddings=[doc["embedding"]],

        metadatas=[

            {

                "source": doc["source"],

                "type": doc["type"],

                "chunk_number": doc["chunk_number"],

                "characters": doc["characters"]

            }

        ]

    )

print(f"Stored : {collection.count()} chunks")

# ==========================================================
# Finish
# ==========================================================

print("\n======================================")
print("Vector Store Created Successfully")
print("======================================")
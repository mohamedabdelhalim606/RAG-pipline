import re
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

# ==========================================================
# Stop Words
# ==========================================================

STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "of","to","in","on","at","for","with","about","into","through",
    "during","before","after","above","below","from","up","down",
    "out","off","over","under","again","further","then","once",
    "here","there","when","where","why","how","all","any","both",
    "each","few","more","most","other","some","such","no","nor",
    "not","only","own","same","so","than","too","very","can","will",
    "just","should","now","what","which","who","whom","this","that",
    "these","those","do","does","did","doing","have","has","had"
}

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_FILE = PROJECT_ROOT / "Data" / "processed" / "chunks.json"

# ==========================================================
# Preprocessing
# ==========================================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    filtered_tokens = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    # حماية من الأسئلة التي قد تحذف كلماتها بالكامل بسبب الـ Stop Words
    return filtered_tokens if filtered_tokens else tokens

# ==========================================================
# Load Chunks
# ==========================================================

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"لم يتم العثور على ملف القطع في المسار: {CHUNKS_FILE}")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\nLoaded Chunks : {len(chunks)}")

# ==========================================================
# Build BM25 Index
# ==========================================================

corpus = [chunk["text"] for chunk in chunks]
tokenized_corpus = [preprocess(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

print("✅ BM25 Index Created")

# ==========================================================
# BM25 Search Function
# ==========================================================

def bm25_search(question, top_k=5):
    query = preprocess(question)
    
    if not query:
        return []

    scores = bm25.get_scores(query)

    # ترتيب النتائج وتصفية القطع ذات التقييم الصفري (الغير مرتبطة إطلاقاً)
    results = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    filtered_results = [(score, chunk) for score, chunk in results if score > 0.0]
    
    # إذا لم تتبق نتائج بعد الفلترة نرجع أفضل النتائج كخيار أخير
    return filtered_results[:top_k] if filtered_results else results[:top_k]

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":
    question = input("\nQuestion : ")
    results = bm25_search(question)

    print("\n========== BM25 RESULTS ==========\n")

    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"Rank      : {rank}")
        print(f"Score     : {score:.4f}")
        print(f"Source    : {chunk.get('source', 'N/A')}")
        print(f"Chunk     : {chunk.get('chunk_number', 'N/A')}")
        print("-" * 70)
        print(chunk["text"][:500])
        print("=" * 70)
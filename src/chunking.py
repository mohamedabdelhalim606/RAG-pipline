import sys
import json
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

# ==========================================================
# Project Path
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ==========================================================
# Paths
# ==========================================================

CLEAN_FOLDER = PROJECT_ROOT / "Data" / "cleaned"

OUTPUT_FOLDER = PROJECT_ROOT / "Data" / "processed"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "chunks.json"

# ==========================================================
# Splitters
# ==========================================================

headers = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers,
    strip_headers=False,
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

# ==========================================================
# Chunking
# ==========================================================

chunks = []
chunk_id = 1

print("\n========== CHUNKING ==========\n")

for file in CLEAN_FOLDER.glob("*"):

    print(f"Processing : {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    # ---------- Markdown Split ----------

    docs = header_splitter.split_text(text)

    document_chunks = []

    for doc in docs:

        document_chunks.extend(
            recursive_splitter.split_text(doc.page_content)
        )

    print(f"Chunks : {len(document_chunks)}")

    for number, chunk in enumerate(document_chunks, start=1):

        chunks.append(
            {
                "id": chunk_id,
                "source": file.stem,
                "type": "documentation",
                "chunk_number": number,
                "characters": len(chunk),
                "text": chunk,
            }
        )

        chunk_id += 1

# ==========================================================
# Research Paper
# ==========================================================

print("\n========== RESEARCH PAPER ==========\n")

paper = PROJECT_ROOT / "Data" / "papers" / "2005.11401v4.pdf"

reader = PdfReader(paper)

paper_text = ""

for page in reader.pages:

    txt = page.extract_text()

    if txt:
        paper_text += txt + "\n"

paper_chunks = recursive_splitter.split_text(paper_text)

print(f"Chunks : {len(paper_chunks)}")

for number, chunk in enumerate(paper_chunks, start=1):

    chunks.append(
        {
            "id": chunk_id,
            "source": "RAG Research Paper",
            "type": "paper",
            "chunk_number": number,
            "characters": len(chunk),
            "text": chunk,
        }
    )

    chunk_id += 1

# ==========================================================
# Save
# ==========================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print("\n=======================================")
print(f"Total Chunks : {len(chunks)}")
print(f"Saved To     : {OUTPUT_FILE}")
print("=======================================\n")

print("First Chunk\n")
print(chunks[0])

print("\nLast Chunk\n")
print(chunks[-1])
import sys
from pathlib import Path

from pypdf import PdfReader

# ==========================================================
# Project Path
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Data.sources import DOCUMENTS, RESEARCH_PAPER

# ==========================================================
# Loader
# ==========================================================

class SourceLoader:

    def load_markdown(self):

        documents = []

        print("\n========== KNOWLEDGE BASE ==========\n")

        for file in DOCUMENTS:

            print(f"Loading : {file.name}")

            with open(file, "r", encoding="utf-8") as f:

                text = f.read()

            documents.append({

                "source": file.stem,

                "type": "documentation",

                "content": text

            })

            print(f"✅ {len(text)} characters\n")

        return documents

    # ======================================================

    def load_research_paper(self):

        print("\n========== RESEARCH PAPER ==========\n")

        reader = PdfReader(RESEARCH_PAPER)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                pages.append(text)

        paper_text = "\n".join(pages)

        print(f"✅ Loaded ({len(paper_text)} characters)\n")

        return {

            "source": "RAG Research Paper",

            "type": "paper",

            "content": paper_text

        }


# ==========================================================

if __name__ == "__main__":

    loader = SourceLoader()

    docs = loader.load_markdown()

    paper = loader.load_research_paper()

    documents = docs + [paper]

    print("=" * 60)

    print(f"Total Documents : {len(documents)}")

    print("=" * 60)

    print()

    for doc in documents:

        print(f"Source : {doc['source']}")

        print(f"Type   : {doc['type']}")

        print(f"Length : {len(doc['content'])}")

        print("-" * 60)

    print("\n========== SAMPLE ==========\n")

    print(documents[0]["content"][:1000])
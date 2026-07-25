import sys
from pathlib import Path

# ==========================================================
# Project Path
# ==========================================================

FILE_PATH = Path(__file__).resolve()

PROJECT_ROOT = FILE_PATH.parent.parent.parent

sys.path.append(str(PROJECT_ROOT))

# ==========================================================
# Import Hybrid Retrieval
# ==========================================================

from src.retrieval.hybrid_retrieval import hybrid_search

# ==========================================================
# Context Retrieval
# ==========================================================

def retrieve_context(question: str, top_k: int = 5):
    """
    Retrieve the most relevant chunks and
    build a formatted context for the LLM.

    Returns
    -------
    context : str
        Formatted context.

    results : list
        Raw retrieval results.
    """

    results = hybrid_search(question, top_k=top_k)

    if not results:

        return "No relevant information was found.", []

    context_blocks = []

    for item in results:

        block = f"""
Source : {item["source"]}
Chunk  : {item["chunk_number"]}

{item["text"]}
"""

        context_blocks.append(block.strip())

    separator = "\n" + ("-" * 80) + "\n"

    context = separator.join(context_blocks)

    return context, results


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    question = input("\nQuestion : ")

    context, results = retrieve_context(question)

    print("\n========== CONTEXT ==========\n")

    print(context)

    print("\n" + "=" * 80)
    print(f"Retrieved Chunks : {len(results)}")
    print("=" * 80)
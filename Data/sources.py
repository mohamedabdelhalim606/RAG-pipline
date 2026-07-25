from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE = BASE_DIR / "Knowledge Base"

PAPERS = BASE_DIR / "papers"

DOCUMENTS = [

    KNOWLEDGE_BASE / "ChromaDB.md",

    KNOWLEDGE_BASE / "LangChain RAG.md",

    KNOWLEDGE_BASE / "LangChain Text Splitters.md",

    KNOWLEDGE_BASE / "OpenRouter.md",

    KNOWLEDGE_BASE / "RecursiveCharacterTextSplitter.md",

    KNOWLEDGE_BASE / "SentenceTransformers.md"

]

RESEARCH_PAPER = PAPERS / "2005.11401v4.pdf"
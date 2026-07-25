import re
import sys
from pathlib import Path

# ==========================================================
# Project Path
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Data.sources import DOCUMENTS

# ==========================================================
# Output Folder
# ==========================================================

CLEAN_FOLDER = PROJECT_ROOT / "Data" / "cleaned"
CLEAN_FOLDER.mkdir(parents=True, exist_ok=True)


def clean(self, text: str):

    # --------------------------------------------------
    # Remove Documentation Index Block
    # --------------------------------------------------

    if "Documentation Index" in text:

        match = re.search(r"\n# ", text)

        if match:
            text = text[match.start() + 1:]

    # --------------------------------------------------
    # Remove Code Blocks
    # --------------------------------------------------

    text = re.sub(r"```[\s\S]*?```", "", text)

    # --------------------------------------------------
    # Remove Inline Code
    # --------------------------------------------------

    text = re.sub(r"`[^`]*`", "", text)

    # --------------------------------------------------
    # Remove URLs
    # --------------------------------------------------

    text = re.sub(r"https?://\S+", "", text)

    # --------------------------------------------------
    # Markdown Links
    # [Text](url) ---> Text
    # --------------------------------------------------

    text = re.sub(r"\[([^\]]+)\]\((.*?)\)", r"\1", text)

    # --------------------------------------------------
    # Remove HTML Tags
    # --------------------------------------------------

    text = re.sub(r"<[^>]+>", "", text)

    # --------------------------------------------------
    # Remove Images
    # --------------------------------------------------

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # --------------------------------------------------
    # Remove Markdown Quotes
    # --------------------------------------------------

    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # --------------------------------------------------
    # Remove Installation Commands
    # --------------------------------------------------

    text = re.sub(r"^pip install.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^npm install.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^uv add.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^poetry add.*$", "", text, flags=re.MULTILINE)

    # --------------------------------------------------
    # Remove Navigation Words
    # --------------------------------------------------

    noise = [
        "Skip to main content",
        "Navigation",
        "Documentation Index",
        "Dashboard",
        "GitHub",
        "Search",
        "Ctrl K",
        "Copy page",
        "On this page"
    ]

    for item in noise:
        text = text.replace(item, "")

    # --------------------------------------------------
    # Remove Extra Spaces
    # --------------------------------------------------

    text = re.sub(r"[ \t]+", " ", text)

    # --------------------------------------------------
    # Remove Many Blank Lines
    # --------------------------------------------------

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
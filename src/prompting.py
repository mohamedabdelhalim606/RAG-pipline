import os
import sys
from pathlib import Path
from openai import OpenAI

# ==========================================================
# Paths & Imports
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Data import sources
from src.retrieval.retrieve_context import retrieve_context

# ==========================================================
# API Configuration
# ==========================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ==========================================================
# LLM Generation
# ==========================================================

def get_llm_response(question: str, api_key: str = None):

    # Retrieve context from Hybrid Search
    context_text, raw_results = retrieve_context(question)

    # API Key
    key_to_use = api_key or OPENROUTER_API_KEY

    if not key_to_use:
        return (
            "Error: OpenRouter API Key was not found.",
            context_text,
            raw_results,
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key_to_use,
    )

    # System Prompt
    system_prompt = """
You are an expert Retrieval-Augmented Generation (RAG) assistant.

Use ONLY the provided context to answer the user's question.

Rules:
- Never use outside knowledge.
- If the answer is not contained in the context, reply exactly:
"I do not have enough information to answer this based on the provided documents."
- Keep the answer concise, accurate, and well structured.
- Combine information from multiple retrieved chunks whenever appropriate.
- Do not mention the context or say "according to the context".
"""

    # User Prompt
    user_message = f"""
Context:
{context_text}

Question:
{question}

Answer:
"""

    try:

        response = client.chat.completions.create(

            model=OPENROUTER_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],

            temperature=0.1,
            top_p=0.9,
            max_tokens=700,

        )

        answer = response.choices[0].message.content

        
            
        
        sources = []
        seen = set()

        for item in raw_results:

            key = (item["source"], item["chunk_number"])

            if key not in seen:

                seen.add(key)
                sources.append(item)

        return answer, context_text, sources

    except Exception as e:

        return (
            f"Error while calling OpenRouter:\n{e}",
            context_text,
            [],
        )


    
    except Exception as e:
        return f"❌ error: {str(e)}", context_text, raw_results


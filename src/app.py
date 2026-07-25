import streamlit as st

from src.prompting import get_llm_response

st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password"
)

model = st.sidebar.selectbox(
    "Model",
    [
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it"
    ]
)

# ==========================================================
# Main Page
# ==========================================================

st.title("🤖 RAG Question Answering System")
st.caption("Ask questions about your documents using Hybrid Retrieval (BM25 + Vector Search)")

st.write(
    "Ask questions about your documents using Hybrid Retrieval (BM25 + Vector Search)."
)

question = st.text_area(
    "Enter your question",
    height=120
)

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching documents and generating answer..."):

        answer, context, sources = get_llm_response(
            question=question,
            api_key=api_key
        )

    st.success("Done!")

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Retrieved Sources")

    for i, source in enumerate(sources[:3], start=1):

        with st.expander(
            f"📄 Source {i} • {source['source']} • Chunk #{source['chunk_number']}"
        ):

            preview = source["text"][:220].replace("\n", " ")

            st.write(preview + "...")

            st.write(f"**RRF Score:** {source['rrf_score']:.4f}")

            if source["bm25_score"] is not None:
                st.write(f"**BM25 Score:** {source['bm25_score']:.4f}")

            if source["vector_distance"] is not None:
                st.write(f"**Vector Distance:** {source['vector_distance']:.4f}")
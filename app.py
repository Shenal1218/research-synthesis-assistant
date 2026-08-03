"""
app.py
======
Streamlit front-end for the Research Paper Methodology & Synthesis Assistant.

Run with:
    streamlit run app.py

API keys are read ONLY from `st.secrets` (see .streamlit/secrets.toml.example)
so they are never hardcoded, committed to git, typed into the UI, or
visible to whoever is using the app. If the keys are missing from
secrets.toml, the app shows a setup instruction instead of a text box
that could leak a key to anyone reading over the user's shoulder.
"""

import streamlit as st
from agents import run_pipeline

st.set_page_config(page_title="Research Synthesis Assistant", page_icon="📚", layout="wide")


def _get_secret(name: str) -> str:
    try:
        return st.secrets[name]
    except Exception:
        return ""


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
OPENROUTER_API_KEY = _get_secret("OPENROUTER_API_KEY")

# --------------------------------------------------------------------------- #
# Sidebar -- non-secret configuration only
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")

reasoning_model_name = st.sidebar.selectbox(
    "Reasoning model (OpenRouter)",
    options=["anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "API keys are configured by the app administrator in "
    "`.streamlit/secrets.toml` and are never entered or shown in this UI."
)

if not GROQ_API_KEY or not OPENROUTER_API_KEY:
    st.error(
        "This app isn't configured yet. Add `GROQ_API_KEY` and "
        "`OPENROUTER_API_KEY` to `.streamlit/secrets.toml` "
        "(copy `.streamlit/secrets.toml.example` as a starting point), "
        "then restart the app."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Main chat interface
# --------------------------------------------------------------------------- #
st.title("📚 Research Paper Methodology & Synthesis Assistant")
st.caption(
    "Ask about the ingested paper collection: e.g. "
    "\"Summarize the papers on transformer efficiency\" or "
    "\"Critique the methodology of the federated learning papers\"."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "trace", "retrieved_docs"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 Behind the scenes: agent-to-agent trace"):
                for step in msg["trace"]:
                    st.markdown(f"**{step['agent']}** · `{step['model']}`")
                    st.json(step["output"])

question = st.chat_input("Ask a question about the paper collection...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing question, retrieving evidence, synthesizing, critiquing..."):
            result_state = run_pipeline(
                question=question,
                groq_api_key=GROQ_API_KEY,
                openrouter_api_key=OPENROUTER_API_KEY,
                reasoning_model_name=reasoning_model_name,
            )

        final_answer = result_state.get("final_answer", "_No answer generated._")
        st.markdown(final_answer)

        with st.expander("🔍 Behind the scenes: agent-to-agent trace"):
            for step in result_state.get("trace", []):
                st.markdown(f"**{step['agent']}** · `{step['model']}`")
                st.json(step["output"])

        with st.expander("📄 Retrieved evidence (RAG)"):
            for d in result_state.get("retrieved_docs", []):
                st.markdown(f"**{d['source']}** (score: {d['relevance_score']})")
                st.text(d["content"][:400] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "trace": result_state.get("trace", []),
    })
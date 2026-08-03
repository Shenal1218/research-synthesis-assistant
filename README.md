# Research Paper Methodology & Synthesis Assistant

**IT41043 — Agentic AI Assignment (Option B: Research Support Tool)**
Horizon Campus

A multi-agent research support tool that helps undergraduate students
summarize, critique, and extract structured information (datasets,
methodologies) from a local collection of AI/CS academic papers, using
Retrieval-Augmented Generation (RAG) and a Router → Researcher → Critic
agent pipeline built with LangGraph.

---
## live demo link

https://research-synthesis-assistant-rx9gdfxvmbrxbjbl5adcel.streamlit.app/

---
Developer

**K.L.A.R.S.Perera** - **ITBIN-2313-0080**

---

## 1. Architecture


<img width="743" height="773" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/301c55a1-040f-41f0-83a8-c20301d9989c" />

---

**Agentic patterns implemented (3 required, 3 delivered):**

| # | Pattern | Where | Model |
|---|---------|-------|-------|
| 1 | **Router** | `agents.py::router_node` classifies the student's question into `summarize` / `critique` / `extract_datasets` / `general` | Groq Llama 3.1 8B |
| 2 | **Tool-Use / ReAct** | `agents.py::retrieval_node` reformulates the query based on intent, then calls `rag_pipeline.retrieve()` as a tool | RAG tool (no LLM call) |
| 3 | **Reflection / Critique** | `agents.py::researcher_node` (Agent 1) drafts a synthesis; `agents.py::critic_node` (Agent 2) reviews it for unbacked claims/gaps and can send it back for one revision | OpenRouter reasoning model |

**Agent-to-agent communication:** Agent 1 and Agent 2 never exchange free
text — they exchange typed JSON objects (`researcher_output`,
`critic_output`) stored as fields on the shared LangGraph `AgentState`.
This is visible live in the Streamlit "behind the scenes" expander.

---

## 2. Model Selection Justification

| Criterion | Groq (Llama 3.1 8B) | OpenRouter (Claude 3.5 Sonnet / GPT-4o-mini) |
|---|---|---|
| **Role in system** | Intent routing, light parsing | Literature synthesis, methodology critique |
| **Latency** | Very low (~100–300ms) — Groq's LPU inference is optimized for speed | Higher (~2–6s) — deeper reasoning takes longer |
| **Cost** | Very low / generous free tier — cheap enough to call on every turn | Higher per-token cost — reserved for 1–2 calls per turn |
| **Context window** | 128K tokens (sufficient for a short question) | 128K–200K tokens (needed to hold multiple retrieved chunks + prior critique) |
| **Reasoning depth** | Adequate for classification, weak for nuanced critique | Strong multi-step reasoning, better at spotting unsupported claims and structuring citations |
| **Why this split** | Routing runs on *every* user turn — a small, fast model keeps the UI responsive without wasting the reasoning budget on a trivial classification task | Synthesis and critique are the two steps that actually determine output quality, so the strongest available model is used only where it matters |

This mirrors a common production pattern: **cheap/fast model for control
flow, expensive/smart model for content generation** — cutting cost and
latency without sacrificing output quality where it counts.

---

## 3. RAG Pipeline

- **Loader:** `TextLoader` / `PyPDFLoader` over `data/papers/` (20+ files)
- **Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=800`, `chunk_overlap=120`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (free, local, 384-dim)
- **Vector store:** Chroma, persisted to `./chroma_db`
- **Retrieval:** top-k similarity search with relevance scores returned as JSON

### 3.1 Retrieval Quality Evaluation (5 queries)

Run:
```bash
python rag_pipeline.py --rebuild
```

This prints a report for 5 fixed evaluation queries (see
`EVALUATION_QUERIES` in `rag_pipeline.py`). For each query, manually
label the top-k results as relevant/irrelevant and fill in this table:

| # | Query | Top-k sources retrieved | Precision@k (manual) | Notes |
|---|---|---|---|---|
| 1 | What evaluation metrics are used to measure model performance? | *(paste from console output)* | e.g. 4/5 | |
| 2 | What datasets were used for training and testing? | | | |
| 3 | What are the limitations mentioned by the authors? | | | |
| 4 | How is the proposed method different from prior baselines? | | | |
| 5 | What preprocessing steps were applied to the raw data? | | | |

---

## 4. Setup

```bash
git clone <your-repo-url>
cd research-synthesis-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# add 20+ .txt or .pdf papers into data/papers/

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real GROQ_API_KEY and OPENROUTER_API_KEY

python rag_pipeline.py --rebuild     # build the vector store + run evaluation
streamlit run app.py                 # launch the UI
```

Get free API keys at:
- Groq: https://console.groq.com
- OpenRouter: https://openrouter.ai

---

## 5. Git Workflow (feature-branch, 15+ semantic commits)

Suggested branch/commit plan — adapt the exact commits to your real work,
but keep each commit small and semantically scoped (this is what a
lecturer checks for in the commit history during marking):

```
main
 ├─ feat/rag-pipeline
 │   1. chore: scaffold project structure and requirements.txt
 │   2. feat(rag): add document loader for txt/pdf papers
 │   3. feat(rag): add recursive chunking with overlap
 │   4. feat(rag): integrate HuggingFace all-MiniLM-L6-v2 embeddings
 │   5. feat(rag): build persistent Chroma vector store
 │   6. feat(rag): implement retrieve() with relevance scores
 │   7. test(rag): add 5-query retrieval evaluation harness
 │   → merge feat/rag-pipeline into main
 │
 ├─ feat/agents
 │   8. feat(agents): define shared AgentState schema
 │   9. feat(agents): implement router_node with Groq Llama 3.1 8B
 │   10. feat(agents): implement retrieval_node (tool-use/ReAct step)
 │   11. feat(agents): implement researcher_node (Agent 1) with JSON schema prompt
 │   12. feat(agents): implement critic_node (Agent 2) with JSON schema prompt
 │   13. feat(agents): add conditional reflection loop (revise vs finalize)
 │   14. feat(agents): wire LangGraph StateGraph and compile()
 │   → merge feat/agents into main
 │
 └─ feat/streamlit-ui
     15. feat(ui): add sidebar secrets configuration
     16. feat(ui): add chat interface with session state
     17. feat(ui): add agent-trace and retrieved-evidence expanders
     18. docs: write README with architecture, justification, eval, git guide
     19. chore: add .gitignore and secrets.toml.example
     → merge feat/streamlit-ui into main
```

Example commands for one feature branch:

```bash
git checkout -b feat/rag-pipeline
# ... write load_documents() ...
git add rag_pipeline.py
git commit -m "feat(rag): add document loader for txt/pdf papers"
# ... write chunk_documents() ...
git add rag_pipeline.py
git commit -m "feat(rag): add recursive chunking with overlap"
# ... repeat for each function ...
git checkout main
git merge --no-ff feat/rag-pipeline
```

Use `--no-ff` merges so the feature branches remain visible in `git log --graph`.

---

## 6. Limitations

- Retrieval quality depends on chunking strategy; very short or very
  long chunks reduce Precision@k (see Section 3.1).
- The critic's `needs_revision` flag is capped at 1 revision loop to
  keep response latency reasonable for a live demo.
- No authentication or multi-user support — designed for a single
  student's local research session.
- The system trusts the ingested PDFs/text as ground truth; it does not
  verify the underlying papers' own claims.

## 7. Academic Integrity Statement

This project was built as an individual submission for IT41043. All
third-party libraries (LangChain, LangGraph, Streamlit, ChromaDB,
sentence-transformers) are used under their respective open-source
licenses and are cited here rather than their code being copied into
this repository. LLM-generated text produced by the tool always includes
inline citations back to the source paper it was retrieved from.

## 8. References

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- LangChain documentation: https://python.langchain.com/
- Groq API: https://console.groq.com/docs
- OpenRouter API: https://openrouter.ai/docs
- Sentence-Transformers `all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- ChromaDB: https://docs.trychroma.com/

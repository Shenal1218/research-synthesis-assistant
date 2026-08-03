"""
agents.py
=========
Defines the multi-agent LangGraph workflow for the Research Paper
Methodology & Synthesis Assistant.

Agentic patterns implemented here:
  1. ROUTER          -- router_node()      (fast model: Groq Llama 3.1 8B)
  2. TOOL-USE / ReAct -- retrieval_node()   (calls rag_pipeline.retrieve)
  3. REFLECTION       -- researcher_node() + critic_node() loop
                         (reasoning model: OpenRouter Claude 3.5 Sonnet / GPT-4o-mini)

Agent-to-agent communication is implemented as structured JSON objects
carried inside a single shared LangGraph `AgentState` (a TypedDict).
LangGraph passes this state between nodes. Agent 1's output and Agent 2's
critique are both explicit JSON fields in the state (not free text), so
they can be logged, displayed in the Streamlit "behind the scenes"
expanders, and reasoned about independently -- this is the "structured
JSON message" requirement.
"""

import os
import json
from typing import List, TypedDict, Literal, Optional

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from rag_pipeline import retrieve

# --------------------------------------------------------------------------- #
# Model clients
# --------------------------------------------------------------------------- #

def get_router_model(groq_api_key: str) -> ChatGroq:
    """Fast model used ONLY for intent classification and light parsing.

    Groq's Llama 3.1 8B is chosen here because routing is a low-complexity,
    high-frequency call (runs on every single user turn) where latency
    matters far more than deep reasoning ability. See the model selection
    justification table in README.md.
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0,
    )


def get_reasoning_model(openrouter_api_key: str,
                         model_name: str = "anthropic/claude-3.5-sonnet") -> ChatOpenAI:
    """Reasoning model used for synthesis (Agent 1) and critique (Agent 2).

    OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI is reused
    with `base_url` pointed at OpenRouter. This model is deliberately NOT
    used for routing -- it is slower and costlier, so it is reserved for
    the two calls per turn that actually need strong reasoning: literature
    synthesis and methodological critique.
    """
    return ChatOpenAI(
        model=model_name,
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
    )


# --------------------------------------------------------------------------- #
# Shared state passed between LangGraph nodes
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    question: str                       # original student question
    intent: str                         # "summarize" | "critique" | "extract_datasets" | "general"
    retrieved_docs: List[dict]          # output of the RAG tool-use step
    researcher_output: dict             # Agent 1's structured JSON output
    critic_output: dict                 # Agent 2's structured JSON output
    revision_count: int                 # how many reflection loops have run
    final_answer: str                   # what gets shown to the student
    trace: List[dict]                   # log of every agent-to-agent message (for the UI)
    # config passed in at invoke-time rather than hardcoded / committed to git
    groq_api_key: str
    openrouter_api_key: str
    reasoning_model_name: str


MAX_REVISIONS = 1  # cap reflection loops so the app stays responsive


# --------------------------------------------------------------------------- #
# Node 1 -- ROUTER (Pattern 1)
# --------------------------------------------------------------------------- #
ROUTER_SYSTEM_PROMPT = """You are an intent classifier for an academic research assistant.
Classify the student's question into EXACTLY one of these labels:
- "summarize"        -> student wants a literature summary/overview of papers
- "critique"         -> student wants a methodology critique / weaknesses analysis
- "extract_datasets" -> student wants datasets, benchmarks or experimental setups extracted
- "general"          -> anything else

Respond with ONLY the label, nothing else."""


def router_node(state: AgentState) -> AgentState:
    llm = get_router_model(state["groq_api_key"])
    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ])
    intent = response.content.strip().lower()
    if intent not in {"summarize", "critique", "extract_datasets", "general"}:
        intent = "general"

    trace_entry = {"agent": "router", "model": "groq/llama-3.1-8b-instant",
                    "output": {"intent": intent}}
    return {
        "intent": intent,
        "trace": state.get("trace", []) + [trace_entry],
    }


# --------------------------------------------------------------------------- #
# Node 2 -- TOOL USE / ReAct retrieval (Pattern 2)
# --------------------------------------------------------------------------- #
def retrieval_node(state: AgentState) -> AgentState:
    """Acts as the 'Act' step of a ReAct loop: given the classified intent,
    reformulate the search query and call the RAG tool."""
    intent_to_query_hint = {
        "summarize": "overview, contributions and findings of the paper",
        "critique": "methodology, evaluation setup and limitations",
        "extract_datasets": "datasets, benchmarks, experimental setup",
        "general": "",
    }
    hint = intent_to_query_hint.get(state["intent"], "")
    search_query = f"{state['question']} {hint}".strip()

    docs = retrieve(search_query, k=5)

    trace_entry = {"agent": "retrieval_tool", "model": "rag_pipeline/retrieve",
                    "output": {"query": search_query, "num_results": len(docs)}}
    return {
        "retrieved_docs": docs,
        "trace": state.get("trace", []) + [trace_entry],
    }


# --------------------------------------------------------------------------- #
# Node 3 -- AGENT 1: Literature Researcher (Reflection pattern, part A)
# --------------------------------------------------------------------------- #
RESEARCHER_SYSTEM_PROMPT = """You are Agent 1, the Literature Researcher.
You will be given a student's question, its classified intent, and a set of
retrieved paper excerpts (each tagged with a source filename).

Your job:
1. Directly answer the student's question using ONLY the retrieved excerpts.
2. Every factual claim MUST be followed by a citation like (Source: <filename>).
3. Do not invent papers, numbers, or claims not present in the excerpts.

Respond ONLY in this JSON schema:
{
  "synthesis": "<the written answer with inline (Source: file.txt) citations>",
  "citations_used": ["file1.txt", "file2.txt"],
  "claims": ["claim 1 with its source", "claim 2 with its source"]
}"""


def researcher_node(state: AgentState) -> AgentState:
    llm = get_reasoning_model(state["openrouter_api_key"], state.get("reasoning_model_name"))

    context_block = "\n\n".join(
        f"[Source: {d['source']}]\n{d['content']}" for d in state["retrieved_docs"]
    )

    critique_feedback = ""
    if state.get("critic_output"):
        critique_feedback = (
            "\n\nA previous critique flagged these issues -- revise your synthesis "
            "to address them:\n" + json.dumps(state["critic_output"], indent=2)
        )

    user_prompt = (
        f"Student question: {state['question']}\n"
        f"Intent: {state['intent']}\n\n"
        f"Retrieved excerpts:\n{context_block}"
        f"{critique_feedback}"
    )

    response = llm.invoke([
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    researcher_output = _safe_json_parse(response.content, fallback_key="synthesis")

    trace_entry = {"agent": "researcher (Agent 1)",
                    "model": "openrouter/" + str(state.get("reasoning_model_name")),
                    "output": researcher_output}
    return {
        "researcher_output": researcher_output,
        "trace": state.get("trace", []) + [trace_entry],
    }


# --------------------------------------------------------------------------- #
# Node 4 -- AGENT 2: Academic Critic (Reflection pattern, part B)
# --------------------------------------------------------------------------- #
CRITIC_SYSTEM_PROMPT = """You are Agent 2, the Academic Critic.
You will receive Agent 1's literature synthesis as structured JSON.

Critically review it for:
- unbacked_claims: statements without a (Source: ...) citation
- gaps: important aspects of the question the synthesis failed to address
- suggestions: concrete improvements Agent 1 should make

Respond ONLY in this JSON schema:
{
  "unbacked_claims": ["..."],
  "gaps": ["..."],
  "suggestions": ["..."],
  "needs_revision": true or false
}
Set "needs_revision" to true only if there is at least one unbacked claim or
a significant gap. Be strict but fair."""


def critic_node(state: AgentState) -> AgentState:
    llm = get_reasoning_model(state["openrouter_api_key"], state.get("reasoning_model_name"))

    user_prompt = (
        f"Original question: {state['question']}\n\n"
        f"Agent 1's synthesis (JSON):\n{json.dumps(state['researcher_output'], indent=2)}"
    )

    response = llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    critic_output = _safe_json_parse(response.content, fallback_key="gaps")

    trace_entry = {"agent": "critic (Agent 2)",
                    "model": "openrouter/" + str(state.get("reasoning_model_name")),
                    "output": critic_output}
    return {
        "critic_output": critic_output,
        "revision_count": state.get("revision_count", 0) + 1,
        "trace": state.get("trace", []) + [trace_entry],
    }


# --------------------------------------------------------------------------- #
# Node 5 -- Finalizer
# --------------------------------------------------------------------------- #
def finalize_node(state: AgentState) -> AgentState:
    synthesis = state["researcher_output"].get("synthesis", "")
    critic = state.get("critic_output", {})

    notes = ""
    if critic.get("suggestions"):
        notes = "\n\n---\n**Reviewer notes (Agent 2):** " + "; ".join(critic["suggestions"])

    return {"final_answer": synthesis + notes}


def _safe_json_parse(raw: str, fallback_key: str) -> dict:
    """LLMs occasionally wrap JSON in prose or code fences -- defensively parse
    instead of letting the whole pipeline crash on a malformed response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[len("json\n"):]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {fallback_key: raw, "parse_error": True}


# --------------------------------------------------------------------------- #
# Conditional edge -- the Reflection loop (Pattern 3)
# --------------------------------------------------------------------------- #
def route_after_critique(state: AgentState) -> Literal["revise", "finalize"]:
    needs_revision = state.get("critic_output", {}).get("needs_revision", False)
    if needs_revision and state.get("revision_count", 0) <= MAX_REVISIONS:
        return "revise"
    return "finalize"


# --------------------------------------------------------------------------- #
# Build the graph
# --------------------------------------------------------------------------- #
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critique,
        {"revise": "researcher", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_pipeline(question: str, groq_api_key: str, openrouter_api_key: str,
                  reasoning_model_name: str = "anthropic/claude-3.5-sonnet") -> AgentState:
    """Convenience wrapper used by app.py -- builds the graph and runs one turn."""
    app_graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "groq_api_key": groq_api_key,
        "openrouter_api_key": openrouter_api_key,
        "reasoning_model_name": reasoning_model_name,
        "revision_count": 0,
        "trace": [],
    }
    return app_graph.invoke(initial_state)

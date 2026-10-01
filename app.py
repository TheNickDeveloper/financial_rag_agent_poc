import streamlit as st
from pathlib import Path

from rag_engine import RAGEngine, LOCAL_EMBEDDING_MODELS
from llm import LLMConfigError
from agent_graph import FinancialAgentGraph

st.set_page_config(
    page_title="Financial Knowledge Agent",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

GRAPH_DIR = "graph_output"

st.title("📊 Financial Knowledge Agent")
st.caption("RAG + Agentic Skills POC for financial research and knowledge Q&A")


# ------------------------------------------------------------------ #
# Cached resources
# ------------------------------------------------------------------ #
@st.cache_resource
def get_rag(embedding_provider, embedding_model):
    return RAGEngine(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )


@st.cache_resource
def get_agent(_rag, provider, model, top_k, use_llm_router):
    return FinancialAgentGraph(
        rag=_rag,
        provider=provider,
        model=model,
        top_k=top_k,
        use_llm_router=use_llm_router,
    )


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox("LLM Provider", ["DeepSeek", "OpenAI"])

    default_model = "deepseek-chat" if provider == "DeepSeek" else "gpt-4o-mini"
    model = st.text_input("Model", value=default_model)

    embedding_provider = st.selectbox(
        "Embedding Provider",
        ["FastEmbed", "OpenAI"]
    )

    if embedding_provider == "FastEmbed":
        embedding_model = st.selectbox(
            "Embedding Model",
            LOCAL_EMBEDDING_MODELS
        )
        st.caption(
            "Local ONNX embedding — runs on CPU, no API key, no cost. "
            "English docs: all-MiniLM-L6-v2 or bge-small-en-v1.5. "
            "Chinese financial docs: nomic-embed-text-v1.5."
        )
    else:
        embedding_model = st.text_input(
            "Embedding Model",
            value="text-embedding-3-small"
        )

    top_k = st.slider("Retrieved Documents", 2, 10, 5)

    use_llm_router = st.toggle(
        "LLM Router",
        value=True,
        help=(
            "Classify the request with a cheap LLM call. "
            "If off (or if the call fails) keyword rules are used."
        )
    )

    st.divider()
    st.subheader("📚 Knowledge Base")

    uploaded = st.file_uploader(
        "Upload research reports / financial documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True
    )

    if uploaded:
        for f in uploaded:
            path = DATA_DIR / f.name
            path.write_bytes(f.getbuffer())
        st.success(f"{len(uploaded)} file(s) uploaded.")

    if st.button("🔄 Build / Update Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents..."):
            rag_for_ingest = get_rag(embedding_provider, embedding_model)
            result = rag_for_ingest.ingest_directory(DATA_DIR)
        st.success(
            f"Indexed {result['documents']} documents / "
            f"{result['chunks']} chunks."
        )
        get_agent.clear()

    st.divider()

    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.saved_graphs = {}
        st.rerun()

    st.divider()
    st.markdown("""
    **Available Skills**

    - 🔎 Knowledge Retrieval
    - 📄 Document Summary
    - 📊 Company Comparison
    - ⚠️ Risk Analysis
    - 🧠 Research Synthesis
    - 🧾 Source Citation
    """)


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "saved_graphs" not in st.session_state:
    # Map from message index -> saved file path, so we don't re-save.
    st.session_state.saved_graphs = {}


# ------------------------------------------------------------------ #
# Build rag + agent
# ------------------------------------------------------------------ #
rag = get_rag(embedding_provider, embedding_model)

try:
    agent = get_agent(rag, provider, model, top_k, use_llm_router)
except LLMConfigError as exc:
    st.error(str(exc))
    st.stop()


# ------------------------------------------------------------------ #
# Helper: graph controls under an assistant message
# ------------------------------------------------------------------ #
def render_graph_controls(message_index: int, agent: FinancialAgentGraph):
    saved = st.session_state.saved_graphs.get(message_index)

    if saved:
        st.caption(f"🖼️ Graph saved: `{saved}`")
        return

    with st.expander("📊 Generate the graph for this run?"):
        st.caption(
            "Render the agent's nodes and edges as a PNG into `graph_output/`."
        )
        if st.button("💾 Generate Graph", key=f"gen_graph_{message_index}"):
            try:
                path = agent.save_graph_image(
                    output_dir=GRAPH_DIR,
                    fmt="png",
                )
                st.session_state.saved_graphs[message_index] = path
                st.success(f"Saved to: {path}")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to generate graph: {exc}")


# ------------------------------------------------------------------ #
# Render chat history (with skill / sources / graph controls)
# ------------------------------------------------------------------ #
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # --- skill / route 提示 ---
            route = msg.get("route") or {}
            if route and msg.get("skill"):
                icon = "🧠" if route.get("source") == "llm" else "🔤"
                st.caption(
                    f"{icon} Skill: **{msg['skill']}** "
                    f"via {route.get('source')} router"
                    f" · {route.get('reason', '')}"
                )

            # --- sources ---
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['source']}** — page/chunk "
                            f"{s.get('page', '-')}\n\n"
                            f"> {s['snippet']}"
                        )

            # --- graph 控件 ---
            render_graph_controls(idx, agent)


# ------------------------------------------------------------------ #
# Chat input
# ------------------------------------------------------------------ #
prompt = st.chat_input(
    "Ask about companies, financial reports, risks, financial metrics..."
)

if prompt:
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Run agent
    with st.spinner("Researching..."):
        try:
            response = agent.run(prompt)
        except Exception as exc:
            st.error(f"Agent failed: {exc}")
            st.stop()

    # 3. Append assistant message with all metadata
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "skill": response.get("skill"),
        "route": response.get("route") or {},
        "sources": response.get("sources") or [],
    })

    # 4. Rerun so the history loop renders everything (including new msg)
    st.rerun()
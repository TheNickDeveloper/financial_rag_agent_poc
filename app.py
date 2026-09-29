import streamlit as st
from pathlib import Path
from rag_engine import RAGEngine, LOCAL_EMBEDDING_MODELS
from agent import FinancialAgent
from llm import LLMConfigError

st.set_page_config(
    page_title="Financial Knowledge Agent",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

st.title("📊 Financial Knowledge Agent")
st.caption("RAG + Agentic Skills POC for financial research and knowledge Q&A")


# ---------- 缓存：避免每次 rerun 都重建向量库 / 加载 ONNX 模型 ----------
@st.cache_resource
def get_rag(embedding_provider, embedding_model):
    return RAGEngine(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )


@st.cache_resource
def get_agent(_rag, provider, model, top_k, use_llm_router):
    return FinancialAgent(
        rag=_rag,
        provider=provider,
        model=model,
        top_k=top_k,
        use_llm_router=use_llm_router
    )


with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        ["DeepSeek", "OpenAI"]
    )

    if provider == "DeepSeek":
        default_model = "deepseek-chat"
    else:
        default_model = "gpt-4o-mini"

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
            f"Indexed {result['documents']} documents / {result['chunks']} chunks."
        )
        # ingest 后清掉 agent 缓存，让下次问答用同一个 rag 实例
        get_agent.clear()

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


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

rag = get_rag(embedding_provider, embedding_model)

try:
    agent = get_agent(rag, provider, model, top_k, use_llm_router)
except LLMConfigError as exc:
    st.error(str(exc))
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Ask about companies, financial reports, risks, financial metrics..."
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            response = agent.run(prompt)

        st.markdown(response["answer"])

        route = response.get("route") or {}
        if route:
            icon = "🧠" if route.get("source") == "llm" else "🔤"
            st.caption(
                f"{icon} Skill: **{response['skill']}** "
                f"via {route.get('source')} router"
                f" · {route.get('reason', '')}"
            )

        if response.get("sources"):
            with st.expander("📚 Sources"):
                for s in response["sources"]:
                    st.markdown(
                        f"**{s['source']}** — page/chunk {s.get('page', '-')}\n\n"
                        f"> {s['snippet']}"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"]
    })
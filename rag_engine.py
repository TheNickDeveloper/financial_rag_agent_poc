from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

CHROMA_DIR = "chroma_db"

DEFAULT_EMBEDDING_MODELS = {
    "FastEmbed": "sentence-transformers/all-MiniLM-L6-v2",  # 默认轻量模型
    "OpenAI": "text-embedding-3-small",
}

LOCAL_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # ~22MB，最轻
    "BAAI/bge-small-en-v1.5",                  # ~33MB，英文检索更好
    "nomic-ai/nomic-embed-text-v1.5",          # ~274MB，长上下文
]

SPECIFIC_MODEL_PATH = r"C:\Users\ZhuanZ（无密码）\Desktop\financial_rag_agent_poc\embedding_models"

class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings 接口的 FastEmbed 包装。"""

    def __init__(self, model_name: str, specific_model_path: str = None):
        from fastembed import TextEmbedding

        self._model_name = model_name

        # 如果提供了本地模型路径，直接用本地文件，不再联网下载
        if specific_model_path:
            self._model = TextEmbedding(
                model_name=model_name,
                specific_model_path=specific_model_path
            )
        else:
            self._model = TextEmbedding(model_name=model_name)

        # Nomic 系列需要 task prefix，否则检索质量下降
        self._needs_prefix = "nomic" in model_name.lower()

    def embed_documents(self, texts):
        if self._needs_prefix:
            texts = [f"search_document: {t}" for t in texts]
        # fastembed 返回 generator，需要转成 list[list[float]]
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text):
        if self._needs_prefix:
            text = f"search_query: {text}"
        return next(self._model.embed([text])).tolist()


def build_embeddings(embedding_provider, embedding_model=None,
                     specific_model_path=None):
    embedding_model = (
        embedding_model or DEFAULT_EMBEDDING_MODELS[embedding_provider]
    )

    if embedding_provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=embedding_model)

    if embedding_provider == "FastEmbed":
        return FastEmbedEmbeddings(
            model_name=embedding_model,
            specific_model_path=specific_model_path
        )

    raise ValueError(f"Unsupported embedding provider: {embedding_provider}")


class RAGEngine:

    def __init__(self, embedding_provider="FastEmbed", embedding_model=None,
                 specific_model_path=None):
        self.embedding_provider = embedding_provider
        self.embedding_model = (
            embedding_model or DEFAULT_EMBEDDING_MODELS[embedding_provider]
        )

        # 如果调用方没传，就用模块级常量
        self.specific_model_path = (
            specific_model_path or SPECIFIC_MODEL_PATH
        )

        self.embeddings = build_embeddings(
            self.embedding_provider,
            self.embedding_model,
            specific_model_path=self.specific_model_path
        )

        # 清洗模型名，避免 "/" 和 "-" 影响 Chroma collection 命名
        safe_model = (
            self.embedding_model
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        self.collection_name = f"financial_knowledge_{safe_model}"

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )

    def load_file(self, path: Path):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return PyPDFLoader(str(path)).load()

        if suffix == ".docx":
            return Docx2txtLoader(str(path)).load()

        if suffix in [".txt", ".md"]:
            return TextLoader(
                str(path),
                encoding="utf-8"
            ).load()

        return []

    def ingest_directory(self, directory):
        directory = Path(directory)
        documents = []

        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in [
                ".pdf", ".txt", ".md", ".docx"
            ]:
                documents.extend(self.load_file(path))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        if chunks:
            # 分批插入，避免超过 ChromaDB 单次操作上限
            batch_size = 5000  # 留出余量，确保低于 5461
            total = len(chunks)
            for i in range(0, total, batch_size):
                batch = chunks[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                print(f"Inserted batch {i // batch_size + 1}: "
                    f"{len(batch)} chunks ({i + len(batch)}/{total})")

            # 部分 Chroma 版本需要显式持久化
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()

        return {
            "documents": len(documents),
            "chunks": len(chunks)
        }

    def search(self, query, k=5):
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )

        output = []

        for doc, score in results:
            metadata = doc.metadata or {}

            output.append({
                "content": doc.page_content,
                "source": Path(
                    metadata.get("source", "unknown")
                ).name,
                "page": metadata.get("page", "-"),
                # Chroma 默认 L2 距离：score 越小越相似
                "score": float(score)
            })

        return output
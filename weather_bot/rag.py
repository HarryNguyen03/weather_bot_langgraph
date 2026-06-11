"""
RAG module cho Weather Bot.
Build vectorstore từ knowledge base và expose retriever tool cho Weather Analyst.

Yêu cầu:
  pip install langchain-community langchain-text-splitters
  ollama pull nomic-embed-text

knowledge/ phải nằm cùng thư mục với file này và chứa ít nhất 1 file .md.
"""

from pathlib import Path

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
EMBED_MODEL = "nomic-embed-text"
HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _load_markdown_docs(knowledge_dir: Path) -> list:
    """Đọc và split tất cả file .md trong knowledge_dir."""
    md_files = list(knowledge_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .md nào trong {knowledge_dir}. "
            "Tạo thư mục knowledge/ và thêm file .md trước khi chạy."
        )

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )

    all_docs = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for chunk in chunks:
            chunk.metadata["source"] = md_file.name
        all_docs.extend(chunks)
        print(f"  [RAG] {md_file.name}: {len(chunks)} chunks")

    print(f"  [RAG] Tổng cộng: {len(all_docs)} chunks từ {len(md_files)} file")
    return all_docs


def build_vectorstore() -> InMemoryVectorStore:
    """Build InMemoryVectorStore từ toàn bộ knowledge base."""
    print("\n[RAG] Đang build vectorstore...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    docs = _load_markdown_docs(KNOWLEDGE_DIR)
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    print("[RAG] Vectorstore sẵn sàng.\n")
    return vectorstore


def build_retriever_tool(vectorstore: InMemoryVectorStore):
    """Tạo retriever tool để Weather Analyst tra cứu knowledge base."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def retrieve_weather_knowledge(query: str) -> str:
        """
        Tra cứu knowledge base về lời khuyên thời tiết, sức khỏe, trang phục, và khí hậu.

        Dùng khi người dùng hỏi về:
        - Lời khuyên sức khỏe liên quan đến thời tiết (nắng nóng, mưa, lạnh)
        - Gợi ý trang phục theo điều kiện thời tiết
        - Hoạt động ngoài trời phù hợp với thời tiết
        - Đặc điểm khí hậu của vùng/thành phố tại Việt Nam

        KHÔNG dùng tool này để lấy dữ liệu thời tiết thực tế —
        dùng get_weather hoặc get_5day_forecast thay thế.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "Không tìm thấy thông tin liên quan trong knowledge base."

        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Nguồn: {source}]\n{doc.page_content}")

        return "\n---\n".join(parts)

    return retrieve_weather_knowledge

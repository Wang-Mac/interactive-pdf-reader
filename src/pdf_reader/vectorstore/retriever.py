from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from pdf_reader.config import Settings, get_settings


def create_retriever(
    vector_store: Chroma,
    k: int | None = None,
    settings: Settings | None = None,
) -> VectorStoreRetriever:
    resolved_settings = settings or get_settings()
    resolved_k = k if k is not None else resolved_settings.retrieval_k
    if resolved_k <= 0:
        raise ValueError("k must be greater than zero")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": resolved_k},
    )


def retrieve_documents(
    vector_store: Chroma,
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    retriever = create_retriever(vector_store, k=k, settings=settings)
    return list(retriever.invoke(query))

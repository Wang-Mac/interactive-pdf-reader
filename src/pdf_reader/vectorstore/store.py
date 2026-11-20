from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pdf_reader.config import Settings, get_settings

DEFAULT_COLLECTION_NAME = "pdf_documents"


def create_vector_store(
    documents: Iterable[Document] | None = None,
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    resolved_settings = settings or get_settings()
    persist_directory = resolved_settings.chroma_persist_directory
    persist_directory.mkdir(parents=True, exist_ok=True)

    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    document_list = list(documents) if documents is not None else []
    if document_list:
        return Chroma.from_documents(
            documents=document_list,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_directory),
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def load_vector_store(
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    return create_vector_store(
        settings=settings,
        embeddings=embeddings,
        collection_name=collection_name,
    )


def add_documents(
    vector_store: Chroma,
    documents: Iterable[Document],
) -> list[str]:
    document_list = list(documents)
    if not document_list:
        return []
    return vector_store.add_documents(document_list)

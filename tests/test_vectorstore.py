from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pdf_reader.config import Settings
from pdf_reader.vectorstore import (
    add_documents,
    create_embeddings,
    create_retriever,
    create_vector_store,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def create_settings(tmp_path: Path, api_key: str | None = "test-key") -> Settings:
    return Settings(
        nvidia_api_key=api_key,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_chat_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_embedding_model="nvidia/nemotron-3-embed-1b",
        chroma_persist_directory=tmp_path / "chroma",
        upload_directory=tmp_path / "uploads",
        chunk_size=1000,
        chunk_overlap=200,
        retrieval_k=4,
    )


@patch("pdf_reader.vectorstore.embeddings.NVIDIAEmbeddings")
def test_create_embeddings_uses_configured_model(nvidia_embeddings: MagicMock) -> None:
    settings = create_settings(Path("/tmp"))

    create_embeddings(settings)

    nvidia_embeddings.assert_called_once_with(
        model="nvidia/nemotron-3-embed-1b",
        nvidia_api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
    )


def test_create_embeddings_requires_api_key(tmp_path: Path) -> None:
    settings = create_settings(tmp_path, api_key=None)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        create_embeddings(settings)


@patch("pdf_reader.vectorstore.store.Chroma")
def test_create_vector_store_from_documents(chroma: MagicMock, tmp_path: Path) -> None:
    settings = create_settings(tmp_path)
    embeddings = FakeEmbeddings()
    documents = [
        Document(
            page_content="First chunk",
            metadata={"source": "/tmp/a.pdf", "page": 1},
        )
    ]

    create_vector_store(
        documents,
        settings=settings,
        embeddings=embeddings,
        collection_name="test_collection",
    )

    chroma.from_documents.assert_called_once_with(
        documents=documents,
        embedding=embeddings,
        collection_name="test_collection",
        persist_directory=str(settings.chroma_persist_directory),
    )
    assert settings.chroma_persist_directory.is_dir()


@patch("pdf_reader.vectorstore.store.Chroma")
def test_create_empty_vector_store(chroma: MagicMock, tmp_path: Path) -> None:
    settings = create_settings(tmp_path)
    embeddings = FakeEmbeddings()

    create_vector_store(
        settings=settings,
        embeddings=embeddings,
        collection_name="test_collection",
    )

    chroma.assert_called_once_with(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_directory),
    )


def test_create_vector_store_rejects_empty_collection_name(tmp_path: Path) -> None:
    settings = create_settings(tmp_path)

    with pytest.raises(ValueError, match="collection_name must not be empty"):
        create_vector_store(settings=settings, collection_name=" ")


def test_add_documents_empty_input() -> None:
    vector_store = MagicMock()

    assert add_documents(vector_store, []) == []
    vector_store.add_documents.assert_not_called()


def test_create_retriever_uses_configured_k(tmp_path: Path) -> None:
    settings = create_settings(tmp_path)
    vector_store = MagicMock()

    create_retriever(vector_store, settings=settings)

    vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


@pytest.mark.parametrize("k", [0, -1])
def test_create_retriever_rejects_invalid_k(k: int, tmp_path: Path) -> None:
    settings = create_settings(tmp_path)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        create_retriever(MagicMock(), k=k, settings=settings)

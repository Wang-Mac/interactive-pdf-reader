import pytest
from langchain_core.documents import Document

from pdf_reader.document import (
    split_documents,
    split_text,
    validate_split_parameters,
)


def test_split_documents_preserves_metadata_and_adds_start_index() -> None:
    document = Document(
        page_content="A short paragraph. Another sentence.",
        metadata={"source": "/tmp/sample.pdf", "file_name": "sample.pdf", "page": 2},
    )

    chunks = split_documents([document], chunk_size=20, chunk_overlap=5)

    assert chunks
    assert all(chunk.metadata["source"] == "/tmp/sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["file_name"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_split_documents_returns_empty_for_empty_input() -> None:
    assert split_documents([]) == []


def test_split_text_with_explicit_parameters() -> None:
    chunks = split_text("Alpha beta gamma delta", chunk_size=10, chunk_overlap=2)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-10, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_validate_split_parameters_rejects_invalid_values(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        validate_split_parameters(chunk_size, chunk_overlap)

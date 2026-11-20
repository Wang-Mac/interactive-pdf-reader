from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_reader.config import get_settings


def validate_split_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def split_documents(
    documents: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    document_list = list(documents)
    if not document_list:
        return []

    settings = get_settings()
    resolved_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    resolved_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    )
    validate_split_parameters(resolved_chunk_size, resolved_chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(document_list)


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    if not text:
        return []

    settings = get_settings()
    resolved_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    resolved_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    )
    validate_split_parameters(resolved_chunk_size, resolved_chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
    )
    return splitter.split_text(text)

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_reader.ui.app import (
    _chat_history_for_qa,
    _collection_name_for_pdf,
    _safe_pdf_name,
    save_uploaded_pdf,
)


def test_safe_pdf_name_accepts_pdf() -> None:
    assert _safe_pdf_name("/tmp/guide.pdf") == "guide.pdf"


@pytest.mark.parametrize("file_name", ["guide.txt", "", "README"])
def test_safe_pdf_name_rejects_non_pdf(file_name: str) -> None:
    with pytest.raises(ValueError):
        _safe_pdf_name(file_name)


def test_save_uploaded_pdf_writes_file(tmp_path: Path) -> None:
    settings = MagicMock()
    settings.upload_directory = tmp_path

    with patch(
        "pdf_reader.ui.app.get_settings",
        return_value=settings,
    ):
        saved_path = save_uploaded_pdf("guide.pdf", b"%PDF-test")

    assert saved_path == tmp_path / "guide.pdf"
    assert saved_path.read_bytes() == b"%PDF-test"


def test_save_uploaded_pdf_rejects_empty_content(tmp_path: Path) -> None:
    settings = MagicMock()
    settings.upload_directory = tmp_path

    with patch(
        "pdf_reader.ui.app.get_settings",
        return_value=settings,
    ):
        with pytest.raises(ValueError, match="PDF is empty"):
            save_uploaded_pdf("guide.pdf", b"")


def test_chat_history_conversion() -> None:
    history = _chat_history_for_qa(
        [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval augmented generation."},
        ]
    )

    assert history == [
        ("user", "What is RAG?"),
        ("assistant", "Retrieval augmented generation."),
    ]


def test_collection_name_isolated_by_pdf_digest() -> None:
    first = _collection_name_for_pdf("guide.pdf", "a" * 64)
    second = _collection_name_for_pdf("guide.pdf", "b" * 64)

    assert first != second
    assert first.startswith("pdf_")

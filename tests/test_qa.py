from unittest.mock import ANY, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from pdf_reader.qa import (
    QAResult,
    answer_question,
    build_chat_history,
    format_context,
    stream_answer_question,
)


def test_format_context_includes_file_page_and_text() -> None:
    documents = [
        Document(
            page_content="PDF reader overview",
            metadata={"file_name": "guide.pdf", "page": 3},
        )
    ]

    context = format_context(documents)

    assert "guide.pdf, page 3" in context
    assert "PDF reader overview" in context


def test_format_context_handles_empty_documents() -> None:
    context = format_context([])

    assert "No relevant context" in context


def test_source_citation_normalizes_string_page() -> None:
    document = Document(
        page_content="Test text",
        metadata={"file_name": "guide.pdf", "page": "7"},
    )

    with (
        patch(
            "pdf_reader.qa.pipeline.retrieve_documents",
            return_value=[document],
        ),
        patch("pdf_reader.qa.pipeline.create_chat_model") as create_model,
    ):
        create_model.return_value.invoke.return_value = AIMessage(content="Answer")

        result = answer_question(MagicMock(), "Test question")

    assert result.sources[0].page == 7


def test_build_chat_history_formats_roles() -> None:
    history = build_chat_history(
        [
            ("user", "What is RAG?"),
            ("assistant", "Retrieval augmented generation."),
        ]
    )

    assert "User: What is RAG?" in history
    assert "Assistant: Retrieval augmented generation." in history


@patch("pdf_reader.qa.pipeline.create_chat_model")
@patch("pdf_reader.qa.pipeline.retrieve_documents")
def test_answer_question_returns_answer_and_sources(
    retrieve: MagicMock,
    create_model: MagicMock,
) -> None:
    retrieve.return_value = [
        Document(
            page_content="The reader uses LangChain.",
            metadata={
                "source": "/tmp/guide.pdf",
                "file_name": "guide.pdf",
                "page": 2,
            },
        )
    ]
    create_model.return_value.invoke.return_value = AIMessage(
        content="The reader uses LangChain."
    )

    result = answer_question(
        MagicMock(),
        "Which framework does it use?",
        chat_history=[("user", "Summarize the PDF.")],
        k=2,
    )

    retrieve.assert_called_once_with(
        ANY,
        "Which framework does it use?",
        k=2,
    )
    assert isinstance(result, QAResult)
    assert result.answer == "The reader uses LangChain."
    assert len(result.sources) == 1
    assert result.sources[0].file_name == "guide.pdf"
    assert result.sources[0].page == 2


def test_answer_question_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        answer_question(MagicMock(), "   ")


@patch("pdf_reader.qa.pipeline.create_chat_model")
@patch("pdf_reader.qa.pipeline.retrieve_documents")
def test_answer_question_rejects_empty_model_content(
    retrieve: MagicMock,
    create_model: MagicMock,
) -> None:
    retrieve.return_value = [
        Document(
            page_content="Context text",
            metadata={"file_name": "guide.pdf", "page": 1},
        )
    ]
    create_model.return_value.invoke.return_value = AIMessage(content="")

    with pytest.raises(RuntimeError, match="returned no final answer"):
        answer_question(MagicMock(), "Summarize the PDF")


@patch("pdf_reader.qa.pipeline.create_chat_model")
@patch("pdf_reader.qa.pipeline.retrieve_documents")
def test_stream_answer_question_yields_sources_then_text(
    retrieve: MagicMock,
    create_model: MagicMock,
) -> None:
    retrieve.return_value = [
        Document(
            page_content="Streaming answer context.",
            metadata={"source": "/tmp/guide.pdf", "file_name": "guide.pdf", "page": 1},
        )
    ]
    create_model.return_value.stream.return_value = iter(
        [
            AIMessage(content="Streaming "),
            AIMessage(content="answer"),
        ]
    )

    events = list(
        stream_answer_question(
            MagicMock(),
            "Explain streaming",
        )
    )

    assert hasattr(events[0], "file_name")
    assert events[1:] == ["Streaming ", "answer"]

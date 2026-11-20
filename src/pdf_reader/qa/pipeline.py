from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from pdf_reader.llm import create_chat_model
from pdf_reader.prompts import qa_prompt
from pdf_reader.vectorstore import retrieve_documents


@dataclass(frozen=True, slots=True)
class SourceCitation:
    source: str | None
    file_name: str | None
    page: int | None
    text: str


@dataclass(frozen=True, slots=True)
class QAResult:
    answer: str
    sources: tuple[SourceCitation, ...]


def format_context(documents: Iterable[Document]) -> str:
    document_list = list(documents)
    if not document_list:
        return "No relevant context was found in the uploaded document."

    sections: list[str] = []
    for index, document in enumerate(document_list, start=1):
        file_name = document.metadata.get("file_name")
        page = document.metadata.get("page")
        source_label = str(file_name) if file_name else "unknown file"
        if page is not None:
            source_label = f"{source_label}, page {page}"
        text = document.page_content.strip()
        if len(text) > 2400:
            text = text[:2400].rstrip() + "..."
        sections.append(f"[Source {index}: {source_label}]\n{text}")
    return "\n\n".join(sections)


def build_chat_history(messages: Iterable[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        normalized_role = role.strip().lower()
        if normalized_role in {"human", "user"}:
            lines.append(f"User: {content.strip()}")
        elif normalized_role in {"ai", "assistant"}:
            lines.append(f"Assistant: {content.strip()}")
    return "\n".join(lines) if lines else "No previous conversation."


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "".join(text_parts)
    return str(content)


def _build_sources(documents: Iterable[Document]) -> tuple[SourceCitation, ...]:
    sources: list[SourceCitation] = []
    for document in documents:
        source = document.metadata.get("source")
        file_name = document.metadata.get("file_name")
        page = document.metadata.get("page")
        normalized_page: int | None = None
        if isinstance(page, int):
            normalized_page = page
        elif isinstance(page, str):
            try:
                normalized_page = int(page)
            except ValueError:
                normalized_page = None
        sources.append(
            SourceCitation(
                source=str(source) if source is not None else None,
                file_name=str(file_name) if file_name is not None else None,
                page=normalized_page,
                text=document.page_content,
            )
        )
    return tuple(sources)


def answer_question(
    vector_store: Chroma,
    question: str,
    chat_history: Iterable[tuple[str, str]] | str | None = None,
    k: int | None = None,
) -> QAResult:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")

    documents = retrieve_documents(
        vector_store,
        normalized_question,
        k=k,
    )
    context = format_context(documents)
    if isinstance(chat_history, str):
        history = chat_history.strip() or "No previous conversation."
    else:
        history = build_chat_history(chat_history or [])

    messages = qa_prompt.format_messages(
        context=context,
        chat_history=history,
        question=normalized_question,
    )
    response = create_chat_model(max_completion_tokens=2048).invoke(messages)
    answer = _content_to_text(response.content).strip()
    if not answer:
        raise RuntimeError(
            "The chat model returned no final answer. Please retry the question."
        )
    return QAResult(answer=answer, sources=_build_sources(documents))


def stream_answer_question(
    vector_store: Chroma,
    question: str,
    chat_history: Iterable[tuple[str, str]] | str | None = None,
    k: int | None = None,
) -> Iterator[str | SourceCitation]:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")

    documents = retrieve_documents(vector_store, normalized_question, k=k)
    yield from _build_sources(documents)

    context = format_context(documents)
    if isinstance(chat_history, str):
        history = chat_history.strip() or "No previous conversation."
    else:
        history = build_chat_history(chat_history or [])

    messages = qa_prompt.format_messages(
        context=context,
        chat_history=history,
        question=normalized_question,
    )
    for chunk in create_chat_model().stream(messages):
        text = _content_to_text(chunk.content)
        if text:
            yield text

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from pdf_reader.config import get_settings
from pdf_reader.document import load_pdf, split_documents
from pdf_reader.qa import SourceCitation, answer_question
from pdf_reader.vectorstore import create_embeddings, create_vector_store


def _safe_pdf_name(file_name: str) -> str:
    name = Path(file_name).name
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if not name:
        raise ValueError("The uploaded PDF must have a file name.")
    return name


def save_uploaded_pdf(file_name: str, content: bytes) -> Path:
    name = _safe_pdf_name(file_name)
    if not content:
        raise ValueError("The uploaded PDF is empty.")

    settings = get_settings()
    upload_directory = settings.upload_directory
    upload_directory.mkdir(parents=True, exist_ok=True)
    path = upload_directory / name
    path.write_bytes(content)
    return path


def _collection_name_for_pdf(file_name: str, content_digest: str) -> str:
    _safe_pdf_name(file_name)
    digest_prefix = hashlib.sha256(content_digest.encode()).hexdigest()[:12]
    return f"pdf_{digest_prefix}"


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("uploaded_file_name", None)
    st.session_state.setdefault("uploaded_file_signature", None)


def _chat_history_for_qa(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(message["role"], message["content"]) for message in messages]


def _render_sources(sources: tuple[SourceCitation, ...]) -> None:
    if not sources:
        st.caption("No source fragments were retrieved.")
        return

    with st.expander("Retrieved sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            location = source.file_name or "Unknown file"
            if source.page is not None:
                location = f"{location}, page {source.page}"
            st.markdown(f"**Source {index}: {location}**")
            st.write(source.text)


def _process_pdf(uploaded_file: Any) -> bool:
    try:
        file_name = _safe_pdf_name(uploaded_file.name)
        content = uploaded_file.getvalue()
        content_digest = hashlib.sha256(content).hexdigest()
        signature = (file_name, content_digest)
        if st.session_state.uploaded_file_signature == signature:
            return True

        with st.spinner("Processing PDF and building the vector index..."):
            saved_path = save_uploaded_pdf(file_name, content)
            pages = load_pdf(saved_path)
            if not pages:
                raise ValueError("No readable text was found in the PDF.")

            chunks = split_documents(pages)
            embeddings = create_embeddings()
            vector_store = create_vector_store(
                chunks,
                embeddings=embeddings,
                collection_name=_collection_name_for_pdf(
                    file_name,
                    content_digest,
                ),
            )

        st.session_state.vector_store = vector_store
        st.session_state.uploaded_file_name = file_name
        st.session_state.uploaded_file_signature = signature
        st.session_state.messages = []
        st.success(
            f"Processed {file_name}: {len(pages)} page(s), {len(chunks)} text chunk(s)."
        )
        return True
    except Exception as error:
        st.session_state.vector_store = None
        st.session_state.uploaded_file_name = None
        st.session_state.uploaded_file_signature = None
        st.session_state.messages = []
        st.error(f"Could not process the PDF: {error}")
        return False


def run_app() -> None:
    st.set_page_config(
        page_title="Interactive PDF Reader",
        page_icon="PDF",
        layout="wide",
    )
    initialize_session_state()

    st.title("Interactive PDF Reader")
    st.caption("Upload a PDF, build a local vector index, and ask questions.")

    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            _process_pdf(uploaded_file)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.caption("Sources were included with this answer.")

    if st.session_state.vector_store is None:
        st.info("Upload a PDF to start asking questions.")
        return

    question = st.chat_input("Ask a question about the uploaded PDF")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
                result = answer_question(
                    st.session_state.vector_store,
                    question,
                    chat_history=_chat_history_for_qa(st.session_state.messages[:-1]),
                )
            st.markdown(result.answer)
            _render_sources(result.sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": [
                    {
                        "file_name": source.file_name,
                        "page": source.page,
                        "text": source.text,
                    }
                    for source in result.sources
                ],
            }
        )
    except Exception as error:
        with st.chat_message("assistant"):
            st.error(f"Could not answer the question: {error}")


if __name__ == "__main__":
    run_app()

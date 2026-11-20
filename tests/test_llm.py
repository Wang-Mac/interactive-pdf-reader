from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from pdf_reader.config import Settings
from pdf_reader.llm import chat, create_chat_model, stream_chat


def create_settings(api_key: str | None = "test-key") -> Settings:
    return Settings(
        nvidia_api_key=api_key,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_chat_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_embedding_model="nvidia/nemotron-3-embed-1b",
        chroma_persist_directory="/tmp/chroma",
        upload_directory="/tmp/uploads",
        chunk_size=1000,
        chunk_overlap=200,
        retrieval_k=4,
    )


@patch("pdf_reader.llm.client.ChatNVIDIA")
def test_create_chat_model_passes_nvidia_configuration(chat_nvidia: MagicMock) -> None:
    settings = create_settings()

    create_chat_model(
        settings,
        temperature=0.2,
        max_completion_tokens=512,
    )

    chat_nvidia.assert_called_once_with(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.2,
        max_completion_tokens=512,
    )


def test_create_chat_model_requires_api_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        create_chat_model(create_settings(api_key=None))


@patch("pdf_reader.llm.client.create_chat_model")
def test_chat_returns_text_content(create_model: MagicMock) -> None:
    model = create_model.return_value
    model.invoke.return_value = AIMessage(content="Mock answer")

    answer = chat("Explain this document")

    assert answer == "Mock answer"
    model.invoke.assert_called_once_with("Explain this document")


@patch("pdf_reader.llm.client.create_chat_model")
def test_stream_chat_yields_text_chunks(create_model: MagicMock) -> None:
    model = create_model.return_value
    model.stream.return_value = iter(
        [
            AIMessage(content="Hello "),
            AIMessage(content="world"),
        ]
    )

    chunks = list(stream_chat("Say hello"))

    assert chunks == ["Hello ", "world"]

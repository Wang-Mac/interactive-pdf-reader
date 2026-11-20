from pathlib import Path

import pytest

from pdf_reader.config import Settings


def test_settings_load_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env({}, project_root=tmp_path)

    assert settings.nvidia_api_key is None
    assert settings.nvidia_chat_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert settings.nvidia_embedding_model == "nvidia/nemotron-3-embed-1b"
    assert settings.chroma_persist_directory == tmp_path / "data/chroma"
    assert settings.upload_directory == tmp_path / "data/uploads"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_k == 4


def test_settings_accepts_overrides(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_CHAT_MODEL": "test-chat",
            "NVIDIA_EMBEDDING_MODEL": "test-embedding",
            "CHROMA_PERSIST_DIRECTORY": "./custom/chroma",
            "UPLOAD_DIRECTORY": "./custom/uploads",
            "CHUNK_SIZE": "800",
            "CHUNK_OVERLAP": "100",
            "RETRIEVAL_K": "6",
        },
        project_root=tmp_path,
    )

    assert settings.require_nvidia_api_key() == "test-key"
    assert settings.nvidia_chat_model == "test-chat"
    assert settings.nvidia_embedding_model == "test-embedding"
    assert settings.chroma_persist_directory == tmp_path / "custom/chroma"
    assert settings.upload_directory == tmp_path / "custom/uploads"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.retrieval_k == 6


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CHUNK_SIZE", "0"),
        ("CHUNK_OVERLAP", "-1"),
        ("CHUNK_OVERLAP", "1000"),
        ("RETRIEVAL_K", "0"),
        ("CHUNK_SIZE", "not-a-number"),
    ],
)
def test_settings_rejects_invalid_values(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        Settings.from_env({name: value}, project_root=tmp_path)

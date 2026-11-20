import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Settings:
    nvidia_api_key: str | None
    nvidia_base_url: str
    nvidia_chat_model: str
    nvidia_embedding_model: str
    chroma_persist_directory: Path
    upload_directory: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        project_root: Path | None = None,
    ) -> "Settings":
        root = (project_root or PROJECT_ROOT).resolve()

        if environ is None:
            load_dotenv(root / ".env")
            values: Mapping[str, str] = os.environ
        else:
            values = environ

        def text(name: str, default: str) -> str:
            value = values.get(name, default).strip()
            return value or default

        def integer(name: str, default: int) -> int:
            raw_value = text(name, str(default))
            try:
                return int(raw_value)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer: {raw_value}") from error

        def resolve_path(name: str, default: str) -> Path:
            path = Path(text(name, default)).expanduser()
            if not path.is_absolute():
                path = root / path
            return path.resolve()

        nvidia_api_key = values.get("NVIDIA_API_KEY", "").strip() or None

        settings = cls(
            nvidia_api_key=nvidia_api_key,
            nvidia_base_url=text(
                "NVIDIA_BASE_URL",
                "https://integrate.api.nvidia.com/v1",
            ),
            nvidia_chat_model=text(
                "NVIDIA_CHAT_MODEL",
                "nvidia/nemotron-3.5-lightning-30b-a3b",
            ),
            nvidia_embedding_model=text(
                "NVIDIA_EMBEDDING_MODEL",
                "nvidia/nemotron-3-embed-1b",
            ),
            chroma_persist_directory=resolve_path(
                "CHROMA_PERSIST_DIRECTORY",
                "./data/chroma",
            ),
            upload_directory=resolve_path(
                "UPLOAD_DIRECTORY",
                "./data/uploads",
            ),
            chunk_size=integer("CHUNK_SIZE", 1000),
            chunk_overlap=integer("CHUNK_OVERLAP", 200),
            retrieval_k=integer("RETRIEVAL_K", 4),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_k <= 0:
            raise ValueError("RETRIEVAL_K must be greater than zero")
        if not self.nvidia_base_url:
            raise ValueError("NVIDIA_BASE_URL must not be empty")

    def require_nvidia_api_key(self) -> str:
        if not self.nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY is required. Copy .env.example to .env and set it."
            )
        return self.nvidia_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()

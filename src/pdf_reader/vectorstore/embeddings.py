from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from pdf_reader.config import Settings, get_settings


def create_embeddings(settings: Settings | None = None) -> NVIDIAEmbeddings:
    resolved_settings = settings or get_settings()
    api_key = resolved_settings.require_nvidia_api_key()
    return NVIDIAEmbeddings(
        model=resolved_settings.nvidia_embedding_model,
        nvidia_api_key=api_key,
        base_url=resolved_settings.nvidia_base_url,
    )

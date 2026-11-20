from collections.abc import Iterator

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from pdf_reader.config import Settings, get_settings


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


def create_chat_model(
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
    max_completion_tokens: int | None = None,
) -> ChatNVIDIA:
    resolved_settings = settings or get_settings()
    api_key = resolved_settings.require_nvidia_api_key()

    kwargs: dict[str, object] = {
        "model": resolved_settings.nvidia_chat_model,
        "nvidia_api_key": api_key,
        "base_url": resolved_settings.nvidia_base_url,
        "temperature": temperature,
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    return ChatNVIDIA(**kwargs)


def chat(
    prompt: str,
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
    max_completion_tokens: int | None = None,
) -> str:
    model = create_chat_model(
        settings,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    response = model.invoke(prompt)
    return _content_to_text(response.content)


def stream_chat(
    prompt: str,
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
    max_completion_tokens: int | None = None,
) -> Iterator[str]:
    model = create_chat_model(
        settings,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    for chunk in model.stream(prompt):
        if chunk.content:
            text = _content_to_text(chunk.content)
            if text:
                yield text

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


def open_chat_model(provider: str, model: str) -> BaseChatModel:
    if provider == "ollama":
        return ChatOllama(model=model, reasoning=True, temperature=0)
    raise ValueError(f"unknown LLM provider {provider!r}; expected: ollama")

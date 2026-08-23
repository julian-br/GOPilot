from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def open_chat_model(provider: str, model: str) -> BaseChatModel:
    if provider == "ollama":
        return ChatOpenAI(
            model=model,
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    raise ValueError(f"unknown LLM provider {provider!r}; expected: ollama")

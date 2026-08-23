import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from src.paths import ROOT


OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def open_chat_model(provider: str, model: str) -> BaseChatModel:
    if provider == "ollama":
        return ChatOpenAI(
            model=model,
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            temperature=0,
        )
    if provider == "openrouter":
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set; add it to .env")
        return ChatOpenRouter(
            model=model,
            temperature=0,
            max_retries=2,
        )
    raise ValueError(f"unknown LLM provider {provider!r}; expected: ollama, openrouter")

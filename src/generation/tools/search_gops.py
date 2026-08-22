from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from src.generation.prompt_inputs import format_candidates


def build_search_gops_tool(retriever: BaseRetriever) -> BaseTool:
    @tool
    def search_gops(query: str) -> str:
        """Search the EBM catalogue for GOPs matching a medical dictation or service."""
        return format_candidates(retriever.invoke(query))

    return search_gops

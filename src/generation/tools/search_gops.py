from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from src.generation.prompt_inputs import format_candidates


def build_search_gops_tool(retriever: BaseRetriever) -> BaseTool:
    @tool
    def search_gops(query: str) -> str:
        """Search the EBM catalogue with a short, focused German billing scenario or question.

        Formulate every query independently from the dictation as a complete, precise statement.
        Do not use isolated keywords, keyword lists, or the complete dictation. Search separately
        for a documented service, a Pauschale or Zuschlag, a contact situation, or an exclusion.
        """
        return format_candidates(retriever.invoke(query))

    return search_gops

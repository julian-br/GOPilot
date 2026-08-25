from langchain_core.tools import BaseTool, tool
from langchain_qdrant import QdrantVectorStore

from src.db import find_gop
from src.generation.prompt_inputs import format_candidate


def build_get_gop_tool(store: QdrantVectorStore, specialty: str) -> BaseTool:
    @tool
    def get_gop(code: str) -> str:
        """Get the exact EBM catalogue entry for a known five-digit GOP code."""
        document = find_gop(store, code)
        if document is None:
            return f"Keine GOP mit der Ziffer {code} gefunden."

        specialties = document.metadata["specialties"]
        billable = not specialties or specialty in specialties
        return "\n".join(
            [
                format_candidate(document),
                f"Fuer Fachgruppe {specialty} abrechenbar: {'ja' if billable else 'nein'}",
            ]
        )

    return get_gop

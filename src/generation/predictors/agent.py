from typing import Callable

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore

from src.generation.prompt_inputs import format_patient_context
from src.generation.prompts import AGENT_SYSTEM_PROMPT
from src.generation.schemas import RecommendationResult
from src.generation.tools import build_get_gop_tool, build_search_gops_tool
from src.patient import Patient


class AgentPredictor:
    def __init__(
        self,
        model: BaseChatModel,
        store: QdrantVectorStore,
        retriever: BaseRetriever,
        specialty: str,
        quarter: str,
        close_resources: Callable[[], None] = lambda: None,
    ) -> None:
        self._agent = create_agent(
            model=model,
            tools=[
                build_search_gops_tool(retriever),
                build_get_gop_tool(store, specialty),
            ],
            system_prompt=AGENT_SYSTEM_PROMPT,
            response_format=ToolStrategy(
                RecommendationResult,
                handle_errors=(
                    "Die letzte Empfehlung ist ungueltig. Jeder GOP-Code muss genau "
                    "fuenf Ziffern haben. Gib nur bestaetigte GOPs zurueck."
                ),
            ),
            name="billing_agent",
        )
        self._quarter = quarter
        self._close_resources = close_resources

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationResult:
        patient_context = format_patient_context(
            patient, current_quarter=self._quarter
        )
        output = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Abrechnungsquartal: {self._quarter}\n\n"
                            f"Diktat:\n{dictation}\n\n"
                            "Patientenkontext:\n"
                            f"{patient_context}"
                        ),
                    }
                ]
            }
        )
        result = output.get("structured_response")
        if not isinstance(result, RecommendationResult):
            raise ValueError("agent did not return a valid structured response")
        return result

    def close(self) -> None:
        self._close_resources()

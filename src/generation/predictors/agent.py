from typing import Callable

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore

from src.generation.prompt_inputs import format_patient_context
from src.generation.prompts import AGENT_SYSTEM_PROMPT
from src.generation.schemas import RecommendationResult, RecommendationRun
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
            response_format=ToolStrategy(RecommendationResult),
            name="billing_agent",
        )
        self._quarter = quarter
        self._close_resources = close_resources

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationRun:
        output = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Abrechnungsquartal: {self._quarter}\n\n"
                            f"Diktat:\n{dictation}\n\n"
                            f"Patientenkontext:\n{format_patient_context(patient)}"
                        ),
                    }
                ]
            }
        )
        reasoning_parts = [
            message.additional_kwargs.get("reasoning_content")
            for message in output["messages"]
            if isinstance(message, AIMessage)
        ]
        reasoning = "\n\n".join(part for part in reasoning_parts if part) or None
        if "structured_response" not in output:
            raise ValueError("agent did not return a structured response")
        return RecommendationRun(
            result=output["structured_response"],
            reasoning=reasoning,
        )

    def close(self) -> None:
        self._close_resources()

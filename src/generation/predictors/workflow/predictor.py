"""Workflow: understand services, retrieve two paths, select, and merge."""

from __future__ import annotations

import json
from typing import Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph

from src.generation.prompt_inputs import format_candidates, format_patient_context
from src.generation.predictors.workflow.prompts import (
    FLAT_SELECTION_PROMPT,
    SERVICE_SELECTION_PROMPT,
    UNDERSTANDING_PROMPT,
)
from src.generation.predictors.workflow.schemas import PerformedServices
from src.generation.schemas import Recommendation, RecommendationResult
from src.patient import Patient

FLAT_RATE_QUERIES = (
    "Versichertenpauschale oder Grundpauschale bei Arzt-Patienten-Kontakt",
    "Zuschlag zu einer Pauschale fuer Behandlung und Betreuung",
)

# TODO: Expand this or source from official glossaries
GENERAL_DEFINITIONS = {
    "Behandlungsfall": (
        "Ein Behandlungsfall ist nicht ein einzelner Kontakt. Er umfasst alle "
        "Kontakte desselben Versicherten durch dieselbe Arztpraxis in einem "
        "Kalendervierteljahr zulasten derselben Krankenkasse. Mehrere Kontakte "
        "im selben Quartal erzeugen keinen neuen Behandlungsfall."
    ),
}


class WorkflowState(TypedDict, total=False):
    dictation: str
    context: str
    services: list[str]
    service_candidates: list[tuple[str, list[Document]]]
    flat_candidates: list[Document]
    service_results: list[RecommendationResult]
    flat_result: RecommendationResult
    result: RecommendationResult


def _case_context(patient: Patient | None, quarter: str) -> str:
    return (
        "Praxisart: Hausarztpraxis\n" # TODO: Avoid hardcoding
        f"Abrechnungsquartal: {quarter}\n"
        f"{format_patient_context(patient, current_quarter=quarter)}"
    )


def _add_unique(
    target: list[Recommendation],
    seen: set[str],
    result: RecommendationResult,
) -> None:
    for recommendation in result.recommendations:
        if recommendation.code not in seen:
            seen.add(recommendation.code)
            target.append(recommendation)


class WorkflowPredictor:
    def __init__(
        self,
        model: BaseChatModel,
        retriever: BaseRetriever,
        quarter: str,
        max_services: int,
        max_candidates_per_path: int,
        close_resources: Callable[[], None] = lambda: None,
    ) -> None:
        self._understanding_chain = (
            UNDERSTANDING_PROMPT
            | model.with_structured_output(
                PerformedServices,
                method="function_calling",
            )
        )
        self._service_selection_chain = (
            SERVICE_SELECTION_PROMPT
            | model.with_structured_output(
                RecommendationResult,
                method="function_calling",
            )
        )
        self._flat_rate_selection_chain = (
            FLAT_SELECTION_PROMPT
            | model.with_structured_output(
                RecommendationResult,
                method="function_calling",
            )
        )
        self._retriever = retriever
        self._quarter = quarter
        self._max_services = max_services
        self._max_candidates = max_candidates_per_path
        self._close_resources = close_resources
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState).add_sequence(
            [
                ("understand_services", self._understand_services),
                ("retrieve_services", self._retrieve_service_candidates),
                ("retrieve_flat_rates", self._retrieve_flat_candidates),
                ("select_services", self._select_services),
                ("select_flat_rates", self._select_flat_rates),
                ("merge_recommendations", self._merge_recommendations),
            ]
        )
        graph.set_entry_point("understand_services")
        graph.set_finish_point("merge_recommendations")
        return graph.compile()

    def _understand_services(self, state: WorkflowState) -> WorkflowState:
        result = self._understanding_chain.invoke({"dictation": state["dictation"]})
        services = result.performed_services[: self._max_services]
        return {"services": [service.description for service in services]}

    def _retrieve_service_candidates(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        candidates = [
            (
                service,
                self._retriever.invoke(
                    service,
                    config={"run_name": "retrieve_workflow_service"},
                )[: self._max_candidates],
            )
            for service in state["services"]
        ]
        return {"service_candidates": candidates}

    def _retrieve_flat_candidates(self, _: WorkflowState) -> WorkflowState:
        candidates: list[Document] = []
        seen: set[str] = set()
        for query in FLAT_RATE_QUERIES:
            for candidate in self._retriever.invoke(
                query,
                config={"run_name": "retrieve_workflow_flat_rate"},
            ):
                code = str(candidate.metadata["code"])
                title = candidate.page_content.splitlines()[0].casefold()
                if code in seen or not any(word in title for word in ("pauschale", "zuschlag")):
                    continue
                seen.add(code)
                candidates.append(candidate)
                if len(candidates) >= self._max_candidates:
                    return {"flat_candidates": candidates}
        return {"flat_candidates": candidates}

    def _select_services(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        results: list[RecommendationResult] = []
        for service, candidates in state["service_candidates"]:
            result = self._service_selection_chain.invoke(
                {
                    "case_context": state["context"],
                    "dictation": state["dictation"],
                    "service": service,
                    "candidates": format_candidates(candidates),
                }
            )
            results.append(result)
        return {"service_results": results}

    def _select_flat_rates(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        candidates = state["flat_candidates"]
        if not candidates:
            return {"flat_result": RecommendationResult(recommendations=[])}
        result = self._flat_rate_selection_chain.invoke(
            {
                "case_context": state["context"],
                "definitions": json.dumps(
                    GENERAL_DEFINITIONS,
                    ensure_ascii=False,
                    indent=2,
                ),
                "dictation": state["dictation"],
                "candidates": format_candidates(candidates),
            }
        )
        return {"flat_result": result}

    def _merge_recommendations(self, state: WorkflowState) -> WorkflowState:
        recommendations: list[Recommendation] = []
        seen: set[str] = set()
        for result in state["service_results"]:
            _add_unique(recommendations, seen, result)
        _add_unique(recommendations, seen, state["flat_result"])
        return {"result": RecommendationResult(recommendations=recommendations)}

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationResult:
        state = self._graph.invoke(
            {
                "dictation": dictation,
                "context": _case_context(patient, self._quarter),
            },
            config={"run_name": "workflow"},
        )
        return state["result"]

    def close(self) -> None:
        self._close_resources()

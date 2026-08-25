"""Workflow: understand services, retrieve two paths, select, and merge."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

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


def _case_context(patient: Patient | None, quarter: str) -> str:
    return (
        "Praxisart: Hausarztpraxis\n" # TODO: Avoid hardcoding
        f"Abrechnungsquartal: {quarter}\n"
        f"{format_patient_context(patient, current_quarter=quarter)}"
    )


def _add_offered(
    target: list[Recommendation],
    seen: set[str],
    result: RecommendationResult,
    candidates: list[Document],
) -> None:
    offered = {str(candidate.metadata["code"]) for candidate in candidates}
    for recommendation in result.recommendations:
        if recommendation.code in offered and recommendation.code not in seen:
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

    def _understand_services(self, dictation: str) -> list[str]:
        result = self._understanding_chain.invoke({"dictation": dictation})
        if not isinstance(result, PerformedServices):
            raise TypeError("workflow did not return performed services")
        descriptions: list[str] = []
        seen: set[str] = set()
        for service in result.performed_services:
            description = " ".join(service.description.split())
            key = description.casefold()
            if description and key not in seen:
                seen.add(key)
                descriptions.append(description)
            if len(descriptions) >= self._max_services:
                break
        return descriptions

    def _retrieve_service_candidates(
        self,
        services: list[str],
    ) -> list[tuple[str, list[Document]]]:
        return [
            (
                service,
                self._retriever.invoke(
                    service,
                    config={"run_name": "retrieve_workflow_service"},
                )[: self._max_candidates],
            )
            for service in services
        ]

    def _retrieve_flat_candidates(self) -> list[Document]:
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
                    return candidates
        return candidates

    def _select_services(
        self,
        service_candidates: list[tuple[str, list[Document]]],
        dictation: str,
        context: str,
    ) -> list[RecommendationResult]:
        def select(item: tuple[str, list[Document]]) -> RecommendationResult:
            service, candidates = item
            result = self._service_selection_chain.invoke(
                {
                    "case_context": context,
                    "dictation": dictation,
                    "service": service,
                    "candidates": format_candidates(candidates),
                }
            )
            if not isinstance(result, RecommendationResult):
                raise TypeError("workflow did not return service recommendations")
            return result

        with ThreadPoolExecutor(max_workers=min(5, len(service_candidates))) as executor:
            return list(executor.map(select, service_candidates))

    def _select_flat_rates(
        self,
        candidates: list[Document],
        dictation: str,
        context: str,
    ) -> RecommendationResult:
        result = self._flat_rate_selection_chain.invoke(
            {
                "case_context": context,
                "definitions": json.dumps(
                    GENERAL_DEFINITIONS,
                    ensure_ascii=False,
                    indent=2,
                ),
                "dictation": dictation,
                "candidates": format_candidates(candidates),
            }
        )
        if not isinstance(result, RecommendationResult):
            raise TypeError("workflow did not return flat-rate recommendations")
        return result

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationResult:
        services = self._understand_services(dictation)
        service_candidates = self._retrieve_service_candidates(services)
        flat_candidates = self._retrieve_flat_candidates()
        context = _case_context(patient, self._quarter)

        with ThreadPoolExecutor(max_workers=2) as executor:
            services_future = (
                executor.submit(
                    self._select_services,
                    service_candidates,
                    dictation,
                    context,
                )
                if service_candidates
                else None
            )
            flat_future = (
                executor.submit(
                    self._select_flat_rates,
                    flat_candidates,
                    dictation,
                    context,
                )
                if flat_candidates
                else None
            )

            service_results = services_future.result() if services_future else []
            flat_result = flat_future.result() if flat_future else None

        recommendations: list[Recommendation] = []
        seen: set[str] = set()
        for result, (_, candidates) in zip(service_results, service_candidates):
            _add_offered(recommendations, seen, result, candidates)
        if flat_result is not None:
            _add_offered(recommendations, seen, flat_result, flat_candidates)
        return RecommendationResult(recommendations=recommendations)

    def close(self) -> None:
        self._close_resources()

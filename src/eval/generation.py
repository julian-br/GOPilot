"""How well does a generation strategy recommend expected GOPs?"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import mlflow.langchain
import mlflow.openai
from mlflow.entities import SpanStatusCode

from src.config import DEFAULT_CONFIG, Config, RetrievalConfig, load_config
from src.eval.cases import Case, load_cases, require_catalogue_quarter
from src.generation.predictors import Predictor, build_predictor

CaseResults = dict[str, dict[str, Any]]


def evaluate_case(
    predictor: Predictor,
    case: Case,
    index: int,
    case_count: int,
) -> tuple[str, dict[str, Any]]:
    start = perf_counter()
    with mlflow.start_span(
        "generation_case",
        attributes={
            "case_id": case.case_id,
            "case_index": index,
            "cases": case_count,
            "patient_id": case.patient.id,
            "expected": list(case.expected),
        },
    ) as span:
        error = None
        try:
            result = predictor.predict(case.dictation, case.patient)
            predictions = [
                {"code": recommendation.code, "reason": recommendation.reason}
                for recommendation in result.recommendations
            ]
        except Exception as exc:
            predictions = []
            error = f"{type(exc).__name__}: {exc}"
            span.record_exception(exc)
            span.set_status(SpanStatusCode.ERROR)

        predicted = {p["code"] for p in predictions}
        expected = set(case.expected)
        case_result = {
            "expected": list(case.expected),
            "predicted": predictions,
            "true_positive": [code for code in case.expected if code in predicted],
            "false_positive": [
                p["code"] for p in predictions if p["code"] not in expected
            ],
            "false_negative": [code for code in case.expected if code not in predicted],
        }
        if error is not None:
            case_result["error"] = error
        elapsed = perf_counter() - start
        span.set_outputs(case_result)
        span.set_attribute("duration_seconds", elapsed)
    status = " ERROR" if error is not None else ""
    print(
        f"[{index}/{case_count}] {case.case_id} {elapsed:.1f}s{status}",
        flush=True,
    )
    return case.case_id, case_result


def evaluate_cases(
    predictor: Predictor,
    cases: list[Case],
    parallel: bool = False,
) -> CaseResults:
    indexed_cases = list(enumerate(cases, 1))
    if not indexed_cases:
        return {}
    if not parallel:
        return dict(
            evaluate_case(predictor, case, index, len(cases))
            for index, case in indexed_cases
        )

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        futures = [
            executor.submit(evaluate_case, predictor, case, index, len(cases))
            for index, case in indexed_cases
        ]
        completed = dict(future.result() for future in as_completed(futures))
    return {case.case_id: completed[case.case_id] for case in cases}


def classification_metrics(results: CaseResults) -> dict[str, float]:
    tp = sum(len(result["true_positive"]) for result in results.values())
    fp = sum(len(result["false_positive"]) for result in results.values())
    fn = sum(len(result["false_negative"]) for result in results.values())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "generation_precision": precision,
        "generation_recall": recall,
        "generation_f1": f1,
    }


def main(
    config: Config,
    limit: int | None = None,
    parallel: bool = False,
) -> dict[str, float]:
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]
    require_catalogue_quarter(cases, config.catalogue_quarter)

    mlflow.set_experiment(config.experiment)
    strategy = (
        "rag-rerank"
        if isinstance(config, RetrievalConfig)
        and config.generation_strategy == "rag"
        and config.reranker
        else config.generation_strategy
    )
    run_name = f"{strategy}-{config.llm_model}"
    with mlflow.start_run(run_name=run_name):
        mlflow.langchain.autolog(log_traces=True)
        mlflow.openai.autolog(log_traces=True)
        config_values = config.model_dump()
        mlflow.log_params(config_values)
        mlflow.log_dict(config_values, "config.json")

        predictor = build_predictor(config)
        try:
            results = evaluate_cases(predictor, cases, parallel=parallel)
        finally:
            predictor.close()
        errors = sum("error" in result for result in results.values())
        metrics = classification_metrics(results) | {
            "generation_error_rate": errors / len(results) if results else 0.0
        }

        mlflow.log_param("cases", len(cases))
        mlflow.log_param(
            "cases_with_expected", sum(1 for case in cases if case.expected)
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(results, "generation_cases.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="evaluate all selected cases concurrently",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name, value in main(
        load_config(args.config), limit=args.limit, parallel=args.parallel
    ).items():
        print(f"  {name:<16} {value:.3f}")

"""How well does a generation strategy recommend expected GOPs?"""

import argparse
from dataclasses import asdict
from time import perf_counter
from typing import Any

import mlflow
import mlflow.langchain
from mlflow.entities import SpanStatusCode

from src.config import Config, load_config
from src.db import get_patient
from src.eval.cases import Case, load_cases
from src.generation.predictors import Predictor, build_predictor

CaseResults = dict[str, dict[str, Any]]


def evaluate_cases(predictor: Predictor, cases: list[Case]) -> CaseResults:
    results = {}
    for index, case in enumerate(cases, 1):
        start = perf_counter()
        with mlflow.start_span(
            "generation_case",
            attributes={
                "case_id": case.case_id,
                "case_index": index,
                "cases": len(cases),
                "patient_id": case.patient_id,
                "quarter": case.quarter,
                "expected": list(case.expected),
            },
        ) as span:
            error = None
            try:
                patient = get_patient(case.patient_id, case.quarter)
                run = predictor.predict(case.dictation, patient)
                predictions = [
                    {"rank": i, "code": r.code, "reason": r.reason}
                    for i, r in enumerate(run.result.recommendations, 1)
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
            results[case.case_id] = case_result
        status = " ERROR" if error is not None else ""
        print(
            f"[{index}/{len(cases)}] {case.case_id} {elapsed:.1f}s{status}",
            flush=True,
        )
    return results


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


def main(config: Config, limit: int | None = None) -> dict[str, float]:
    mlflow.set_experiment(config.experiment)
    run_name = f"{config.generation_strategy}-{config.llm_model}"
    with mlflow.start_run(run_name=run_name):
        mlflow.langchain.autolog(log_traces=True)
        mlflow.log_params(asdict(config))
        mlflow.log_dict(asdict(config), "config.json")

        cases = load_cases()
        if limit is not None:
            cases = cases[:limit]
        predictor = build_predictor(config)
        try:
            results = evaluate_cases(predictor, cases)
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
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name, value in main(load_config(), limit=args.limit).items():
        print(f"  {name:<16} {value:.3f}")

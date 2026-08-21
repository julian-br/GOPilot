"""How well does the LLM alone recommend expected GOPs?"""

import argparse
from dataclasses import asdict
from time import perf_counter
from typing import Any

import mlflow
import mlflow.langchain

from src.config import Config, load_config
from src.db import get_patient
from src.eval.cases import Case, load_cases
from src.generation.predictors import Predictor, build_predictor

CUTOFFS = (1, 5, 10)

CaseResults = dict[str, dict[str, Any]]
Ranks = dict[str, dict[str, int | None]]


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
            patient = get_patient(case.patient_id, case.quarter)
            run = predictor.predict(case.dictation, patient)
            predictions = [
                {"rank": i, "code": r.code, "reason": r.reason}
                for i, r in enumerate(run.result.recommendations, 1)
            ]
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
            elapsed = perf_counter() - start
            span.set_outputs(case_result)
            span.set_attribute("duration_seconds", elapsed)
            results[case.case_id] = case_result
        print(f"[{index}/{len(cases)}] {case.case_id} {elapsed:.1f}s", flush=True)
    return results


def ranks_per_case(results: CaseResults) -> Ranks:
    ranks = {}
    for case_id, result in results.items():
        position = {p["code"]: p["rank"] for p in result["predicted"]}
        ranks[case_id] = {code: position.get(code) for code in result["expected"]}
    return ranks


def recall_at(ranks: Ranks, k: int) -> float:
    expected = [r for case in ranks.values() for r in case.values()]
    if not expected:
        return 0.0
    return sum(1 for r in expected if r is not None and r <= k) / len(expected)


def classification_metrics(results: CaseResults) -> dict[str, float]:
    tp = sum(len(result["true_positive"]) for result in results.values())
    fp = sum(len(result["false_positive"]) for result in results.values())
    fn = sum(len(result["false_negative"]) for result in results.values())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "llm_precision": precision,
        "llm_recall": recall,
        "llm_f1": f1,
    }


def main(config: Config, limit: int | None = None) -> dict[str, float]:
    with mlflow.start_run(run_name=config.experiment):
        mlflow.langchain.autolog(log_traces=True)
        mlflow.log_params(asdict(config))
        mlflow.log_dict(asdict(config), "config.json")

        cases = load_cases()
        if limit is not None:
            cases = cases[:limit]
        predictor = build_predictor(config)
        results = evaluate_cases(predictor, cases)
        ranks = ranks_per_case(results)
        metrics = classification_metrics(results) | {
            f"llm_recall_at_{k}": recall_at(ranks, k) for k in CUTOFFS
        }

        mlflow.log_param("cases", len(cases))
        mlflow.log_param(
            "cases_with_expected", sum(1 for case in cases if case.expected)
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(results, "llm_cases.json")
        mlflow.log_dict(ranks, "llm_ranks.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name, value in main(load_config(), limit=args.limit).items():
        print(f"  {name:<16} {value:.3f}")

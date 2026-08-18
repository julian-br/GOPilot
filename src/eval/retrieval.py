"""How well does retrieval alone surface the expected GOPs?"""

from dataclasses import asdict

import mlflow
from langchain_chroma import Chroma

from src.config import Config, load_config
from src.db import open_index
from src.eval.cases import Case, load_cases
from src.retrieval import find_candidates

CUTOFFS = (1, 5, 10, 20, 50)

Ranks = dict[str, dict[str, int | None]]


def ranks_per_case(index: Chroma, cases: list[Case], specialty: str, depth: int) -> Ranks:
    """Rank of each expected GOP in the candidate list, or None if it never showed up."""
    ranks = {}
    for case in cases:
        found = find_candidates(index, case.dictation, specialty, depth)
        position = {d.metadata["code"]: i for i, d in enumerate(found, 1)}
        ranks[case.case_id] = {gop: position.get(gop) for gop in case.expected}
    return ranks


def recall_at(ranks: Ranks, k: int) -> float:
    expected = [r for case in ranks.values() for r in case.values()]
    return sum(1 for r in expected if r is not None and r <= k) / len(expected)


def main(config: Config) -> dict[str, float]:
    cases = [c for c in load_cases() if c.expected]
    index = open_index(config.embedding_model)
    cutoffs = tuple(sorted({*CUTOFFS, config.top_k}))
    ranks = ranks_per_case(index, cases, config.practice_specialty, max(cutoffs))
    metrics = {f"recall_at_{k}": recall_at(ranks, k) for k in cutoffs}

    with mlflow.start_run(run_name=config.experiment):
        mlflow.log_params(asdict(config))
        mlflow.log_param("cases", len(cases))
        mlflow.log_metrics(metrics)
        mlflow.log_dict(ranks, "ranks.json")
    return metrics


if __name__ == "__main__":
    for name, value in main(load_config()).items():
        print(f"  {name:<12} {value:.3f}")

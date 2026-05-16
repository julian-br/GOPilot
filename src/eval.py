"""
Evaluation pipeline: compares GOP recommendation quality across conditions.

Conditions (configured in YAML):
  basic   = dictation + patient data from SQLite
  rag     = dictation + patient data + ChromaDB top-k
  agent   = tool-calling agent with ChromaDB tools

Usage:
    python -m src.eval
    python -m src.eval --config configs/default.yaml --verbose
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Literal

os.environ.pop("SSLKEYLOGFILE", None)

import chromadb
import yaml

from src.agent import run_agent
from src.db import get_patient_context
from src.inference import ask_llm, build_prompt, parse_gops
from src.ingest import CHROMA_PATH, COLLECTION_NAME, OllamaEmbedder

CASES_DIR = Path("data/test_dictations")
REPORTS_DIR = Path("reports")

Condition = Literal["basic", "rag", "agent"]


@dataclass
class EvalConfig:
    experiment: str = "default"
    model: str = "qwen3.5:9b"
    top_k: int = 10
    thinking: bool = False
    practice_fachgruppe: str = "Hausärztlicher Versorgungsbereich"
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    conditions: list[Condition] = field(default_factory=lambda: ["basic", "rag", "agent"])

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_cases() -> list[dict]:
    return sorted(
        [json.loads(p.read_text(encoding="utf-8")) for p in CASES_DIR.glob("case_*.json")],
        key=lambda c: c["case_id"],
    )


def patient_summary(patient_id: str, quartal: str, already_billed_gops: list[str] | None = None) -> str:
    ctx = get_patient_context(patient_id, quartal)
    if ctx is None:
        return f"Patient {patient_id} (unbekannt)"
    billed = ctx["gops_already_billed"] if already_billed_gops is None else already_billed_gops
    already = ", ".join(billed) or "keine"
    return (
        f"{ctx['name']}, {ctx['age']} Jahre, {ctx['gender']}, {ctx['insurance']}, "
        f"Quartal {quartal}, bereits abgerechnet: {already}"
    )


def retrieve_gops(col, query: str, n: int) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    res = col.query(query_texts=[query], n_results=n, include=["documents", "metadatas"])
    for meta, doc in zip(res["metadatas"][0], res["documents"][0]):
        gop = meta["gop"]
        if gop not in seen:
            seen.add(gop)
            lines.append(f"  GOP {gop}: {doc[:220]}")
    return "\n".join(lines)


def metrics(predicted: list[str], expected: list[str]) -> dict:
    pred_set, exp_set = set(predicted), set(expected)
    if not exp_set and not pred_set:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(exp_set) if exp_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def run_condition(
    case: dict,
    condition: Condition,
    col,
    cfg: EvalConfig,
) -> dict:
    dictation = case["dictation"]
    if condition == "agent":
        result = run_agent(
            dictation=dictation,
            patient_id=case["patient_id"],
            quartal=case["quartal"],
            model=cfg.model,
            think=cfg.thinking,
            practice_fachgruppe=cfg.practice_fachgruppe,
            reranker_model=cfg.reranker_model,
            already_billed_gops=case.get("already_billed_gops"),
        )
        raw = result["response"]
        predicted = parse_gops(raw)
        return {
            "predicted": predicted,
            "raw_response": raw,
            "search_plan": result.get("search_plan", []),
            "hypothetical_document": result.get("hypothetical_document"),
            "reranker": result.get("reranker"),
            "practice_fachgruppe": cfg.practice_fachgruppe,
            "tool_log": result["tool_log"],
            "steps": result["steps"],
            "error": result.get("error"),
            "metrics": metrics(predicted, case["expected_gops"]),
        }

    patient_ctx = patient_summary(
        case["patient_id"],
        case["quartal"],
        case.get("already_billed_gops"),
    )
    gop_ctx = retrieve_gops(col, dictation, cfg.top_k) if condition == "rag" else None
    raw = ask_llm(build_prompt(dictation, patient_ctx, gop_ctx), model=cfg.model, think=cfg.thinking)
    predicted = parse_gops(raw)
    return {
        "predicted": predicted,
        "raw_response": raw,
        "metrics": metrics(predicted, case["expected_gops"]),
    }


def run(cfg: EvalConfig, verbose: bool = False) -> dict:
    cases = load_cases()
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(
        name=COLLECTION_NAME, embedding_function=OllamaEmbedder()
    )

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['case_id']} ...", flush=True)
        row = {"case_id": case["case_id"], "scenario": case["scenario"], "expected": case["expected_gops"]}
        for cond in cfg.conditions:
            print(f"  {cond} ...", end=" ", flush=True)
            row[cond] = run_condition(case, cond, col, cfg)
            print(f"-> {row[cond]['predicted']}", flush=True)
        rows.append(row)

        if verbose:
            print(f"\n{'='*70}")
            print(f"{case['case_id']}: {case['scenario']}")
            print(f"  Expected : {case['expected_gops']}")
            for cond in cfg.conditions:
                m = row[cond]["metrics"]
                print(f"  {cond:<8}: {row[cond]['predicted']}  (P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f})")

    # Summary
    n = len(rows)
    summary = {
        cond: round(sum(r[cond]["metrics"]["f1"] for r in rows) / n, 4)
        for cond in cfg.conditions
    }

    _print_table(rows, cfg, summary)

    return {
        "experiment": cfg.experiment,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "summary": {**{"n_cases": n}, **{f"avg_f1_{c}": v for c, v in summary.items()}},
        "cases": rows,
    }


def _print_table(rows: list[dict], cfg: EvalConfig, summary: dict) -> None:
    conds = cfg.conditions
    col_w = 9
    header_conds = "  ".join(f"{c.upper()[:7]:>{col_w}}" for c in conds)
    sep = "-" * 90
    print(f"\n{sep}")
    print(f"Experiment: {cfg.experiment}  |  model: {cfg.model}  |  top_k: {cfg.top_k}")
    print(sep)
    print(f"{'Case':<10} {'Expected':<18}  {header_conds}  Scenario")
    print(sep)
    for r in rows:
        f1s = "  ".join(f"{r[c]['metrics']['f1']:>{col_w}.2f}" for c in conds)
        print(f"{r['case_id']:<10} {str(r['expected']):<18}  {f1s}  {r['scenario'][:38]}")
    print(sep)
    avg_line = "  ".join(f"{summary[c]:>{col_w}.2f}" for c in conds)
    print(f"{'AVERAGE':<10} {'':<18}  {avg_line}")
    if len(conds) >= 2:
        deltas = []
        for i in range(1, len(conds)):
            d = summary[conds[i]] - summary[conds[i - 1]]
            deltas.append(f"{conds[i-1]}->{conds[i]}: {d:+.2f}")
        print(f"  Delta  {',  '.join(deltas)}")
    print(f"{sep}\n")


def save_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{report['experiment']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GOPilot across RAG conditions")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to experiment config YAML (default: configs/default.yaml)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    report = run(cfg, verbose=args.verbose)
    path = save_report(report)
    print(f"Report saved: {path}")


if __name__ == "__main__":
    main()

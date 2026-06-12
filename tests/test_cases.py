"""
Structural validation of the test dictation cases.

Checks machine-verifiable properties only (no semantic judgement):
consistency with the patient DB, the billing context, and the EBM catalogue.

Usage:
    python -m unittest tests.test_cases
"""

import json
import os
import re
import unittest
from pathlib import Path

os.environ.pop("SSLKEYLOGFILE", None)

import chromadb

from src.db import get_patient_context
from src.ingest import CHROMA_PATH, COLLECTION_NAME

CASES_DIR = Path("data/test_dictations")
REQUIRED_FIELDS = {
    "case_id", "patient_id", "quartal", "already_billed_gops",
    "scenario", "dictation", "expected_gops", "notes",
}
# Chapters billable by the Hausarzt practice all cases assume:
# own chapter (03) plus arztgruppenübergreifende chapters.
_HAUSARZT_BILLABLE_PREFIXES = ("01", "02", "03", "30", "31", "32", "33", "34", "35", "36", "37", "38", "40")


def load_cases() -> list[dict]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(CASES_DIR.glob("case_*.json"))
    ]


class CaseStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection(name=COLLECTION_NAME)
        cls.catalogue_gops = {
            meta["gop"] for meta in col.get(include=["metadatas"])["metadatas"]
        }

    def test_cases_exist(self):
        self.assertGreater(len(self.cases), 0)

    def test_required_fields_and_id_matches_filename(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                self.assertTrue(REQUIRED_FIELDS.issubset(case.keys()),
                                f"missing: {REQUIRED_FIELDS - case.keys()}")
                self.assertEqual(case["case_id"], path.stem)

    def test_patient_exists_in_db(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                ctx = get_patient_context(case["patient_id"], case["quartal"])
                self.assertIsNotNone(ctx, f"unknown patient {case['patient_id']}")

    def test_quartal_format(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                self.assertRegex(case["quartal"], r"^[1-4]/20\d{2}$")

    def test_gop_codes_are_five_digits_and_unique(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                for gop in case["expected_gops"] + case["already_billed_gops"]:
                    self.assertRegex(gop, r"^\d{5}$")
                self.assertEqual(len(case["expected_gops"]), len(set(case["expected_gops"])))

    def test_expected_not_already_billed(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                overlap = set(case["expected_gops"]) & set(case["already_billed_gops"])
                self.assertFalse(overlap, f"expected GOPs already billed: {overlap}")

    def test_expected_gops_exist_in_catalogue(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                missing = [g for g in case["expected_gops"] if g not in self.catalogue_gops]
                self.assertFalse(missing, f"not in EBM catalogue: {missing}")

    def test_expected_gops_billable_by_hausarzt_practice(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                bad = [
                    g for g in case["expected_gops"]
                    if not g.startswith(_HAUSARZT_BILLABLE_PREFIXES)
                ]
                self.assertFalse(bad, f"not billable by Hausarzt practice: {bad}")

    def test_dictation_age_matches_patient_db(self):
        for path, case in self.cases:
            with self.subTest(case=path.name):
                ctx = get_patient_context(case["patient_id"], case["quartal"])
                m = re.search(r"(\d+)\s+Jahre", case["dictation"])
                if ctx and m:
                    self.assertEqual(int(m.group(1)), ctx["age"],
                                     f"dictation age {m.group(1)} != DB age {ctx['age']}")


if __name__ == "__main__":
    unittest.main()

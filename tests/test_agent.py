import unittest

from src.agent import (
    _candidate_basis_gops,
    _enforce_base_dependencies,
    _parse_judge_response,
    _resolve_exclusions,
)


class ParseJudgeResponseTest(unittest.TestCase):
    def test_select_true_with_evidence(self):
        self.assertTrue(_parse_judge_response('{"evidence": "Spirographie durchgeführt", "select": true}'))

    def test_select_true_without_evidence_is_rejected(self):
        self.assertFalse(_parse_judge_response('{"evidence": "", "select": true}'))
        self.assertFalse(_parse_judge_response('{"select": true}'))

    def test_select_false(self):
        self.assertFalse(_parse_judge_response('{"evidence": "", "select": false}'))

    def test_last_valid_json_wins(self):
        text = 'Beispiel: {"evidence": "x", "select": false}\nFinal: {"evidence": "Infusion intravenös", "select": true}'
        self.assertTrue(_parse_judge_response(text))

    def test_no_json_is_rejection(self):
        self.assertFalse(_parse_judge_response("Die GOP passt, true."))

    def test_surrounding_prose_is_tolerated(self):
        self.assertTrue(_parse_judge_response('Begründung ... {"evidence": "Hausbesuch", "select": true} fertig'))


class ResolveExclusionsTest(unittest.TestCase):
    @staticmethod
    def _candidate(punkte: int, ausschluesse: list[str]) -> dict:
        return {"details": {"punkte": punkte, "ausschluesse": ausschluesse}}

    def test_higher_punkte_wins_on_conflict(self):
        by_gop = {
            "02300": self._candidate(68, ["02301"]),
            "02301": self._candidate(133, ["02300"]),
        }
        self.assertEqual(_resolve_exclusions(["02300", "02301"], by_gop), ["02301"])

    def test_no_conflict_keeps_all_in_rank_order(self):
        by_gop = {
            "03000": self._candidate(225, []),
            "03330": self._candidate(53, []),
        }
        self.assertEqual(_resolve_exclusions(["03000", "03330"], by_gop), ["03000", "03330"])

    def test_one_sided_exclusion_counts(self):
        by_gop = {
            "11111": self._candidate(10, []),
            "22222": self._candidate(99, ["11111"]),
        }
        self.assertEqual(_resolve_exclusions(["11111", "22222"], by_gop), ["22222"])


class BasisGopsTest(unittest.TestCase):
    @staticmethod
    def _cand(gop: str, volltext: str) -> dict:
        return {"gop": gop, "details": {"volltext": volltext}}

    def test_zuschlag_to_single_base(self):
        c = self._cand("03221", "GOP 03221: Zuschlag zu der Gebührenordnungsposition 03220 "
                                 "für die intensive Behandlung Obligater Leistungsinhalt - ...")
        self.assertEqual(_candidate_basis_gops(c), ["03220"])

    def test_im_zusammenhang_mit_multiple_bases(self):
        c = self._cand("01952", "GOP 01952: Zuschlag im Zusammenhang mit den "
                                 "Gebührenordnungspositionen 01949, 01950, 01953 oder 01955 "
                                 "für 154 Punkte das therapeutische Gespräch Obligater Leistungsinhalt")
        self.assertEqual(_candidate_basis_gops(c), ["01949", "01950", "01953", "01955"])

    def test_standalone_has_no_base(self):
        c = self._cand("03330", "GOP 03330: Spirographische Untersuchung. Obligater "
                                 "Leistungsinhalt - Darstellung der Flussvolumenkurve")
        self.assertEqual(_candidate_basis_gops(c), [])

    def test_exclusion_reference_not_mistaken_for_base(self):
        # "nicht neben ... 01410" appears only in the notes, after Obligater Leistungsinhalt
        c = self._cand("02300", "GOP 02300: Kleinchirurgischer Eingriff I. Obligater "
                                 "Leistungsinhalt - Operativer Eingriff. Die "
                                 "Gebührenordnungsposition 02300 ist nicht neben den "
                                 "Gebührenordnungspositionen 01410 berechnungsfähig.")
        self.assertEqual(_candidate_basis_gops(c), [])


class EnforceBaseDependenciesTest(unittest.TestCase):
    @staticmethod
    def _cand(gop: str, volltext: str) -> dict:
        return {"gop": gop, "details": {"volltext": volltext}}

    def setUp(self):
        self.zuschlag = self._cand("01952", "GOP 01952: Zuschlag im Zusammenhang mit der "
                                            "Gebührenordnungsposition 01950 Obligater Leistungsinhalt")
        self.base = self._cand("01950", "GOP 01950: Substitutionsbehandlung Obligater Leistungsinhalt")
        self.standalone = self._cand("03330", "GOP 03330: Spirographie Obligater Leistungsinhalt")

    def test_drop_when_base_absent(self):
        by_gop = {"01952": self.zuschlag, "03330": self.standalone}
        self.assertEqual(
            _enforce_base_dependencies(["01952", "03330"], by_gop, set()),
            ["03330"],
        )

    def test_keep_when_base_billed(self):
        by_gop = {"01952": self.zuschlag}
        self.assertEqual(
            _enforce_base_dependencies(["01952"], by_gop, {"01950"}),
            ["01952"],
        )

    def test_keep_when_base_selected(self):
        by_gop = {"01952": self.zuschlag, "01950": self.base}
        self.assertEqual(
            _enforce_base_dependencies(["01950", "01952"], by_gop, set()),
            ["01950", "01952"],
        )

    def test_fixpoint_propagates_through_chain(self):
        # A(top) -> B(mid) -> C(base); only A and B selected, C absent => both dropped
        a = self._cand("00003", "GOP 00003: Zuschlag zu der Gebührenordnungsposition 00002 Obligater")
        b = self._cand("00002", "GOP 00002: Zuschlag zu der Gebührenordnungsposition 00001 Obligater")
        by_gop = {"00003": a, "00002": b}
        self.assertEqual(_enforce_base_dependencies(["00003", "00002"], by_gop, set()), [])


if __name__ == "__main__":
    unittest.main()

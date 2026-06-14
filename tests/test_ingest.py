import unittest

from src.ingest import _expand_gop_range, _extract_exclusions


class ExtractExclusionsTest(unittest.TestCase):
    def test_single_line_list(self):
        text = ("Die Gebührenordnungsposition 02300 ist nicht neben den "
                "Gebührenordnungspositionen 01741, 02301, 02302 berechnungsfähig.")
        self.assertEqual(_extract_exclusions(text), ["01741", "02301", "02302"])

    def test_phrase_wrapped_across_line_breaks(self):
        # The "nicht neben den" / "Gebührenordnungspositionen" split is the real PDF layout.
        text = ("Die Gebührenordnungsposition 02300 ist nicht neben den\n"
                "Gebührenordnungspositionen 01741, 02301, 02302 berechnungsfähig.")
        self.assertEqual(_extract_exclusions(text), ["01741", "02301", "02302"])

    def test_range_expansion(self):
        text = ("Die Gebührenordnungsposition 02300 ist nicht neben den "
                "Gebührenordnungspositionen 02321 bis 02323, 02325 berechnungsfähig.")
        self.assertEqual(_extract_exclusions(text), ["02321", "02322", "02323", "02325"])

    def test_singular_form(self):
        text = ("Die Gebührenordnungsposition 02321 ist im Behandlungsfall nicht neben der "
                "Gebührenordnungsposition 34291 berechnungsfähig.")
        self.assertEqual(_extract_exclusions(text), ["34291"])

    def test_section_references_yield_no_codes(self):
        text = ("Die Gebührenordnungsposition 02300 ist nicht neben den "
                "Gebührenordnungspositionen der Abschnitte 18.3, 30.5 berechnungsfähig.")
        self.assertEqual(_extract_exclusions(text), [])

    def test_no_exclusion_sentence(self):
        self.assertEqual(_extract_exclusions("Obligater Leistungsinhalt - Infusion intravenös"), [])


class ExpandGopRangeTest(unittest.TestCase):
    def test_normal_range(self):
        self.assertEqual(_expand_gop_range("06350", "06352"), ["06350", "06351", "06352"])

    def test_oversized_range_falls_back_to_endpoints(self):
        self.assertEqual(_expand_gop_range("01000", "09999"), ["01000", "09999"])


if __name__ == "__main__":
    unittest.main()

import unittest

from src.inference import parse_gops


class ParseGopsTest(unittest.TestCase):
    def test_parse_last_json_array(self):
        self.assertEqual(parse_gops('details ["03000"]\n["03220", "03324"]'), ["03220", "03324"])

    def test_empty_list_is_valid_answer(self):
        self.assertEqual(parse_gops("Keine abrechenbare Leistung. []"), [])

    def test_free_text_without_json_array_yields_nothing(self):
        self.assertEqual(parse_gops("GOP 03000 und 03324 waeren denkbar"), [])

    def test_duplicates_are_removed(self):
        self.assertEqual(parse_gops('["03000", "03000", "03324"]'), ["03000", "03324"])

    def test_non_gop_entries_are_ignored(self):
        self.assertEqual(parse_gops('["03000", "abc", "123"]'), ["03000"])


if __name__ == "__main__":
    unittest.main()

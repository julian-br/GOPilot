import unittest

from src.inference import parse_gops


class ParseGopsTest(unittest.TestCase):
    def test_parse_last_json_array(self):
        self.assertEqual(parse_gops('details ["03000"]\n["03220", "03324"]'), ["03220", "03324"])

    def test_no_text_fallback_by_default(self):
        self.assertEqual(parse_gops("GOP 03000 und 03324 waeren denkbar"), [])

    def test_optional_text_fallback_is_deduplicated(self):
        self.assertEqual(
            parse_gops("GOP 03000, GOP 03000 und GOP 03324", allow_text_fallback=True),
            ["03000", "03324"],
        )


if __name__ == "__main__":
    unittest.main()

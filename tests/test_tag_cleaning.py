import unittest
import pandas as pd
from pipeline.generate_semantic_vectors import clean_tag_string

class TestTagCleaning(unittest.TestCase):
    def test_clean_tag_string(self):
        # Dictionary string
        self.assertEqual(clean_tag_string("{'Action': 100, 'Indie': 50}"), "Action, Indie")
        # Empty
        self.assertEqual(clean_tag_string("{}"), "")
        self.assertEqual(clean_tag_string(""), "")
        # Non-dict string (should return as is or stringified)
        self.assertEqual(clean_tag_string("Action"), "Action")
        # NaN
        self.assertEqual(clean_tag_string(pd.NA), "")

if __name__ == "__main__":
    unittest.main()

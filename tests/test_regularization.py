import unittest
import pandas as pd
import numpy as np
import os
import json
from pipeline.calculate_regularization import calculate_global_positive_rate, solve_tag_vector_k, solve_playtime_regularization

class TestRegularization(unittest.TestCase):
    def test_calculate_global_positive_rate(self):
        # Create synthetic data
        data = {
            'positive': [90, 10, 50],
            'negative': [10, 90, 50]
        }
        df = pd.DataFrame(data)
        
        # With high threshold, no games meet it, should fallback
        # With threshold 0, all games meet it
        # Let's mock a case where some meet it. 
        # REG_RATE_MIN_REVIEWS_THRESHOLD is usually 100 in constants.py? Let's check.
        
        rate = calculate_global_positive_rate(df)
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_solve_tag_vector_k_minimal(self):
        # Minimal data for tag vector K solver
        data = {
            'tags': [
                "{'Action': 100, 'Indie': 50}",
                "{'Action': 10, 'RPG': 90}",
                "{'Indie': 100, 'Casual': 100}"
            ]
        }
        df = pd.DataFrame(data)
        k = solve_tag_vector_k(df)
        self.assertIsInstance(k, float)
        self.assertGreaterEqual(k, 0.0)

    def test_solve_playtime_regularization_no_file(self):
        # Should return default if file doesn't exist
        c = solve_playtime_regularization("non_existent_reviews.csv")
        self.assertEqual(c, 100.0)

    def test_solve_playtime_regularization_with_data(self):
        # Create a temp csv
        filename = "test_reviews_temp.csv"
        data = {
            'appid': [1, 1, 1, 2, 2],
            'author_playtime_forever': [100, 200, 150, 1000, 50],
            'voted_up': [True, True, True, True, False]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        try:
            # Threshold 1 -> both games are reliable (if voted_up=True)
            # Game 1: 3 positive reviews
            # Game 2: 1 positive review
            c = solve_playtime_regularization(filename, threshold=1)
            self.assertIsInstance(c, float)
            self.assertGreater(c, 0.0)
        finally:
            # Small delay to ensure Windows releases the file handle if there was any lingering access
            import time
            time.sleep(0.1)
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except PermissionError:
                    pass # Windows file locking can be flaky in tests

if __name__ == "__main__":
    unittest.main()

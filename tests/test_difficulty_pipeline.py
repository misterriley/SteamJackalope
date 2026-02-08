import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.generate_difficulty_model import normalize_title, rank_int, calculate_bic, has_numeric_mismatch

class TestDifficultyPipeline(unittest.TestCase):

    def test_normalize_title(self):
        # Test basic normalization
        self.assertEqual(normalize_title("The Witcher 3: Wild Hunt"), "the witcher 3 wild hunt")
        self.assertEqual(normalize_title("Half-Life 2"), "half life 2")
        
        # Test numeral conversion
        self.assertEqual(normalize_title("Final Fantasy VII"), "final fantasy 7")
        self.assertEqual(normalize_title("Resident Evil 4 Remake"), "resident evil 4")
        self.assertEqual(normalize_title("Age of Empires II: Definitive Edition"), "age of empires 2")
        
        # Test edition removal
        self.assertEqual(normalize_title("Skyrim Special Edition"), "skyrim special")
        self.assertEqual(normalize_title("Batman: Arkham City - Game of the Year Edition"), "batman arkham city")

    def test_has_numeric_mismatch(self):
        # Should match
        self.assertFalse(has_numeric_mismatch("doom", "doom"))
        self.assertFalse(has_numeric_mismatch("doom 2016", "doom 2016"))
        
        # Should mismatch
        self.assertTrue(has_numeric_mismatch("doom", "doom 2"))
        self.assertTrue(has_numeric_mismatch("half life", "half life 2"))
        self.assertTrue(has_numeric_mismatch("civilization 5", "civilization 6"))

    def test_rank_int(self):
        data = np.array([1, 2, 3, 4, 5])
        transformed = rank_int(data)
        
        # Check if output is roughly standard normal
        self.assertAlmostEqual(transformed.mean(), 0, places=1)
        self.assertAlmostEqual(transformed.std(), 1, places=0)
        
        # Check order preservation
        self.assertTrue(np.all(np.diff(transformed) > 0))

    def test_calculate_bic(self):
        n = 100
        rss = 10.0
        k = 2
        bic = calculate_bic(n, rss, k)
        
        # Manual calc: 2 * log(100) + 100 * log(10/100)
        # 2 * 4.605 + 100 * -2.302 = 9.21 - 230.2 = -220.99
        expected = k * np.log(n) + n * np.log(rss / n)
        self.assertAlmostEqual(bic, expected)

if __name__ == '__main__':
    unittest.main()

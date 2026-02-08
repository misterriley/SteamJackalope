import unittest
import pandas as pd
import numpy as np
from pipeline.generate_metadata import clean_release_date
from common.utils import to_z

class TestMetadataProcessing(unittest.TestCase):
    def test_clean_release_date(self):
        # YYYY-MM-DD
        self.assertEqual(clean_release_date("2023-01-01").year, 2023)
        # YYYY
        self.assertEqual(clean_release_date("2023").month, 7)
        # Month YYYY
        self.assertEqual(clean_release_date("Jan 2023").year, 2023)
        self.assertEqual(clean_release_date("Jan 2023").month, 1)
        # YYYY Month
        self.assertEqual(clean_release_date("2023 January").year, 2023)
        # Malformed
        self.assertTrue(pd.isna(clean_release_date("invalid date")))

    def test_to_z(self):
        x = np.array([1, 2, 3, 4, 5])
        z = to_z(x)
        self.assertAlmostEqual(np.mean(z), 0.0)
        self.assertAlmostEqual(np.std(z), 1.0)
        
        # Test zero variance
        x2 = np.array([1, 1, 1])
        z2 = to_z(x2)
        self.assertTrue(np.all(z2 == 0.0))

if __name__ == "__main__":
    unittest.main()

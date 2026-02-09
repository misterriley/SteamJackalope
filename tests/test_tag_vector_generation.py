import unittest
import numpy as np
import pandas as pd
import os
import ast
from unittest.mock import patch
from pipeline.generate_tag_vectors import generate_tag_vectors

class TestTagVectorGeneration(unittest.TestCase):
    def setUp(self):
        self.test_csv = "test_games.csv"
        self.output_vectors = "test_steam_tag_vectors.npy"
        self.output_constants = "test_regularization_constants.json"
        self.w_tag_file = "test_w_tag.npy"
        self.tag_norms_file = "test_tag_vectors_norms.npy"
        
        # Create a small test CSV
        data = {
            'appid': [1, 2, 3],
            'tags': [
                "{'Action': 100, 'Indie': 50}",
                "{'Action': 10, 'RPG': 90}",
                "{'Indie': 100, 'Casual': 100}"
            ]
        }
        pd.DataFrame(data).to_csv(self.test_csv, index=False)

    def tearDown(self):
        import gc
        gc.collect()  # Force garbage collection to close any dangling file handles
        files_to_remove = [
            self.test_csv, 
            self.output_vectors, 
            self.output_constants, 
            self.w_tag_file,
            self.tag_norms_file
        ]
        
        for f in files_to_remove:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_generate_tag_vectors(self):
        # This function writes to steam_tag_vectors.npy
        generate_tag_vectors(
            self.test_csv, 
            output_vectors=self.output_vectors, 
            output_constants=self.output_constants,
            output_norms=self.tag_norms_file,
            w_tag_path=self.w_tag_file
        )
        
        self.assertTrue(os.path.exists(self.output_vectors))
        vectors = np.load(self.output_vectors)
        
        # 3 games. The number of components may be less than the number of tags (4)
        # due to singularity truncation in CLR space.
        self.assertEqual(vectors.shape[0], 3)
        
        # Check for NaN or Inf
        self.assertFalse(np.isnan(vectors).any())
        self.assertFalse(np.isinf(vectors).any())
        
        # Since it's ZCA whitened (uncentered covariance), the mean may not be exactly 0
        self.assertLess(abs(np.mean(vectors)), 1.0)
        
        # Verify norms file was also created
        self.assertTrue(os.path.exists(self.tag_norms_file))
        norms = np.load(self.tag_norms_file)
        self.assertEqual(len(norms), 3)

if __name__ == "__main__":
    unittest.main()

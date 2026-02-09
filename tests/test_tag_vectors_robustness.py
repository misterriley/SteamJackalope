import unittest
import numpy as np
import pandas as pd
import os
import ast
from pipeline.generate_tag_vectors import generate_tag_vectors, whiten

class TestTagVectorsRobustness(unittest.TestCase):
    def setUp(self):
        self.test_csv = "robustness_test_games.csv"
        self.output_vectors = "robustness_tag_vectors.npy"
        self.output_constants = "robustness_constants.json"
        self.w_tag_file = "robustness_w_tag.npy"
        self.tag_norms_file = "robustness_tag_norms.npy"

    def tearDown(self):
        import gc
        gc.collect()
        for f in [self.test_csv, self.output_vectors, self.output_constants, self.w_tag_file, self.tag_norms_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_whitening_stability_with_singular_data(self):
        """
        Test that whitening remains stable even when data is perfectly singular 
        (which happens with CLR transforms).
        """
        # Create 100 games with 10 tags
        num_games = 100
        num_tags = 10
        
        # Random data
        X = np.random.rand(num_games, num_tags)
        
        # Force a linear constraint: sum(rows) = 0
        X = X - X.mean(axis=1, keepdims=True)
        
        # Whiten with variance threshold (new API)
        variance_threshold = 0.80
        whitened, W = whiten(X, variance_threshold=variance_threshold)
        
        # Check shape - should be reduced but at least 1 dimension
        self.assertGreater(whitened.shape[1], 0)
        self.assertLess(whitened.shape[1], num_tags + 1)
        
        # Check that no value is 'exploding''
        # With unit variance, values shouldn't be massive
        self.assertLess(np.max(np.abs(whitened)), 10.0) 
        
        # Check that variance is indeed close to 1 for active dimensions
        stds = np.std(whitened, axis=0)
        for s in stds:
            self.assertAlmostEqual(s, 1.0, places=1)

    def test_disparate_games_similarity(self):
        """
        Create synthetic disparate games and ensure they don't get high 
        similarity due to noise amplification.
        """
        # Game A: Tags 0, 1, 2
        # Game B: Tags 3, 4, 5
        # They share NO tags.
        data = {
            'appid': list(range(1, 21)),
            'tags': []
        }
        for i in range(10):
            data['tags'].append("{'Action': 100, 'Indie': 50}") # Type A
        for i in range(10):
            data['tags'].append("{'RPG': 100, 'Strategy': 50}") # Type B
            
        pd.DataFrame(data).to_csv(self.test_csv, index=False)
        
        # Run generation
        generate_tag_vectors(
            self.test_csv, 
            output_vectors=self.output_vectors,
            output_constants=self.output_constants,
            output_norms=self.tag_norms_file,
            w_tag_path=self.w_tag_file
        )
        
        vectors = np.load(self.output_vectors)
        
        # Type A indices: 0-9, Type B indices: 10-19
        v_a = vectors[0]
        v_b = vectors[10]
        
        dot_product = np.dot(v_a, v_b)
        
        # In a healthy whitened space, the dot product should be negative 
        # for disparate items because they are on opposite sides of the mean.
        # At the very least, it shouldn't be a huge positive number.
        self.assertLess(dot_product, 5.0) 
        
        # Check norms
        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)
        
        # Physical dim will be min(128, num_tags)
        # Here tags are: Action, Indie, RPG, Strategy -> 4 tags
        # But wait, CLR makes it singular, so 3 active dimensions
        expected_norm = np.sqrt(3) 
        self.assertAlmostEqual(norm_a, expected_norm, delta=2.0)
        self.assertAlmostEqual(norm_b, expected_norm, delta=2.0)

if __name__ == "__main__":
    unittest.main()

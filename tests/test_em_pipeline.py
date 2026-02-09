import unittest
import numpy as np
import scipy.sparse as sp
import sys
import os

# Add parent directory to path to import generate_tag_vectors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.generate_tag_vectors import iterative_em_imputation, optimize_k_stochastic, apply_tag_transform, whiten

class TestEMPipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        # 100 games, 50 tags
        np.random.seed(42)
        self.num_games = 100
        self.num_tags = 50
        
        # Create dense counts with some structure
        # Use a few latent factors to create correlation
        factors = np.random.rand(self.num_games, 5)
        loadings = np.random.rand(5, self.num_tags)
        lambda_ = np.dot(factors, loadings) * 100
        
        # Generate counts
        self.counts_dense = np.random.poisson(lambda_)
        
        # Make it sparse (add zeros)
        mask = np.random.rand(self.num_games, self.num_tags) > 0.7
        self.counts_dense[~mask] = 0
        
        # Ensure some reliable games (>1000 votes)
        self.counts_dense[0:10] += 200 # boost counts for first 10
        
        self.sparse_counts = sp.csr_matrix(self.counts_dense)
        self.original_votes = self.counts_dense.sum(axis=1)

    def test_iterative_em(self):
        print("\nTesting Iterative EM...")
        # Run 2 iterations for speed
        augmented_counts, G = iterative_em_imputation(self.sparse_counts, max_iter=2)
        
        # Check shapes
        self.assertEqual(augmented_counts.shape, (self.num_games, self.num_tags))
        self.assertEqual(G.shape, (self.num_tags,))
        
        # Check consistency
        # Imputation should fill zeros (if correlation suggests)
        # Check if any zero became non-zero (likely)
        # Note: zeros might stay zero if correlation is weak or cap is 0
        
        # Verify G sums to 1 (approx)
        self.assertAlmostEqual(G.sum(), 1.0, places=5)
        
        # Verify reliable games (first 10) likely have close to original counts?
        # Imputation only affects "tags outside top 20".
        # If reliable games have > 20 tags, the top 20 are fixed.
        # But reliable games usually have stable distributions.
        pass

    def test_optimize_k(self):
        print("\nTesting K Optimization...")
        # Need augmented counts and G
        augmented_counts = self.counts_dense.astype(float) # Mock augmented
        G_prior = augmented_counts.sum(axis=0) / augmented_counts.sum()
        
        # Run optimization
        k = optimize_k_stochastic(augmented_counts, self.sparse_counts, G_prior)
        
        print(f"Optimal K: {k}")
        self.assertGreater(k, 0)
        self.assertLess(k, 10000)

    def test_anscombe_and_dampening(self):
        print("\nTesting Anscombe Transform and Dampening...")
        G_prior = np.ones(self.num_tags) / self.num_tags
        K = 100.0
        
        # apply_tag_transform expects original_total_votes as (N,)
        anscombe = apply_tag_transform(self.counts_dense, G_prior, self.original_votes, K, transform_type='anscombe')
        self.assertEqual(anscombe.shape, (self.num_games, self.num_tags))
        
        # Whiten
        whitened, _ = whiten(anscombe)
        # Whitening might drop singular dimensions, especially in small synthetic datasets
        self.assertEqual(whitened.shape[0], self.num_games)
        self.assertLessEqual(whitened.shape[1], self.num_tags)

if __name__ == '__main__':
    unittest.main()

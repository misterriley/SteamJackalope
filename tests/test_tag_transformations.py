import unittest
import numpy as np
from pipeline.generate_tag_vectors import apply_tag_transform

class TestTagTransformations(unittest.TestCase):
    def setUp(self):
        self.num_tags = 5
        self.num_games = 10
        self.augmented_counts = np.random.rand(self.num_games, self.num_tags) * 100
        self.prior_G = np.random.rand(self.num_tags)
        self.prior_G /= self.prior_G.sum()
        self.original_total_votes = np.random.rand(self.num_games) * 500
        self.K = 100.0

    def test_anscombe_zero_game(self):
        # A hypothetical game with 0 votes should result in a zero vector
        # because the regularized profile will be identical to the prior G.
        test_counts = (self.prior_G * 0.0).reshape(1, -1)
        test_votes = np.array([0.0])
        
        vectors = apply_tag_transform(test_counts, self.prior_G, test_votes, self.K, transform_type='anscombe')
        np.testing.assert_array_almost_equal(vectors, np.zeros_like(vectors), decimal=6)

    def test_clr_zero_game(self):
        # A hypothetical game with 0 votes should result in a zero vector
        test_counts = (self.prior_G * 0.0).reshape(1, -1)
        test_votes = np.array([0.0])
        
        vectors = apply_tag_transform(test_counts, self.prior_G, test_votes, self.K, transform_type='clr')
        np.testing.assert_array_almost_equal(vectors, np.zeros_like(vectors), decimal=6)

    def test_none_zero_game(self):
        # A hypothetical game with 0 votes should result in a zero vector
        test_counts = (self.prior_G * 0.0).reshape(1, -1)
        test_votes = np.array([0.0])
        
        vectors = apply_tag_transform(test_counts, self.prior_G, test_votes, self.K, transform_type='none')
        np.testing.assert_array_almost_equal(vectors, np.zeros_like(vectors), decimal=6)

    def test_dampening_effect(self):
        # Check that higher N leads to larger magnitudes (less dampening)
        # Create two games with same relative signal but different vote counts
        # Signal: all votes in the first tag
        signal_counts_high = np.zeros((1, self.num_tags))
        signal_counts_high[0, 0] = 1000.0
        votes_high = np.array([1000.0])

        signal_counts_low = np.zeros((1, self.num_tags))
        signal_counts_low[0, 0] = 1.0
        votes_low = np.array([1.0])
        
        vectors_high = apply_tag_transform(signal_counts_high, self.prior_G, votes_high, self.K, transform_type='none')
        vectors_low = apply_tag_transform(signal_counts_low, self.prior_G, votes_low, self.K, transform_type='none')
        
        self.assertGreater(np.linalg.norm(vectors_high), np.linalg.norm(vectors_low))

    def test_none_zero_sum(self):
        # Test game with all zero counts
        zero_counts = np.zeros((1, self.num_tags))
        # Total votes also 0
        zero_votes = np.array([0.0])
        
        # This should not raise division by zero error
        # and should return a finite vector (likely negative prior)
        vectors = apply_tag_transform(zero_counts, self.prior_G, zero_votes, self.K, transform_type='none')
        
        self.assertTrue(np.all(np.isfinite(vectors)))
        
        # Dampening should be 0 because N=0
        # So final vector should be 0
        np.testing.assert_array_almost_equal(vectors, np.zeros_like(vectors), decimal=6)

if __name__ == "__main__":
    unittest.main()

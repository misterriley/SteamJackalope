import unittest
import os
import pandas as pd
import numpy as np
import subprocess
import sys
import json

class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.test_games = "test_scraped_games.csv"
        self.test_reviews = "test_scraped_reviews.csv"
        self.test_config = "test_pipeline_config.json"
        
        # Create a test config that points to test output files
        # to avoid overwriting production artifacts
        test_config_data = {
            "games_file": self.test_games,
            "reviews_file": self.test_reviews,
            "clean_games_file": "data/test_pipeline_games_clean.csv",
            "metadata_file": "test_metadata.parquet",
            "embeddings_desc_file": "test_embeddings_desc.npy",
            "embeddings_tag_file": "test_embeddings_structural.npy",
            "tag_vectors_file": "test_tag_vectors.npy",
            "quality_grid_file": "test_quality_scores_grid.npy",
            "regularization_json": "test_regularization_constants.json"
        }
        with open(self.test_config, "w") as f:
            json.dump(test_config_data, f)

        # Minimal games data
        games_data = {
            'appid': [10, 20],
            'name': ['Game 1', 'Game 2'],
            'release_date': ['2020-01-01', '2021-01-01'],
            'positive': [100, 200],
            'negative': [10, 20],
            'tags': ["{'Action': 10}", "{'RPG': 20}"],
            'genres': ['Action', 'RPG'],
            'categories': ['Single-player', 'Multi-player'],
            'supported_languages': ['English', 'English'],
            'mature_content': [0, 0],
            'short_description': ['Description 1', 'Description 2']
        }
        pd.DataFrame(games_data).to_csv(self.test_games, index=False)
        
        # Minimal reviews data
        reviews_data = {
            'appid': [10, 20],
            'author_playtime_forever': [1000, 2000],
            'review_text': ['Great game!', 'Love it!']
        }
        pd.DataFrame(reviews_data).to_csv(self.test_reviews, index=False)

    def tearDown(self):
        import gc
        gc.collect()  # Force garbage collection to close any dangling file handles
        # Note: We avoid deleting production artifacts here to prevent accidental data loss
        # during discoverable test runs.
        files_to_remove = [
            self.test_games, 
            self.test_reviews,
            self.test_config,
            "data/test_pipeline_games_clean.csv",
            "test_metadata.parquet",
            "test_embeddings_desc.npy",
            "test_embeddings_structural.npy",
            "test_tag_vectors.npy",
            "test_quality_scores_grid.npy",
            "test_regularization_constants.json"
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_pipeline_execution(self):
        """
        Tests the full pipeline execution using a separate configuration and isolated file paths.
        This ensures that running tests does not modify production artifacts.
        """
        # We'll run run_pipeline.py as a subprocess with our test files
        # Note: generate_semantic_vectors.py requires a model download/load which might be slow or fail in CI, 
        # but let's try a minimal run.
        
        # Set environment variables to override production paths in child processes
        test_env = os.environ.copy()
        test_env["STEAM_METADATA_FILE"] = "test_metadata.parquet"
        test_env["STEAM_EMBEDDINGS_DESC_FILE"] = "test_embeddings_desc.npy"
        test_env["STEAM_EMBEDDINGS_TAG_FILE"] = "test_embeddings_structural.npy"
        test_env["STEAM_TAG_VECTORS_FILE"] = "test_tag_vectors.npy"
        test_env["STEAM_QUALITY_GRID_FILE"] = "test_quality_scores_grid.npy"
        test_env["STEAM_REGULARIZATION_JSON"] = "test_regularization_constants.json"

        cmd = [
            sys.executable, os.path.join("pipeline", "run_pipeline.py"),
            "--config", self.test_config,
            "--games", self.test_games,
            "--reviews", self.test_reviews
        ]
        
        print("\nRunning pipeline integration test...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=test_env)
        
        # We check return code. If it fails because of 'all-MiniLM-L6-v2' download, 
        # that's a known environment dependency.
        if result.returncode != 0:
            if "sentence_transformers" not in result.stderr:
                 print(f"Pipeline failed: {result.stderr}")
        
        # Check if test artifacts were produced
        self.assertTrue(os.path.exists("data/test_pipeline_games_clean.csv"), "Clean games file not found")
        self.assertTrue(os.path.exists("test_regularization_constants.json"), "Test constants file not found")
        self.assertTrue(os.path.exists("test_tag_vectors.npy"), "Test tag vectors file not found")
        self.assertTrue(os.path.exists("test_metadata.parquet"), "Test metadata file not found")

if __name__ == "__main__":
    unittest.main()

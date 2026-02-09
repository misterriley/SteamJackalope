import unittest
import os
import pandas as pd
import numpy as np
from pipeline.generate_metadata import generate_metadata
from pipeline.generate_quality_scores_grid import generate_quality_grid

class TestMetadataSync(unittest.TestCase):
    def setUp(self):
        self.test_games_csv = "test_games_sync.csv"
        self.test_metadata_parquet = "test_metadata_sync.parquet"
        self.test_grid_npy = "test_grid_sync.npy"
        
        # Create test data with a game that has no name
        data = {
            'appid': [1, 2, 3],
            'name': ['Game 1', None, 'Game 3'], # Game 2 has no name
            'release_date': ['2020-01-01', '2020-01-01', '2020-01-01'],
            'positive': [1000, 2000, 3000],
            'negative': [100, 200, 300],
            'tags': ['{}', '{}', '{}'],
            'genres': ['Action', 'Action', 'Action']
        }
        pd.DataFrame(data).to_csv(self.test_games_csv, index=False)

    def tearDown(self):
        for f in [self.test_games_csv, self.test_metadata_parquet, self.test_grid_npy]:
            if os.path.exists(f):
                os.remove(f)

    def test_nameless_game_dropping(self):
        """
        Verify that games with missing names are dropped, ensuring index synchronization.
        """
        # 1. Generate metadata
        generate_metadata(self.test_games_csv, output_path=self.test_metadata_parquet)
        
        # Check metadata length
        df_meta = pd.read_parquet(self.test_metadata_parquet)
        self.assertEqual(len(df_meta), 2, "Metadata should only have 2 games (nameless one dropped)")
        self.assertNotIn(2, df_meta['appid'].values, "AppID 2 should have been dropped")
        
        # 2. Generate quality grid
        generate_quality_grid(self.test_metadata_parquet, output_path=self.test_grid_npy)
        
        # Check grid shape
        grid = np.load(self.test_grid_npy)
        self.assertEqual(grid.shape[1], 2, "Grid should have 2 columns (matching metadata)")

if __name__ == "__main__":
    unittest.main()

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    DISC_SLIDER_VALUES, 
    QUALITY_SCORE_S_CONST, 
    QUALITY_SCORE_S_BASE, 
    QUALITY_SCORE_MIN_VOTES_FOR_RELIABLE, 
    QUALITY_SCORE_CLIP, 
    Z_SCORE_CLAMP_MAX, 
    Z_SCORE_CLAMP_MIN, 
    METADATA_FILE, 
    QUALITY_GRID_FILE
)
from common.utils import safe_save_npy

def generate_quality_grid_quantile(metadata_path, output_path):
    """
    Quality Grid Generation with Quantile Normalization.
    Forces each grid step into a perfect Standard Normal distribution N(0, 1).
    This solves the "bunching" and extreme kurtosis issues while 
    preserving the ranking order exactly.
    """
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)
    df.drop_duplicates(subset=['appid'], inplace=True)
    df.dropna(subset=['appid', 'name'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    p = df['positive'].fillna(0).values
    n = df['negative'].fillna(0).values
    
    reliable_mask = (p + n) >= QUALITY_SCORE_MIN_VOTES_FOR_RELIABLE
    reliable_ps = p[reliable_mask]/(p[reliable_mask] + n[reliable_mask])
    
    a = np.mean(reliable_ps)

    slider_values = DISC_SLIDER_VALUES
    num_games = len(df)
    num_steps = len(slider_values)
    
    grid = np.zeros((num_steps, num_games), dtype=np.float64)

    print("Computing quality scores grid with Quantile Normalization...")
    for i, val in enumerate(slider_values):
        s = QUALITY_SCORE_S_CONST * (QUALITY_SCORE_S_BASE ** (-val))
        prob = np.clip((p + s * a) / (p + n + s), QUALITY_SCORE_CLIP, 1 - QUALITY_SCORE_CLIP)
        
        # Probit transform gives the raw "quality" signal
        raw_scores = norm.ppf(prob)

        # Tie-Aware Quantile Normalization
        # 1. Get ranks using the 'average' method to handle ties consistently
        from scipy.stats import rankdata
        ranks = rankdata(raw_scores, method='average')
        
        # 2. Convert ranks to quantiles [0, 1]
        # Use (rank - 0.5) / n to avoid 0.0 and 1.0 which map to +/- inf
        quantiles = (ranks - 0.5) / len(raw_scores)
        
        # 3. Map to Normal Distribution
        final_scores = norm.ppf(quantiles)
        
        grid[i] = final_scores.astype(np.float64)
        
        # Statistics for logging
        k = ((final_scores**4).mean() / (final_scores**2).mean()**2 - 3)
        max_z = np.max(final_scores)
        min_z = np.min(final_scores)
        print(f"  Slider {val:+.1f}: s={s:8.2f} | Kurtosis={k:8.4f} | Max Z={max_z:8.2f}")

    print(f"Saving quantile-normalized grid to {output_path}...")
    safe_save_npy(output_path, grid.astype(np.float16))
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate quality scores grid with Quantile Normalization.")
    parser.add_argument("--metadata", default=METADATA_FILE, help="Path to metadata.parquet")
    parser.add_argument("--output", default=QUALITY_GRID_FILE, help="Output .npy file")
    args = parser.parse_args()
    
    generate_quality_grid_quantile(args.metadata, args.output)

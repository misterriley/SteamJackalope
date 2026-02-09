import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import AP_SLIDER_VALUES, QUALITY_SCORE_S_CONST, QUALITY_SCORE_S_BASE, QUALITY_SCORE_MIN_VOTES_FOR_RELIABLE, QUALITY_SCORE_CLIP, QUALITY_SCORE_PIN_GROUP, PIN_QUALITY_DISTRIBUTION, Z_SCORE_CLAMP_MAX, Z_SCORE_CLAMP_MIN

def generate_quality_grid(metadata_path, output_path):
    """
    Generates a grid of quality scores for different "Popularity Preference" settings.
    
    The algorithm uses a Bayesian approach where the prior strength (s) is varied 
    based on the user's preference for mainstream vs. niche games.
    
    - High Popularity Preference (val=1): High 's' (e.g., 245,000) makes it hard for 
      games with few reviews to have high scores.
    - Low Popularity Preference (val=-1): Low 's' (e.g., 50) allows niche games with 
      high ratios to climb the rankings.
      
    The resulting scores are Probit-transformed (converted to z-scores) for 
    consistent blending with other components in app.py.

    Args:
        metadata_path (str): Path to the metadata parquet file.
        output_path (str): Path to save the generated numpy grid.
    """
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)
    df.drop_duplicates(subset=['appid'], inplace=True)
    # Ensure name is present to maintain index synchronization with other pipeline stages
    df.dropna(subset=['appid', 'name'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    p = df['positive'].fillna(0).values
    n = df['negative'].fillna(0).values
    
    reliable_mask = (p + n) >= QUALITY_SCORE_MIN_VOTES_FOR_RELIABLE
    reliable_ps = p[reliable_mask]/(p[reliable_mask] + n[reliable_mask])
    
    a = np.mean(reliable_ps)

    slider_values = AP_SLIDER_VALUES
    num_games = len(df)
    num_steps = len(slider_values)
    
    # Use float64 to prevent sorting ties when scores are extremely close (high regularization)
    grid = np.zeros((num_steps, num_games), dtype=np.float64)

    # Calculate baseline stats for s = S_CONST
    print(f"Calculating baseline stats for s = {QUALITY_SCORE_S_CONST}...")
    s_baseline = QUALITY_SCORE_S_CONST
    prob_baseline = (p + s_baseline * a) / (p + n + s_baseline)
    prob_baseline = np.clip(prob_baseline, QUALITY_SCORE_CLIP, 1 - QUALITY_SCORE_CLIP)
    scores_baseline = norm.ppf(prob_baseline)
    
    sorted_baseline = np.sort(scores_baseline)
    target_top_mean = np.mean(sorted_baseline[-QUALITY_SCORE_PIN_GROUP:])
    target_bottom_mean = np.mean(sorted_baseline[:QUALITY_SCORE_PIN_GROUP])
    print(f"Baseline Target Means -> Top {QUALITY_SCORE_PIN_GROUP}: {target_top_mean:.4f}, Bottom {QUALITY_SCORE_PIN_GROUP}: {target_bottom_mean:.4f}")

    print("Computing quality scores grid...")
    for i, val in enumerate(slider_values):
        # New formula: s = 3500 * 70^(val)
        s = QUALITY_SCORE_S_CONST * (QUALITY_SCORE_S_BASE ** (val))
        # Bayesian score
        prob = (p + s * a) / (p + n + s)
        print(f"Computing scores for slider value {val} (s={s:.2f})...")

        # Probit transform (z-score of probability)
        # Clamp probability to avoid inf in ppf
        prob = np.clip(prob, QUALITY_SCORE_CLIP, 1 - QUALITY_SCORE_CLIP)
        raw_scores = norm.ppf(prob)

        # Normalize scores
        if PIN_QUALITY_DISTRIBUTION:
            # Normalize scores to match baseline spread
            sorted_raw = np.sort(raw_scores)
            current_top_mean = np.mean(sorted_raw[-QUALITY_SCORE_PIN_GROUP:])
            current_bottom_mean = np.mean(sorted_raw[:QUALITY_SCORE_PIN_GROUP])

            if abs(current_top_mean - current_bottom_mean) > 1e-9:
                m = (target_top_mean - target_bottom_mean) / (current_top_mean - current_bottom_mean)
                c = target_top_mean - m * current_top_mean
                scores = raw_scores * m + c
                print(f"  Applied pinning normalization: m={m:.4f}, c={c:.4f} (Raw Top: {current_top_mean:.4f}, Raw Bottom: {current_bottom_mean:.4f})")
            else:
                print(f"  Warning: skipping normalization due to collapsed scores (Top: {current_top_mean:.4f}, Bottom: {current_bottom_mean:.4f})")
                scores = raw_scores
        else:
            # Standard z-score
            mean_raw = np.mean(raw_scores)
            std_raw = np.std(raw_scores)
            if std_raw > 1e-9:
                scores = (raw_scores - mean_raw) / std_raw
                print(f"  Applied standard z-score normalization (Mean: {mean_raw:.4f}, Std: {std_raw:.4f})")
            else:
                print(f"  Warning: skipping normalization due to zero variance (Mean: {mean_raw:.4f}, Std: {std_raw:.4f})")
                scores = raw_scores
            
        # print how many games are beyond constants.
        num_beyond_min = np.sum(scores < Z_SCORE_CLAMP_MIN)
        num_beyond_max = np.sum(scores > Z_SCORE_CLAMP_MAX)
        if num_beyond_min > 0 or num_beyond_max > 0:
            print(f"  Clamped {num_beyond_min} games below {Z_SCORE_CLAMP_MIN} and {num_beyond_max} games above {Z_SCORE_CLAMP_MAX}")

        grid[i] = scores.astype(np.float64)
        print(f"  Slider {val}: s={s:.2f}")

        print_grid = sorted(grid[i], reverse=True)[:5]
        game_names_to_print = [df.iloc[np.where(grid[i] == score)[0][0]]['name'] for score in print_grid]
        print(f"  Top 5 games: {game_names_to_print}")  
        print(f"  Top 5 scores: {print_grid}")

    print(f"Saving grid to {output_path}...")
    np.save(output_path, grid)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate quality scores grid.")
    parser.add_argument("--metadata", default="metadata.parquet", help="Path to metadata.parquet")
    parser.add_argument("--output", default="quality_scores_grid.npy", help="Output .npy file")
    args = parser.parse_args()
    
    generate_quality_grid(args.metadata, args.output)

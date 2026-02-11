import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import GLOBAL_POSITIVE_RATE, QUALITY_GRID_FILE, METADATA_FILE

def check_s():
    grid = np.load(QUALITY_GRID_FILE)
    df = pd.read_parquet(METADATA_FILE)
    df.drop_duplicates(subset=['appid'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    a = GLOBAL_POSITIVE_RATE
    
    # Check index 0 (all the way to the left)
    print(f"Checking grid index 0 (Slider = -1.0), Global Positive Rate={a}:")
    for i in range(5):
        p = df.iloc[i]['positive']
        n = df.iloc[i]['negative']
        score = grid[0][i]
        prob = norm.cdf(score)
        
        # Bayesian formula: prob = (p + s*a) / (p + n + s)
        # s = (p - prob*(p + n)) / (prob - a)
        
        if abs(prob - a) < 1e-6:
            s = np.nan
        else:
            s = (p - prob*(p + n)) / (prob - a)
        
        print(f"  Game: {df.iloc[i]['name']}, p={p}, n={n}, score={score:.4f}, prob={prob:.4f}, inferred s={s:.2f}")

    # Check index 5 (center)
    print(f"\nChecking grid index 5 (Slider = 0.0), Global Positive Rate={a}:")
    for i in range(5):
        p = df.iloc[i]['positive']
        n = df.iloc[i]['negative']
        score = grid[5][i]
        prob = norm.cdf(score)
        if abs(prob - a) < 1e-6:
            s = np.nan
        else:
            s = (p - prob*(p + n)) / (prob - a)
        print(f"  Game: {df.iloc[i]['name']}, p={p}, n={n}, score={score:.4f}, prob={prob:.4f}, inferred s={s:.2f}")

if __name__ == "__main__":
    check_s()
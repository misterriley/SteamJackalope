import pandas as pd
import numpy as np
import os

def prepare_subsets(steamid):
    gt_path = f"data/user_{steamid}_ground_truth.csv"
    df = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    
    # Shuffle and take 150
    df_150 = df.sample(n=min(150, len(df)), random_state=42)
    # Take 50 from that 150
    df_50 = df_150.sample(n=min(50, len(df_150)), random_state=42)
    
    df_150.to_csv(f"data/user_{steamid}_test_150.csv", index=False)
    df_50.to_csv(f"data/user_{steamid}_test_50.csv", index=False)
    
    print(f"Created subsets: 150 games and 50 games from {len(df)} total ratings.")

if __name__ == "__main__":
    prepare_subsets('76561198039155404')

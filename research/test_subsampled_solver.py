import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.solve_user_taste import solve_user_taste

def run_subsampled_test(user_id="76561198039155404", sample_size=100):
    print("--- Subsampled Solver Test: N=" + str(sample_size) + " ---")
    
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    
    df_sub = df.sample(n=sample_size, random_state=42)
    sub_gt_path = "data/user_" + user_id + "_subsampled_gt.csv"
    df_sub.to_csv(sub_gt_path, index=False)
    
    print("Running solver on " + str(sample_size) + " random ratings...")
    profile = solve_user_taste(sub_gt_path)
    
    topic_match = profile['metadata']['topic_match']
    print("\nFinal Topic Match Weight: " + f"{topic_match:.6f}")

if __name__ == "__main__":
    run_subsampled_test()

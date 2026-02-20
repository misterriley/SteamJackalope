import numpy as np
import os
from scipy.stats import kurtosis
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import ROOT_DIR

def analyze_quality_grid():
    grid_path = os.path.join(ROOT_DIR, 'data', 'production', 'quality_scores_grid.npy')
    if not os.path.exists(grid_path):
        print(f"Error: {grid_path} not found.")
        return
        
    quality_grid = np.load(grid_path).astype(np.float32)
    print(f"Analyzing Quality Grid: {quality_grid.shape}")
    print(f"{'Step':>4} | {'Kurtosis':>10} | {'Max Z':>10} | {'Min Z':>10} | {'Mean':>10} | {'Std':>10}")
    print("-" * 65)
    
    for i in range(quality_grid.shape[0]):
        q = quality_grid[i]
        mean = np.mean(q)
        std = np.std(q)
        
        # Avoid division by zero
        if std < 1e-9:
            q_scaled = q - mean
        else:
            q_scaled = (q - mean) / std
            
        k = kurtosis(q)
        max_z = np.max(q_scaled)
        min_z = np.min(q_scaled)
        
        print(f"{i:4d} | {k:10.4f} | {max_z:10.4f} | {min_z:10.4f} | {mean:10.4f} | {std:10.4f}")

if __name__ == "__main__":
    analyze_quality_grid()

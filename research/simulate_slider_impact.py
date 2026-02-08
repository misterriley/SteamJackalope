import pandas as pd
import numpy as np
import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE,
    QUALITY_GRID_FILE,
    AP_SLIDER_VALUES,
    AP_SLIDER_MIN,
    AP_SLIDER_STEP,
    QUALITY_WEIGHT_MULTIPLIER,
    AGE_WEIGHT_MULTIPLIER,
    POPULARITY_WEIGHT_MULTIPLIER,
    LENGTH_WEIGHT_MULTIPLIER,
    DIFFICULTY_WEIGHT_MULTIPLIER,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)
from common.utils import calculate_hybrid_score

def run_simulation(num_samples=20000):
    print(f"Loading data from {METADATA_FILE} and {QUALITY_GRID_FILE}...")
    metadata = pd.read_parquet(METADATA_FILE)
    quality_grid = np.load(QUALITY_GRID_FILE)
    
    num_games = len(metadata)
    print(f"Number of games: {num_games}")
    
    # Pre-process z-scores
    z_date = np.clip(metadata['date_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_pop = np.clip(metadata['pop_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_length = np.clip(metadata['playtime_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    if 'difficulty_z' in metadata.columns:
        z_difficulty = np.clip(metadata['difficulty_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    else:
        print("Warning: difficulty_z not found in metadata. Using zeros.")
        z_difficulty = np.zeros(num_games)
    
    # Pre-process quality grid
    q_grid_clamped = np.clip(quality_grid, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # Semantic and Tag components are 0 for this simulation
    z_semantic = np.zeros(num_games)
    z_tag = np.zeros(num_games)
    w_semantic = 0.0
    w_tag = 0.0
    
    slider_names = ['quality', 'age', 'popularity', 'discovery', 'length', 'difficulty']
    impacts = {name: [] for name in slider_names}
    
    top_k = 10  # Focused on top 100 as per orientation.md
    print(f"Running {num_samples} simulation samples (Top {top_k} rank focus) using 24 threads...")

    def get_scores(prefs):
        w_spps = QUALITY_WEIGHT_MULTIPLIER * prefs['quality']
        w_date = AGE_WEIGHT_MULTIPLIER * prefs['age']
        w_pop = POPULARITY_WEIGHT_MULTIPLIER * prefs['popularity']
        w_length = LENGTH_WEIGHT_MULTIPLIER * prefs['length']
        w_difficulty = DIFFICULTY_WEIGHT_MULTIPLIER * prefs['difficulty']
        
        grid_idx = int(round((prefs['discovery'] - AP_SLIDER_MIN) / AP_SLIDER_STEP))
        grid_idx = max(0, min(len(AP_SLIDER_VALUES) - 1, grid_idx))
        z_spps = q_grid_clamped[grid_idx]
        
        return calculate_hybrid_score(
            z_semantic, w_semantic,
            z_tag, w_tag,
            z_spps, w_spps,
            z_date, w_date,
            z_pop, w_pop,
            z_length, w_length,
            z_difficulty, w_difficulty
        )

    def simulate_task(_):
        # 1. Pick random initial positions for all sliders
        current_prefs = {
            'quality': random.choice(AP_SLIDER_VALUES),
            'age': random.choice(AP_SLIDER_VALUES),
            'popularity': random.choice(AP_SLIDER_VALUES),
            'discovery': random.choice(AP_SLIDER_VALUES),
            'length': random.choice(AP_SLIDER_VALUES),
            'difficulty': random.choice(AP_SLIDER_VALUES)
        }
        
        # Calculate initial scores
        old_scores = get_scores(current_prefs)
        
        # 2. Identify top K games and their initial ranks
        old_ranks = np.argsort(np.argsort(-old_scores))
        top_k_mask = old_ranks < top_k
        
        # 3. Pick one random slider to change
        slider_to_change = random.choice(slider_names)
        
        # 4. Move it one notch
        current_val = current_prefs[slider_to_change]
        current_idx = AP_SLIDER_VALUES.index(current_val)
        
        if current_idx == 0:
            # Must go up
            new_idx = 1
        elif current_idx == len(AP_SLIDER_VALUES) - 1:
            # Must go down
            new_idx = current_idx - 1
        else:
            # Randomly go up or down
            new_idx = current_idx + random.choice([-1, 1])
            
        new_prefs = current_prefs.copy()
        new_prefs[slider_to_change] = AP_SLIDER_VALUES[new_idx]
        
        # Calculate new scores
        new_scores = get_scores(new_prefs)
        
        # 4. Calculate new ranks
        new_ranks = np.argsort(np.argsort(-new_scores))
        
        # 5. Calculate average rank shift for the top K
        rank_shifts = (new_ranks[top_k_mask].astype(float) - old_ranks[top_k_mask].astype(float)) ** 2
        avg_shift = np.mean(rank_shifts)
        
        return slider_to_change, avg_shift

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(simulate_task, i) for i in range(num_samples)]
        for i, future in enumerate(as_completed(futures)):
            slider_to_change, mse = future.result()
            impacts[slider_to_change].append(mse)
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples...")
        
    print(f"\nSimulation Results (Average Rank Shift per Notch Change - TOP {top_k}):")
    for name in slider_names:
        avg_shift = np.mean(impacts[name])
        rmse = np.sqrt(avg_shift)        
        print(f"  {name.capitalize():<12}: RMSE = {rmse:.2f} positions (MSE = {avg_shift:.2f})")

if __name__ == "__main__":
    run_simulation()

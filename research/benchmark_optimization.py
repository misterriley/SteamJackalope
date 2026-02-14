
import pandas as pd
import numpy as np
import os
import sys
import time
from itertools import product
from multiprocessing import Pool
from functools import partial

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.analyze_playtime_sentiment import estimate_ppp_vectorized
from common.constants import GLOBAL_POSITIVE_RATE

def compute_game_log_likelihood(playtimes, voted_up, gamma, s, a=0.80):
    ppp = estimate_ppp_vectorized(playtimes, voted_up, gamma=gamma, s=s, a=a)
    ll = np.sum(voted_up * np.log(ppp + 1e-12) + (~voted_up) * np.log(1 - ppp + 1e-12))
    return ll

def evaluate_parameters_on_games(params_tuple, games_data):
    gamma, s = params_tuple
    total_ll = 0.0
    for playtimes, voted_up in games_data:
        ll = compute_game_log_likelihood(playtimes, voted_up, gamma, s, GLOBAL_POSITIVE_RATE)
        total_ll += ll
    return total_ll

def main():
    print("Loading reviews...")
    df = pd.read_csv('scraped_reviews.csv')
    
    # Filter for valid reviews
    valid = df[(df['author_playtime_forever'] > 0) & (df['voted_up'].isin([True, False]))]
    counts = valid.groupby('appid').size()
    eligible_games = counts[counts >= 2].index.tolist()
    
    print(f"Eligible games: {len(eligible_games)}")
    
    # Sample 100 games
    np.random.seed(42)
    sample_games = np.random.choice(eligible_games, 100, replace=False)
    
    games_data = []
    for appid in sample_games:
        game_reviews = valid[valid['appid'] == appid]
        playtimes = game_reviews['author_playtime_forever'].values.astype(float)
        voted_up = game_reviews['voted_up'].values.astype(bool)
        
        if len(playtimes) > 200:
            idx = np.random.choice(len(playtimes), 200, replace=False)
            playtimes = playtimes[idx]
            voted_up = voted_up[idx]
        
        games_data.append((playtimes, voted_up))
    
    # Grid size (25x25 = 625)
    gammas = np.logspace(-3, 2, 25)
    s_values = np.logspace(-3, 3, 25)
    param_grid = list(product(gammas, s_values))
    
    print(f"Evaluating {len(param_grid)} parameter combinations on 100 games...")
    
    start_time = time.time()
    
    # Sequential evaluation for 10 combos to get a baseline for one core
    baseline_combos = 10
    for i in range(baseline_combos):
        evaluate_parameters_on_games(param_grid[i], games_data)
    
    end_time = time.time()
    avg_per_combo_per_100_games = (end_time - start_time) / baseline_combos
    
    print(f"Average time per 100 games per combo (1 core): {avg_per_combo_per_100_games:.4f}s")
    
    # Estimate total time for 625 combos on N cores
    n_cores = os.cpu_count() - 1
    total_time_100_games = (avg_per_combo_per_100_games * len(param_grid)) / n_cores
    print(f"Estimated total time for 100 games ({n_cores} cores): {total_time_100_games:.2f}s")
    
    # Target 600s (10 minutes)
    target_games = int(100 * (600 / total_time_100_games))
    print(f"Target number of games for 10 minutes: {target_games}")

if __name__ == "__main__":
    main()

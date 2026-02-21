import pandas as pd
import numpy as np
import os
import sys
import random
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE,
    QUALITY_GRID_FILE,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    EMBEDDINGS_TAG_FILE,
    EMBEDDINGS_STRUCTURAL_NORMS_FILE,
    TAG_VECTORS_FILE,
    TAG_NORMS_FILE,
    AP_SLIDER_VALUES,
    DISC_SLIDER_VALUES,
    AP_SLIDER_MIN,
    AP_SLIDER_STEP,
    DISC_SLIDER_MIN,
    DISC_SLIDER_STEP,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    DOT_PRODUCT_LAMBDA,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    SEMANTIC_SIMILARITY_MEAN,
    SEMANTIC_SIMILARITY_STD,
    EPSILON
)
from common.utils import calculate_hybrid_score

def run_simulation(num_samples=1000):
    print(f"Loading data...")
    metadata = pd.read_parquet(METADATA_FILE)
    num_games = len(metadata)
    
    # Load all vectors
    quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
    embeddings_desc = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    embeddings_desc_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    
    # Metadata z-scores
    z_date = np.clip(metadata['date_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).astype(np.float32)
    z_pop = np.clip(metadata['pop_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).astype(np.float32)
    z_length = np.clip(metadata['playtime_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).astype(np.float32)
    z_difficulty = np.clip(metadata['difficulty_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).astype(np.float32)
    z_price = np.clip(metadata['price_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).astype(np.float32)
    
    slider_names = ['semantic', 'tag', 'quality', 'age', 'popularity', 'discovery', 'length', 'difficulty', 'price']
    impacts = {name: [] for name in slider_names}
    
    print(f"Running {num_samples} simulation iterations...")

    for _ in tqdm(range(num_samples)):
        # 1. Pick a random seed game
        seed_idx = random.randint(0, num_games - 1)
        
        # Calculate semantic similarity to seed
        seed_desc = embeddings_desc[seed_idx].astype(np.float32)
        # Normalize seed vector to unit length as server does
        seed_mag = np.linalg.norm(seed_desc)
        seed_desc_unit = seed_desc / (seed_mag if seed_mag > EPSILON else 1.0)
        
        sem_sims = np.dot(embeddings_desc, seed_desc_unit)
        denom_desc = embeddings_desc_norms + SEMANTIC_DOT_PRODUCT_LAMBDA
        z_semantic = (sem_sims / denom_desc) * SEMANTIC_GLOBAL_SCALING_FACTOR
        z_semantic = (z_semantic - SEMANTIC_SIMILARITY_MEAN) / (SEMANTIC_SIMILARITY_STD + EPSILON)
        
        # Calculate tag similarity to seed
        seed_tag = tag_vectors[seed_idx].astype(np.float32)
        seed_norm = tag_norms[seed_idx].astype(np.float32)
        beta_seed = seed_tag / (seed_norm + DOT_PRODUCT_LAMBDA)
        
        tag_dots = np.dot(tag_vectors, beta_seed)
        denom_tag = tag_norms + DOT_PRODUCT_LAMBDA
        z_tag = (tag_dots / denom_tag) * TAG_GLOBAL_SCALING_FACTOR
        
        # 2. Randomize all sliders
        current_prefs = {
            'semantic': random.choice(AP_SLIDER_VALUES),
            'tag': random.choice(AP_SLIDER_VALUES),
            'quality': random.choice(AP_SLIDER_VALUES),
            'age': random.choice(AP_SLIDER_VALUES),
            'popularity': random.choice(AP_SLIDER_VALUES),
            'discovery': random.choice(DISC_SLIDER_VALUES),
            'length': random.choice(AP_SLIDER_VALUES),
            'difficulty': random.choice(AP_SLIDER_VALUES),
            'price': random.choice(AP_SLIDER_VALUES)
        }
        
        def get_scores(prefs):
            # Discovery grid lookup
            grid_idx = int(round((prefs['discovery'] - DISC_SLIDER_MIN) / DISC_SLIDER_STEP))
            grid_idx = max(0, min(len(DISC_SLIDER_VALUES) - 1, grid_idx))
            z_spps = np.clip(quality_grid[grid_idx], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
            
            return calculate_hybrid_score(
                z_semantic, prefs['semantic'],
                z_tag, prefs['tag'],
                z_spps, prefs['quality'],
                z_date, prefs['age'],
                z_pop, prefs['popularity'],
                z_length, prefs['length'],
                z_difficulty, prefs['difficulty'],
                z_price, prefs['price']
            )
            
        # Initial scores and ranks
        old_scores = get_scores(current_prefs)
        # Exclude the seed game from recommendations as the server does
        old_scores[seed_idx] = -1e12
        
        # argsort twice gives ranks (0-indexed)
        # We want descending order for scores, so -old_scores
        old_ranks = np.argsort(np.argsort(-old_scores))
        
        # 3. For each slider, move it and measure
        for slider in slider_names:
            vals = DISC_SLIDER_VALUES if slider == 'discovery' else AP_SLIDER_VALUES
            current_val = current_prefs[slider]
            current_idx = vals.index(current_val)
            
            # Pick direction
            if current_idx == 0:
                new_idx = 1
            elif current_idx == len(vals) - 1:
                new_idx = current_idx - 1
            else:
                new_idx = current_idx + random.choice([-1, 1])
                
            new_prefs = current_prefs.copy()
            new_prefs[slider] = vals[new_idx]
            
            new_scores = get_scores(new_prefs)
            new_scores[seed_idx] = -1e12
            new_ranks = np.argsort(np.argsort(-new_scores))
            
            # Loss calculation
            # "Only measure for games in the top 30 before or after the move."
            relevant_mask = (old_ranks < 30) | (new_ranks < 30)
            
            # "For any game that is beyond rank 30, say that it is rank 31."
            # In 0-indexing: rank 1 is 0, rank 30 is 29, rank 31 is 30.
            old_clamped = np.minimum(old_ranks[relevant_mask], 30)
            new_clamped = np.minimum(new_ranks[relevant_mask], 30)
            
            sq_diff = np.sum((old_clamped - new_clamped)**2)
            impacts[slider].append(sq_diff)
            
    print(f"\nSimulation Results (Sum of Squares Rank Shift per Notch Change):")
    for name in slider_names:
        avg_ss = np.mean(impacts[name])
        print(f"  {name.capitalize():<12}: Mean Sum of Squares = {avg_ss:.2f}")

if __name__ == "__main__":
    run_simulation(num_samples=500)

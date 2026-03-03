import pandas as pd
import numpy as np
import os
import sys
import json
import ast
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    PRODUCTION_DATA_DIR,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)
from common.utils import calculate_jackalope_kernel, softmin_blend

def explore_neighborhood(steamid="76561198039155404"):
    print(f"--- Neighborhood Explorer for {steamid} ---")
    
    # 1. Load Profile
    profile_path = f"data/user_{steamid}_taste_profile.json"
    if not os.path.exists(profile_path):
        print(f"Error: Profile {profile_path} not found. Run solver first.")
        return
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    # 2. Load Metadata and Vectors
    print("Loading population data...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): i for i, aid in enumerate(full_metadata['appid'])}
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_global = np.clip(quality_grid[0], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    diff_z = np.clip(full_metadata['difficulty_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

    STRICT_ANCHORS = ["Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", "Action-Adventure", "Adventure"]
    HORROR_MARKERS = ["Horror", "Survival Horror", "Psychological Horror", "Gore", "Violent"]
    ALL_ANCHORS = STRICT_ANCHORS + HORROR_MARKERS
    
    anchor_masks = {a: full_metadata['tags'].fillna('').astype(str).str.contains(rf"'{re.escape(a)}':", regex=True).values for a in ALL_ANCHORS}

    # 3. Interactive Loop
    while True:
        try:
            target_input = input("\nEnter Seed AppID (or name, or 'exit'): ").strip()
            if target_input.lower() in ['exit', 'quit', 'q']:
                break
            
            if target_input.isdigit():
                seed_appid = int(target_input)
            else:
                matches = full_metadata[full_metadata['name'].str.contains(target_input, case=False, na=False)]
                if matches.empty:
                    print("No matches found.")
                    continue
                print("\nMatches:")
                for _, m in matches.head(5).iterrows():
                    print(f"  {m['appid']}: {m['name']}")
                seed_appid = int(matches.iloc[0]['appid'])
                print(f"Using: {matches.iloc[0]['name']} ({seed_appid})")

            if seed_appid not in appid_to_idx:
                print(f"AppID {seed_appid} not found in production data.")
                continue
            
            idx_s = appid_to_idx[seed_appid]
            
            # 4. Neighborhood
            print(f"Calculating neighborhood for {full_metadata.iloc[idx_s]['name']}...")
            tags_s = full_metadata.iloc[idx_s]['tags']
            if isinstance(tags_s, str): tags_s = ast.literal_eval(tags_s)
            max_s = max(tags_s.values()) if tags_s else 1.0
            seed_anchors = [a for a in STRICT_ANCHORS if tags_s.get(a, 0) / max_s > 0.25]
            
            # Identify Active Narrative Seeds for floor reduction
            NARRATIVE_TAGS = ["Story Rich", "Choices Matter", "Multiple Endings", "Visual Novel", "Atmospheric", "Emotional"]
            active_narrative = [t for t in NARRATIVE_TAGS if tags_s.get(t, 0) / max_s > 0.3]

            sims = calculate_jackalope_kernel(
                tag_vectors=tag_vectors, tag_norms=tag_norms,
                seed_tag_vec=tag_vectors[idx_s], seed_tag_norm=tag_norms[idx_s],
                sem_vectors=sem_vectors, sem_norms=sem_norms,
                seed_sem_vec=sem_vectors[idx_s], seed_sem_norm=sem_norms[idx_s],
                topic_distributions=topic_distributions, seed_topic_dist=topic_distributions[idx_s],
                topic_means=t_means, topic_stds=t_stds,
                tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
                sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
                seed_anchors=seed_anchors,
                active_narrative_seed=active_narrative,
                candidate_anchor_masks=anchor_masks
            )
            
            neighbor_indices = np.argsort(-sims)[:100]
            
            # 5. Scoring
            print("Applying your taste profile to neighborhood...")
            weights = profile['metadata']
            anchors = profile['kernel_anchors']
            
            q_part = q_global[neighbor_indices] * weights.get('quality', 0)
            diff_part = diff_z[neighbor_indices] * weights.get('difficulty', 0)
            age_part = np.clip(full_metadata.iloc[neighbor_indices]['date_z'].values, -3, 3) * weights.get('age', 0)
            pop_part = np.clip(full_metadata.iloc[neighbor_indices]['pop_z'].values, -3, 3) * weights.get('popularity', 0)
            
            kernel_contrib = np.zeros(100)
            for anchor in anchors:
                a_appid = anchor['appid']
                a_weight = anchor['weight']
                if a_appid not in appid_to_idx: continue
                idx_a = appid_to_idx[a_appid]
                t_sim = (np.dot(tag_vectors[neighbor_indices], tag_vectors[idx_a]) / 
                        ((tag_norms[neighbor_indices] + DOT_PRODUCT_LAMBDA) * (tag_norms[idx_a] + DOT_PRODUCT_LAMBDA))) * TAG_GLOBAL_SCALING_FACTOR
                s_sim = (np.dot(sem_vectors[neighbor_indices], sem_vectors[idx_a]) / 
                        ((sem_norms[neighbor_indices] + SEMANTIC_DOT_PRODUCT_LAMBDA) * (sem_norms[idx_a] + SEMANTIC_DOT_PRODUCT_LAMBDA))) * SEMANTIC_GLOBAL_SCALING_FACTOR
                kernel_contrib += (t_sim * 0.5 + s_sim * 0.5) * a_weight

            predicted_ratings = q_part + diff_part + age_part + pop_part + kernel_contrib + profile.get('intercept', 5.0)
            
            results_df = full_metadata.iloc[neighbor_indices][['appid', 'name']].copy()
            results_df['similarity'] = sims[neighbor_indices]
            results_df['predicted'] = np.clip(predicted_ratings, 0, 10)
            
            print(f"\n--- Top 100 Neighbors for '{full_metadata.iloc[idx_s]['name']}' (Sorted by your Taste) ---")
            sorted_res = results_df.sort_values('predicted', ascending=False)
            
            print(f"{'AppID':<10} | {'Name':<40} | {'Sim':<6} | {'Rating':<6}")
            print("-" * 75)
            for _, row in sorted_res.head(30).iterrows():
                print(f"{row['appid']:<10} | {row['name'][:40]:<40} | {row['similarity']:<6.3f} | {row['predicted']:<6.2f}")
            print(f"\n(Showing top 30 of 100 neighbors. Seed was {full_metadata.iloc[idx_s]['name']})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "76561198039155404"
    explore_neighborhood(sid)

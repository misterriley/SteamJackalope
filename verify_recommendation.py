
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.server import DataManager, RecommendationRequest, recommend, ensure_python_types

def run_verification():
    print("Initializing DataManager...")
    dm = DataManager()
    dm.load_data()
    
    steam_id = "76561198039155404"
    profile_path = f"data/user_{steam_id}_taste_profile.json"
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    
    if not os.path.exists(profile_path):
        print(f"Error: Profile not found at {profile_path}")
        return
        
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        
    df_gt = pd.read_csv(gt_path)
    # Only keep rated games for the linear model (just like the UI does)
    # The UI sends rated_appids and library_details
    rated_df = df_gt[df_gt['status'] == 'rated'].copy()
    rated_appids = rated_df['appid'].tolist()
    
    # Construct library_details
    lib_details = {}
    for _, row in df_gt.iterrows():
        lib_details[int(row['appid'])] = {
            'actual_rating': float(row['actual_rating']) if not pd.isna(row['actual_rating']) else 5.0,
            'p_plus_t': float(row['p_plus_t']) if not pd.isna(row['p_plus_t']) else 0.5
        }
        
    # Get all owned appids to exclude them
    all_owned_appids = df_gt['appid'].tolist()

    # Construct the request matching the Taste DNA "Analyze" state
    meta = profile['metadata']
    req = RecommendationRequest(
        alpha=meta.get('semantic', 1.0),
        beta=meta.get('tag_match', 1.0),
        gamma_topic=meta.get('topic_match', 0.1),
        quality_pref=meta.get('quality', 1.0),
        age_pref=meta.get('age', 0.0),
        pop_pref=meta.get('popularity', 0.0),
        length_pref=meta.get('length', 0.0),
        difficulty_pref=meta.get('difficulty', 0.0),
        price_pref=meta.get('price', 0.0),
        disc_pref=0.0, # Neutral discovery for verification
        vibe_vector=profile['vibe_vector'],
        semantic_vibe_vector=profile['semantic_vibe_vector'],
        topic_vibe_vector=profile['topic_vibe_vector'],
        metadata_weights=profile['metadata'], # Contains kernel_match etc.
        intercept=profile.get('intercept', 5.0),
        scaling_factor=profile.get('scaling_factor', 1.0),
        rated_appids=rated_appids,
        library_appids=all_owned_appids,
        library_details=lib_details,
        profile_filter="all", # Filter out all owned games
        top_k=10
    )
    
    # Inject profile metadata weights (kernel_anchors)
    # The recommend function expects f"MIG_{group}" in weights if they aren't in metadata_weights
    # But wait, recommend() already handles the mapping if metadata_weights is provided.
    
    print("\nExecuting Recommendation...")
    # We call recommend() directly. We need to mock the data_manager global if it's used inside.
    # Actually, recommend() uses the global data_manager. We'll patch it.
    import app.server
    app.server.data_manager = dm
    
    results = recommend(req)
    
    print(f"\nTop 10 Recommendations for {steam_id}:")
    print("-" * 50)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['name']} (Match: {res['match_percent']:.1f}%, Score: {res['weighted_score']:.2f})")
        print(f"   AppID: {res['appid']}")
        
if __name__ == "__main__":
    run_verification()

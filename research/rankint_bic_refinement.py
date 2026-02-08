import sys
import os
import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from scipy.stats import norm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def get_rare_tags(csv_path, threshold=500):
    """Identify tags that appear in fewer than threshold games."""
    df = pd.read_csv(csv_path, low_memory=False)
    tag_counts = {}
    for tags_str in df['tags']:
        if pd.isna(tags_str):
            continue
        try:
            t_data = ast.literal_eval(tags_str)
            t_list = t_data.keys() if isinstance(t_data, dict) else t_data
            for t in t_list:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        except:
            continue
    return [tag for tag, count in tag_counts.items() if count < threshold]

# --- Refinement Constraints ---
EXCLUDED_TAGS = ["Soundtrack", "Great Soundtrack", "Beautiful", "Blood", "Futuristic", "Stylized", "Cyberpunk", "Cartoony", "Cartoon", "VR", "Dystopian"]
INITIAL_FEATURES = [
           "Fast-Paced",
 "Precision Platformer",    
       "Co-op Campaign",    
    "Quick-Time Events",    
                "Ninja",    
           "Platformer",    
   "Twin Stick Shooter",    
           "Mouse only",    
          "Perma Death",    
                 "2.5D",    
          "Bullet Time",    
    "Gun Customization",    
                "Retro",    
           "Souls-like",    
              "Sokoban",    
              "Classic",    
              "Parkour",    
          "Bullet Hell",    
                "Music",    
          "Competitive",    
         "Score Attack",    
         "Level Editor",    
        "Side Scroller",    
               "Rhythm",    
    "Time Manipulation",    
"Character Action Game",    
    "Spectacle fighter",    
           "Controller",    
              "Physics",    
                "Mechs",    
         "Replay Value",    
       "Automobile Sim",    
               "Robots",    
     "Vehicular Combat",    
           "Minimalist",    
                  "PvE",    
               "Combat",    
                "Space",    
           "2D Fighter",    
         "Third Person",    
              "Shooter",    
         "Martial Arts",    
               "Flight",    
               "Aliens",    
 "Third-Person Shooter",    
                "1980s",    
       "Hack and Slash",    
    "Puzzle-Platformer",    
           "3D Fighter",    
     "Top-Down Shooter",    
            "Swordplay",    
          "Destruction",    
       "4 Player Local",    
                  "PvP",    
         "Online Co-Op",      
           "Team-Based",    
        "Arena Shooter",    
             "Tutorial",    
               "Sports",    
             "Abstract",    
              "Driving",    
         "Hero Shooter",    
        "Battle Royale",     
          "Beat 'em up",     
        "2D Platformer",     
        "3D Platformer",     
               "1990's",
]   
MIN_CORRELATION = 0

def rank_int(y):
    """
    Rank-Based Inverse Normal Transformation (Rank-INT).
    Forces data into a standard normal distribution based on ranks.
    Uses Blom's constant (0.375) as a standard offset.
    """
    n = len(y)
    # Calculate ranks (handling ties by average)
    ranks = pd.Series(y).rank(method='average')
    # Blom's constant 3/8 = 0.375
    # Formula: Phi^-1((rank - 3/8) / (n + 1/4))
    probs = (ranks - 0.375) / (n + 0.25)
    return norm.ppf(probs)

def calculate_bic(n, rss, k):
    if rss <= 0: return -np.inf
    return k * np.log(n) + n * np.log(rss / n)

def fit_and_score_bic(X, y, current_indices, candidate_idx):
    indices = current_indices + [candidate_idx]
    X_subset = X[:, indices]
    reg = LinearRegression()
    reg.fit(X_subset, y)
    rss = np.sum((y - reg.predict(X_subset)) ** 2)
    return calculate_bic(X.shape[0], rss, len(indices) + 1), candidate_idx

def fit_and_score_removal_bic(X, y, current_indices, remove_idx):
    indices = [i for i in current_indices if i != remove_idx]
    if not indices:
        rss = np.sum((y - np.mean(y)) ** 2)
        k = 1
    else:
        X_subset = X[:, indices]
        reg = LinearRegression()
        reg.fit(X_subset, y)
        rss = np.sum((y - reg.predict(X_subset)) ** 2)
        k = len(indices) + 1
    return calculate_bic(X.shape[0], rss, k), remove_idx

def main():
    print(f"Initializing Stepwise BIC Refinement with Rank-INT...")
    
    # 1. Load Data
    csv_path = "data/pipeline_games_clean.csv"
    
    # Identify and add rare tags to EXCLUDED_TAGS
    print(f"Identifying tags appearing in fewer than 500 games...")
    rare_tags = get_rare_tags(csv_path, threshold=500)
    EXCLUDED_TAGS.extend(rare_tags)
    print(f"Added {len(rare_tags)} rare tags to EXCLUDED_TAGS.")

    print(f"Exclusions: {EXCLUDED_TAGS}")
    print(f"Constraint: |Zero-Order Correlation| >= {MIN_CORRELATION}")

    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # Generate vectors
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_rankint.npy"
    vectors, appids = gtv.generate_tag_vectors(csv_path, output_vectors="research/temp_vectors_rankint.npy")
    
    # 2. Setup Target Variable (y) with Rank-INT
    target_tags = ["Difficult", "Unforgiving"]
    target_indices = [tag_to_idx[t] for t in target_tags]
    y_raw = vectors[:, target_indices]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y_blended = np.sum(y_z, axis=1)
    
    print("Applying Rank-Based Inverse Normal Transformation (Rank-INT) to y...")
    y = rank_int(y_blended)
    print(f"New y stats - Mean: {y.mean():.4f}, Std: {y.std():.4f}, Min: {y.min():.4f}, Max: {y.max():.4f}")
    
    # 3. Setup Features (X) with Zero-Sum Maintenance and Rank-INT
    removed_sums = np.sum(vectors[:, target_indices], axis=1)
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in target_indices: feature_mask[idx] = False
    
    X_full_raw = vectors[:, feature_mask].copy()
    n_features_full = X_full_raw.shape[1]
    X_full_raw = X_full_raw + (removed_sums / n_features_full).reshape(-1, 1)
    feature_names = np.array(unique_tags)[feature_mask]

    print("Applying Rank-Based Inverse Normal Transformation (Rank-INT) to all feature columns...")
    X_full = np.zeros_like(X_full_raw)
    for i in range(n_features_full):
        X_full[:, i] = rank_int(X_full_raw[:, i])
    
    # 4. Correlation Constraint (on Rank-INT y)
    print("Calculating zero-order correlations with Rank-INT y...")
    correlations = np.array([np.corrcoef(X_full[:, i], y)[0, 1] for i in range(X_full.shape[1])])
    correlation_mask = np.abs(correlations) >= MIN_CORRELATION
    print(f"Features passing correlation threshold (>= {MIN_CORRELATION}): {correlation_mask.sum()}")
    
    # Final exclusion list
    exclude_indices = []
    for i, name in enumerate(feature_names):
        if not correlation_mask[i] or name in EXCLUDED_TAGS or any(ex in name for ex in ["Football", "Soccer"]):
            exclude_indices.append(i)
    
    print(f"Total features excluded: {len(exclude_indices)}")
    
    # 5. Iterative BIC Selection (Starting from INITIAL_FEATURES)
    print(f"Initializing model with: {INITIAL_FEATURES}")
    selected_indices = [np.where(feature_names == t)[0][0] for t in INITIAL_FEATURES if t in feature_names]
    
    # Calculate starting BIC
    if not selected_indices:
        current_bic = calculate_bic(X_full.shape[0], np.sum((y - np.mean(y))**2), 1)
    else:
        reg_start = LinearRegression()
        reg_start.fit(X_full[:, selected_indices], y)
        current_bic = calculate_bic(X_full.shape[0], np.sum((y - reg_start.predict(X_full[:, selected_indices]))**2), len(selected_indices) + 1)
    
    print(f"Initial Model BIC: {current_bic:.4f}")
    
    n_jobs = 24
    n_features = X_full.shape[1]
    
    while True:
        changed = False
        
        # --- Forward Step ---
        candidate_indices = [i for i in range(n_features) if i not in selected_indices and i not in exclude_indices]
        if candidate_indices:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score_bic)(X_full, y, selected_indices, idx) 
                for idx in candidate_indices
            )
            results.sort(key=lambda x: x[0])
            best_candidate_bic, best_candidate_idx = results[0]
            
            if best_candidate_bic < current_bic:
                current_bic = best_candidate_bic
                selected_indices.append(best_candidate_idx)
                changed = True
                print(f"[+] Added '{feature_names[best_candidate_idx]}' | BIC: {current_bic:.4f}")
        
        # --- Backward Step ---
        if selected_indices:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score_removal_bic)(X_full, y, selected_indices, idx)
                for idx in selected_indices
            )
            results.sort(key=lambda x: x[0])
            best_removal_bic, best_removal_idx = results[0]
            
            if best_removal_bic < current_bic:
                current_bic = best_removal_bic
                selected_indices.remove(best_removal_idx)
                changed = True
                print(f"[-] Removed '{feature_names[best_removal_idx]}' | BIC: {current_bic:.4f}")
        
        if not changed:
            print("Convergence reached.")
            break
            
    # 6. Finalize and Save
    selected_features = [feature_names[i] for i in selected_indices]
    X_final = X_full[:, selected_indices]
    reg = LinearRegression()
    reg.fit(X_final, y)
    
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': reg.coef_
    }).sort_values('coefficient', ascending=False)
    
    coef_df.to_csv("research/bic_rankint_coefficients.csv", index=False)
    
    results = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids],
        'actual_raw': y_blended,
        'actual_rankint': y,
        'predicted_rankint': reg.predict(X_final)
    })
    results.to_csv("research/bic_rankint_predictions.csv", index=False)
    
    print(f"\nFinal model selected {len(selected_features)} features.")
    print("Predictors (Rank-INT Scale):")
    print(coef_df.to_string(index=False))
    
    # Cleanup
    for f in ["research/temp_w_tag_rankint.npy", "research/temp_vectors_rankint.npy"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()

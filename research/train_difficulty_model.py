import pandas as pd
import numpy as np
import ast
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from common.constants import DIFFICULTY_PREDICTIONS_FILE

def rank_int(y):
    """Rank-Based Inverse Normal Transformation."""
    n = len(y)
    ranks = pd.Series(y).rank(method='average')
    # Blom's constant 0.375
    probs = (ranks - 0.375) / (n + 0.25)
    return norm.ppf(probs)

def calculate_bic(n, rss, k):
    """Bayesian Information Criterion."""
    if rss <= 0: return -np.inf
    return k * np.log(n) + n * np.log(rss / n)

def parse_tags(df):
    """Extract tags from dataframe and filter rare ones."""
    print("Parsing tags...")
    all_game_tags = []
    tag_counts = {}
    
    for tag_str in tqdm(df['tags'], desc="Scanning tags", smoothing=0):
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            all_game_tags.append({})
            continue
        try:
            # Try to eval as dict (typical format)
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                all_game_tags.append(tags_dict)
                for t in tags_dict.keys():
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            elif isinstance(tags_dict, list):
                # Fallback for list format
                d = {t: 1 for t in tags_dict}
                all_game_tags.append(d)
                for t in tags_dict:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            else:
                all_game_tags.append({})
        except:
            all_game_tags.append({})
            
    # Filter rare tags (< 100 games)
    surviving_tags = sorted([t for t, c in tag_counts.items() if c >= 100])
    print(f"Total unique tags: {len(tag_counts)}")
    print(f"Surviving tags (>= 100 occurrences): {len(surviving_tags)}")
    
    tag_to_idx = {tag: i for i, tag in enumerate(surviving_tags)}
    num_tags = len(surviving_tags)
    num_games = len(df)
    
    # Create dense matrix of proportions
    X = np.zeros((num_games, num_tags), dtype=np.float32)
    for i, tags in enumerate(tqdm(all_game_tags, desc="Building matrix", smoothing=0)):
        row_sum = sum(tags.values()) if tags else 0
        if row_sum > 0:
            for t, c in tags.items():
                if t in tag_to_idx:
                    X[i, tag_to_idx[t]] = c / row_sum
                    
    return X, surviving_tags

def main():
    # 1. Load Data
    print("Loading data...")
    steam_df = pd.read_csv('data/pipeline_games_clean.csv', low_memory=False)
    match_df = pd.read_csv('data/gamefaqs_to_steam_match.csv')
    
    # 2. Extract Tags
    X_full, tag_names = parse_tags(steam_df)
    
    # 3. Align training data
    # Create mapping from appid to index in steam_df
    appid_to_idx = {appid: i for i, appid in enumerate(steam_df['appid'])}
    
    train_indices = []
    y_raw = []
    
    for _, row in match_df.iterrows():
        appid = row['steam_appid']
        if appid in appid_to_idx:
            idx = appid_to_idx[appid]
            train_indices.append(idx)
            y_raw.append(row['gf_difficulty'])
            
    X_train_raw = X_full[train_indices]
    y_train_raw = np.array(y_raw)
    
    print(f"Training samples: {len(y_train_raw)}")
    
    # 4. Rank-INT Transformation for features (X)
    # We keep y in its raw 1-5 scale as requested
    print("Applying Rank-INT transformation to features...")
    y_train = y_train_raw
    
    X_train = np.zeros_like(X_train_raw)
    for i in range(X_train_raw.shape[1]):
        X_train[:, i] = rank_int(X_train_raw[:, i])
        
    # Full dataset Rank-INT for later prediction
    X_all = np.zeros_like(X_full)
    for i in range(X_full.shape[1]):
        X_all[:, i] = rank_int(X_full[:, i])
        
    # 5. Stepwise Selection (BIC)
    print("Starting stepwise feature selection...")
    n, p = X_train.shape
    selected_indices = []
    current_bic = calculate_bic(n, np.sum((y_train - np.mean(y_train))**2), 1)
    
    print(f"Initial BIC: {current_bic:.4f}")
    
    while True:
        best_bic = current_bic
        best_candidate = None
        mode = None # 'add' or 'remove'
        
        # Forward Step
        for i in range(p):
            if i not in selected_indices:
                cand_indices = selected_indices + [i]
                reg = LinearRegression().fit(X_train[:, cand_indices], y_train)
                rss = np.sum((y_train - reg.predict(X_train[:, cand_indices]))**2)
                bic = calculate_bic(n, rss, len(cand_indices) + 1)
                if bic < best_bic:
                    best_bic = bic
                    best_candidate = i
                    mode = 'add'
        
        # Backward Step (only if we have indices to remove)
        if selected_indices:
            for i in selected_indices:
                cand_indices = [idx for idx in selected_indices if idx != i]
                if not cand_indices:
                    rss = np.sum((y_train - np.mean(y_train))**2)
                    bic = calculate_bic(n, rss, 1)
                else:
                    reg = LinearRegression().fit(X_train[:, cand_indices], y_train)
                    rss = np.sum((y_train - reg.predict(X_train[:, cand_indices]))**2)
                    bic = calculate_bic(n, rss, len(cand_indices) + 1)
                
                if bic < best_bic:
                    best_bic = bic
                    best_candidate = i
                    mode = 'remove'
        
        if mode == 'add':
            selected_indices.append(best_candidate)
            current_bic = best_bic
            print(f"[+] Added '{tag_names[best_candidate]}' | BIC: {current_bic:.4f}")
        elif mode == 'remove':
            selected_indices.remove(best_candidate)
            current_bic = best_bic
            print(f"[-] Removed '{tag_names[best_candidate]}' | BIC: {current_bic:.4f}")
        else:
            print("Convergence reached.")
            break
            
    # 6. Final Model
    final_features = [tag_names[i] for i in selected_indices]
    reg_final = LinearRegression().fit(X_train[:, selected_indices], y_train)
    
    print("\nModel Coefficients:")
    coef_df = pd.DataFrame({'Tag': final_features, 'Coef': reg_final.coef_}).sort_values('Coef', ascending=False)
    print(coef_df.to_string(index=False))
    
    # 7. Predictions and Contributions
    print("\nCalculating contributions and saving predictions...")
    X_final_all = X_all[:, selected_indices]
    
    # Calculate raw prediction
    # raw_pred = intercept + sum(coef * X)
    intercept = reg_final.intercept_
    coefs = reg_final.coef_
    
    contributions = X_final_all * coefs
    preds = np.sum(contributions, axis=1) + intercept
    
    # Create detailed output
    output_df = pd.DataFrame({
        'appid': steam_df['appid'],
        'name': steam_df['name'],
        'intercept': intercept,
        'difficulty_predicted_raw': preds
    })
    
    # Add contribution columns
    for i, tag in enumerate(final_features):
        col_name = f"contrib_{tag.lower().replace(' ', '_')}"
        output_df[col_name] = contributions[:, i]
        
    # Final clamped prediction
    output_df['difficulty_predicted_clamped'] = np.clip(preds, 1.0, 5.0)
    
    output_df.to_csv(DIFFICULTY_PREDICTIONS_FILE, index=False)
    print(f"Detailed predictions saved to {DIFFICULTY_PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()

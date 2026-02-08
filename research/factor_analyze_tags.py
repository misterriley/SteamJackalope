import pandas as pd
import numpy as np
import ast
from collections import Counter
import sklearn.utils.validation
import factor_analyzer.factor_analyzer
from factor_analyzer import FactorAnalyzer
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

# Monkeypatch sklearn for factor_analyzer compatibility
original_check_array = sklearn.utils.validation.check_array
def patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return original_check_array(*args, **kwargs)

sklearn.utils.validation.check_array = patched_check_array
sklearn.utils.check_array = patched_check_array
factor_analyzer.factor_analyzer.check_array = patched_check_array

def run_analysis():
    print("Loading data...")
    df = pd.read_csv('data/games_march2025_cleaned.csv')
    
    print("Parsing tags...")
    all_game_tags = []
    global_tag_counts = Counter()
    for tag_str in df['tags']:
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            all_game_tags.append({})
            continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                all_game_tags.append(tags_dict)
                global_tag_counts.update(tags_dict.keys())
            else:
                all_game_tags.append({})
        except:
            all_game_tags.append({})
            
    unique_tags = sorted(global_tag_counts.keys())
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    num_tags = len(unique_tags)
    print(f"Number of unique tags: {num_tags}")
    print(f"Number of games: {len(all_game_tags)}")

    # 1. Initial Processing and Global Distribution G
    print("Calculating Global Tag Distribution G...")
    # Calculate G from all observed votes
    total_votes_per_tag = np.zeros(num_tags)
    for tags in all_game_tags:
        for t, c in tags.items():
            total_votes_per_tag[tag_to_idx[t]] += c
    
    G = total_votes_per_tag / (total_votes_per_tag.sum() + 1e-9)
    
    # 2. Preparation for Iterative Imputation
    print("Preparing tag matrix for iterative imputation...")
    # We store counts and original totals
    orig_tag_counts = np.zeros((len(all_game_tags), num_tags))
    lod_thresholds = np.zeros(len(all_game_tags))
    is_observed = np.zeros((len(all_game_tags), num_tags), dtype=bool)
    
    for i, tags in enumerate(all_game_tags):
        if not tags:
            # Policy: if no tags, setting threshold to 1 as baseline
            lod_thresholds[i] = 1.0
            continue
        
        sorted_votes = sorted(tags.values(), reverse=True)
        v_min = sorted_votes[min(len(sorted_votes), 20) - 1]
        lod_thresholds[i] = v_min
        
        for t, c in tags.items():
            idx = tag_to_idx[t]
            orig_tag_counts[i, idx] = c
            is_observed[i, idx] = True

    # 3. Iterative Imputation with Factor Model
    # Start with baseline LOD imputation
    print("Initializing with LOD imputation...")
    T = orig_tag_counts.copy()
    for i in range(len(all_game_tags)):
        impute_val = lod_thresholds[i] / np.sqrt(2)
        T[i, ~is_observed[i, :]] = impute_val
    
    # Standardize to probabilities
    T_probs = T / T.sum(axis=1, keepdims=True)

    print("Refining imputation using factor structure...")
    # Use a subset of reliable games to build the initial model
    reliable_mask = (orig_tag_counts.sum(axis=1) > 100)
    if reliable_mask.sum() < 100: reliable_mask[:] = True
    
    n_impute_factors = 15
    for iteration in range(1): # Reduced for speed
        print(f"  Iteration {iteration + 1}...")
        # Fit FA on reliable games or all games if small enough
        sample_indices = np.random.choice(np.where(reliable_mask)[0], min(2000, reliable_mask.sum()), replace=False)
        # Use a simpler correlation-based affinity for speed
        C = np.corrcoef(T_probs[sample_indices], rowvar=False)
        np.fill_diagonal(C, 0)
        C = np.clip(C, 0, 1)
        
        for i in range(len(all_game_tags)):
            unobs = ~is_observed[i, :]
            obs = is_observed[i, :]
            if not any(obs): continue
            
            w_obs = T_probs[i, obs]
            affinity = (w_obs @ C[obs, :][:, unobs]) / (w_obs.sum() + 1e-9)
            boost = np.clip(0.1 + 0.9 * affinity, 0.1, 1.0)
            T[i, unobs] = (lod_thresholds[i] / np.sqrt(2)) * boost
            
        T_probs = T / T.sum(axis=1, keepdims=True)

    # 4. Bayesian Regularization (Pseudocounts)
    print("Finding optimal Bayesian pseudocount K...")
    # Prepare CV data
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    cv_indices = np.random.choice(len(T_probs), min(1000, len(T_probs)), replace=False)
    heldout_probs = []
    train_T_imputed = []
    orig_totals_cv = []
    
    for i in cv_indices:
        tags = all_game_tags[i]
        total = sum(tags.values())
        if total < 5: continue
        
        vec = np.zeros(num_tags)
        for t, c in tags.items(): vec[tag_to_idx[t]] = c
        
        num_h = max(1, int(total * 0.2))
        h_votes = np.random.multinomial(num_h, vec / total)
        t_votes = vec - h_votes
        
        # Apply imputation logic to t_votes
        t_total = t_votes.sum()
        num_nonzero = np.count_nonzero(t_votes)
        if t_total == 0 or num_nonzero == 0:
            t_imputed = np.ones(num_tags)
        else:
            v_min_t = sorted(t_votes[t_votes > 0], reverse=True)[min(num_nonzero, 20)-1]
            t_imputed = np.full(num_tags, v_min_t / np.sqrt(2))
            # Just use the raw votes for simplicity in CV
            t_imputed[t_votes > 0] = t_votes[t_votes > 0]
            
        train_T_imputed.append(t_imputed / t_imputed.sum())
        heldout_probs.append(h_votes / h_votes.sum())
        orig_totals_cv.append(t_total)
        
    train_T_imputed = np.array(train_T_imputed)
    heldout_probs = np.array(heldout_probs)
    orig_totals_cv = np.array(orig_totals_cv).reshape(-1, 1)

    def log_likelihood(K):
        if K < 0: return 1e12
        k_vals = K / (orig_totals_cv + K + 1e-9)
        probs = (1 - k_vals) * train_T_imputed + k_vals * G
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.sum(heldout_probs * np.log(probs))

    res = minimize_scalar(log_likelihood, bounds=(0, 2000), method='bounded')
    print(f"Optimal pseudocount K: {res.x:.4f}")

if __name__ == "__main__":
    run_analysis()

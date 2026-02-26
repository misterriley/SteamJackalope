import pandas as pd
import numpy as np
import os
import re
import ast
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import json
import scipy.sparse as sp

# Add parent directory to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.generate_tag_vectors import iterative_em_imputation, optimize_k_stochastic, apply_tag_transform

# Reuse normalization logic from original pipeline
ROMAN_MAP = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
WORD_MAP = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

def normalize_title(title):
    if not isinstance(title, str): return ""
    title = title.lower()
    title = re.sub(r"\b(edition|enhanced|goty|game of the year|complete|anniversary|remastered|remake|collection|bundle|definitive|director's cut|hd)\b", '', title)
    for roman, arabic in ROMAN_MAP.items():
        title = re.sub(r'\b' + roman.lower() + r'\b', str(arabic), title)
    for word, arabic in WORD_MAP.items():
        title = re.sub(r'\b' + word + r'\b', str(arabic), title)
    title = re.sub(r'[^\w\s]', ' ', title)
    title = ' '.join(title.split())
    return title

def parse_gamefaqs_directory(directory):
    all_data = []
    if not os.path.exists(directory):
        return pd.DataFrame(columns=['gf_title', 'difficulty'])
    files = [f for f in os.listdir(directory) if f.endswith('.html')]
    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            table = soup.find('table')
            if not table: continue
            tbody = table.find('tbody')
            if not tbody: continue
            for row in tbody.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 3:
                    a_tag = cols[1].find('a')
                    if not a_tag: continue
                    all_data.append({'gf_title': a_tag.text.strip(), 'difficulty': cols[2].text.strip()})
    df = pd.DataFrame(all_data)
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
    df = df.dropna(subset=['difficulty'])
    df['difficulty'] = (df['difficulty'] - 1.0) * 2.5
    return df.groupby('gf_title')['difficulty'].mean().reset_index()

def main():
    print("Loading Data...")
    games_csv = "data/pipeline_games_clean.csv"
    gf_dir = "data/GameFAQs"
    steam_df = pd.read_csv(games_csv, low_memory=False)
    gf_df = parse_gamefaqs_directory(gf_dir)
    
    print("Matching GameFAQs to Steam...")
    steam_df['norm_name'] = steam_df['name'].apply(normalize_title)
    norm_to_steam = {row['norm_name']: row['appid'] for _, row in steam_df.iterrows() if row['norm_name']}
    
    matched_data = []
    for _, row in gf_df.iterrows():
        norm_gf = normalize_title(row['gf_title'])
        if norm_gf in norm_to_steam:
            matched_data.append({'appid': norm_to_steam[norm_gf], 'difficulty': row['difficulty']})
    
    train_targets_df = pd.DataFrame(matched_data).drop_duplicates('appid')
    print("Matched " + str(len(train_targets_df)) + " unique games for training.")

    # --- Feature Extraction ---
    print("Generating Shrunken + CLR Tag Data (5 EM Iterations)...")
    all_game_tags = []
    global_tags = set()
    for tag_str in tqdm(steam_df['tags'], desc="Parsing tags"):
        if pd.isna(tag_str) or not tag_str:
            all_game_tags.append({})
            continue
        try:
            d = ast.literal_eval(tag_str)
            if isinstance(d, dict):
                all_game_tags.append(d)
                global_tags.update(d.keys())
            else: all_game_tags.append({})
        except: all_game_tags.append({})
            
    unique_tags = sorted(list(global_tags))
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    row_ind, col_ind, data = [], [], []
    for i, tags in enumerate(all_game_tags):
        for t, c in tags.items():
            row_ind.append(i)
            col_ind.append(tag_to_idx[t])
            data.append(c)
    sparse_counts = sp.csr_matrix((data, (row_ind, col_ind)), shape=(len(steam_df), len(unique_tags)), dtype=np.float32)
    original_total_votes = np.array(sparse_counts.sum(axis=1)).flatten()

    augmented_counts, G_final = iterative_em_imputation(sparse_counts, max_iter=5)
    K = optimize_k_stochastic(augmented_counts, sparse_counts, G_final)
    X_tags, _ = apply_tag_transform(augmented_counts, G_final, original_total_votes, K, transform_type='clr')

    print("Loading Topics...")
    X_topics = np.load("data/production/topic_distributions.npy")
    
    X_all = np.hstack([X_tags, X_topics])
    feature_names = unique_tags + ["topic_" + str(i) for i in range(X_topics.shape[1])]

    # Prepare Dataset
    appid_to_row = {aid: i for i, aid in enumerate(steam_df['appid'])}
    train_indices = [appid_to_row[aid] for aid in train_targets_df['appid'] if aid in appid_to_row]
    y = train_targets_df[train_targets_df['appid'].isin(appid_to_row)]['difficulty'].values
    X = X_all[train_indices]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    results = {}

    print("\nTraining Lasso (L1)...")
    lasso = LassoCV(cv=20, random_state=42, n_jobs=-1, max_iter=5000)
    lasso.fit(X_train, y_train)
    results['Lasso (L1)'] = lasso

    print("Training Ridge (L2)...")
    ridge = RidgeCV(cv=20)
    ridge.fit(X_train, y_train)
    results['Ridge (L2)'] = ridge

    print("Training Elastic Net...")
    enet = ElasticNetCV(cv=20, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=42, n_jobs=-1, max_iter=5000)
    enet.fit(X_train, y_train)
    results['Elastic Net'] = enet

    print("\n" + "="*80)
    print("Model                | Train R2   | Test R2    | Test RMSE  | Non-Zero Coefs")
    print("-" * 80)
    
    for name, model in results.items():
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        tr_r2 = r2_score(y_train, train_pred)
        te_r2 = r2_score(y_test, test_pred)
        te_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        non_zero = np.sum(model.coef_ != 0)
        
        print(name.ljust(20) + " | " + 
              format(tr_r2, ".4f").ljust(10) + " | " + 
              format(te_r2, ".4f").ljust(10) + " | " + 
              format(te_rmse, ".4f").ljust(10) + " | " + 
              str(non_zero))
    
    print("="*80)
    print("Elastic Net Optimal L1 Ratio: " + str(enet.l1_ratio_))
    print("Elastic Net Optimal Alpha:    " + str(enet.alpha_))

if __name__ == "__main__":
    main()

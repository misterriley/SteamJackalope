import pandas as pd
import numpy as np
import os
import re
import ast
import json
import scipy.sparse as sp
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, norm
import pyarrow.parquet as pq
import pyarrow as pa

# Add parent directory to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    METADATA_FILE, 
    TOPIC_DISTRIBUTIONS_FILE, 
    DIFFICULTY_PREDICTIONS_FILE,
    TAG_EM_ITERATIONS,
    ROOT_DIR
)
from pipeline.generate_tag_vectors import iterative_em_imputation, optimize_k_stochastic, apply_tag_transform

def has_numeric_mismatch(s1, s2):
    """
    Returns True if s1 and s2 contain different numbers.
    Used to prevent sequels from matching the original in title-matching rescues.
    """
    nums1 = set(re.findall(r'\d+', s1))
    nums2 = set(re.findall(r'\d+', s2))
    return nums1 != nums2

def calculate_bic(n, rss, k):
    """
    Calculates the Bayesian Information Criterion.
    n: number of samples
    rss: residual sum of squares
    k: number of parameters
    """
    if n <= 0: return 0
    # Use the formula: n * ln(RSS/n) + k * ln(n)
    # We use log10 or ln? Usually ln.
    if rss <= 0: rss = 1e-12
    return n * np.log(rss / n) + k * np.log(n)

def rank_int(data, c=3.0/8.0):
    """
    Rank-based Inverse Normal Transformation.
    Maps data to a normal distribution.
    """
    # data: (n_samples, n_features) or (n_samples,)
    data_array = np.asarray(data)
    is_1d = data_array.ndim == 1
    if is_1d:
        data_array = data_array[:, np.newaxis]
        
    n = data_array.shape[0]
    transformed = np.zeros_like(data_array, dtype=np.float32)
    for i in range(data_array.shape[1]):
        col = data_array[:, i]
        # Handle constant columns
        if np.all(col == col[0]):
            transformed[:, i] = 0
            continue
        # Get ranks (handle ties by averaging)
        ranks = rankdata(col, method='average')
        # Map to 0-1
        prob = (ranks - c) / (n - 2*c + 1)
        # Map to normal
        transformed[:, i] = norm.ppf(prob)
        
    return transformed.flatten() if is_1d else transformed

def normalize_title(title):
    if not isinstance(title, str): return ""
    ROMAN_MAP = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
    WORD_MAP = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
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
    df['difficulty'] = (df['difficulty'] - 1.0) * 2.5 # Scale 1-5 to 0-10
    return df.groupby('gf_title')['difficulty'].mean().reset_index()

def main():
    print("--- Difficulty Model Generation (Build 68 - Rank-INT) ---")
    games_csv = os.path.join(ROOT_DIR, "data", "pipeline_games_clean.csv")
    gf_dir = os.path.join(ROOT_DIR, "data", "GameFAQs")
    
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

    # 2. Feature Extraction (Tags)
    print("Generating Shrunken + CLR Tag Data (" + str(TAG_EM_ITERATIONS) + " EM Iterations)...")
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

    augmented_counts, G_final = iterative_em_imputation(sparse_counts, max_iter=TAG_EM_ITERATIONS)
    K = optimize_k_stochastic(augmented_counts, sparse_counts, G_final)
    X_tags, _ = apply_tag_transform(augmented_counts, G_final, original_total_votes, K, transform_type='clr')

    # 3. Topic Data
    print("Loading Topics...")
    X_topics = np.load(TOPIC_DISTRIBUTIONS_FILE)
    
    # 4. Prepare Training Data
    X_all_raw = np.hstack([X_tags, X_topics])
    feature_names = unique_tags + ["topic_" + str(i) for i in range(X_topics.shape[1])]

    # --- Rank-INT Transformation ---
    # We Rank-INT the FULL dataset to ensure everyone is on the same normal scale.
    # This prevents extrapolation errors for extreme values.
    print("Applying Rank-INT to all features...")
    X_all = rank_int(X_all_raw)
    
    appid_to_row = {aid: i for i, aid in enumerate(steam_df['appid'])}
    train_indices = [appid_to_row[aid] for aid in train_targets_df['appid'] if aid in appid_to_row]
    y_train = train_targets_df[train_targets_df['appid'].isin(appid_to_row)]['difficulty'].values
    
    # Target Rank-INT? 
    # Usually we don't Rank-INT the target if we want to preserve the 0-10 scale.
    # But we can standard scale it.
    X_train = X_all[train_indices]
    
    # 5. Train Final Model (L1)
    print("Training Final L1 Model (Rank-INT features)...")
    model = LassoCV(cv=20, random_state=42, n_jobs=-1, max_iter=10000)
    model.fit(X_train, y_train)
    
    # 6. Predict for ALL Games
    print("Generating predictions for all games...")
    predictions = model.predict(X_all)
    
    # Clamp 0-10
    predictions = np.clip(predictions, 0, 10)
    
    # 7. Save Artifacts
    pred_df = pd.DataFrame({
        'appid': steam_df['appid'],
        'difficulty_predicted': predictions
    })
    pred_df.to_csv(DIFFICULTY_PREDICTIONS_FILE, index=False)
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    coef_df = coef_df[coef_df['coefficient'] != 0].sort_values('coefficient', ascending=False)
    
    COEF_FILE = os.path.join(os.path.dirname(DIFFICULTY_PREDICTIONS_FILE), "difficulty_coefficients.json")
    with open(COEF_FILE, 'w') as f:
        json.dump(coef_df.to_dict(orient='records'), f, indent=4)

    # 8. Update Metadata Parquet
    print("Updating metadata.parquet...")
    metadata = pd.read_parquet(METADATA_FILE)
    if 'difficulty_predicted' in metadata.columns:
        metadata.drop(columns=['difficulty_predicted'], inplace=True)
    if 'difficulty_z' in metadata.columns:
        metadata.drop(columns=['difficulty_z'], inplace=True)
        
    metadata = metadata.merge(pred_df, on='appid', how='left')
    metadata['difficulty_predicted'] = metadata['difficulty_predicted'].fillna(5.0)
    
    diff_scores = metadata['difficulty_predicted'].values
    mean_diff = np.mean(diff_scores)
    std_diff = np.std(diff_scores)
    metadata['difficulty_z'] = (metadata['difficulty_predicted'] - mean_diff) / (std_diff + 1e-9)
    
    table = pa.Table.from_pandas(metadata)
    pq.write_table(table, METADATA_FILE)
    
    print("Difficulty Stats: Mean=" + str(mean_diff) + ", Std=" + str(std_diff))
    print("Success!")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import ast
import os
import re
import difflib
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import DIFFICULTY_PREDICTIONS_FILE

# Roman to Arabic numeral mapping
ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
}

# Word to Arabic numeral mapping
WORD_MAP = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

def normalize_title(title):
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r'\b(edition|enhanced|goty|game of the year|complete|anniversary|remastered|remake|collection|bundle|definitive|director\'s cut|hd)\b', '', title)
    for roman, arabic in ROMAN_MAP.items():
        title = re.sub(r'\b' + roman.lower() + r'\b', str(arabic), title)
    for word, arabic in WORD_MAP.items():
        title = re.sub(r'\b' + word + r'\b', str(arabic), title)
    title = re.sub(r'[^\w\s]', ' ', title)
    title = ' '.join(title.split())
    return title

def has_numeric_mismatch(s1, s2):
    nums1 = set(re.findall(r'\d+', s1))
    nums2 = set(re.findall(r'\d+', s2))
    return nums1 != nums2

def rank_int(y):
    n = len(y)
    ranks = pd.Series(y).rank(method='average')
    probs = (ranks - 0.375) / (n + 0.25)
    return norm.ppf(probs)

def calculate_bic(n, rss, k):
    if rss <= 0: return -np.inf
    return k * np.log(n) + n * np.log(rss / n)

def parse_gamefaqs_directory(directory):
    all_data = []
    if not os.path.exists(directory):
        print(f"Warning: {directory} not found.")
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

    print("Difficulty distribution before rescaling:")    
    print(df['difficulty'].describe())

    # Rescale from 1-5 to 0-10: new = (old - 1) * 2.5
    df['difficulty'] = (df['difficulty'] - 1.0) * 2.5

    print("Difficulty distribution after rescaling:")
    print(df['difficulty'].describe())
    
    return df.groupby('gf_title')['difficulty'].mean().reset_index()

def parse_steam_tags(df):
    all_game_tags = []
    tag_counts = {}
    for tag_str in df['tags']:
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            all_game_tags.append({})
            continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                all_game_tags.append(tags_dict)
                for t in tags_dict.keys():
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            elif isinstance(tags_dict, list):
                d = {t: 1 for t in tags_dict}
                all_game_tags.append(d)
                for t in tags_dict:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            else:
                all_game_tags.append({})
        except:
            all_game_tags.append({})
    return all_game_tags, tag_counts

def main():
    print("Starting difficulty model generation...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/pipeline_games_clean.csv")
    parser.add_argument("--gamefaqs", default="data/GameFAQs")
    parser.add_argument("--output", default=DIFFICULTY_PREDICTIONS_FILE)
    args = parser.parse_args()

    print("Step 1: Parsing Data...")
    steam_df = pd.read_csv(args.games, low_memory=False)
    gf_df = parse_gamefaqs_directory(args.gamefaqs)
    
    print("Step 2: Matching GameFAQs to Steam...")
    steam_df['norm_name'] = steam_df['name'].apply(normalize_title)
    norm_to_steam = {}
    for _, row in steam_df.iterrows():
        norm = row['norm_name']
        if norm:
            if norm not in norm_to_steam: norm_to_steam[norm] = []
            norm_to_steam[norm].append((row['appid'], row['name']))
    
    all_steam_norms = list(norm_to_steam.keys())
    matched_data = []
    for _, row in gf_df.iterrows():
        norm_gf = normalize_title(row['gf_title'])
        if not norm_gf: continue
        
        match = None
        if norm_gf in norm_to_steam:
            match = norm_to_steam[norm_gf][0]
        else:
            is_single_word = re.match(r'^[a-z]+$', norm_gf)
            if not is_single_word:
                # Conservative substring
                for sn in all_steam_norms:
                    if (norm_gf in sn or sn in norm_gf) and not has_numeric_mismatch(norm_gf, sn):
                        ratio = min(len(norm_gf), len(sn)) / max(len(norm_gf), len(sn))
                        if ratio > 0.85:
                            match = norm_to_steam[sn][0]
                            break
        if match:
            matched_data.append({'appid': match[0], 'difficulty': row['difficulty']})
    
    train_df = pd.DataFrame(matched_data)
    if train_df.empty:
        # If no matches, create empty DataFrame with expected columns to avoid KeyErrors
        train_df = pd.DataFrame(columns=['appid', 'difficulty'])
        
    print(f"Matched {len(train_df)} games for training.")

    print("Step 3: Preparing Tag Matrix...")
    all_game_tags, tag_counts = parse_steam_tags(steam_df)
    surviving_tags = sorted([t for t, c in tag_counts.items() if c >= 100])
    tag_to_idx = {tag: i for i, tag in enumerate(surviving_tags)}
    X_full = np.zeros((len(steam_df), len(surviving_tags)), dtype=np.float32)
    for i, tags in enumerate(all_game_tags):
        row_sum = sum(tags.values()) if tags else 0
        if row_sum > 0:
            for t, c in tags.items():
                if t in tag_to_idx: X_full[i, tag_to_idx[t]] = c / row_sum
    
    # Rank-INT features
    X_all_rankint = np.zeros_like(X_full)
    for i in range(X_full.shape[1]):
        X_all_rankint[:, i] = rank_int(X_full[:, i])

    # Filter training
    appid_to_idx = {appid: i for i, appid in enumerate(steam_df['appid'])}
    
    if not train_df.empty:
        train_indices = [appid_to_idx[aid] for aid in train_df['appid'] if aid in appid_to_idx]
        y_train = train_df[train_df['appid'].isin(appid_to_idx)]['difficulty'].values
        X_train = X_all_rankint[train_indices]
    else:
        # Handle case with no training data (e.g. tests or missing GameFAQs data)
        print("Warning: No training data found. Skipping model training.")
        train_indices = []
        y_train = np.array([])
        X_train = np.array([]).reshape(0, X_all_rankint.shape[1])

    print("Step 4: Stepwise BIC Selection...")
    n, p = X_train.shape
    selected = []
    
    if n > 10: # Only train if we have enough data
        current_bic = calculate_bic(n, np.sum((y_train - np.mean(y_train))**2), 1)
        
        while True:
            best_bic, best_cand, mode = current_bic, None, None
            for i in range(p):
                if i not in selected:
                    cand = selected + [i]
                    reg = LinearRegression().fit(X_train[:, cand], y_train)
                    rss = np.sum((y_train - reg.predict(X_train[:, cand]))**2)
                    bic = calculate_bic(n, rss, len(cand) + 1)
                    if bic < best_bic: best_bic, best_cand, mode = bic, i, 'add'
            if selected:
                for i in selected:
                    cand = [idx for idx in selected if idx != i]
                    rss = np.sum((y_train - np.mean(y_train))**2) if not cand else np.sum((y_train - LinearRegression().fit(X_train[:, cand], y_train).predict(X_train[:, cand]))**2)
                    bic = calculate_bic(n, rss, len(cand) + 1)
                    if bic < best_bic: best_bic, best_cand, mode = bic, i, 'remove'
            
            if mode == 'add':
                selected.append(best_cand)
                current_bic = best_bic
            elif mode == 'remove':
                selected.remove(best_cand)
                current_bic = best_bic
            else: break
        
        print(f"Selected {len(selected)} tags.")
        reg_final = LinearRegression().fit(X_train[:, selected], y_train)
        intercept = reg_final.intercept_
        coefs = reg_final.coef_
    else:
        print("Insufficient training data. Using default prediction (5.0).")
        selected = []
        intercept = 5.0
        coefs = []
    
    print("Step 5: Saving Predictions...")
    if selected:
        X_final_all = X_all_rankint[:, selected]
        contributions = X_final_all * coefs
        preds = np.sum(contributions, axis=1) + intercept
    else:
        contributions = np.zeros((len(steam_df), 0))
        preds = np.full(len(steam_df), intercept)
    
    print(f"Predicted difficulty range before clipping: {preds.min():.2f} to {preds.max():.2f}")
    print("Predicted difficulty distribution before clipping:")
    print(pd.Series(preds).describe())
    
    output_df = pd.DataFrame({
        'appid': steam_df['appid'],
        'intercept': intercept,
        'difficulty_predicted_raw': preds
    })
    
    # Add contribution columns
    final_features = [surviving_tags[i] for i in selected]
    for i, tag in enumerate(final_features):
        col_name = f"contrib_{tag.lower().replace(' ', '_')}"
        output_df[col_name] = contributions[:, i]
        
    output_df['difficulty_predicted'] = np.clip(preds, 0.0, 10.0)
    
    output_df.to_csv(args.output, index=False)

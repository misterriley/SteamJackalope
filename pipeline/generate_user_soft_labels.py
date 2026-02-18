import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys
import json

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    API_KEY, 
    GLOBAL_POSITIVE_RATE, 
    QUALITY_SCORE_S_CONST,
    PLAYTIME_SENTIMENT_GAMMA,
    PLAYTIME_SENTIMENT_S,
    METADATA_FILE,
    QUALITY_TO_RATING_SLOPE,
    QUALITY_TO_RATING_INTERCEPT
)
from common.utils import calculate_personalized_quality
from research.analyze_playtime_sentiment import estimate_ppp_vectorized

def generate_soft_labels(user_library_path, reviews_path='scraped_reviews.csv', output_path=None):
    """
    Generates predicted ratings (soft labels) for a user's library.
    """
    if not os.path.exists(user_library_path):
        print(f"Error: User library file not found: {user_library_path}")
        return None
        
    print(f"Loading user library from {user_library_path}...")
    user_df = pd.read_csv(user_library_path)
    user_appids = user_df['appid'].unique()
    
    if 'playtime_forever' not in user_df.columns:
        print("Error: user_library.csv must contain 'playtime_forever' (in minutes).")
        return None

    print(f"Loading metadata from {METADATA_FILE}...")
    # We need appid, name, positive, negative
    metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'positive', 'negative'])
    
    # Merge user library with metadata
    user_games = user_df.merge(metadata, on='appid', how='inner', suffixes=('', '_global'))
    
    # Track playtime presence for UI filtering
    user_games['has_playtime'] = user_games['playtime_forever'] > 0
    
    print(f"Found metadata for {len(user_games)} games in library ({np.sum(~user_games['has_playtime'])} with zero playtime).")

    if user_games.empty:
        print("No games in library matched the metadata.")
        return None

    # Load reviews - this is the slow part. We only load what we need.
    print(f"Loading and filtering reviews from {reviews_path} (this may take a minute)...")
    # Use chunksize to avoid loading 1.2GB at once if possible, or just use usecols
    needed_cols = ['appid', 'voted_up', 'author_playtime_forever']
    
    # Efficient loading: only games in user library with playtime
    user_active_appids = user_games[user_games['has_playtime']]['appid'].unique()
    reviews_iter = pd.read_csv(reviews_path, usecols=needed_cols, chunksize=100000)
    relevant_reviews_list = []
    for chunk in reviews_iter:
        relevant_reviews_list.append(chunk[chunk['appid'].isin(user_active_appids)])
    
    reviews_df = pd.concat(relevant_reviews_list) if relevant_reviews_list else pd.DataFrame(columns=needed_cols)
    print(f"Loaded {len(reviews_df)} relevant reviews for {reviews_df['appid'].nunique()} games.")

    results = []

    for idx, row in user_games.iterrows():
        appid = row['appid']
        user_playtime = row['playtime_forever']
        p = row['positive']
        n = row['negative']
        user_voted_up = row.get('user_voted_up', np.nan)
        user_review_text = row.get('user_review_text', "")
        
        # 1. Calculate Global Quality Q (probit)
        # Bayesian smoothing toward global prior
        s = QUALITY_SCORE_S_CONST
        a = GLOBAL_POSITIVE_RATE
        prob = (p + s * a) / (p + n + s)
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        q_global = norm.ppf(prob)
        
        if not row['has_playtime']:
            # For zero-playtime games, use a neutral default and global Q
            results.append({
                'appid': appid,
                'name': row['name'],
                'playtime_forever': 0,
                'user_voted_up': np.nan,
                'user_review_text': "",
                'global_q': q_global,
                'p_plus_t': prob,
                'personalized_q': q_global,
                'predicted_rating': 5, # Neutral middle
                'has_playtime': False
            })
            continue

        # 2. Calculate Personalized Probability p+(t)
        if pd.notna(user_voted_up):
            # Use real user review as ground truth (Hard Label)
            p_plus_t = 1.0 if user_voted_up else 0.0
            print(f"Using user review for {row['name']}: {user_voted_up}")
        else:
            game_reviews = reviews_df[reviews_df['appid'] == appid]
            
            if game_reviews.empty:
                 # If no reviews found for a game you've played, we'll keep it but warn
                 # (This shouldn't happen often if our scrape is good)
                 p_plus_t = prob
            else:
                playtimes = game_reviews['author_playtime_forever'].values.astype(float)
                
                voted_up = game_reviews['voted_up'].values.astype(bool)
                
                # Use global parameters
                gamma = PLAYTIME_SENTIMENT_GAMMA
                s_reg = PLAYTIME_SENTIMENT_S
                
                # Predict p+ at user's playtime
                # Custom kernel prediction for the user's playtime
                from research.analyze_playtime_sentiment import lognormal_kernel_vectorized
                weights = lognormal_kernel_vectorized(user_playtime, playtimes, gamma)
                
                numerator = np.sum(weights * voted_up) + s_reg * a
                denominator = np.sum(weights) + s_reg
                p_plus_t = numerator / denominator

        # 3. Calculate Personalized Quality
        personalized_q = calculate_personalized_quality(np.array([q_global]), np.array([p_plus_t]))[0]
        
        # 4. Map to 0-10 scale using calibrated anchors
        rating = QUALITY_TO_RATING_INTERCEPT + QUALITY_TO_RATING_SLOPE * personalized_q
        rating = int(np.clip(np.round(rating), 0, 10))
        
        results.append({
            'appid': appid,
            'name': row['name'],
            'playtime_forever': user_playtime,
            'user_voted_up': user_voted_up,
            'user_review_text': user_review_text,
            'global_q': q_global,
            'p_plus_t': p_plus_t,
            'personalized_q': personalized_q,
            'predicted_rating': rating,
            'has_playtime': True
        })

    results_df = pd.DataFrame(results)
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Soft labels saved to {output_path}")
        
    return results_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/generate_user_soft_labels.py <user_library_csv_or_steamid>")
        sys.exit(1)
        
    input_val = sys.argv[1]
    
    # Handle being passed a full path or just the ID
    if input_val.endswith('.csv'):
        user_lib = input_val
        steamid = os.path.basename(user_lib).replace('user_', '').replace('_library.csv', '')
    else:
        # Assume it's a SteamID
        steamid = input_val
        user_lib = f"data/user_{steamid}_library.csv"
        
    output = f"data/user_{steamid}_soft_labels.csv"
    
    generate_soft_labels(user_lib, output_path=output)

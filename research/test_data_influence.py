import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# Constants
MODELS = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
TEST_WORDS = [
    "happiness", 
    "melancholy", 
    "adrenaline", 
    "cozy", 
    "challenging", 
    "atmospheric", 
    "freedom", 
    "betrayal", 
    "friendship", 
    "terror"
]

def run_influence_test():
    print("Loading data...")
    # Load enough games to find 100 with reviews
    games_df = pd.read_csv('data/pipeline_games_clean.csv').head(500) 
    reviews_df = pd.read_csv('scraped_reviews.csv')
    
    # Bundle reviews (limit to first 5)
    reviews_bundled = reviews_df.groupby('appid')['review_text'].apply(lambda x: " | ".join(map(str, x.head(5)))).reset_index()
    df = games_df.merge(reviews_bundled, on='appid', how='inner').head(100)
    
    print(f"Testing with {len(df)} games.")
    
    desc_only = df['short_description'].fillna('').tolist()
    # Manual string concatenation to avoid formatting issues in script writing
    desc_plus_rev = []
    for i in range(len(df)):
        d = str(df.iloc[i]['short_description']) if pd.notna(df.iloc[i]['short_description']) else ""
        r = str(df.iloc[i]['review_text']) if pd.notna(df.iloc[i]['review_text']) else ""
        desc_plus_rev.append("Description: " + d + " Reviews: " + r)
    
    results = {}

    for model_name in MODELS:
        print(f"\n--- Testing Model: {model_name} ---")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        # 1. Embed Desc Only
        print("Generating 'Description Only' embeddings...")
        emb_desc = model.encode(desc_only, show_progress_bar=True)
        emb_desc = emb_desc / (np.linalg.norm(emb_desc, axis=1, keepdims=True) + 1e-9)
        
        # 2. Embed Desc + Reviews
        print("Generating 'Description + Reviews' embeddings...")
        emb_plus = model.encode(desc_plus_rev, show_progress_bar=True)
        emb_plus = emb_plus / (np.linalg.norm(emb_plus, axis=1, keepdims=True) + 1e-9)
        
        model_results = {}
        for query in TEST_WORDS:
            query_vec = model.encode([query])[0]
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            
            # Match against Desc Only
            sims_desc = np.dot(emb_desc, query_vec)
            idx_desc = np.argmax(sims_desc)
            
            # Match against Desc + Rev
            sims_plus = np.dot(emb_plus, query_vec)
            idx_plus = np.argmax(sims_plus)
            
            model_results[query] = {
                "desc": {"name": str(df.iloc[idx_desc]['name']), "score": float(sims_desc[idx_desc])},
                "plus": {"name": str(df.iloc[idx_plus]['name']), "score": float(sims_plus[idx_plus])}
            }
        
        results[model_name] = model_results

    # Print Comparison Table
    for model_name in MODELS:
        if model_name not in results: continue
        print("\n" + "="*90)
        print(f"MODEL: {model_name}")
        print("-"*90)
        print(f"{'QUERY':<15} | {'DESC ONLY':<35} | {'DESC + REVIEWS':<35}")
        print("-"*90)
        for query in TEST_WORDS:
            res = results[model_name][query]
            d_name = res['desc']['name'][:33]
            p_name = res['plus']['name'][:33]
            print(f"{query:<15} | {d_name:<35} | {p_name:<35}")
            print(f"{'':<15} | ({res['desc']['score']:.3f}) | ({res['plus']['score']:.3f})")
            print("-"*90)

if __name__ == "__main__":
    if not os.path.exists('data/pipeline_games_clean.csv') or not os.path.exists('scraped_reviews.csv'):
        print("Required CSV files missing.")
    else:
        run_influence_test()

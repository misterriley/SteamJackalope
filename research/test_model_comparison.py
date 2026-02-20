import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# Constants
NEW_MODEL = 'all-mpnet-base-v2'
OLD_MODEL = 'all-MiniLM-L6-v2'
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

def run_comparison_test():
    print(f"Loading data...")
    # Load games and reviews
    games_df = pd.read_csv('data/pipeline_games_clean.csv').head(500) 
    reviews_df = pd.read_csv('scraped_reviews.csv')
    
    # Bundle reviews (limit to first 5 for speed in this test)
    reviews_bundled = reviews_df.groupby('appid')['review_text'].apply(lambda x: " | ".join(map(str, x.head(5)))).reset_index()
    df = games_df.merge(reviews_bundled, on='appid', how='inner').head(100)
    
    print(f"Testing with {len(df)} games.")
    
    # Construct descriptive text
    df['text'] = "Description: " + df['short_description'].fillna('') + " Reviews: " + df['review_text'].fillna('')
    texts = df['text'].tolist()
    
    results = {}

    for model_name in [OLD_MODEL, NEW_MODEL]:
        print(f"\n--- Testing Model: {model_name} ---")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        print(f"Generating embeddings...")
        game_embs = model.encode(texts, show_progress_bar=True)
        # Unit normalize
        game_embs = game_embs / (np.linalg.norm(game_embs, axis=1, keepdims=True) + 1e-9)
        
        model_results = {}
        for query in TEST_WORDS:
            query_vec = model.encode([query])[0]
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            
            sims = np.dot(game_embs, query_vec)
            top_indices = np.argsort(-sims)[:3]
            
            matches = []
            for idx in top_indices:
                matches.append({
                    "name": df.iloc[idx]['name'],
                    "score": float(sims[idx])
                })
            model_results[query] = matches
        
        results[model_name] = model_results

    # Print Comparison
    print("\n" + "="*80)
    print(f"{'QUERY':<15} | {'MiniLM (Old)':<30} | {'MPNet (New)':<30}")
    print("-"*80)
    
    for query in TEST_WORDS:
        if OLD_MODEL in results and NEW_MODEL in results:
            old_top = results[OLD_MODEL][query][0]
            new_top = results[NEW_MODEL][query][0]
            print(f"{query:<15} | {old_top['name'][:28]:<30} | {new_top['name'][:28]:<30}")
            print(f"{'':<15} | ({old_top['score']:.3f}) | ({new_top['score']:.3f})")
            print("-"*80)

if __name__ == "__main__":
    if not os.path.exists('data/pipeline_games_clean.csv') or not os.path.exists('scraped_reviews.csv'):
        print("Required CSV files missing.")
    else:
        run_comparison_test()

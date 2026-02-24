
import pandas as pd
import numpy as np
import os
import sys
import json
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, 
    EMBEDDINGS_DESC_FILE,
    ROOT_DIR
)

def run_bertopic_research(sample_size=10000):
    print(f"--- BERTopic Research: {sample_size} Game Sample ---")
    
    # 1. Load Data
    csv_path = os.path.join(ROOT_DIR, "data", "pipeline_games_clean.csv")
    reviews_path = os.path.join(ROOT_DIR, "scraped_reviews.csv")
    
    print("Loading games and reviews...")
    df = pd.read_csv(csv_path)
    
    # Group reviews to join with games
    if os.path.exists(reviews_path):
        reviews_df = pd.read_csv(reviews_path)
        # Use a subset of reviews to keep text manageable but informative
        reviews_bundled = reviews_df.groupby('appid')['review_text'].apply(lambda x: " ".join(map(str, list(x)[:5]))).reset_index()
        df = df.merge(reviews_bundled, on='appid', how='left')
        df['review_text'] = df['review_text'].fillna('')
    else:
        print("Warning: Reviews not found. Using descriptions only.")
        df['review_text'] = ''
        
    # Clean text: combine, lowercase, remove prefixes that might pollute topics
    df['clean_text'] = (
        df['short_description'].fillna('') + " " + df['review_text']
    ).str.lower()
    
    # 2. Sampling and Vector Selection
    print("Selecting sample and aligned embeddings...")
    all_embeddings = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    
    # Pick random indices
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)
    
    sample_texts = df.iloc[sample_indices]['clean_text'].tolist()
    sample_names = df.iloc[sample_indices]['name'].tolist()
    sample_vectors = all_embeddings[sample_indices].astype(np.float32)
    
    # 3. Initialize Vectorizer with Stopwords
    # We filter out common English junk words and project-specific prefixes
    custom_stopwords = ["description", "reviews", "game", "play", "steam", "players", "get", "use"]
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
    
    # 4. Run BERTopic
    print("Fitting BERTopic model...")
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=True,
        min_topic_size=30 # Increased for more cohesive themes
    )
    
    topics, probs = topic_model.fit_transform(sample_texts, sample_vectors)
    
    # 5. Analyze Results
    print("\n--- Topic Analysis (Top 15 Topics) ---")
    topic_info = topic_model.get_topic_info()
    
    # We want to find representative games for each topic
    # probs shape: (num_samples, num_topics)
    
    for i in range(min(15, len(topic_info)-1)):
        row = topic_info.iloc[i+1] # skip -1 (outliers)
        topic_num = row['Topic']
        count = row['Count']
        
        # Get Top Words
        words = topic_model.get_topic(topic_num)
        word_str = ", ".join([w[0] for w in words[:10]])
        
        print(f"\nTOPIC {topic_num} ({count} games): {word_str}")
        
        # Get Top 10 representative games by probability
        topic_probs = probs[:, topic_num]
        top_game_indices = np.argsort(-topic_probs)[:10]
        
        print("  Representative Games:")
        for idx in top_game_indices:
            name = sample_names[idx]
            p = topic_probs[idx]
            print(f"    - {name} (conf: {p:.2f})")

if __name__ == "__main__":
    try:
        run_bertopic_research()
    except ImportError:
        print("\nERROR: BERTopic not installed. Run: pip install bertopic")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nAn error occurred: {e}")

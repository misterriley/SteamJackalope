import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# Component imports for parallelization
from umap import UMAP
from hdbscan import HDBSCAN

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, 
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_RAW_FILE,
    TOPIC_DISTRIBUTIONS_FILE,
    TOPIC_MODEL_FILE,
    ROOT_DIR
)

def generate_full_topic_model(df_path=None):
    print("--- SteamJackalope: High-Resolution Topic Modeling ---")
    
    # 1. Load Data
    csv_path = df_path if df_path else os.path.join(ROOT_DIR, "data", "pipeline_games_clean.csv")
    reviews_path = os.path.join(ROOT_DIR, "scraped_reviews.csv")
    embeddings_path = EMBEDDINGS_DESC_RAW_FILE
    
    print("Loading games dataset...")
    df = pd.read_csv(csv_path)
    
    print("Loading and bundling reviews (5 per game)...")
    if os.path.exists(reviews_path):
        reviews_iter = pd.read_csv(reviews_path, chunksize=200000)
        bundled_list = []
        for chunk in tqdm(reviews_iter, desc="Reading reviews"):
            chunk['review_text'] = chunk['review_text'].fillna('').astype(str)
            bundled_list.append(chunk[['appid', 'review_text']])
        
        full_reviews = pd.concat(bundled_list)
        reviews_bundled = full_reviews.groupby('appid')['review_text'].apply(lambda x: " ".join(list(x)[:5])).reset_index()
        df = df.merge(reviews_bundled, on='appid', how='left')
        df['review_text'] = df['review_text'].fillna('')
    else:
        print("Warning: Reviews not found. Using descriptions only.")
        df['review_text'] = ''
        
    df['clean_text'] = (
        df['short_description'].fillna('') + " " + df['review_text']
    ).str.lower()
    
    # 2. Load Pre-computed Embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    
    # 3. Configure Parallelized Components
    print("Configuring UMAP and HDBSCAN for parallel execution...")
    
    # Check for GPU acceleration (RAPIDS cuML)
    try:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        from cuml.manifold import UMAP as cuUMAP
        print(">>> GPU Acceleration detected! Using cuML for UMAP and HDBSCAN.")
        umap_model = cuUMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', verbose=True)
        hdbscan_model = cuHDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    except ImportError:
        print(">>> cuML not found. Using multi-core CPU (n_jobs=-1).")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', n_jobs=-1, low_memory=True, verbose=True)
        hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom', prediction_data=True, core_dist_n_jobs=-1)

    vectorizer_model = CountVectorizer(stop_words="english", min_df=5)
    
    # 4. Initialize and Fit BERTopic
    print("Fitting BERTopic model (Target: ~250 topics)...")
    # We turn off HDBSCAN probabilities to avoid the serial bottleneck.
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=False, 
        min_topic_size=30,
        nr_topics=250, 
        low_memory=True
    )
    
    docs = df['clean_text'].tolist()
    
    # 3.5 Tiny Dataset Handling (for Tests)
    if len(docs) < 50:
        print("Warning: Tiny dataset detected. Skipping HDBSCAN and generating uniform distribution.")
        # Create a dummy probs matrix of shape (num_docs, 250)
        num_docs = len(docs)
        num_topics = 250
        probs = np.full((num_docs, num_topics), 1.0 / num_topics)
        
        dist_path = TOPIC_DISTRIBUTIONS_FILE
        np.save(dist_path, probs.astype(np.float16))
        
        labels_path = os.path.join(os.path.dirname(TOPIC_DISTRIBUTIONS_FILE), "topic_labels.json")
        with open(labels_path, "w") as f:
            json.dump({i: "dummy_topic" for i in range(num_topics)}, f)
            
        print("Success! Dummy artifacts saved for small dataset.")
        return

    topics, _ = topic_model.fit_transform(docs, embeddings)
    
    # 4.5 Optimized Vector-Space Soft Assignment
    print("Calculating topic distributions via Vector-Space Soft Assignment...")
    # BERTopic stores centroids in topic_embeddings_. 
    # Index 0 is Topic -1 (outliers), valid topics are from index 1 onwards.
    topic_embeddings = topic_model.topic_embeddings_[1:]
    
    # Calculate Cosine Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings, topic_embeddings)
    
    # Apply Softmax with a WARMER temperature (T=0.2)
    T = 0.2
    # Subtract max for numerical stability
    sim_matrix = sim_matrix - np.max(sim_matrix, axis=1, keepdims=True)
    exp_sim = np.exp(sim_matrix / T)
    probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    
    # 5. Save Artifacts
    dist_path = TOPIC_DISTRIBUTIONS_FILE
    print(f"Saving normalized distributions to {dist_path}...")
    np.save(dist_path, probs.astype(np.float16))
    
    model_path = TOPIC_MODEL_FILE
    print(f"Saving trained model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(topic_model, f)
        
    # 6. Generate Summary CSV
    print("Generating Topic Summary CSV...")
    topic_info = topic_model.get_topic_info()
    summary_data = []
    
    # To find top games per topic properly, we use the probability matrix
    # Note: Topic indices in info match the columns in probs (excluding -1)
    
    for i in tqdm(range(len(topic_info)), desc="Summarizing topics"):
        row = topic_info.iloc[i]
        topic_num = row['Topic']
        if topic_num == -1: continue 
        
        # Top 20 Words
        words = topic_model.get_topic(topic_num)
        word_list = [w[0] for w in words[:20]]
        
        # Top 10 Games by Probability
        # If probs is (N, K), the i-th row in topic_info (after -1) corresponds to column i-1
        topic_col_idx = i - 1
        game_probs = probs[:, topic_col_idx]
        top_game_indices = np.argsort(-game_probs)[:10]
        top_games = df.iloc[top_game_indices]['name'].tolist()
        
        summary_data.append({
            'topic_id': topic_num,
            'count': row['Count'],
            'top_words': ", ".join(word_list),
            'top_games': " | ".join(top_games)
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(os.path.dirname(TOPIC_DISTRIBUTIONS_FILE), "topic_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # 7. Save Topic Labels JSON
    labels_path = os.path.join(os.path.dirname(TOPIC_DISTRIBUTIONS_FILE), "topic_labels.json")
    topic_labels = {int(t): ", ".join([w[0] for w in topic_model.get_topic(t)[:5]]) for t in topic_model.get_topics() if t != -1}
    with open(labels_path, "w") as f:
        json.dump(topic_labels, f, indent=4)

    print(f"\nSuccess! Found {len(topic_info)-1} topics.")
    print(f"Artifacts saved to data/production/")

if __name__ == "__main__":
    df_arg = sys.argv[1] if len(sys.argv) > 1 else None
    generate_full_topic_model(df_arg)

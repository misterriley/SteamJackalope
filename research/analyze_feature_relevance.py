import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, 
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, TAG_NORMS_FILE, PRODUCTION_DATA_DIR,
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR, SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_GLOBAL_SCALING_FACTOR
)

def analyze_relevance(user_id="76561198039155404"):
    # 1. Load User Data
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    n_samples = len(y)
    
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid'])
    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    
    # 2. Extract and Preprocess Features (Exactly like solver)
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices]
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    tag_feat = (tag_vectors / (tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices]
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    sem_feat = (sem_vectors / (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy"))
    topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy"))
    topic_feat = ((topic_dist - topic_means) / (topic_stds + 1e-10)) * np.sqrt(0.5) * TOPIC_GLOBAL_SCALING_FACTOR

    all_thematic = np.hstack([tag_feat, sem_feat, topic_feat])
    group_names = (['Tag'] * tag_feat.shape[1] + 
                   ['Semantic'] * sem_feat.shape[1] + 
                   ['Topic'] * topic_feat.shape[1])
    
    print("Total thematic features: " + str(all_thematic.shape[1]))
    print("Sample size (N): " + str(n_samples))

    # 3. Calculate Zero-Order Correlations
    correlations = []
    for i in range(all_thematic.shape[1]):
        feat = all_thematic[:, i]
        if np.std(feat) > 1e-9:
            corr = np.corrcoef(feat, y)[0, 1]
        else:
            corr = 0.0
        correlations.append({
            'index': i,
            'group': group_names[i],
            'corr': corr,
            'abs_corr': abs(corr)
        })
    
    correlations = sorted(correlations, key=lambda x: x['abs_corr'], reverse=True)
    
    allowed_thematic = n_samples - 6
    top_features = correlations[:allowed_thematic]
    
    survivor_counts = {'Tag': 0, 'Semantic': 0, 'Topic': 0}
    for feat in top_features:
        survivor_counts[feat['group']] += 1
        
    print("\n--- Feature Relevance Analysis (Top " + str(allowed_thematic) + " survivors) ---")
    print("{:<10} | {:<6} | {:<15}".format("Group", "Count", "% of Survivors"))
    print("-" * 40)
    for group, count in survivor_counts.items():
        print("{:<10} | {:<6} | {:>6.1%}".format(group, count, count/allowed_thematic))
        
    print("\n--- Top 10 Most Relevant Dimensions ---")
    for i in range(10):
        f = correlations[i]
        print(str(i+1) + ". " + f"{f['group']:<10}" + " (idx " + str(f['index']) + "): r=" + f"{f['corr']:+.4f}")

if __name__ == "__main__":
    analyze_relevance()

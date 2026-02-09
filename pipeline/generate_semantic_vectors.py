import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os
import sys
import ast

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE, 
    EMBEDDINGS_TAG_FILE, 
    W_DESC_FILE, 
    W_STRUCTURAL_FILE,
    MEAN_DESC_FILE,
    MEAN_STRUCTURAL_FILE,
    EPSILON
)

def whiten(vectors, variance_threshold=0.80):
    print("Whitening vectors (Centered ZCA with dimensionality reduction)...")
    n_samples = vectors.shape[0]
    
    # Calculate and subtract mean
    mean = np.mean(vectors, axis=0)
    centered_vectors = vectors - mean
    
    # Calculate covariance matrix M (centered)
    M = np.dot(centered_vectors.T, centered_vectors) / n_samples
    U, S, Vt = np.linalg.svd(M)
    
    # Compute cumulative explained variance
    cumvar = np.cumsum(S) / np.sum(S)
    n_components = np.argmax(cumvar >= variance_threshold) + 1
    if n_components <= 0:
        n_components = max(1, int(variance_threshold * len(S)))  # fallback
    print(f"Retaining {n_components} dimensions for {variance_threshold:.1%} variance (original: {len(S)})")
    
    # Keep top components
    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    
    epsilon = 1e-7
    # ZCA whitening matrix: U_reduced @ diag(1/sqrt(S_reduced + epsilon))
    W = np.dot(U_reduced, np.diag(1.0 / np.sqrt(S_reduced + epsilon)))
    whitened = np.dot(centered_vectors, W)
    return whitened, W, mean

def clean_tag_string(tag_str):
    """
    Converts a Steam tag dictionary string into a comma-separated list of tag names.
    e.g., "{'Action': 100, 'Indie': 50}" -> "Action, Indie"
    """
    if pd.isna(tag_str) or tag_str == "" or tag_str == "{}":
        return ""
    try:
        # Steam tags are stored as a string representation of a dict
        tags_dict = ast.literal_eval(tag_str)
        if isinstance(tags_dict, dict):
            return ", ".join(tags_dict.keys())
        return str(tag_str)
    except:
        return str(tag_str)

def generate_embeddings(csv_path, reviews_path, embeddings_desc_out, embeddings_tag_out, metadata_out):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset=['appid'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    reviews_df = None
    if reviews_path and os.path.exists(reviews_path):
        print(f"Loading reviews from {reviews_path}...")
        reviews_df = pd.read_csv(reviews_path)
        # Group reviews by appid and join with a separator
        reviews_bundled = reviews_df.groupby('appid')['review_text'].apply(lambda x: " | ".join(map(str, x))).reset_index()
        df = df.merge(reviews_bundled, on='appid', how='left')
        df['review_text'] = df['review_text'].fillna('')
    else:
        print("No reviews file found. Proceeding without reviews.")
        df['review_text'] = ''

    # Identify relevant columns for embedding
    print("Preprocessing data for embeddings...")
    df['short_description'] = df['short_description'].fillna('')
    df['genres'] = df['genres'].fillna('')
    
    # Clean tags for cleaner semantic matching (extract keys from dict string)
    tqdm_available = False
    try:
        from tqdm import tqdm
        tqdm_available = True
    except ImportError:
        pass

    print("Cleaning tags for structural embeddings...")
    if tqdm_available:
        df['clean_tags'] = df['tags'].apply(clean_tag_string)
    else:
        df['clean_tags'] = df['tags'].apply(clean_tag_string)
    
    # Vector A: Structural/Categorical (Genres + Tags)
    df['structural_text'] = (
        "Genres: " + df['genres'] + 
        " Tags: " + df['clean_tags']
    )
    
    # Vector B: Narrative/Descriptive (Description + Reviews)
    df['desc_text'] = (
        "Description: " + df['short_description'] + 
        " Reviews: " + df['review_text']
    )
    
    from common.constants import MODEL_NAME
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating structural embeddings...")
    structural_texts = df['structural_text'].tolist()
    # Identify empty inputs to zero them out later
    # Empty structural text is "Genres:  Tags: "
    is_empty_struct = df['structural_text'].str.strip() == "Genres:  Tags:"
    # Using a larger batch size for speed
    embeddings_structural = model.encode(structural_texts, show_progress_bar=True, batch_size=128)
    embeddings_structural[is_empty_struct] = 0
    
    print("Generating descriptive embeddings...")
    desc_texts = df['desc_text'].tolist()
    # Empty descriptive text is "Description:  Reviews: "
    is_empty_desc = df['desc_text'].str.strip() == "Description:  Reviews:"
    embeddings_desc = model.encode(desc_texts, show_progress_bar=True, batch_size=128)
    embeddings_desc[is_empty_desc] = 0

    print("Whitening embeddings...")
    embeddings_structural, W_structural, mean_structural = whiten(embeddings_structural, variance_threshold=0.80)
    embeddings_desc, W_desc, mean_desc = whiten(embeddings_desc, variance_threshold=0.80)

    print("Normalizing embeddings for memory-mapping...")
    def normalize(m):
        norms = np.linalg.norm(m.astype(np.float32), axis=1, keepdims=True)
        norms[norms == 0] = EPSILON
        return (m / norms).astype(np.float16)

    embeddings_structural = normalize(embeddings_structural)
    embeddings_desc = normalize(embeddings_desc)
    
    print(f"Saving structural embeddings to {embeddings_tag_out}...")
    np.save(embeddings_tag_out, embeddings_structural)
    print(f"Saving structural whitening matrix to {W_STRUCTURAL_FILE}...")
    np.save(W_STRUCTURAL_FILE, W_structural.astype(np.float16))
    print(f"Saving structural mean vector to {MEAN_STRUCTURAL_FILE}...")
    np.save(MEAN_STRUCTURAL_FILE, mean_structural.astype(np.float16))
    
    print(f"Saving descriptive embeddings to {embeddings_desc_out}...")
    np.save(embeddings_desc_out, embeddings_desc)
    print(f"Saving descriptive whitening matrix to {W_DESC_FILE}...")
    np.save(W_DESC_FILE, W_desc.astype(np.float16))
    print(f"Saving descriptive mean vector to {MEAN_DESC_FILE}...")
    np.save(MEAN_DESC_FILE, mean_desc.astype(np.float16))

    # Run distribution analysis
    try:
        from research.analyze_vector_distributions import analyze_distribution
        analyze_distribution(embeddings_structural, "Whitened Structural Embeddings")
        analyze_distribution(embeddings_desc, "Whitened Descriptive Embeddings")
    except ImportError:
        print("Warning: could not import analyze_distribution from research.analyze_vector_distributions")
    
    print(f"Saving metadata to {metadata_out}...")
    
    # Save metadata
    metadata_df = df[['appid', 'name', 'genres', 'tags', 'categories', 'supported_languages']]
    metadata_df.to_parquet(metadata_out, compression='snappy')
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate split embeddings for Steam games.")
    parser.add_argument("--csv", default="scraped_games.csv", help="Input CSV file")
    parser.add_argument("--reviews", default="scraped_reviews.csv", help="Input reviews CSV file")
    parser.add_argument("--embeddings_desc", default=EMBEDDINGS_DESC_FILE, help="Output .npy file for description embeddings")
    parser.add_argument("--embeddings_tag", default=EMBEDDINGS_TAG_FILE, help="Output .npy file for structural embeddings")
    parser.add_argument("--metadata", default="metadata.parquet", help="Output .parquet file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        generate_embeddings(args.csv, args.reviews, args.embeddings_desc, args.embeddings_tag, args.metadata)
    else:
        print(f"Error: {args.csv} not found.")

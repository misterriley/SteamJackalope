import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os
import sys
import ast
import json

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE, 
    EMBEDDINGS_DESC_NORMS_FILE,
    EMBEDDINGS_TAG_FILE, 
    EMBEDDINGS_STRUCTURAL_NORMS_FILE,
    W_DESC_FILE,
    W_STRUCTURAL_FILE,
    MEAN_DESC_FILE,
    MEAN_STRUCTURAL_FILE,
    EPSILON,
    MODEL_NAME,
    SENTENCE_TRANSFORMER_BACKEND,
    SENTENCE_TRANSFORMER_MODEL_KWARGS,
    METADATA_FILE,
    REGULARIZATION_FILE
)
from common.utils import safe_save_npy, calculate_dot_product_lambda

def whiten(vectors, variance_threshold=0.95):
    print(f"Whitening vectors (Uncentered ZCA with {variance_threshold:.0%} variance threshold)...")
    n_samples = vectors.shape[0]
    
    # Skip mean subtraction to keep the origin at (0,0,...)
    mean = np.zeros(vectors.shape[1], dtype=np.float32)
    
    # Calculate raw second moment matrix M (uncentered)
    M = np.dot(vectors.T, vectors) / n_samples
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
    whitened = np.dot(vectors, W)
    
    # Unit Normalization: Ensure all vectors have length 1.0
    print("Performing unit normalization on whitened vectors...")
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms[norms < EPSILON] = 1.0 # Avoid division by zero for empty entries
    whitened = whitened / norms
    
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

def generate_embeddings(csv_path, reviews_path, embeddings_desc_out, embeddings_tag_out, metadata_out,
                        w_desc_out=None, w_structural_out=None, mean_desc_out=None, mean_structural_out=None,
                        desc_norms_out=None, structural_norms_out=None):
    """
    Generate semantic embeddings with optional custom output paths for whitening matrices and means.
    If the weight/mean outputs are None, they will not be saved (to prevent overwriting production files).
    """
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
        tqdm.pandas(desc="Cleaning tags", smoothing=0)
        df['clean_tags'] = df['tags'].progress_apply(clean_tag_string)
    else:
        df['clean_tags'] = df['tags'].apply(clean_tag_string)
    
    # Vector A: Structural/Categorical (Genres + Tags) - LOWERCASE
    df['structural_text'] = (
        "genres: " + df['genres'].str.lower() + 
        " tags: " + df['clean_tags'].str.lower()
    )
    
    # Vector B: Narrative/Descriptive (Description + Reviews) - LOWERCASE
    df['desc_text'] = (
        "description: " + df['short_description'].str.lower() + 
        " reviews: " + df['review_text'].str.lower()
    )
    
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Optimization: Enable tokenizer parallelism to speed up batch preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        backend=SENTENCE_TRANSFORMER_BACKEND,
        model_kwargs=SENTENCE_TRANSFORMER_MODEL_KWARGS
    )
    
    # Use a larger batch size to saturate the GPU
    BATCH_SIZE = 256 if device == "cuda" else 32
    
    print("Generating structural embeddings...")
    structural_texts = df['structural_text'].tolist()
    # Identify empty inputs to zero them out later
    is_empty_struct = df['structural_text'].str.strip() == "genres:  tags:"
    embeddings_structural = model.encode(structural_texts, show_progress_bar=True, batch_size=BATCH_SIZE)
    embeddings_structural[is_empty_struct] = 0
    
    print("Generating descriptive embeddings...")
    desc_texts = df['desc_text'].tolist()
    # Empty descriptive text is "description:  reviews: "
    is_empty_desc = df['desc_text'].str.strip() == "description:  reviews:"
    embeddings_desc = model.encode(desc_texts, show_progress_bar=True, batch_size=BATCH_SIZE)
    embeddings_desc[is_empty_desc] = 0

    print("Whitening embeddings...")
    embeddings_structural, W_structural, mean_structural = whiten(embeddings_structural, variance_threshold=0.95)
    embeddings_desc, W_desc, mean_desc = whiten(embeddings_desc, variance_threshold=0.95)

    print("Processing embeddings for storage...")
    # Calculate norms before float16 conversion
    structural_norms = np.linalg.norm(embeddings_structural.astype(np.float32), axis=1).astype(np.float16)
    desc_norms = np.linalg.norm(embeddings_desc.astype(np.float32), axis=1).astype(np.float16)

    # Cast to float16 for memory mapping
    embeddings_structural_f16 = embeddings_structural.astype(np.float16)
    embeddings_desc_f16 = embeddings_desc.astype(np.float16)
    
    print(f"Saving structural embeddings to {embeddings_tag_out}...")
    safe_save_npy(embeddings_tag_out, embeddings_structural_f16)
    
    struct_norms_path = structural_norms_out if structural_norms_out else EMBEDDINGS_STRUCTURAL_NORMS_FILE
    print(f"Saving structural norms to {struct_norms_path}...")
    safe_save_npy(struct_norms_path, structural_norms)
    
    # Save whitening matrices and means only if paths are provided
    if w_structural_out:
        print(f"Saving structural whitening matrix to {w_structural_out}...")
        safe_save_npy(w_structural_out, W_structural.astype(np.float16))
    if mean_structural_out:
        print(f"Saving structural mean vector to {mean_structural_out}...")
        safe_save_npy(mean_structural_out, mean_structural.astype(np.float16))
    
    print(f"Saving descriptive embeddings to {embeddings_desc_out}...")
    safe_save_npy(embeddings_desc_out, embeddings_desc_f16)
    
    desc_norms_path = desc_norms_out if desc_norms_out else EMBEDDINGS_DESC_NORMS_FILE
    print(f"Saving descriptive norms to {desc_norms_path}...")
    safe_save_npy(desc_norms_path, desc_norms)
    
    if w_desc_out:
        print(f"Saving descriptive whitening matrix to {w_desc_out}...")
        safe_save_npy(w_desc_out, W_desc.astype(np.float16))
    if mean_desc_out:
        print(f"Saving descriptive mean vector to {mean_desc_out}...")
        safe_save_npy(mean_desc_out, mean_desc.astype(np.float16))

    # Calculate and save constants
    print("Calculating semantic regularization constants (Uncentered Unit Pathway)...")
    from common.utils import calculate_dot_product_lambda
    semantic_lambda = calculate_dot_product_lambda(embeddings_desc)
    
    # Calculate SEMANTIC_GLOBAL_SCALING_FACTOR to match Tag variance parity.
    # This ensures LASSO treats both modalities equally.
    try:
        from common.constants import TAG_VECTORS_FILE, TAG_NORMS_FILE, DOT_PRODUCT_LAMBDA, TAG_GLOBAL_SCALING_FACTOR
        if os.path.exists(TAG_VECTORS_FILE) and os.path.exists(TAG_NORMS_FILE):
            print("Calculating Semantic scaling factor based on Tag variance...")
            # Load a sample to calculate variance
            tag_vecs = np.load(TAG_VECTORS_FILE, mmap_mode='r')
            tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
            
            sample_size = min(10000, len(tag_vecs))
            idx = np.random.choice(len(tag_vecs), sample_size, replace=False)
            
            t_scaled = (tag_vecs[idx].astype(np.float32) / (tag_norms[idx].reshape(-1, 1).astype(np.float32) + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
            s_scaled_unit = embeddings_desc[idx].astype(np.float32) # Already unit normalized
            
            tag_std = np.std(t_scaled, axis=0).mean()
            sem_std_unit = np.std(s_scaled_unit, axis=0).mean()
            
            semantic_scaling = float(tag_std / (sem_std_unit + EPSILON))
            print(f"Parity Match: Tag Std={tag_std:.6f}, Sem Unit Std={sem_std_unit:.6f} -> Scaling={semantic_scaling:.4f}")
        else:
            print("Tag files not found. Using fallback scaling factor 11.25")
            semantic_scaling = 11.25
    except Exception as e:
        print(f"Error calculating semantic scaling parity: {e}. Using fallback 11.25")
        semantic_scaling = 11.25
    
    # Establish natural range for semantic similarities
    print("Establishing natural range for semantic similarities (10,000 random pairs)...")
    n_samples = embeddings_desc.shape[0]
    if n_samples > 1:
        indices1 = np.random.randint(0, n_samples, 10000)
        indices2 = np.random.randint(0, n_samples, 10000)
        mask = indices1 == indices2
        indices2[mask] = (indices2[mask] + 1) % n_samples
        
        # Descriptive dot products
        v1_desc = embeddings_desc[indices1].astype(np.float32)
        v2_desc = embeddings_desc[indices2].astype(np.float32)
        dot_desc = np.sum(v1_desc * v2_desc, axis=1)
        
        # Structural dot products
        v1_struct = embeddings_structural[indices1].astype(np.float32)
        v2_struct = embeddings_structural[indices2].astype(np.float32)
        dot_struct = np.sum(v1_struct * v2_struct, axis=1)
        
        # Use ONLY Descriptive dot products (dropping structural for now to reduce noise)
        dot_combined = dot_desc
        
        sim_mean = float(np.mean(dot_combined))
        sim_std = float(np.std(dot_combined))
        print(f"Semantic Similarity Natural Range (Descriptive Only): Mean={sim_mean:.4f}, Std={sim_std:.4f}")
    else:
        sim_mean = 0.0
        sim_std = 1.0

    print(f"Set SEMANTIC_DOT_PRODUCT_LAMBDA: {semantic_lambda:.4f}")
    print(f"Set SEMANTIC_GLOBAL_SCALING_FACTOR: {semantic_scaling:.4f}")

    if os.path.exists(REGULARIZATION_FILE):
        try:
            with open(REGULARIZATION_FILE, "r") as f:
                reg_constants = json.load(f)
        except:
            reg_constants = {}
    else:
        reg_constants = {}

    reg_constants["SEMANTIC_DOT_PRODUCT_LAMBDA"] = semantic_lambda
    reg_constants["SEMANTIC_GLOBAL_SCALING_FACTOR"] = semantic_scaling
    reg_constants["SEMANTIC_SIMILARITY_MEAN"] = sim_mean
    reg_constants["SEMANTIC_SIMILARITY_STD"] = sim_std

    with open(REGULARIZATION_FILE, "w") as f:
        json.dump(reg_constants, f, indent=4)

    # Run distribution analysis
    try:
        from research.analyze_vector_distributions import analyze_distribution
        analyze_distribution(embeddings_structural.astype(np.float32), "Whitened Structural Embeddings")
        analyze_distribution(embeddings_desc.astype(np.float32), "Whitened Descriptive Embeddings")
    except ImportError:
        print("Warning: could not import analyze_distribution from research.analyze_vector_distributions")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate split embeddings for Steam games.")
    parser.add_argument("--csv", default="scraped_games.csv", help="Input CSV file")
    parser.add_argument("--reviews", default="scraped_reviews.csv", help="Input reviews CSV file")
    parser.add_argument("--embeddings_desc", default=EMBEDDINGS_DESC_FILE, help="Output .npy file for description embeddings")
    parser.add_argument("--embeddings_tag", default=EMBEDDINGS_TAG_FILE, help="Output .npy file for structural embeddings")
    parser.add_argument("--metadata", default=None, help="Deprecated: metadata is now handled by generate_metadata.py")
    parser.add_argument("--w_desc", default=W_DESC_FILE, help="Output .npy file for desc whitening matrix")
    parser.add_argument("--w_structural", default=W_STRUCTURAL_FILE, help="Output .npy file for structural whitening matrix")
    parser.add_argument("--mean_desc", default=MEAN_DESC_FILE, help="Output .npy file for desc mean vector")
    parser.add_argument("--mean_structural", default=MEAN_STRUCTURAL_FILE, help="Output .npy file for structural mean vector")
    parser.add_argument("--desc_norms", default=EMBEDDINGS_DESC_NORMS_FILE, help="Output .npy file for desc norms")
    parser.add_argument("--structural_norms", default=EMBEDDINGS_STRUCTURAL_NORMS_FILE, help="Output .npy file for structural norms")
    
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        generate_embeddings(
            args.csv, 
            args.reviews, 
            args.embeddings_desc, 
            args.embeddings_tag, 
            args.metadata,
            args.w_desc,
            args.w_structural,
            args.mean_desc,
            args.mean_structural,
            args.desc_norms,
            args.structural_norms
        )
    else:
        print(f"Error: {args.csv} not found.")

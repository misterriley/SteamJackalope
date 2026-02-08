"""
Game Matching Function

This script provides a function to match a game to the database using its
embedding vectors. It calculates cosine similarity between the game's vectors
and all games in the database, showing results for:
- Descriptive vectors (matching description/reviews)
- Structural vectors (matching genres/tags)
- Tag vectors (matching Steam tags)
- Average of the three categories
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_TAG_FILE,
    METADATA_FILE,
    TAG_VECTORS_FILE,
    DOT_PRODUCT_LAMBDA
)

def normalize(m):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return m / norms

def match_game_to_database(game_id, top_k=10):
    """
    Match a game to the database using its embedding vectors.

    Args:
        game_id (int): The Steam appid of the game to match
        top_k (int): Number of top results to return (default: 10)

    Returns:
        dict: Results containing descriptive, structural, tag, and average matches
    """
    # Load data
    print(f"Loading data...")
    try:
        metadata = pd.read_parquet(METADATA_FILE)
        embeddings_desc = np.load(EMBEDDINGS_DESC_FILE)
        embeddings_structural = np.load(EMBEDDINGS_TAG_FILE)
        tag_vectors = np.load(TAG_VECTORS_FILE)
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. {e}")
        return None

    # Normalize embeddings
    print("Normalizing embeddings...")
    embeddings_desc_norm = normalize(embeddings_desc)
    embeddings_structural_norm = normalize(embeddings_structural)
    tag_vectors_norm = normalize(tag_vectors)

    # Find the game in the database
    game_idx = metadata[metadata['appid'] == game_id].index
    if len(game_idx) == 0:
        print(f"Error: Game with ID {game_id} not found in database.")
        return None

    game_idx = game_idx[0]
    print(f"Found game: {metadata.iloc[game_idx]['name']} (ID: {game_id})")

    # Get the game's vectors
    game_desc_vec = embeddings_desc_norm[game_idx]
    game_structural_vec = embeddings_structural_norm[game_idx]
    game_tag_vec = tag_vectors_norm[game_idx]

    # Calculate similarities (excluding the game itself)
    mask = np.ones(len(metadata), dtype=bool)
    mask[game_idx] = False

    # Calculate similarities (excluding the game itself)
    # Using raw dot products (unscaled) as requested.
    # Note: Descriptive and Structural embeddings are already unit-normalized in the files,
    # so their dot products are equivalent to cosine similarity [0, 1].
    # Tag vectors are NOT unit-normalized, so their dot product reflects both 
    # directional similarity and the "strength/reliability" of the tags.

    # Descriptive similarity
    desc_sims = np.dot(embeddings_desc, embeddings_desc[game_idx])
    desc_sims[game_idx] = -1.0 # Exclude self
    desc_top_idx = np.argsort(desc_sims)[::-1][:top_k]
    desc_results = [
        (metadata.iloc[idx]['name'], desc_sims[idx])
        for idx in desc_top_idx
    ]

    # Structural similarity
    structural_sims = np.dot(embeddings_structural, embeddings_structural[game_idx])
    structural_sims[game_idx] = -1.0 # Exclude self
    structural_top_idx = np.argsort(structural_sims)[::-1][:top_k]
    structural_results = [
        (metadata.iloc[idx]['name'], structural_sims[idx])
        for idx in structural_top_idx
    ]

    # Tag similarity (Regularized Cosine)
    # Sim(A,B) = A.B / (|A||B| + lambda)
    tag_vectors = np.load(TAG_VECTORS_FILE)
    target_vec = tag_vectors[game_idx]
    
    tag_norms = np.linalg.norm(tag_vectors, axis=1)
    target_norm = np.linalg.norm(target_vec)
    
    dot_products = np.dot(tag_vectors, target_vec)
    denom = (tag_norms * target_norm) + DOT_PRODUCT_LAMBDA
    denom[denom == 0] = 1e-12
    
    tag_sims = dot_products / denom
    tag_sims[game_idx] = -1e12 # Exclude self
    
    tag_top_idx = np.argsort(tag_sims)[::-1][:top_k]
    tag_results = [
        (metadata.iloc[idx]['name'], tag_sims[idx])
        for idx in tag_top_idx
    ]

    # Average similarity
    # Re-normalize embeddings just in case they weren't perfectly unit length
    d_norm = normalize(embeddings_desc)
    s_norm = normalize(embeddings_structural)
    d_sims = np.dot(d_norm, d_norm[game_idx])
    s_sims = np.dot(s_norm, s_norm[game_idx])
    
    # Average of the three categories
    avg_sims = (d_sims + s_sims + tag_sims) / 3.0
    avg_sims[game_idx] = -1.0 # Exclude self
    
    avg_top_idx = np.argsort(avg_sims)[::-1][:top_k]
    avg_results = [
        (metadata.iloc[idx]['name'], avg_sims[idx])
        for idx in avg_top_idx
    ]

    return {
        'game_id': game_id,
        'game_name': metadata.iloc[game_idx]['name'],
        'descriptive_results': desc_results,
        'structural_results': structural_results,
        'tag_results': tag_results,
        'average_results': avg_results
    }

def display_results(results, top_k=10):
    """Display the matching results in a formatted table."""
    if not results:
        print("No results to display.")
        return

    col_width = 30

    print("\n" + "="*120)
    print(f"MATCHING RESULTS FOR: {results['game_name']} (ID: {results['game_id']})")
    print("="*120)

    # Display descriptive results
    print(f"\n{'='*120}")
    print(f"TOP {top_k} DESCRIPTIVE MATCHES (Description/Reviews Similarity)")
    print(f"{'='*120}")
    print(f"{'Rank':<4} | {'Game Name':<{col_width}} | {'Score':<8}")
    print("-"*120)
    for i, (name, score) in enumerate(results['descriptive_results'][:top_k]):
        print(f"{i+1:<4} | {name[:col_width-1]:<{col_width}} | {score:.4f}")
    print(f"Highest Score: {results['descriptive_results'][0][1]:.4f}")

    # Display structural results
    print(f"\n{'='*120}")
    print(f"TOP {top_k} STRUCTURAL MATCHES (Genres/Tags Similarity)")
    print(f"{'='*120}")
    print(f"{'Rank':<4} | {'Game Name':<{col_width}} | {'Score':<8}")
    print("-"*120)
    for i, (name, score) in enumerate(results['structural_results'][:top_k]):
        print(f"{i+1:<4} | {name[:col_width-1]:<{col_width}} | {score:.4f}")
    print(f"Highest Score: {results['structural_results'][0][1]:.4f}")

    # Display tag results
    print(f"\n{'='*120}")
    print(f"TOP {top_k} TAG MATCHES (Steam Tags Similarity)")
    print(f"{'='*120}")
    print(f"{'Rank':<4} | {'Game Name':<{col_width}} | {'Score':<8}")
    print("-"*120)
    for i, (name, score) in enumerate(results['tag_results'][:top_k]):
        print(f"{i+1:<4} | {name[:col_width-1]:<{col_width}} | {score:.4f}")
    print(f"Highest Score: {results['tag_results'][0][1]:.4f}")

    # Display average results
    print(f"\n{'='*120}")
    print(f"TOP {top_k} AVERAGE MATCHES (Average of all three)")
    print(f"{'='*120}")
    print(f"{'Rank':<4} | {'Game Name':<{col_width}} | {'Score':<8}")
    print("-"*120)
    for i, (name, score) in enumerate(results['average_results'][:top_k]):
        print(f"{i+1:<4} | {name[:col_width-1]:<{col_width}} | {score:.4f}")
    print(f"Highest Score: {results['average_results'][0][1]:.4f}")

    print("="*120 + "\n")

def main():
    """Command-line interface for the matching function."""
    print("--- Steam Game Matching Tool ---")
    print("Matches a game to the database using its embedding vectors.")
    print("Enter a Steam appid (game ID) to find similar games.\n")

    while True:
        try:
            game_id_input = input("Enter a Steam appid (or 'exit' to quit): ").strip()

            if game_id_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break

            if not game_id_input:
                continue

            try:
                game_id = int(game_id_input)
            except ValueError:
                print("Error: Please enter a valid numeric appid.")
                continue

            results = match_game_to_database(game_id, top_k=10)
            if results:
                display_results(results)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
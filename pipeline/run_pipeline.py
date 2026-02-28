import os
import subprocess
import sys
import argparse
import json

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import constants for default file paths
from common.constants import (
    W_DESC_FILE,
    MEAN_DESC_FILE,
    W_TAG_FILE,
    DIFFICULTY_PREDICTIONS_FILE,
    ROOT_DIR,
    PRODUCTION_DATA_DIR
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name, args):
    """
    Executes a python script as a subprocess.

    Args:
        script_name (str): The name of the python file to execute.
        args (list): A list of command-line arguments to pass to the script.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    print(f"\n>>> Running {script_name} {' '.join(args)}...")
    result = subprocess.run([sys.executable, script_name] + args)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with return code {result.returncode}")
        sys.exit(1)

def load_config(config_path=None):
    """Loads pipeline configuration from a JSON file."""
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "pipeline_config.json")
        
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def main():
    """
    Main entry point for the data pipeline.
    
    Orchestrates the sequence of data processing steps:
    1. Pre-cleans input CSV data.
    2. Generates tag vectors.
    3. Generates semantic embeddings.
    4. Generates consolidated metadata.
    5. Generates quality scores grid.
    """
    # Load defaults from config
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Steam Hybrid Recommender Data Pipeline")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "pipeline_config.json"), help="Path to pipeline_config.json")
    parser.add_argument("--games", default=config.get("games_file", "scraped_games.csv"), help="Path to scraped_games.csv")
    parser.add_argument("--reviews", default=config.get("reviews_file", "scraped_reviews.csv"), help="Path to scraped_reviews.csv")
    parser.add_argument("--tag_names", default=None, help="Path to output tag_names.json")
    parser.add_argument("--download", action="store_true", help="Step 1: Download raw data from Steam")
    parser.add_argument("--build", action="store_true", help="Step 2: Build CSVs from raw data")
    args = parser.parse_args()

    # Re-load config if a different path was provided
    if args.config != os.path.join(SCRIPT_DIR, "pipeline_config.json"):
        config = load_config(args.config)

    # 1. (Optional) Download Data
    if args.download:
        run_script(os.path.join(SCRIPT_DIR, "..", "scraping", "download_steam_data.py"), [])

    # 2. (Optional) Build Dataset
    if args.build:
        run_script(os.path.join(SCRIPT_DIR, "..", "scraping", "build_scraped_dataset.py"), ["--output", args.games, "--reviews_output", args.reviews])

    if not os.path.exists(args.games):
        print(f"Error: {args.games} not found.")
        return

    # 0. Pre-clean data for consistency
    clean_games_path = config.get("clean_games_file", "data/pipeline_games_clean.csv")
    import pandas as pd
    print(f"Pre-cleaning {args.games} for consistency...")
    df = pd.read_csv(args.games, low_memory=False)
    initial_len = len(df)
    
    # Filter DLCs if the column exists
    if 'is_dlc' in df.columns:
        dlc_count = df['is_dlc'].fillna(False).sum()
        print(f"Excluding {dlc_count} DLCs...")
        df = df[df['is_dlc'] != True]
        
    df.drop_duplicates(subset=['appid'], inplace=True)
    df.dropna(subset=['appid', 'name'], inplace=True)
    # Ensure tags is not null for tag vectors
    df['tags'] = df['tags'].fillna('{}')
    clean_games = clean_games_path
    df.to_csv(clean_games, index=False)
    print(f"Cleaned data saved to {clean_games}. Rows: {initial_len} -> {len(df)}")
    args.games = clean_games

    # 0.5. Calculate Regularization Constants
    reg_json = config.get("regularization_json", os.path.join(SCRIPT_DIR, "regularization_constants.json"))
    run_script(os.path.join(SCRIPT_DIR, "calculate_regularization.py"), ["--csv", args.games, "--reviews", args.reviews, "--output", reg_json])

    # 0.6. Generate Playtime Sentiment Parameters
    run_script(os.path.join(SCRIPT_DIR, "generate_playtime_sentiment_params.py"), [
        "--output", reg_json
    ])

    # 1. Generate Tag Vectors
    tag_vectors_path = config.get("tag_vectors_file", "steam_tag_vectors.npy")
    tag_norms_path = config.get("tag_norms_file", "tag_vectors_norms.npy")
    w_tag_path = config.get("w_tag_file", W_TAG_FILE)
    tag_names_out = args.tag_names if args.tag_names else config.get("tag_names_file")
    
    tag_script_args = [
        args.games, 
        "--output", tag_vectors_path, 
        "--constants", reg_json,
        "--norms", tag_norms_path,
        "--w_tag", w_tag_path
    ]
    if tag_names_out:
        tag_script_args.extend(["--tag_names", tag_names_out])
        
    run_script(os.path.join(SCRIPT_DIR, "generate_tag_vectors.py"), tag_script_args)

    # 2. Generate Semantic Vectors
    # This takes both games and reviews.
    embeddings_desc_path = config.get("embeddings_desc_file", "embeddings_desc.npy")
    embeddings_desc_norms_path = config.get("embeddings_desc_norms_file", "embeddings_desc_norms.npy")
    w_desc_path = config.get("w_desc_file", W_DESC_FILE)
    mean_desc_path = config.get("mean_desc_file", MEAN_DESC_FILE)
    temp_metadata_path = "temp_metadata_for_embeddings.parquet"
    run_script(os.path.join(SCRIPT_DIR, "generate_semantic_vectors.py"), [
        "--csv", args.games, 
        "--reviews", args.reviews,
        "--embeddings_desc", embeddings_desc_path,
        "--metadata", temp_metadata_path,
        "--w_desc", w_desc_path,
        "--mean_desc", mean_desc_path,
        "--desc_norms", embeddings_desc_norms_path
    ])
    if os.path.exists(temp_metadata_path):
        os.remove(temp_metadata_path)

    # 2.5 Generate Topic Model
    # Uses the raw embeddings produced in the previous step
    run_script(os.path.join(SCRIPT_DIR, "generate_topic_model.py"), [clean_games_path])
    
    # 2.6 Generate Topic Standardization Stats
    # Required for the Taste Solver to equalize thematic feature variance
    run_script(os.path.join(SCRIPT_DIR, "generate_topic_stats.py"), [clean_games_path])

    # 1.5 Generate Difficulty Model (Now after Topics)
    difficulty_preds_path = config.get("difficulty_preds_file", DIFFICULTY_PREDICTIONS_FILE)
    gamefaqs_dir = config.get("gamefaqs_dir", "data/GameFAQs")
    run_script(os.path.join(SCRIPT_DIR, "generate_difficulty_model.py"), ["--games", args.games, "--gamefaqs", gamefaqs_dir, "--output", difficulty_preds_path])

    # 2.7 Calculate Modality Multipliers
    # Calibrates relative weighting of Tags, Semantics, and Topics based on population variance
    run_script(os.path.join(ROOT_DIR, "research", "calculate_modality_variance.py"), [])

    # 3. Generate Metadata
    # This ensures we have the most complete metadata, including z-scores and playtime info.
    metadata_path = config.get("metadata_file", "metadata.parquet")
    run_script(os.path.join(SCRIPT_DIR, "generate_metadata.py"), [args.games, args.reviews, "--output", metadata_path])

    # 4. Generate Quality Scores Grid
    # This takes the metadata.parquet produced in step 3
    quality_grid_path = config.get("quality_grid_file", "quality_scores_grid.npy")
    run_script(os.path.join(SCRIPT_DIR, "generate_quality_scores_grid.py"), ["--metadata", metadata_path, "--output", quality_grid_path])

    # 5. Validate Outputs
    if validate_outputs(args.games):
        print("\nPipeline completed successfully and validated!")
    else:
        print("\nPipeline completed but validation FAILED!")
        sys.exit(1)

def validate_outputs(clean_games_path):
    """
    Validates that all generated data files have consistent row counts.
    """
    import numpy as np
    import pandas as pd
    from common.constants import (
        EMBEDDINGS_DESC_FILE, 
        METADATA_FILE, 
        TAG_VECTORS_FILE, 
        TAG_NORMS_FILE,
        QUALITY_GRID_FILE,
        TOPIC_DISTRIBUTIONS_FILE,
        REGULARIZATION_FILE
    )
    
    print("\n>>> Validating pipeline outputs...")
    
    if not os.path.exists(clean_games_path):
        print(f"Error: {clean_games_path} missing.")
        return False
        
    df = pd.read_csv(clean_games_path)
    expected_len = len(df)
    print(f"Expected number of games (from {clean_games_path}): {expected_len}")
    
    errors = []
    
    # Check Metadata
    if os.path.exists(METADATA_FILE):
        try:
            m_df = pd.read_parquet(METADATA_FILE)
            if len(m_df) != expected_len:
                errors.append(f"{METADATA_FILE} has {len(m_df)} rows, expected {expected_len}")
        except Exception as e:
            errors.append(f"Error reading {METADATA_FILE}: {e}")
    else:
        errors.append(f"{METADATA_FILE} is missing")
        
    # Check Embeddings
    for emb_file in [EMBEDDINGS_DESC_FILE]:
        if os.path.exists(emb_file):
            try:
                emb = np.load(emb_file)
                if len(emb) != expected_len:
                    errors.append(f"{emb_file} has {len(emb)} rows, expected {expected_len}")
            except Exception as e:
                errors.append(f"Error reading {emb_file}: {e}")
        else:
            errors.append(f"{emb_file} is missing")

    # Check Tag Vectors
    if os.path.exists(TAG_VECTORS_FILE):
        try:
            tags = np.load(TAG_VECTORS_FILE)
            if len(tags) != expected_len:
                errors.append(f"{TAG_VECTORS_FILE} has {len(tags)} rows, expected {expected_len}")
        except Exception as e:
            errors.append(f"Error reading {TAG_VECTORS_FILE}: {e}")
    else:
        errors.append(f"{TAG_VECTORS_FILE} is missing")

    # Check Tag Norms
    if os.path.exists(TAG_NORMS_FILE):
        try:
            norms = np.load(TAG_NORMS_FILE)
            if len(norms) != expected_len:
                errors.append(f"{TAG_NORMS_FILE} has {len(norms)} elements, expected {expected_len}")
        except Exception as e:
            errors.append(f"Error reading {TAG_NORMS_FILE}: {e}")
    else:
        errors.append(f"{TAG_NORMS_FILE} is missing")

    # Check Quality Grid
    if os.path.exists(QUALITY_GRID_FILE):
        try:
            from common.constants import DISC_SLIDER_VALUES
            grid = np.load(QUALITY_GRID_FILE)
            # Quality grid is (num_steps, num_games)
            if grid.ndim != 2:
                errors.append(f"{QUALITY_GRID_FILE} is not 2D")
            else:
                if grid.shape[1] != expected_len:
                    errors.append(f"{QUALITY_GRID_FILE} has {grid.shape[1]} columns, expected {expected_len} (one for each game)")
                if grid.shape[0] != len(DISC_SLIDER_VALUES):
                    errors.append(f"{QUALITY_GRID_FILE} has {grid.shape[0]} rows, expected {len(DISC_SLIDER_VALUES)} (one for each slider step)")
        except Exception as e:
            errors.append(f"Error reading {QUALITY_GRID_FILE}: {e}")
    else:
        errors.append(f"{QUALITY_GRID_FILE} is missing")

    # Check Topic Distributions
    if os.path.exists(TOPIC_DISTRIBUTIONS_FILE):
        try:
            topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE)
            if len(topic_dist) != expected_len:
                errors.append(f"{TOPIC_DISTRIBUTIONS_FILE} has {len(topic_dist)} rows, expected {expected_len}")
            
            # Check Stats
            means_path = os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")
            stds_path = os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")
            if not os.path.exists(means_path) or not os.path.exists(stds_path):
                errors.append("Topic standardization stats (means/stds) are missing")
            else:
                means = np.load(means_path)
                if len(means) != topic_dist.shape[1]:
                    errors.append(f"Topic means count ({len(means)}) does not match topic count ({topic_dist.shape[1]})")
        except Exception as e:
            errors.append(f"Error reading Topic artifacts: {e}")
    else:
        errors.append(f"{TOPIC_DISTRIBUTIONS_FILE} is missing")

    # Check Regularization Constants
    if os.path.exists(REGULARIZATION_FILE):
        try:
            import json
            with open(REGULARIZATION_FILE, 'r') as f:
                reg_data = json.load(f)
            required_keys = ["TAG_GLOBAL_SCALING_FACTOR", "SEMANTIC_GLOBAL_SCALING_FACTOR", "TOPIC_GLOBAL_SCALING_FACTOR"]
            for key in required_keys:
                if key not in reg_data:
                    errors.append(f"{REGULARIZATION_FILE} is missing key: {key}")
        except Exception as e:
            errors.append(f"Error reading {REGULARIZATION_FILE}: {e}")
    else:
        errors.append(f"{REGULARIZATION_FILE} is missing")
        
    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f" - {err}")
        return False
    else:
        print("Validation PASSED: All files have consistent sizes.")
        return True

if __name__ == "__main__":
    main()

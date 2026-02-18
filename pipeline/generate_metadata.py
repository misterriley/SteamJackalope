import pandas as pd
import numpy as np
import os
import sys
import re
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import PLAYTIME_REGULARIZATION_C, DIFFICULTY_PREDICTIONS_FILE, METADATA_FILE, DIFFICULTY_NEUTRAL_FALLBACK
from common.utils import to_z

def clean_release_date(date_str):
    """
    Parses various Steam release date formats into a standard datetime object.
    """
    if pd.isna(date_str) or date_str == "":
        return pd.NaT
    
    s = str(date_str).strip()
    
    # Match "YYYY-MM-DD"
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return pd.to_datetime(s, errors='coerce')
    
    # Match "YYYY" -> YYYY-07-01
    if re.match(r'^\d{4}$', s):
        return pd.to_datetime(s + "-07-01", errors='coerce')
    
    # Match "Month YYYY" or "YYYY Month" (assume 15th)
    m1 = re.match(r'^([A-Za-z]+)\s+(\d{4})$', s)
    if m1:
        return pd.to_datetime(f"{m1.group(1)} 15, {m1.group(2)}", errors='coerce')
    
    m2 = re.match(r'^(\d{4})\s+([A-Za-z]+)$', s)
    if m2:
        return pd.to_datetime(f"{m2.group(2)} 15, {m2.group(1)}", errors='coerce')

    # Default fallback for "DD Mon, YYYY" etc.
    return pd.to_datetime(s, errors='coerce')

def calculate_date_z_scores(df):
    """
    Calculates z-scores for release dates, clamping future dates and handling unknowns.
    """
    # Clamp future dates to today to avoid scoring them as newer than today
    now = pd.Timestamp.now().normalize()
    
    # We need to operate on a copy or modify inplace safely
    # If df has NaT in parsed_date, logical ops might fail if not handled
    
    # Create a working copy of the date series to modify
    working_dates = df['parsed_date'].copy()
    
    # Clamp future dates
    working_dates[working_dates > now] = now
    
    # Convert to numeric timestamp for z-scoring
    # We convert to float to support NaNs for unknown dates
    ts = pd.to_numeric(working_dates, errors='coerce').astype(float)
    
    # Ensure NaT values are treated as NaN (handles int64 NaT representation if any)
    ts[working_dates.isna()] = np.nan
    
    # Calculate stats on valid dates only
    valid_ts = ts.dropna()
    
    if len(valid_ts) > 0:
        mean_ts = valid_ts.mean()
        std_ts = valid_ts.std()
        
        if std_ts < 1e-12:
            std_ts = 1.0
    else:
        mean_ts = 0
        std_ts = 1.0

    # Initialize z-scores to 0.0 (handles unknowns)
    df['date_z'] = 0.0
    
    # Calculate z-scores for valid dates
    df.loc[ts.notna(), 'date_z'] = (ts[ts.notna()] - mean_ts) / std_ts
    
    # Update parsed_date in the dataframe to reflect clamping? 
    # The requirement didn't explicitly say to update the stored date, but it makes sense for consistency.
    # However, "parsed_date" is an intermediate column. "release_year" was already extracted.
    # We'll update parsed_date just in case.
    df['parsed_date'] = working_dates
    
    return df

def generate_metadata(games_path, reviews_path=None, output_path=None):
    if not os.path.exists(games_path):
        print(f"Error: {games_path} not found.")
        return

    # Use constant default if not specified
    if output_path is None:
        output_path = METADATA_FILE
    
    print(f"Loading games data from {games_path}...")
    df = pd.read_csv(games_path)
    
    # Exclude DLCs if the column exists
    if 'is_dlc' in df.columns:
        dlc_count = df['is_dlc'].fillna(False).sum()
        if dlc_count > 0:
            print(f"Excluding {dlc_count} DLCs...")
            df = df[df['is_dlc'] != True]
            
    df.drop_duplicates(subset=['appid'], inplace=True)
    # Ensure name is present to maintain index synchronization with other pipeline stages
    df.dropna(subset=['appid', 'name'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Use release_date from games_path as the final date
    df['final_release_date'] = df['release_date']

    print("Calculating z-scores for release dates and popularity...")
    # Process release date and its z-score
    tqdm.pandas(desc="Cleaning release dates")
    df['parsed_date'] = df['final_release_date'].progress_apply(clean_release_date)
    
    # Extract release year
    df['release_year'] = df['parsed_date'].dt.year
    mean_year = df['release_year'].mean()
    df['release_year'] = df['release_year'].fillna(mean_year)

    df = calculate_date_z_scores(df)

    # Ensure positive/negative columns are numeric
    for col in ['positive', 'negative']:
        if col in df.columns:
            # Remove commas if present and convert to numeric
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # Process reviews if provided (for both counts and playtime)
    if reviews_path and os.path.exists(reviews_path):
        print(f"Processing reviews from {reviews_path}...")
        try:
            # Load reviews
            df_revs = pd.read_csv(reviews_path, usecols=['appid', 'author_playtime_forever', 'voted_up'])
            
            # --- Repair Review Counts ---
            print("Repairing review counts using scraped reviews...")
            # Count positive (True) and negative (False) reviews per appid
            review_counts = df_revs.groupby(['appid', 'voted_up']).size().unstack(fill_value=0)
            
            # Rename columns if they exist (True -> scraped_positive, False -> scraped_negative)
            # Note: columns might not exist if all reviews are one type
            if True in review_counts.columns:
                review_counts.rename(columns={True: 'scraped_positive'}, inplace=True)
            else:
                review_counts['scraped_positive'] = 0
                
            if False in review_counts.columns:
                review_counts.rename(columns={False: 'scraped_negative'}, inplace=True)
            else:
                review_counts['scraped_negative'] = 0
            
            # Merge scraped counts into main dataframe
            df = df.merge(review_counts[['scraped_positive', 'scraped_negative']], on='appid', how='left')
            df['scraped_positive'] = df['scraped_positive'].fillna(0)
            df['scraped_negative'] = df['scraped_negative'].fillna(0)
            
            # Update official counts with max of official vs scraped
            # This handles cases where scraped_games.csv has 0 or stale data
            df['positive'] = df[['positive', 'scraped_positive']].max(axis=1)
            df['negative'] = df[['negative', 'scraped_negative']].max(axis=1)
            
            # Clean up temporary columns
            df.drop(columns=['scraped_positive', 'scraped_negative'], inplace=True)

            # --- Calculate Playtime ---
            print("Calculating median playtime...")
            # Filter for positive reviews and valid playtime
            df_revs_pos = df_revs[df_revs['voted_up'] == True]
            df_revs_pos = df_revs_pos[df_revs_pos['author_playtime_forever'] > 0]
            
            playtime_stats = df_revs_pos.groupby('appid')['author_playtime_forever'].agg(['median', 'count']).reset_index()
            playtime_stats.rename(columns={'median': 'median_playtime', 'count': 'review_count'}, inplace=True)
            
            df = df.merge(playtime_stats, on='appid', how='left')
            
            # Fill missing review counts with 0
            df['review_count'] = df['review_count'].fillna(0)
            
            # Calculate global mean of (log-transformed) medians for shrinkage
            # We only consider games that actually have some reviews for the global prior
            valid_medians = df[df['review_count'] > 0]['median_playtime']
            global_log_median = np.log1p(valid_medians).mean()
            
            # If no valid medians at all, fallback to 0
            if pd.isna(global_log_median):
                global_log_median = 0.0
            
            # Fill missing median_playtime with exp(global_log_median) - 1
            df['median_playtime'] = df['median_playtime'].fillna(np.expm1(global_log_median))
            
            # Apply Bayesian shrinkage to log-transformed playtime
            log_playtime = np.log1p(df['median_playtime'])
            shrunken_log_playtime = (df['review_count'] * log_playtime + PLAYTIME_REGULARIZATION_C * global_log_median) / (df['review_count'] + PLAYTIME_REGULARIZATION_C)
            
            # Calculate z-score for shrunken log-transformed playtime
            df['playtime_z'] = to_z(shrunken_log_playtime)
            
            # Store estimated playtime (exponential of shrunken log-median)
            df['estimated_playtime'] = np.expm1(shrunken_log_playtime)
            
        except Exception as e:
            print(f"Warning: Failed to process reviews: {e}")
            # Ensure playtime columns exist even if reviews fail, to prevent saving errors
            if 'median_playtime' not in df.columns: df['median_playtime'] = 0
            if 'playtime_z' not in df.columns: df['playtime_z'] = 0
            if 'estimated_playtime' not in df.columns: df['estimated_playtime'] = 0
            
    else:
        if reviews_path:
            print(f"Warning: Reviews file {reviews_path} not found.")
        # Ensure playtime columns exist
        df['median_playtime'] = 0
        df['playtime_z'] = 0
        df['estimated_playtime'] = 0

    # Process popularity (reviews) - done AFTER potential repair
    reviews_count = df['positive'] + df['negative']
    log_rev = np.log1p(reviews_count)
    df['pop_z'] = to_z(log_rev)

    # --- Process Difficulty ---
    difficulty_preds_path = DIFFICULTY_PREDICTIONS_FILE
    if os.path.exists(difficulty_preds_path):
        print("Integrating difficulty predictions...")
        try:
            diff_df = pd.read_csv(difficulty_preds_path)
            # Merge all difficulty related columns (predictions + contributions)
            # Exclude 'name' to avoid collision
            diff_cols = [c for c in diff_df.columns if c != 'name']
            df = df.merge(diff_df[diff_cols], on='appid', how='left')
            df['difficulty_predicted'] = df['difficulty_predicted'].fillna(DIFFICULTY_NEUTRAL_FALLBACK) # Neutral fallback
            
            # Identify games with valid tags for z-score calculation
            # Games with blank tag lists are ignored to prevent skewing the distribution
            has_tags = df['tags'].apply(lambda x: x != '{}' and x != '[]' and x != '' and pd.notna(x))
            
            subset = df[has_tags]['difficulty_predicted'].values
            if len(subset) > 0:
                mean_diff = np.mean(subset)
                std_diff = np.std(subset)
                print(f"Difficulty Z-Scoring: mean={mean_diff:.4f}, std={std_diff:.4f} (from {len(subset)} tagged games)")
                df['difficulty_z'] = (df['difficulty_predicted'] - mean_diff) / (std_diff if std_diff > 1e-9 else 1.0)
            else:
                df['difficulty_z'] = 0.0
        except Exception as e:
            print(f"Warning: Failed to integrate difficulty predictions: {e}")
            df['difficulty_predicted'] = DIFFICULTY_NEUTRAL_FALLBACK
            df['difficulty_z'] = 0.0
    else:
        print("Warning: difficulty_predictions.csv not found. Using defaults.")
        df['difficulty_predicted'] = DIFFICULTY_NEUTRAL_FALLBACK
        df['difficulty_z'] = 0.0

    print("Saving metadata to metadata.parquet...")
    # Ensure all required columns exist (including dynamic contribution columns)
    required_cols = ['appid', 'name', 'short_description', 'genres', 'tags', 'categories', 'supported_languages', 'final_release_date', 'positive', 'negative', 'mature_content', 'price', 'date_z', 'pop_z', 'median_playtime', 'playtime_z', 'estimated_playtime', 'release_year', 'difficulty_predicted', 'difficulty_z', 'intercept', 'difficulty_predicted_raw']
    # Add contribution columns if they exist
    contrib_cols = [c for c in df.columns if c.startswith('contrib_')]
    required_cols.extend(contrib_cols)
    
    available_cols = [c for c in required_cols if c in df.columns]
    metadata_df = df[available_cols].copy()
    
    # Ensure string columns are actually strings (preventing 'double' inference for empty columns)
    string_cols = ['name', 'short_description', 'genres', 'tags', 'categories', 'supported_languages', 'price', 'final_release_date']
    for col in string_cols:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].fillna('').astype(str)

    # Ensure numeric columns don't have NaNs which force them to double/float
    int_cols = ['appid', 'positive', 'negative', 'mature_content', 'release_year']
    for col in int_cols:
        if col in metadata_df.columns:
            metadata_df[col] = pd.to_numeric(metadata_df[col], errors='coerce').fillna(0).astype(np.int64)

    metadata_df.rename(columns={'final_release_date': 'release_date'}, inplace=True)
    metadata_df.to_parquet(output_path, compression='snappy')
    print(f"Metadata saved to {output_path}")
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate consolidated metadata for Steam games.")
    parser.add_argument("games", default="scraped_games.csv", nargs='?', help="Path to scraped_games.csv")
    parser.add_argument("reviews", default="scraped_reviews.csv", nargs='?', help="Path to scraped_reviews.csv")
    parser.add_argument("--output", default=None, help="Output .parquet file")
    
    args = parser.parse_args()
    
    generate_metadata(args.games, args.reviews, args.output)
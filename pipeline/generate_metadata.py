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
    Handle "coming soon" by setting it to 1 year in the future.
    """
    if pd.isna(date_str) or date_str == "":
        return pd.NaT
    
    s = str(date_str).strip()
    
    # Handle "coming soon", "TBD", "Maybe", etc.
    placeholders = ['coming soon', 'to be announced', 'maybe', 'tbd']
    if s.lower() in placeholders:
        return pd.Timestamp.now().normalize() + pd.DateOffset(years=1)
    
    # Handle extreme placeholder dates (e.g., 9998, 6969, 9000)
    extreme_match = re.search(r'\b(9998|6969|9000|2099)\b', s)
    if extreme_match:
        return pd.Timestamp.now().normalize() + pd.DateOffset(years=1)

    # Handle Quarterly dates (e.g., Q1 2026) -> Map to end of that quarter or just 1yr out
    if re.match(r'^[Qq][1-4]\s+\d{4}$', s):
        return pd.Timestamp.now().normalize() + pd.DateOffset(years=1)
    
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
    Future dates are clamped to today to prevent distribution skew.
    """
    from common.utils import to_z
    # Clamp future dates to today
    now = pd.Timestamp.now().normalize()
    working_dates = df['parsed_date'].copy()
    working_dates[working_dates > now] = now
    
    # Convert to numeric timestamp for z-scoring
    ts = pd.to_numeric(working_dates, errors='coerce').astype(float)
    ts[working_dates.isna()] = np.nan
    
    # Calculate z-scores for valid dates
    df['date_z'] = 0.0
    valid_mask = ts.notna()
    if valid_mask.any():
        df.loc[valid_mask, 'date_z'] = to_z(ts[valid_mask])
    
    # Update parsed_date in the dataframe to reflect clamping
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
    tqdm.pandas(desc="Cleaning release dates", smoothing=0)
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

    # --- Process Price ---
    print("Calculating price z-scores...")
    def parse_price(p):
        if pd.isna(p) or p == "": return np.nan
        s = str(p).lower().strip()
        
        # If it contains non-price keywords, it's neutral information
        if any(word in s for word in ["demo", "pass", "preview", "alpha", "beta", "coming soon", "n/a"]):
            return np.nan
            
        if "free" in s: return 0.0
        
        # Look for patterns that look like prices: $19.99, 19.99, $5
        match = re.search(r'(\d+[.,]\d{2})', s) # XX.XX or XX,XX
        if match:
            val = match.group(1).replace(',', '.')
            try: return float(val)
            except: pass
            
        # Just a number? Only if it's small (e.g. < $200) and doesn't look like an ID (long)
        match = re.search(r'^\s*\$?(\d+)\s*$', s)
        if match:
            try:
                val = float(match.group(1))
                if val < 200:
                    return val
            except: pass
                
        return np.nan

    df['price_numeric'] = df['price'].apply(parse_price)
    
    # Use log-transform for z-scoring to handle lognormal distribution
    # We calculate z-scores only on games that actually HAVE a price.
    # Games without a price are assigned a z-score of 0.0 (the mean / neutral information).
    valid_mask = df['price_numeric'].notna()
    log_price = np.log1p(df.loc[valid_mask, 'price_numeric'])
    
    # Calculate stats on valid prices
    lp_mean = log_price.mean()
    lp_std = log_price.std()
    
    # Assign z-scores: (x - mean) / std for valid games, 0.0 for others
    df['price_z'] = 0.0
    if lp_std > 0:
        df.loc[valid_mask, 'price_z'] = (log_price - lp_mean) / lp_std

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

    # --- Pre-calculate Boolean Features ---
    print("Pre-calculating boolean flags...")
    df['is_vr_only'] = df['categories'].fillna('').astype(str).str.contains('VR Only', case=False).astype(bool)
    
    langs = df['supported_languages'].fillna('').astype(str)
    if (langs == '').all():
        df['is_english'] = True
    else:
        df['is_english'] = langs.str.contains('English', case=False).values
        
    df['is_utility'] = df['tags'].fillna('').astype(str).str.contains('Utilities', case=False).astype(bool)
    
    nsfw_tags_pattern = r"'Hentai':"
    df['is_nsfw'] = (
        (df['mature_content'] > 0) | 
        (df['tags'].fillna('').astype(str).str.contains(nsfw_tags_pattern, regex=True, case=False))
    ).values
    
    df['is_delisted'] = (
        df['price'].fillna('').astype(str).str.contains('delisted', case=False) |
        df['name'].fillna('').astype(str).str.contains('DELISTED', case=False)
    ).values
    
    df['is_hollow'] = (
        (df['short_description'].fillna('').str.len() < 10) & 
        (df['tags'].fillna('') == '{}') &
        (df['genres'].fillna('') == '')
    ).values

    print("Saving metadata to metadata.parquet...")
    # Ensure all required columns exist (including dynamic contribution columns)
    required_cols = [
        'appid', 'name', 'short_description', 'genres', 'tags', 'categories', 'supported_languages', 
        'final_release_date', 'parsed_date', 'positive', 'negative', 'mature_content', 'price', 
        'date_z', 'pop_z', 'median_playtime', 'playtime_z', 'estimated_playtime', 'release_year', 
        'difficulty_predicted', 'difficulty_z', 'price_z',
        'is_vr_only', 'is_english', 'is_utility', 'is_nsfw', 'is_delisted', 'is_hollow', 'header_image',
        'tone_z'
    ]
    # Add contribution columns if they exist
    contrib_cols = [c for c in df.columns if c.startswith('contrib_')]
    required_cols.extend(contrib_cols)
    
    available_cols = [c for c in required_cols if c in df.columns]
    metadata_df = df[available_cols].copy()
    
    # Ensure string columns are actually strings (preventing 'double' inference for empty columns)
    string_cols = ['name', 'short_description', 'genres', 'tags', 'categories', 'supported_languages', 'price', 'final_release_date', 'header_image']
    for col in string_cols:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].fillna('').astype(str)

    # Ensure numeric columns don't have NaNs which force them to double/float
    int_cols = ['appid', 'positive', 'negative', 'mature_content', 'release_year']
    for col in int_cols:
        if col in metadata_df.columns:
            metadata_df[col] = pd.to_numeric(metadata_df[col], errors='coerce').fillna(0).astype(np.int64)

    # Boolean columns to int8 for space
    bool_cols = ['is_vr_only', 'is_english', 'is_utility', 'is_nsfw', 'is_delisted', 'is_hollow']
    for col in bool_cols:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].astype(np.int8)

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
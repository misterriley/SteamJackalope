import pandas as pd
import numpy as np
from scipy import stats
import ast
import os
from common.constants import METADATA_FILE

def analyze_user_tags(user_id):
    # Load user ground truth
    ground_truth_file = f"data/user_{user_id}_ground_truth.csv"
    if not os.path.exists(ground_truth_file):
        print(f"Error: {ground_truth_file} not found.")
        return

    user_df = pd.read_csv(ground_truth_file)
    # actual_rating is what we want
    user_df = user_df[['appid', 'actual_rating']].dropna()
    
    # Load metadata
    metadata_df = pd.read_parquet(METADATA_FILE)
    
    # Merge
    merged_df = pd.merge(user_df, metadata_df[['appid', 'tags']], on='appid')
    
    # Parse tags
    def parse_tags(tag_str):
        try:
            if isinstance(tag_str, dict):
                return list(tag_str.keys())
            return list(ast.literal_eval(tag_str).keys())
        except:
            return []
            
    merged_df['tag_list'] = merged_df['tags'].apply(parse_tags)
    
    # Get all unique tags
    all_tags = set()
    for tags in merged_df['tag_list']:
        all_tags.update(tags)
    
    results = []
    
    for tag in all_tags:
        # Create bins
        present_mask = merged_df['tag_list'].apply(lambda x: tag in x)
        
        present_ratings = merged_df[present_mask]['actual_rating']
        absent_ratings = merged_df[~present_mask]['actual_rating']
        
        if len(present_ratings) < 2 or len(absent_ratings) < 2:
            continue # Need some variance for t-test
            
        t_stat, p_val = stats.ttest_ind(present_ratings, absent_ratings, equal_var=False)
        
        # Check for NaN p-value (e.g. all values same in both groups)
        if pd.isna(p_val):
            continue
            
        if p_val < 0.05:
            mean_present = present_ratings.mean()
            mean_absent = absent_ratings.mean()
            diff = mean_present - mean_absent
            
            results.append({
                'tag': tag,
                'mean_present': mean_present,
                'mean_absent': mean_absent,
                'mean_diff': diff,
                'p_value': p_val,
                'count_present': len(present_ratings),
                'count_absent': len(absent_ratings)
            })
            
    # Sort results
    results_df = pd.DataFrame(results).sort_values(by='mean_diff', ascending=False)
    
    print(f"Analysis for user {user_id}:")
    print(f"Total games with ratings: {len(merged_df)}")
    print("\nTop Positively Associated Tags (p < 0.05):")
    pos_tags = results_df[results_df['mean_diff'] > 0]
    if not pos_tags.empty:
        print(pos_tags[['tag', 'mean_diff', 'p_value', 'count_present', 'count_absent']].to_string(index=False))
    else:
        print("None found.")
        
    print("\nTop Negatively Associated Tags (p < 0.05):")
    neg_tags = results_df[results_df['mean_diff'] < 0].sort_values(by='mean_diff', ascending=True)
    if not neg_tags.empty:
        print(neg_tags[['tag', 'mean_diff', 'p_value', 'count_present', 'count_absent']].to_string(index=False))
    else:
        print("None found.")

if __name__ == "__main__":
    analyze_user_tags("76561198039155404")

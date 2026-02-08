import pandas as pd
import ast
from collections import Counter

def process_tags():
    # Load the dataset
    print("Loading games_march2025_cleaned.csv...")
    df = pd.read_csv('games_march2025_cleaned.csv')
    
    global_tag_counts = Counter()
    utilities_tag_counts = Counter()
    
    print("Processing tags...")
    for index, row in df.iterrows():
        tags_str = row['tags']
        if pd.isna(tags_str) or tags_str == '':
            continue
            
        try:
            # Use ast.literal_eval since the data looks like a Python dictionary string
            tags_dict = ast.literal_eval(tags_str)
            
            if isinstance(tags_dict, dict):
                current_tags = list(tags_dict.keys())
                global_tag_counts.update(current_tags)
                
                if 'Utilities' in current_tags:
                    utilities_tag_counts.update(current_tags)
        except (ValueError, SyntaxError):
            # Fallback for potentially malformed strings if any
            continue

    # Print Global Tag Counts
    print("\n" + "="*50)
    print("GLOBAL TAG COUNTS (Sorted by Frequency):")
    print("="*50)
    # Sort by count (descending), then by name (ascending)
    for tag, count in sorted(global_tag_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{tag}: {count}")
        
    # Print Utilities Tag Counts
    print("\n" + "="*50)
    print("TAG COUNTS FOR GAMES CONTAINING 'Utilities' (Sorted by Frequency):")
    print("="*50)
    if not utilities_tag_counts:
        print("No entries found with the 'Utilities' tag.")
    else:
        for tag, count in sorted(utilities_tag_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"{tag}: {count}")

if __name__ == "__main__":
    process_tags()

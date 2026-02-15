import pandas as pd
import ast
import json
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import METADATA_FILE

def parse_list_safe(x):
    if pd.isna(x): return []
    x = str(x)
    if x.startswith('[') and x.endswith(']'):
        try:
            return ast.literal_eval(x)
        except:
            pass
    return [g.strip() for g in x.split(',') if g.strip()]

def parse_tags_safe(x):
    if pd.isna(x): return []
    x = str(x)
    try:
        # Handle dict format: {'Tag': count, ...}
        if x.startswith('{') and x.endswith('}'):
            data = ast.literal_eval(x)
            if isinstance(data, dict):
                return list(data.keys())
            return []
        # Handle list format: ['Tag1', 'Tag2', ...]
        if x.startswith('[') and x.endswith(']'):
            return ast.literal_eval(x)
        # Fallback to comma-separated
        return [g.strip() for g in x.split(',') if g.strip()]
    except:
        return []

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found.")
        return

    print(f"Loading metadata from {METADATA_FILE}...")
    df = pd.read_parquet(METADATA_FILE, columns=['genres', 'tags', 'categories'])
    
    all_genres = set()
    all_tags = set()
    all_categories = set()
    
    print("Extracting genres...")
    for g_str in df['genres'].dropna().unique():
        all_genres.update(parse_list_safe(g_str))
        
    print("Extracting tags...")
    for t_str in df['tags'].dropna().unique():
        all_tags.update(parse_tags_safe(t_str))

    print("Extracting categories...")
    if 'categories' in df.columns:
        for c_str in df['categories'].dropna().unique():
            all_categories.update(parse_list_safe(c_str))
        
    results = {
        "genres": sorted(list(all_genres)),
        "tags": sorted(list(all_tags)),
        "categories": sorted(list(all_categories))
    }
    
    output_file = "data/unique_terms.json"
    os.makedirs("data", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Extracted {len(all_genres)} genres, {len(all_tags)} tags, and {len(all_categories)} categories.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

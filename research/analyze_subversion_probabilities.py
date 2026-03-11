import pandas as pd
import numpy as np

def get_list(val):
    if pd.isna(val).all() if hasattr(val, 'all') else pd.isna(val): return []
    if isinstance(val, dict): return list(val.keys())
    if isinstance(val, str):
        try:
            d = eval(val)
            if isinstance(d, dict): return list(d.keys())
        except: pass
        return [x.strip() for x in val.split(',')]
    if hasattr(val, 'tolist'): return val.tolist()
    if isinstance(val, np.ndarray) and len(val.shape) == 0 and isinstance(val.item(), dict):
        return list(val.item().keys())
    return list(val)

def main():
    df = pd.read_parquet('data/production/metadata.parquet')
    df['parsed_tags'] = df['tags'].apply(lambda x: set(get_list(x)))
    df['total_reviews'] = df['positive'].fillna(0) + df['negative'].fillna(0)
    
    meta_tags = ['Psychological Horror', 'Fourth Wall', 'Surreal', 'Satire', 'Parody', 'Illuminati', 'Mind-Bending']
    innocent_tags = ['Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development']
    
    print("=== Subversion Intersection Analysis ===\n")
    
    for meta in meta_tags:
        mask_meta = df['parsed_tags'].apply(lambda x: meta in x)
        total_meta = mask_meta.sum()
        
        print(f"[{meta.upper()}] (Total games: {total_meta})")
        
        # See how it interacts with ANY innocent tag
        mask_any_innocent = df['parsed_tags'].apply(lambda x: len(x.intersection(innocent_tags)) > 0)
        intersection_all = df[mask_meta & mask_any_innocent].sort_values('total_reviews', ascending=False)
        
        print(f"-> Overlap with ANY Innocent tag: {len(intersection_all)} games ({len(intersection_all)/total_meta*100:.1f}%)")
        print("-> Top 5 games in this overlap:")
        for _, row in intersection_all.head(5).iterrows():
            overlapping_innocent = list(row['parsed_tags'].intersection(innocent_tags))
            print(f"   * {str(row['name'])[:35]:<35} | Innocent tags: {overlapping_innocent}")
            
        print("-" * 60)

    # Let's do a specific cross-tabulation for the most potent combos
    print("\n=== Specific Tag Pair Frequencies ===")
    pairs_to_check = [
        ('Psychological Horror', 'Dating Sim'),
        ('Psychological Horror', 'Cute'),
        ('Fourth Wall', 'Cute'),
        ('Surreal', 'Cute'),
        ('Satire', 'Farming Sim'),
        ('Illuminati', 'Education')
    ]
    
    for m_tag, i_tag in pairs_to_check:
        mask = df['parsed_tags'].apply(lambda x: m_tag in x and i_tag in x)
        subset = df[mask].sort_values('total_reviews', ascending=False)
        print(f"\n{m_tag} + {i_tag}: {len(subset)} games")
        if len(subset) > 0:
            for _, row in subset.head(5).iterrows():
                print(f"   - {str(row['name'])[:45]}")

if __name__ == "__main__":
    main()

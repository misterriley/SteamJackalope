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

df = pd.read_parquet('data/production/metadata.parquet')
illuminati_games = df[df['tags'].apply(lambda x: 'Illuminati' in get_list(x))].copy()
illuminati_games['total_reviews'] = illuminati_games['positive'].fillna(0) + illuminati_games['negative'].fillna(0)
illuminati_games = illuminati_games.sort_values('total_reviews', ascending=False)

print(f'Total games with Illuminati tag: {len(illuminati_games)}')
print('\nTop 30 most reviewed Illuminati games:')
for _, row in illuminati_games.head(30).iterrows():
    print(f"{str(row['name'])[:40]:<40} | Reviews: {int(row['total_reviews']):,}")

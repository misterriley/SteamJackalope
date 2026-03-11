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
appids = [1194840, 698780, 2380360]
for appid in appids:
    row = df[df['appid'] == appid]
    if not row.empty:
        r = row.iloc[0]
        tags = get_list(r.get('tags', []))
        tone = r.get('tone_z', np.nan)
        print(f'{r["name"]} (AppID: {appid}) - Tone Z: {tone:.2f}')
        print(f'Tags: {tags[:15]}\n')

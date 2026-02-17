import pandas as pd
import ast

gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
df = pd.read_parquet('data/production/metadata.parquet')
user_df = df[df['appid'].isin(gt['appid'])]

def has_tag(tags_str, tag_name):
    if not tags_str: return False
    try:
        t_dict = ast.literal_eval(tags_str)
        return tag_name in t_dict
    except: return False

print("Total games in metadata match:", len(user_df))
print("Mahjong count:", user_df['tags'].apply(lambda x: has_tag(x, 'Mahjong')).sum())
print("NSFW count:", user_df['tags'].apply(lambda x: has_tag(x, 'NSFW')).sum())
print("Sci-fi count:", user_df['tags'].apply(lambda x: has_tag(x, 'Sci-fi')).sum())
print("Hentai count:", user_df['tags'].apply(lambda x: has_tag(x, 'Hentai')).sum())

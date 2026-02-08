import pandas as pd
import ast

print("Loading data...")
df = pd.read_csv('data/pipeline_games_clean.csv', low_memory=False)

tag_counts = {}

# Count number of games that have at least one tag
games_with_tags = df[df['tags'].notna() & (df['tags'] != '') & df['tags'].apply(lambda x: len(ast.literal_eval(x)) > 0)].shape[0]
print(f"Number of games with at least one tag: {games_with_tags}")

print("Counting tags...")
for tags_str in df['tags']:
    if pd.isna(tags_str):
        continue
    try:
        t_data = ast.literal_eval(tags_str)
        t_list = t_data.keys() if isinstance(t_data, dict) else t_data
        for t in t_list:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    except:
        continue

print("Sorting and saving...")
sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

# Save to CSV
with open('research/tag_counts.csv', 'w', encoding='utf-8') as f:
    f.write("tag,count\n")
    for tag, count in sorted_tags:
        # Simple CSV escape for tags with commas
        f.write(f'"{tag}",{count}\n')

print("\nTop common tags:")
for tag, count in sorted_tags:
    print(f"{tag}: {count} - {count/games_with_tags:.4%} of games with tags")

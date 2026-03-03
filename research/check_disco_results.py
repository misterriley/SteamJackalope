import json
with open('data/user_76561198039155404_taste_profile.json', 'r') as f:
    data = json.load(f)
for f in data['favorite_game_recommendations']:
    if 'Disco Elysium' in f['seed_name']:
        print(f"{f['seed_name']} Recs: {[g['name'] for g in f['top_games']]}")

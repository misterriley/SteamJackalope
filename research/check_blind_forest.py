import json
with open('data/user_76561198039155404_taste_profile.json', 'r') as f:
    data = json.load(f)
for f in data['favorite_game_recommendations']:
    if 'Ori and the Blind Forest' in f['seed_name']:
        print(f"Blind Forest Recs: {[g['name'] for g in f['top_games']]}")

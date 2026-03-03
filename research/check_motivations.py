import json
import os

with open('data/user_76561198039155404_taste_profile.json', 'r') as f:
    data = json.load(f)

for m in data['gamer_motivations']:
    names = [g['name'] for g in m['top_games']]
    print(f"{m['motivation']:<15}: {names}")

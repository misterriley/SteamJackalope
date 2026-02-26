import json

with open('playnite_steam_games.json', 'r') as f:
    data = json.load(f)

print("Top 20 Enriched Playnite Games:")
for g in data[:20]:
    print(f"- {g['name']} ({g['appid']}): {g['playtime_hrs']} hours")

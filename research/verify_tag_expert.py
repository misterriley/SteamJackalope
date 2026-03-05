import json

def verify_tag_expert():
    path = 'data/user_76561198039155404_taste_profile.json'
    with open(path, 'r') as f:
        p = json.load(f)
    
    top_tags = p['associative_tags']['top']
    if len(top_tags) > 0:
        tag = top_tags[0]
        print(f"Tag: {tag['tag']}")
        print(f"Has top_games: {'top_games' in tag}")
        if 'top_games' in tag:
            print(f"Top Games Count: {len(tag['top_games'])}")
            for g in tag['top_games']:
                print(f"  - {g['name']}")
    else:
        print("No top tags found.")

if __name__ == '__main__':
    verify_tag_expert()

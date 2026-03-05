import json

def verify_profile():
    path = 'data/user_76561198039155404_taste_profile.json'
    with open(path, 'r') as f:
        p = json.load(f)
    print(f"North Stars: {len(p['north_stars'])}")
    print(f"Favorite Recs Groups: {len(p['favorite_game_recommendations'])}")
    
    # Just to be sure, let's see which favorites are being used
    fav_names = [fav['seed_name'] for fav in p['favorite_game_recommendations']]
    print(f"Seeds used for similar games: {fav_names}")

if __name__ == '__main__':
    verify_profile()

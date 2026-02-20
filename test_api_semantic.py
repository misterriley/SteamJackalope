import requests
import json

def test_api():
    payload = {
        "alpha": 2.0,
        "beta": 0.0,
        "quality_pref": 0.0,
        "age_pref": 0.0,
        "pop_pref": 0.0,
        "disc_pref": 0.0,
        "length_pref": 0.0,
        "difficulty_pref": 0.0,
        "price_pref": 0.0,
        "prompt": "a relaxing farming simulator",
        "seed_games": [],
        "genres": [],
        "tags": [],
        "top_k": 10,
        "remove_vr": True,
        "english_only": True,
        "remove_nsfw": True,
        "remove_utilities": True,
        "remove_unreleased": True
    }
    
    try:
        r = requests.post("http://127.0.0.1:8000/recommend", json=payload)
        if r.status_code != 200:
            print(f"Error: {r.status_code} - {r.text}")
            return
            
        data = r.json()
        print(f"Results for 'a relaxing farming simulator':")
        for item in data:
            print(f" - {item['name'][:40]:40} | Score: {item['weighted_score']:6.3f} | Semantic: {item['semantic_match']:6.3f}")
            
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_api()

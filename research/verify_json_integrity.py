import json

def verify_json():
    path = 'data/user_76561198039155404_taste_profile.json'
    with open(path, 'r') as f:
        p = json.load(f)
    print(f"DNA OOS R2: {p['metadata']['oos_r2']:.4f}")
    print(f"Kernel Weight: {p['metadata']['kernel_match']:.4f}")
    print(f"Graph Weight: {p['metadata']['graph_match']:.4f}")

if __name__ == '__main__':
    verify_json()

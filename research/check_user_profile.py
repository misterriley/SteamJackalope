import json
import os

def extract_profile_summary(user_id):
    path = "data/user_" + user_id + "_taste_profile.json"
    if not os.path.exists(path):
        print("File " + path + " not found.")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    print("--- Profile Summary for " + user_id + " ---")
    print("\nMetadata Weights:")
    for k, v in data['metadata'].items():
        print("  " + k.ljust(12) + ": " + format(v, "+.4f"))
    
    print("\nTop Topic Predictors:")
    topics = data.get('topics', {}).get('top_topics', [])
    for t in topics[:10]:
        print("  Topic " + str(t['index']).ljust(3) + " (" + format(t['weight'], "+.4f") + "): " + str(t['label']))

    print("\nTop Tag Dimensions:")
    dims = data.get('tag_dimensions', {}).get('top_dimensions', [])
    for d in dims[:10]:
        print("  Dim " + str(d['index']).ljust(3) + " (" + format(d['weight'], "+.4f") + "): " + str(d['label']))

if __name__ == "__main__":
    extract_profile_summary("76561198039155404")

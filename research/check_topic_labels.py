
import json
import os

def check_topics(topics=[3, 0, 196, 18, 135]):
    path = 'data/production/topic_descriptions.json'
    if not os.path.exists(path):
        print(f"Error: {path} missing.")
        return
        
    with open(path, 'r') as f:
        descriptions = json.load(f)
        
    for t in topics:
        print(f"Topic {t}: {descriptions.get(str(t))}")

if __name__ == "__main__":
    check_topics()

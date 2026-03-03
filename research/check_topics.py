import json
with open('data/production/topic_descriptions.json', 'r') as f:
    d = json.load(f)
for t in ['18', '155', '59', '111', '219']:
    print(f"Topic {t}: {d.get(t)}")

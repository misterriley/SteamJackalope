import json

with open("tag_categories.json", "r") as f:
    categories = json.load(f)

verbs = set(categories["verbs"])
nouns = set(categories["nouns"])
adjectives = set(categories["adjectives"])
other = set(categories["other"])

tags_to_promote = [
    # From Nouns
    "Exploration", "Conversation", "Diplomacy", "Trading", "Gambling", "Destruction",
    "Automation", "Multiple Endings", "Collectathon", "Narration", "Audio Production",
    "Video Production", "Photo Editing", 
    # From Adjectives
    "Cinematic", "Emotional", "Horror", "Psychological Horror", "Story Rich", "Relaxing",
    "Competitive", "Violent",
    # From Other
    "Choices Matter", "Comedy"
]

for t in tags_to_promote:
    verbs.add(t)
    if t in nouns: nouns.remove(t)
    if t in adjectives: adjectives.remove(t)
    if t in other: other.remove(t)

out = {
    "verbs": sorted(list(verbs)),
    "nouns": sorted(list(nouns)),
    "adjectives": sorted(list(adjectives)),
    "other": sorted(list(other))
}

with open("tag_categories.json", "w") as f:
    json.dump(out, f, indent=4)

print(f"Promoted {len(tags_to_promote)} tags to verbs.")

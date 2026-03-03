import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import re
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TAG_NAMES_FILE

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:7b"

def strip_titles(text):
    titles_to_remove = [
        "The Sims", "Call of Duty", "Battlefield", "Halo", "Street Fighter", 
        "Injustice", "BIT.TRIP RUNNER", "Starcraft", "League of Legends", 
        "World of Warcraft", "Facebook", "Portal 2", "Mario Kart", 
        "Dark Souls", "XCOM", "Fire Emblem", "Civilization", "Cities: Skylines", 
        "Europa Universalis", "Skyrim", "Fallout", "Mass Effect", "Dragon Age", 
        "The Last of Us", "BioShock", "MineCraft"
    ]
    if not isinstance(text, str): return text
    text = re.sub(r'(They gravitate towards|titles like|titles such as|games like|found in games like|found in titles like|found in|found in titles like|scenarios in games like|missions in games like|locations in|games such as|titles like) [^.]*?\.', '.', text)
    for title in titles_to_remove:
        text = re.sub(re.escape(title), "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.+', '.', text)
    return text.strip()

def generate_tags_for_motivation(motivation, description, valid_tags):
    valid_tags_lower = {t.lower(): t for t in valid_tags}
    
    prompt = "You are an expert in video game design and Steam player psychology.\n"
    prompt += "I have a gamer motivation category called '" + motivation + "'.\n"
    prompt += "Description: " + description + "\n\n"
    prompt += "From the following list of valid Steam tags, select exactly 20 tags that best represent this motivation.\n"
    prompt += "Valid Tags:\n" + ", ".join(valid_tags) + "\n\n"
    prompt += "Respond ONLY with a JSON list of the 20 tags you selected.\n"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        raw_resp = result['response']
        data = json.loads(raw_resp)
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break
        if isinstance(data, list):
            found = []
            for t in data:
                t_str = str(t).strip().lower()
                if t_str in valid_tags_lower:
                    found.append(valid_tags_lower[t_str])
            return found[:20]
        return []
    except Exception as e:
        print("Error for " + motivation + ": " + str(e))
        return []

def main():
    csv_path = "research/GamerMotivationDescriptions_Fixed.csv"
    if not os.path.exists(csv_path):
        print("Error: " + csv_path + " not found.")
        return
    df_mot = pd.read_csv(csv_path)
    with open(TAG_NAMES_FILE, 'r') as f:
        valid_tags = json.load(f)
    motivation_profiles = {}
    print("Generating profiles...")
    for _, row in tqdm(df_mot.iterrows(), total=len(df_mot)):
        mot = str(row['Motivation'])
        desc = str(row['Short Description']) + ". " + strip_titles(str(row['Long Description']))
        tags = generate_tags_for_motivation(mot, desc, valid_tags)
        motivation_profiles[mot] = {"description": desc, "synthetic_tags": tags}
    output_path = "research/synthetic_motivation_tags_fixed.json"
    with open(output_path, 'w') as f:
        json.dump(motivation_profiles, f, indent=4)
    print("Saved to " + output_path)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TAG_NAMES_FILE

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:7b"

def generate_tags_for_motivation(motivation, description, valid_tags):
    valid_tags_lower = {t.lower(): t for t in valid_tags}
    
    prompt = "You are an expert in video game design and Steam player psychology.\n"
    prompt += "I have a gamer motivation category called '" + motivation + "'.\n"
    prompt += "Description: " + description + "\n\n"
    prompt += "From the following list of valid Steam tags, select exactly 20 tags that best represent this motivation. "
    prompt += "Choose tags that describe the core gameplay mechanics, themes, and player experience associated with '" + motivation + "'.\n\n"
    prompt += "Valid Tags:\n" + ", ".join(valid_tags) + "\n\n"
    prompt += "Respond ONLY with a JSON list of the 20 tags you selected.\n"
    prompt += 'Example: ["Action", "Strategy", "Open World", ...]'
    
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
        # print("DEBUG: raw response for " + motivation + ": " + raw_resp)
        data = json.loads(raw_resp)
        
        # If model returns {"tags": [...]} instead of [...]
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
        print("Error generating tags for " + motivation + ": " + str(e))
        return []

def main():
    csv_path = "research/GamerMotivationDescriptions_Clean.csv"
    if not os.path.exists(csv_path):
        print("Error: " + csv_path + " not found.")
        return
        
    df_mot = pd.read_csv(csv_path)
    
    with open(TAG_NAMES_FILE, 'r') as f:
        valid_tags = json.load(f)
        
    motivation_profiles = {}
    
    print("Generating synthetic tag profiles using Ollama (" + MODEL_NAME + ")...")
    for _, row in tqdm(df_mot.iterrows(), total=len(df_mot)):
        mot = str(row['Motivation'])
        desc = str(row['Short Description']) + ". " + str(row['Long Description'])
        
        tags = generate_tags_for_motivation(mot, desc, valid_tags)
        motivation_profiles[mot] = {
            "description": desc,
            "synthetic_tags": tags
        }
        
    output_path = "research/synthetic_motivation_tags.json"
    with open(output_path, 'w') as f:
        json.dump(motivation_profiles, f, indent=4)
        
    print("Saved synthetic profiles to " + output_path)
    
    for mot, data in motivation_profiles.items():
        print("\n[" + mot.upper() + "]")
        print("Tags: " + ", ".join(data['synthetic_tags']))

if __name__ == "__main__":
    main()

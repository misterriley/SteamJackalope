import pandas as pd
import numpy as np
import ast
import re

def analyze_tone():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    topics = np.load('data/production/topic_distributions.npy', mmap_mode='r')
    
    # --- Define Tonal Markers ---
    # Score 1-5 (Serious to Bizarre)
    
    TONE_TOPICS = {
        # Serious (1)
        15: 1,  # Russian & Soviet Narrative
        21: 1,  # Military Tank Combat
        74: 1,  # Medieval Kingdom Strategy
        87: 1,  # Military FPS Combat
        200: 1, # Delta Force Tactical Rescue
        
        # Laid Back (2)
        44: 2,  # Mental Health & Meditation
        136: 2, # Gardening & Flower Cozy
        150: 2, # Daily Life Simulation
        170: 2, # Astral & ASMR Experiences
        
        # Humorous (3)
        9: 3,   # Hidden Object Cat Games
        53: 3,  # Cute Bunny & Panda
        141: 3, # Paper & Drawing Cozy
        172: 3, # Capybara & Cute Adventure
        
        # Absurd (4)
        43: 4,  # Mecha & Meme Games
        116: 4, # Corporate Office Workplace Satire
        160: 4, # Multiplayer Party Brawlers
        179: 4, # Historical WWII Nazi Satire
        
        # Bizarre (5)
        75: 5,  # Abstract Indie Projects
        91: 5,  # Midnight Surreal Horror
        118: 5, # Dreamscape Nightmare Horror
        184: 5, # Multiverse & Meta Narrative
    }
    
    TONE_TAGS = {
        "Simulation": 1, "Strategy": 1, "Historical": 1, "Realistic": 1, "Gritty": 1,
        "Relaxing": 2, "Casual": 2, "Cozy": 2, "Family Friendly": 2,
        "Funny": 3, "Cartoon": 3, "Colorful": 3, "Cute": 3,
        "Comedy": 4, "Satire": 4, "Memes": 4, "Parody": 4,
        "Surreal": 5, "Abstract": 5, "Experimental": 5, "Psychological Horror": 5, "Stylized": 4
    }

    print("Calculating Tone Axis...")
    
    # 1. Topic Tone
    topic_tone = np.zeros(len(df))
    topic_weight_sum = np.zeros(len(df))
    for tid, score in TONE_TOPICS.items():
        topic_tone += topics[:, tid] * score
        topic_weight_sum += topics[:, tid]
    
    # Normalize by active tonal topics
    topic_tone_norm = np.where(topic_weight_sum > 0.01, topic_tone / topic_weight_sum, 2.5) # Default to center

    # 2. Tag Tone
    tag_tone = np.zeros(len(df))
    tag_count_sum = np.zeros(len(df))
    
    tag_series = df['tags'].fillna('').astype(str)
    for tag, score in TONE_TAGS.items():
        # Match pattern "'Tag': Value"
        pattern = rf"'{re.escape(tag)}':\s*(\d+)"
        matches = tag_series.str.extract(pattern).fillna(0).astype(float).values.flatten()
        tag_tone += matches * score
        tag_count_sum += matches
        
    tag_tone_norm = np.where(tag_count_sum > 0, tag_tone / tag_count_sum, 2.5)

    # 3. Combined Tone Axis (Hybrid)
    # We weigh tags more heavily for tone as they are explicit
    combined_tone = (0.7 * tag_tone_norm) + (0.3 * topic_tone_norm)
    
    df['tone_score'] = combined_tone
    
    # --- Verify Results ---
    targets = [
        1194840, # Frog Fractions
        1379510, # Algebra Ridge
        440,     # TF2 (Humorous)
        730,     # CS (Serious)
        413150,  # Stardew (Laid Back)
        105600,  # Terraria
        620,     # Portal 2 (Humorous/Absurd)
        350310,  # Sky Hill
        504230,  # Celeste
        1145360, # Hades
        1091500, # Cyberpunk
    ]
    
    print("\nTone Spectrum Analysis (1=Serious, 5=Bizarre):")
    print("-" * 60)
    for tid in targets:
        m = df[df['appid'] == tid]
        if not m.empty:
            print(f"{m.iloc[0]['tone_score']:.2f} | {m.iloc[0]['name']}")
            
    # Show Top Bizarre
    print("\nTop Bizarre Games:")
    print(df.sort_values('tone_score', ascending=False)[['tone_score', 'name']].head(10))
    
    # Show Top Serious
    print("\nTop Serious Games:")
    print(df.sort_values('tone_score', ascending=True)[['tone_score', 'name']].head(10))

if __name__ == "__main__":
    analyze_tone()

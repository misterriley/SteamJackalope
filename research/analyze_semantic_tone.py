import pandas as pd
import numpy as np
import os

def analyze_semantic_tone():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    sem_vectors = np.load('data/production/embeddings_desc.npy', mmap_mode='r')
    
    # --- Define Semantic Anchors ---
    SERIOUS_IDS = [
        107410,  # Arma 3
        223750,  # DCS World
        393380,  # Squad
        394360,  # Hearts of Iron IV
        236850,  # Europa Universalis IV
        1250410, # MS Flight Sim
        244210,  # Assetto Corsa
        730,     # CS
        581320,  # Insurgency: Sandstorm
        736220,  # Post Scriptum
    ]
    
    ABSURD_IDS = [
        1194840, # Frog Fractions
        250260,  # Jazzpunk
        265930,  # Goat Simulator
        224480,  # Octodad
        557340,  # Getting Over It
        327890,  # I Am Bread
        233720,  # Surgeon Simulator
        469820,  # Genital Jousting
        1388770, # Cruelty Squad
        1284710, # Hylics 2
    ]
    
    def get_vecs(ids):
        indices = df[df['appid'].isin(ids)].index.tolist()
        if not indices: return np.array([])
        return sem_vectors[indices].astype(np.float32)

    vecs_ser = get_vecs(SERIOUS_IDS)
    vecs_abs = get_vecs(ABSURD_IDS)
    
    if vecs_ser.size == 0 or vecs_abs.size == 0:
        print("Error: Could not find anchor IDs in metadata.")
        return
        
    mean_ser = np.mean(vecs_ser, axis=0)
    mean_abs = np.mean(vecs_abs, axis=0)
    
    # Define Tone Vector (Serious -> Absurd)
    tone_vector = mean_abs - mean_ser
    tone_vector /= np.linalg.norm(tone_vector) + 1e-9
    
    print("Projecting games onto Tone Axis...")
    # Batch processing for projection
    projections = np.zeros(len(df))
    batch_size = 50000
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        batch = sem_vectors[i:end].astype(np.float32)
        projections[i:end] = np.dot(batch, tone_vector)
        
    df['tone_projection'] = projections
    
    # Normalize to 0-1 range for visibility
    p_min = df['tone_projection'].min()
    p_max = df['tone_projection'].max()
    df['tone_normalized'] = (df['tone_projection'] - p_min) / (p_max - p_min)
    
    # --- Verify Results ---
    targets = [
        1194840, # Frog Fractions
        1379510, # Algebra Ridge
        440,     # TF2
        730,     # CS
        413150,  # Stardew
        620,     # Portal 2
        1091500, # Cyberpunk
        219890,  # Antichamber
        1388770, # Cruelty Squad
    ]
    
    print("\nSemantic Tone Axis (0=Serious, 1=Absurd/Bizarre):")
    print("-" * 60)
    for tid in targets:
        m = df[df['appid'] == tid]
        if not m.empty:
            print(f"{m.iloc[0]['tone_normalized']:.4f} | {m.iloc[0]['name']}")
            
    # Show Extremes
    print("\nTop Absurd/Bizarre (Description):")
    print(df.sort_values('tone_normalized', ascending=False)[['tone_normalized', 'name']].head(10))
    
    print("\nTop Serious (Description):")
    print(df.sort_values('tone_normalized', ascending=True)[['tone_normalized', 'name']].head(10))

if __name__ == "__main__":
    analyze_semantic_tone()

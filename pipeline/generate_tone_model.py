import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, EMBEDDINGS_DESC_FILE, PRODUCTION_DATA_DIR

def generate_tone_model():
    print("Loading production artifacts...")
    df = pd.read_parquet(METADATA_FILE)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    
    # --- Robust Anchor IDs ---
    # Serious (Simulation, Historical, Tactical)
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
    
    # Absurd (Surreal, Satire, Slapstick, Meta)
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
    
    def get_indices(ids):
        return df[df['appid'].isin(ids)].index.tolist()

    idx_ser = get_indices(SERIOUS_IDS)
    idx_abs = get_indices(ABSURD_IDS)
    
    if not idx_ser or not idx_abs:
        print(f"Error: Missing anchors. Serious found: {len(idx_ser)}, Absurd found: {len(idx_abs)}")
        return

    # Calculate centroid vectors
    vecs_ser = sem_vectors[idx_ser].astype(np.float32)
    vecs_abs = sem_vectors[idx_abs].astype(np.float32)
    
    mean_ser = np.mean(vecs_ser, axis=0)
    mean_abs = np.mean(vecs_abs, axis=0)
    
    # Define TONE AXIS (Serious -> Absurd)
    tone_vector = mean_abs - mean_ser
    tone_vector /= np.linalg.norm(tone_vector) + 1e-9
    
    print(f"Projecting {len(df)} games onto Tone Axis...")
    tone_scores = np.zeros(len(df), dtype=np.float32)
    batch_size = 50000
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        batch = sem_vectors[i:end].astype(np.float32)
        tone_scores[i:end] = np.dot(batch, tone_vector)
        
    # Standardize to Z-scores
    mean_score = np.mean(tone_scores)
    std_score = np.std(tone_scores)
    tone_z = (tone_scores - mean_score) / (std_score + 1e-9)
    
    # Save to production
    output_path = os.path.join(PRODUCTION_DATA_DIR, "tone_z.npy")
    np.save(output_path, tone_z)
    print(f"Saved Tone Z-scores to {output_path}")
    print(f"Axis Stats: Mean={mean_score:.4f}, Std={std_score:.4f}")
    
    # Update Metadata Parquet with Tone Z for easy access
    df['tone_z'] = tone_z
    df.to_parquet(METADATA_FILE, index=False)
    print("Updated metadata.parquet with tone_z column.")

if __name__ == "__main__":
    generate_tone_model()

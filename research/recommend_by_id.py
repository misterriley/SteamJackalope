import pandas as pd
import numpy as np
import sys
import os
import ast
import json
import re
from common.utils import calculate_jackalope_kernel, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA,
    EPSILON
)

def main():
    print("Loading production data (one-time setup)...")
    df = pd.read_parquet(METADATA_FILE)
    
    # Load artifacts once
    verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    print("Pre-calculating anchor masks...")
    anchor_masks = {}
    all_anchor_tags = set()
    for tags in MIGS.values(): all_anchor_tags.update(tags)
    all_anchor_tags.update(NARRATIVE_TAGS)
    all_anchor_tags.update(HORROR_MARKERS)
    all_anchor_tags.update(HARD_ANCHORS)
    all_anchor_tags.add("Isometric")
    all_anchor_tags.add("CRPG")
    
    # Tonal/Dissonance Markers (Soul Rescues)
    all_anchor_tags.update({"Education", "Math", "Science", "Typing", "Spelling", "Programming", "Logic"})
    all_anchor_tags.update({"Surreal", "Comedy", "Funny", "Satire", "Parody", "Memes", "Abstract"})
    all_anchor_tags.update({"Cute", "Colorful", "Family Friendly", "Relaxing", "Anime"})
    all_anchor_tags.update({"Horror", "Psychological Horror", "Survival Horror", "Gore", "Violent"})
    
    tag_series = df['tags'].fillna('').astype(str)
    for tag in all_anchor_tags:
        pattern = rf"'{re.escape(tag)}':"
        anchor_masks[tag] = tag_series.str.contains(pattern, regex=True).values

    print("\nJackalope Recommendation Research Tool (V5 - Bugfix)")
    print("Type an AppID, a Game Name, or 'exit' to quit.")

    while True:
        try:
            user_input = input("\nEnter ID or Name > ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input:
                continue
                
            # 1. Resolve Seed
            if user_input.isdigit():
                seed_appid = int(user_input)
                match = df[df['appid'] == seed_appid]
            else:
                # Fuzzy name match
                pattern = re.escape(user_input.lower())
                matches = df[df['name'].str.lower().str.contains(pattern, na=False)]
                if matches.empty:
                    print(f"Error: No game found matching '{user_input}'")
                    continue
                if len(matches) > 1:
                    print(f"Multiple matches found. Using first: {matches.iloc[0]['name']} ({matches.iloc[0]['appid']})")
                    print("Matches: " + ", ".join(matches['name'].head(5).tolist()))
                match = matches.head(1)
                seed_appid = match.iloc[0]['appid']

            seed_idx = match.index[0]
            seed_name = match.iloc[0]['name']
            print(f"Calculating for: {seed_name}...")

            # 2. Prepare Seed Metadata
            tags_s_str = match.iloc[0]['tags']
            tags_s_dict = ast.literal_eval(tags_s_str)
            max_s = max(tags_s_dict.values()) if tags_s_dict else 1.0
            
            # Strict for mechanical anchors and vetoes (avoid ghosting)
            seed_tags_strict = {t for t, v in tags_s_dict.items() if v / max_s > 0.35}
            seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_strict for t in tags)}
            
            # Broader for "Soul" markers (Comedy, Surreal, etc.)
            seed_tags_soul = {t for t, v in tags_s_dict.items() if v / max_s > 0.15}
            seed_tags = seed_tags_soul # Kernel uses this for Identity Intersection Rescue
            
            active_narrative = [t for t in NARRATIVE_TAGS if t in seed_tags_soul]
            
            # 3. TITLE HIJACK DETECTION
            def get_keywords(name):
                kws = set(re.findall(r'\b\w{4,}\b', name.lower()))
                if 'frog' in name.lower(): kws.add('frog')
                return kws
            
            kw_s = get_keywords(seed_name)
            STOPWORDS = {'edition', 'game', 'decade', 'remaster', 'deluxe', 'pack', 'collection', 'complete', 'director', 'remake', 'remastered'}
            kw_s = kw_s - STOPWORDS
            
            hijack_mask = np.zeros(len(df), dtype=bool)
            if kw_s:
                name_series = df['name'].str.lower()
                for kw in kw_s:
                    pattern = rf'\b{re.escape(kw)}'
                    hijack_mask |= name_series.str.contains(pattern, regex=True).values

            if hijack_mask.any():
                suspect_indices = np.where(hijack_mask)[0]
                s_set_all = set(tags_s_dict.keys())
                for idx in suspect_indices:
                    t_tags_str = df.iloc[idx]['tags']
                    t_tags_set = set(re.findall(r"'([^']+)':", t_tags_str))
                    jaccard = len(s_set_all & t_tags_set) / len(s_set_all | t_tags_set) if (s_set_all or t_tags_set) else 0.0
                    if jaccard > 0.20:
                        hijack_mask[idx] = False

            # 4. KERNEL CALCULATION
            total_sim, components = calculate_jackalope_kernel(
                verb_profiles=verb_profiles,
                seed_verb_profile=verb_profiles[seed_idx],
                sem_vectors=sem_vectors, sem_norms=sem_norms,
                seed_sem_vec=sem_vectors[seed_idx], seed_sem_norm=sem_norms[seed_idx],
                topic_distributions=topic_distributions, seed_topic_dist=topic_distributions[seed_idx],
                topic_means=topic_means, topic_stds=topic_stds,
                tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
                sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
                mature_content_flags=df['mature_content'].values > 0,
                seed_mature_content=bool(df.iloc[seed_idx]['mature_content'] > 0),
                seed_migs=seed_migs,
                seed_tags=seed_tags,
                candidate_anchor_masks=anchor_masks,
                active_narrative_seed=active_narrative,
                precalculated_masks={'title_hijack': hijack_mask},
                difficulty_z=df['difficulty_z'].values,
                seed_difficulty_z=df.iloc[seed_idx]['difficulty_z'],
                tone_z=df['tone_z'].values if 'tone_z' in df.columns else None,
                seed_tone_z=df.iloc[seed_idx]['tone_z'] if 'tone_z' in df.columns else None,
                return_components=True
            )
            
            # 5. HIGH-VALUE NOUN RESCUE (Research Boost)
            HIGH_VALUE_NOUNS = {'Education', 'Math', 'Comedy', 'Surreal', 'Typing', 'Spelling', 'Mystery', 'Word Game'}
            shared_hv_nouns = set(tags_s_dict.keys()) & HIGH_VALUE_NOUNS
            if shared_hv_nouns:
                shared_counts = np.zeros(len(df), dtype=int)
                for noun in shared_hv_nouns:
                    if noun in anchor_masks:
                        shared_counts += anchor_masks[noun].astype(int)
                total_sim += np.where(total_sim > 0.02, 0.10 * shared_counts, 0.0)

            results = pd.DataFrame({
                'appid': df['appid'],
                'name': df['name'],
                'score': total_sim,
                'vibe': components['combined'],
                'mech': components['vibe'] * 0.1,
                'diff': components['difficulty'],
                'tone_sim': components['tone'],
                'tone_z': df['tone_z'].values if 'tone_z' in df.columns else 0.0
            })
            
            # Diagnostic for Algebra Ridge
            diag_a = results[results['appid'] == 1379510]
            if not diag_a.empty:
                row = diag_a.iloc[0]
                print(f"DIAG: Algebra Ridge Score={row['score']:.4f} (Vibe={row['vibe']:.4f}, Mech={row['mech']:.4f}, ToneSim={row['tone_sim']:.4f}, ToneZ={row['tone_z']:.4f})")

            results = results[results['appid'] != seed_appid].sort_values(by='score', ascending=False).head(20)

            print(f"\nTop 20 Recommendations for '{seed_name}':")
            print("-" * 130)
            print(f"{'Rank':<5} | {'AppID':<10} | {'Score':<8} | {'Vibe':<8} | {'Mech':<8} | {'Diff':<8} | {'ToneS':<8} | {'ToneZ':<8} | {'Name'}")
            print("-" * 130)
            for i, (idx, row) in enumerate(results.iterrows(), 1):
                print(f"{i:<5} | {row['appid']:<10} | {row['score']:.4f} | {row['vibe']:.4f} | {row['mech']:.4f} | {row['diff']:.4f} | {row['tone_sim']:.4f} | {row['tone_z']:.4f} | {row['name']}")

        except Exception as e:
            print(f"Error during calculation: {e}")
            import traceback
            traceback.print_exc()
            print("The tool is still alive. Try another ID.")

if __name__ == "__main__":
    main()

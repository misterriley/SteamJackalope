
import pandas as pd
import numpy as np
import os
import json
import sys
import re
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from common.constants import (
    PRODUCTION_DATA_DIR, METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA, SEMANTIC_GLOBAL_SCALING_FACTOR,
    SEMANTIC_DOT_PRODUCT_LAMBDA, TOPIC_GLOBAL_SCALING_FACTOR, TOPIC_DOT_PRODUCT_LAMBDA,
    SOFTMIN_TEMPERATURE
)
from common.utils import calculate_jackalope_kernel, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS, extract_seed_metadata, calculate_title_hijack_mask

def analyze_ground_truth():
    print("Loading data artifacts...")
    metadata = pd.read_parquet(METADATA_FILE)
    verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    
    means_path = os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")
    stds_path = os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")
    topic_means = np.load(means_path).astype(np.float32) if os.path.exists(means_path) else None
    topic_stds = np.load(stds_path).astype(np.float32) if os.path.exists(stds_path) else None
    
    tone_path = os.path.join(PRODUCTION_DATA_DIR, "tone_z.npy")
    tone_z = np.load(tone_path, mmap_mode='r') if os.path.exists(tone_path) else None
    
    # Pre-calculate AppID to Index mapping
    appid_to_idx = {int(appid): i for i, appid in enumerate(metadata['appid'])}
    
    # Pre-calculate Anchor Masks
    print("Calculating anchor masks...")
    anchor_masks = {}
    all_anchor_tags = set()
    for tags in MIGS.values(): all_anchor_tags.update(tags)
    all_anchor_tags.update(NARRATIVE_TAGS)
    all_anchor_tags.update(HORROR_MARKERS)
    all_anchor_tags.update(HARD_ANCHORS)
    all_anchor_tags.add("Isometric")
    all_anchor_tags.add("CRPG")
    all_anchor_tags.add("Cinematic")
    
    tag_series = metadata['tags'].fillna('').astype(str)
    for tag in all_anchor_tags:
        pattern = rf"'{re.escape(tag)}':"
        anchor_masks[tag] = tag_series.str.contains(pattern, regex=True).values

    # Load Ground Truth
    gt_path = os.path.join(ROOT_DIR, "data", "kernel_ground_truth.json")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    results = {
        "successfully_sorted": [],
        "not_successfully_sorted": []
    }

    print("\nEvaluating Ground Truth Pairs...")
    print("-" * 80)

    # SUCCESS THRESHOLD: 
    # For Positive: Score > 0.1 (Some structural similarity exists)
    # For Negative: Score < 0.05 (Structural similarity is effectively 0)
    POS_THRESHOLD = 0.1
    NEG_THRESHOLD = 0.05

    for seed_appid_str, labels in gt_data.items():
        seed_appid = int(seed_appid_str)
        if seed_appid not in appid_to_idx:
            print(f"Skipping unknown seed: {seed_appid}")
            continue
            
        seed_idx = appid_to_idx[seed_appid]
        seed_name = metadata.iloc[seed_idx]['name']
        
        # Extract seed metadata
        seed_meta = extract_seed_metadata([seed_idx], metadata)
        
        # Title Hijack Mask for this seed
        title_hijack_mask = calculate_title_hijack_mask([seed_name], metadata)

        # Run kernel for entire library relative to this seed
        scores, components = calculate_jackalope_kernel(
            verb_profiles=verb_profiles,
            seed_verb_profile=verb_profiles[seed_idx],
            sem_vectors=sem_vectors,
            sem_norms=sem_norms,
            seed_sem_vec=sem_vectors[seed_idx],
            seed_sem_norm=sem_norms[seed_idx],
            topic_distributions=topic_distributions,
            seed_topic_dist=topic_distributions[seed_idx],
            topic_means=topic_means,
            topic_stds=topic_stds,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
            dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR,
            sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            mature_content_flags=metadata['mature_content'].values > 0,
            seed_mature_content=seed_meta['mature_flags'][0],
            seed_migs=seed_meta['migs_list'][0],
            seed_tags=seed_meta['all_soul_tags'],
            candidate_anchor_masks=anchor_masks,
            active_narrative_seed=seed_meta['active_narrative'],
            is_cinematic_seed=seed_meta['is_cinematic'],
            precalculated_masks={"title_hijack": title_hijack_mask},
            difficulty_z=metadata['difficulty_z'].values,
            seed_difficulty_z=metadata.iloc[seed_idx]['difficulty_z'],
            tone_z=tone_z if tone_z is not None else metadata['tone_z'].values,
            seed_tone_z=tone_z[seed_idx] if tone_z is not None else metadata.iloc[seed_idx]['tone_z'],
            temperature=0.05,
            return_components=True
        )

        def format_components(idx):
            c = {k: float(v[idx]) if isinstance(v, np.ndarray) else float(v) for k, v in components.items()}
            return f"ID:{c['identity']:.2f} Mech:{c['mechanical']:.2f} Vibe:{c['vibe']:.2f} Tone:{c['tone']:.2f} Theme:{c['theme']:.2f}"

        # Evaluate Positive Labels
        for target_appid in labels.get("positive", []):
            if target_appid not in appid_to_idx: continue
            target_idx = appid_to_idx[target_appid]
            target_name = metadata.iloc[target_idx]['name']
            score = float(scores[target_idx])
            
            entry = {
                "seed": seed_name,
                "target": target_name,
                "type": "positive",
                "score": score,
                "status": "PASS" if score >= POS_THRESHOLD else "FAIL",
                "breakdown": format_components(target_idx)
            }
            
            if entry["status"] == "PASS":
                results["successfully_sorted"].append(entry)
            else:
                results["not_successfully_sorted"].append(entry)

        # Evaluate Negative Labels
        for target_appid in labels.get("negative", []):
            if target_appid not in appid_to_idx: continue
            target_idx = appid_to_idx[target_appid]
            target_name = metadata.iloc[target_idx]['name']
            score = float(scores[target_idx])
            
            entry = {
                "seed": seed_name,
                "target": target_name,
                "type": "negative",
                "score": score,
                "status": "PASS" if score <= NEG_THRESHOLD else "FAIL",
                "breakdown": format_components(target_idx)
            }
            
            if entry["status"] == "PASS":
                results["successfully_sorted"].append(entry)
            else:
                results["not_successfully_sorted"].append(entry)

    # Summary Report
    print("\n" + "=" * 80)
    print("JACKALOPE KERNEL GROUND TRUTH ANALYSIS")
    print("=" * 80)
    
    print(f"\n✅ SUCCESSFULLY SORTED ({len(results['successfully_sorted'])} total):")
    for r in sorted(results["successfully_sorted"], key=lambda x: (x['seed'], x['type'])):
        print(f"  [{r['type'].upper()}] {r['seed']} -> {r['target']}: {r['score']:.4f} ({r['status']}) | {r['breakdown']}")

    print(f"\n❌ NOT SUCCESSFULLY SORTED ({len(results['not_successfully_sorted'])} total):")
    for r in sorted(results["not_successfully_sorted"], key=lambda x: (x['seed'], x['type'])):
        print(f"  [{r['type'].upper()}] {r['seed']} -> {r['target']}: {r['score']:.4f} ({r['status']}) | {r['breakdown']}")

    total = len(results["successfully_sorted"]) + len(results["not_successfully_sorted"])
    accuracy = len(results["successfully_sorted"]) / total if total > 0 else 0
    print(f"\nOverall Kernel G.T. Accuracy: {accuracy:.1%}")

if __name__ == "__main__":
    analyze_ground_truth()

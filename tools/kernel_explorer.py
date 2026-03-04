import streamlit as st
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
from common.utils import calculate_jackalope_kernel, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS

# --- Page Config ---
st.set_page_config(page_title="Jackalope Kernel Explorer", layout="wide", page_icon="🫎")

# --- Ground Truth Storage ---
GROUND_TRUTH_FILE = os.path.join(ROOT_DIR, "data", "kernel_ground_truth.json")

def load_ground_truth():
    if os.path.exists(GROUND_TRUTH_FILE):
        try:
            with open(GROUND_TRUTH_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_ground_truth(data):
    os.makedirs(os.path.dirname(GROUND_TRUTH_FILE), exist_ok=True)
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Data Loading ---
@st.cache_resource
def load_data():
    st.info("Loading data artifacts...")
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
    
    quality_grid_path = os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy")
    quality_grid = np.load(quality_grid_path, mmap_mode='r') if os.path.exists(quality_grid_path) else None

    profile_path = os.path.join(ROOT_DIR, "data", "user_76561198039155404_taste_profile.json")
    pred_ratings_path = os.path.join(ROOT_DIR, "data", "user_76561198039155404_predicted_ratings.npy")
    user_profile = None
    predicted_ratings = None
    
    if os.path.exists(profile_path):
        with open(profile_path, 'r') as f:
            user_profile = json.load(f)
    if os.path.exists(pred_ratings_path):
        try:
            predicted_ratings = np.load(pred_ratings_path)
        except:
            predicted_ratings = None
            
    # Pre-calculate AppID to Index mapping
    appid_to_idx = {int(appid): i for i, appid in enumerate(metadata['appid'])}
    
    # Pre-calculate Anchor Masks (MIGs, Narrative, etc.)
    st.info("Calculating anchor masks...")
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
        
    return metadata, verb_profiles, sem_vectors, sem_norms, topic_distributions, topic_means, topic_stds, tone_z, quality_grid, user_profile, predicted_ratings, appid_to_idx, anchor_masks

metadata, verb_profiles, sem_vectors, sem_norms, topic_distributions, topic_means, topic_stds, tone_z, quality_grid, user_profile, predicted_ratings, appid_to_idx, anchor_masks = load_data()

# --- Helpers ---
def get_game_info(appid):
    if int(appid) not in appid_to_idx:
        return None
    idx = appid_to_idx[int(appid)]
    return metadata.iloc[idx]

def get_steam_img(appid):
    return f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"

def display_game_card(game, column, key_suffix, is_seed=False, is_gt=False, score=None):
    with column:
        st.image(get_steam_img(game['appid']), use_container_width=True)
        st.markdown(f"**[{game['name']}](https://store.steampowered.com/app/{game['appid']})**")
        st.caption(f"AppID: {game['appid']}")
        if score is not None:
            st.caption(f"Kernel Similarity: {score:.4f}")
        
        if not is_seed:
            cols = st.columns(2)
            if cols[0].button("👍 Pos", key=f"pos_{game['appid']}_{key_suffix}"):
                add_to_gt(st.session_state.seed_appid, game['appid'], "positive")
            if cols[1].button("👎 Neg", key=f"neg_{game['appid']}_{key_suffix}"):
                add_to_gt(st.session_state.seed_appid, game['appid'], "negative")
            if is_gt:
                if st.button("❌ Remove", key=f"rem_{game['appid']}_{key_suffix}"):
                    remove_from_gt(st.session_state.seed_appid, game['appid'])

def add_to_gt(seed_appid, target_appid, label):
    gt = load_ground_truth()
    seed_s = str(seed_appid)
    if seed_s not in gt:
        gt[seed_s] = {"positive": [], "negative": []}
    
    # Remove from other list if present
    other = "negative" if label == "positive" else "positive"
    if target_appid in gt[seed_s][other]:
        gt[seed_s][other].remove(target_appid)
    
    if target_appid not in gt[seed_s][label]:
        gt[seed_s][label].append(int(target_appid))
    
    save_ground_truth(gt)
    st.rerun()

def remove_from_gt(seed_appid, target_appid):
    gt = load_ground_truth()
    seed_s = str(seed_appid)
    if seed_s in gt:
        if target_appid in gt[seed_s]["positive"]:
            gt[seed_s]["positive"].remove(target_appid)
        if target_appid in gt[seed_s]["negative"]:
            gt[seed_s]["negative"].remove(target_appid)
        save_ground_truth(gt)
        st.rerun()

# --- App Layout ---
st.title("🫎 Jackalope Kernel Explorer")
st.markdown("Diagnostic tool for examining the Mechanical Identity Group (MIG) similarity kernel.")

with st.sidebar:
    st.header("Search")
    search_appid = st.text_input("Enter Seed AppID", value=st.session_state.get("seed_appid", "1091500")) # Default: Cyberpunk 2077
    if search_appid:
        try:
            st.session_state.seed_appid = int(search_appid)
        except ValueError:
            st.error("Please enter a valid numeric AppID")
            
    st.divider()
    st.header("Quick Exemplars (9+ Ratings)")
    
    # Pre-calculated list of favorites based on your Taste DNA
    exemplars = [
        {'appid': 427520, 'name': 'Factorio'},
        {'appid': 632470, 'name': 'Disco Elysium'},
        {'appid': 1222140, 'name': 'Detroit: Become Human'},
        {'appid': 3240220, 'name': 'GTA V Enhanced'},
        {'appid': 1328670, 'name': 'Mass Effect Legendary Edition'},
        {'appid': 489830, 'name': 'Skyrim SE'},
        {'appid': 524220, 'name': 'NieR:Automata'},
        {'appid': 620, 'name': 'Portal 2'},
        {'appid': 282140, 'name': 'SOMA'},
        {'appid': 292030, 'name': 'The Witcher 3'},
        {'appid': 264710, 'name': 'Subnautica'},
        {'appid': 1082430, 'name': 'Before Your Eyes'},
        {'appid': 57300, 'name': 'Amnesia: TDD'},
        {'appid': 219890, 'name': 'Antichamber'},
        {'appid': 1593500, 'name': 'God of War'},
        {'appid': 304430, 'name': 'INSIDE'},
        {'appid': 92800, 'name': 'SpaceChem'},
        {'appid': 1687950, 'name': 'Persona 5 Royal'},
        {'appid': 261570, 'name': 'Ori & The Blind Forest'},
        {'appid': 1147550, 'name': 'Not For Broadcast'},
        {'appid': 1332010, 'name': 'Stray'},
        {'appid': 835960, 'name': 'The Talos Principle 2'},
        {'appid': 874260, 'name': 'The Forgotten City'},
        {'appid': 210970, 'name': 'The Witness'},
        {'appid': 1794680, 'name': 'Vampire Survivors'}
    ]
    
    for ex in exemplars:
        if st.button(f"{ex['name']}", key=f"ex_{ex['appid']}", use_container_width=True):
            st.session_state.seed_appid = ex['appid']
            st.rerun()

    st.divider()
    st.header("Kernel Settings")
    temperature = st.slider("Temperature (Veto Sharpness)", 0.001, 0.5, 0.05, step=0.001)
    top_k = st.slider("Show Top K", 10, 100, 20)
    
    st.divider()
    st.header("Quick Add Ground Truth")
    qa_appid = st.text_input("Enter AppID to Label", key="qa_appid")
    if qa_appid:
        try:
            qa_id = int(qa_appid)
            game_to_add = get_game_info(qa_id)
            if game_to_add is not None:
                st.image(get_steam_img(qa_id), use_container_width=True)
                st.write(f"**{game_to_add['name']}**")
                qa_cols = st.columns(2)
                if qa_cols[0].button("👍 Add Pos", key="qa_pos"):
                    add_to_gt(st.session_state.seed_appid, qa_id, "positive")
                if qa_cols[1].button("👎 Add Neg", key="qa_neg"):
                    add_to_gt(st.session_state.seed_appid, qa_id, "negative")
            else:
                st.error("AppID not found")
        except ValueError:
            st.error("Invalid AppID")

    st.divider()
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

if "seed_appid" in st.session_state:
    seed_info = get_game_info(st.session_state.seed_appid)
    if seed_info is not None:
        seed_idx = appid_to_idx[st.session_state.seed_appid]
        
        # Display Seed Info
        st.header(f"Seed: {seed_info['name']}")
        col_img, col_info = st.columns([1, 3])
        with col_img:
            st.image(get_steam_img(seed_info['appid']), use_container_width=True)
        with col_info:
            st.markdown(f"**AppID:** {seed_info['appid']}")
            st.markdown(f"**Steam:** [View on Store](https://store.steampowered.com/app/{seed_info['appid']})")
            st.markdown(f"**Genres:** {seed_info['genres']}")
            # Extract tags for metadata
            tags_str = seed_info['tags']
            tags_dict = {}
            if isinstance(tags_str, str) and tags_str.startswith('{'):
                try:
                    # Simple regex to get tags since they are in dict format
                    tag_keys = re.findall(r"'([^']+)':", tags_str)
                    tags_dict = {t: 1 for t in tag_keys}
                except:
                    pass
            st.markdown(f"**Tags:** {', '.join(list(tags_dict.keys())[:20])}...")
            
        st.divider()
        
        # Load Ground Truth
        gt = load_ground_truth()
        seed_s = str(st.session_state.seed_appid)
        seed_gt = gt.get(seed_s, {"positive": [], "negative": []})
        
        # Display Ground Truth
        st.subheader("🎯 Ground Truth (Labels)")
        gt_pos_col, gt_neg_col = st.columns(2)
        
        with gt_pos_col:
            st.write("✅ **Positive Examples** (Similar)")
            if not seed_gt["positive"]:
                st.info("No positive examples yet.")
            else:
                pos_cols = st.columns(3)
                for i, appid in enumerate(seed_gt["positive"]):
                    game = get_game_info(appid)
                    if game is not None:
                        display_game_card(game, pos_cols[i % 3], f"gt_pos_{i}", is_gt=True)

        with gt_neg_col:
            st.write("❌ **Negative Examples** (Dissimilar)")
            if not seed_gt["negative"]:
                st.info("No negative examples yet.")
            else:
                neg_cols = st.columns(3)
                for i, appid in enumerate(seed_gt["negative"]):
                    game = get_game_info(appid)
                    if game is not None:
                        display_game_card(game, neg_cols[i % 3], f"gt_neg_{i}", is_gt=True)
        
        st.divider()
        
        # Calculate Similarities
        st.subheader(f"🔍 Most Similar Games to '{seed_info['name']}'")
        
        with st.spinner("Calculating kernel similarities..."):
            # Prepare seed data
            seed_verb = verb_profiles[seed_idx]
            seed_sem = sem_vectors[seed_idx]
            seed_sem_norm = sem_norms[seed_idx]
            seed_topic = topic_distributions[seed_idx]
            seed_diff_z = seed_info.get('difficulty_z', 0.0)
            seed_tone_z = seed_info.get('tone_z', 0.0)
            
            # Extract MIGs and Tags for seed
            seed_tags_set = set(tags_dict.keys())
            active_seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_set for t in tags)}
            seed_tags_hard = seed_tags_set & HARD_ANCHORS
            active_narrative_seed = [t for t in NARRATIVE_TAGS if t in seed_tags_set]
            is_cinematic_seed = "Cinematic" in seed_tags_set
            
            # Title Hijack Detection
            title_hijack_mask = np.zeros(len(metadata), dtype=bool)
            keywords = [k for k in re.split(r'[^a-zA-Z0-9]', seed_info['name'].lower()) if len(k) > 3]
            if keywords:
                pattern = '|'.join(keywords)
                title_match = metadata['name'].str.lower().str.contains(pattern, regex=True).values
                title_hijack_mask |= title_match
            
            # Calculate kernel
            scores = calculate_jackalope_kernel(
                verb_profiles=verb_profiles,
                seed_verb_profile=seed_verb,
                sem_vectors=sem_vectors,
                sem_norms=sem_norms,
                seed_sem_vec=seed_sem,
                seed_sem_norm=seed_sem_norm,
                topic_distributions=topic_distributions,
                seed_topic_dist=seed_topic,
                topic_means=topic_means,
                topic_stds=topic_stds,
                tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
                dot_product_lambda=DOT_PRODUCT_LAMBDA,
                sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR,
                sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
                topic_scaling_factor=TOPIC_GLOBAL_SCALING_FACTOR,
                mature_content_flags=metadata['mature_content'].values > 0,
                seed_mature_content=bool(seed_info['mature_content'] > 0),
                seed_migs=active_seed_migs,
                seed_tags=seed_tags_hard,
                candidate_anchor_masks=anchor_masks,
                active_narrative_seed=active_narrative_seed,
                is_cinematic_seed=is_cinematic_seed,
                precalculated_masks={"title_hijack": title_hijack_mask},
                difficulty_z=metadata['difficulty_z'].values if 'difficulty_z' in metadata else None,
                seed_difficulty_z=seed_diff_z,
                tone_z=tone_z,
                seed_tone_z=seed_tone_z,
                temperature=temperature
            )
            
            if user_profile is not None and quality_grid is not None and predicted_ratings is not None:
                st.info("Loading pre-calculated Predicted Ratings from out-of-sample NW Kernel Regression...")
                
                # Exclude seed
                scores[seed_idx] = -1.0
                
                # We still need to sort by something. Do we sort by kernel score to find the TOP 1000 structurally similar, 
                # then display their predicted ratings? Yes, that is the intention.
                valid_candidates = scores > 0.05
                scores[~valid_candidates] = -1.0
                top_1000_indices = np.argsort(scores)[::-1][:1000]
                
                pred_scores = []
                for idx in top_1000_indices:
                    if scores[idx] < 0:
                        continue # Skip invalid candidates
                    # Look up exact rating generated by the cross-validation pipeline out-of-sample prediction
                    pred = predicted_ratings[idx]
                    pred_scores.append((idx, np.clip(pred, 0, 10)))
                
                # Sort by predicted rating descending
                pred_scores.sort(key=lambda x: x[1], reverse=True)
                display_indices = [x[0] for x in pred_scores[:top_k]]
                display_scores = {x[0]: x[1] for x in pred_scores}
                score_label = "Predicted Rating"
            else:
                st.warning("User Taste Profile or Predicted Ratings not found. Showing raw kernel scores.")
                scores[seed_idx] = -1.0
                top_indices = np.argsort(scores)[::-1][:top_k]
                display_indices = top_indices
                display_scores = {idx: scores[idx] for idx in display_indices}
                score_label = "Kernel Similarity"
            
            # Display results in a grid
            res_cols = st.columns(5)
            for i, idx in enumerate(display_indices):
                game = metadata.iloc[idx]
                score_val = display_scores[idx]
                
                with res_cols[i % 5]:
                    st.image(get_steam_img(game['appid']), use_container_width=True)
                    st.markdown(f"**[{game['name']}](https://store.steampowered.com/app/{game['appid']})**")
                    st.caption(f"AppID: {game['appid']}")
                    st.caption(f"{score_label}: {score_val:.4f}")
                    
                    cols = st.columns(2)
                    if cols[0].button("👍 Pos", key=f"res_pos_{game['appid']}"):
                        add_to_gt(st.session_state.seed_appid, game['appid'], "positive")
                    if cols[1].button("👎 Neg", key=f"res_neg_{game['appid']}"):
                        add_to_gt(st.session_state.seed_appid, game['appid'], "negative")

    else:
        st.error(f"AppID {st.session_state.seed_appid} not found in metadata.")
else:
    st.info("Enter an AppID in the sidebar to begin.")

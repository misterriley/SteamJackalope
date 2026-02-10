
import streamlit as st
import pandas as pd
import requests
import os
import sys
import ast
import random
import logging
from lists import render_lists_page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    ABG_NOTCHES_ON_SLIDER,
    REMOVE_NSFW_DEFAULT,
    SEMANTIC_WEIGHT_MULTIPLIER,
    TAG_WEIGHT_MULTIPLIER,
    QUALITY_WEIGHT_MULTIPLIER,
    AGE_WEIGHT_MULTIPLIER,
    POPULARITY_WEIGHT_MULTIPLIER,
    LENGTH_WEIGHT_MULTIPLIER,
    DIFFICULTY_WEIGHT_MULTIPLIER,
    AP_SLIDER_VALUES,
    AP_SLIDER_MIN,
    AP_SLIDER_MAX,
    AP_SLIDER_STEP,
    DISCOVERY_LABEL_LEFT,
    DISCOVERY_LABEL_RIGHT,
    
    APP_TITLE,
    APP_HEADER,
    METHODOLOGY_TAB,
    RECOMMENDER_TAB,
    METHODOLOGY_ERROR,
    SIDEBAR_HEADER,
    SEMANTIC_WEIGHT_LABEL,
    TAG_WEIGHT_LABEL,
    QUALITY_PREF_LABEL,
    QUALITY_SLIDER_LABEL,
    QUALITY_LOVED_LABEL,
    QUALITY_HATED_LABEL,
    AGE_PREF_LABEL,
    AGE_OLD_LABEL,
    AGE_NEW_LABEL,
    POP_PREF_LABEL,
    POP_SLIDER_LABEL,
    POP_NICHE_LABEL,
    POP_MAINSTREAM_LABEL,
    DISC_PREF_LABEL,
    DISC_SLIDER_LABEL,
    LENGTH_PREF_LABEL,
    LENGTH_SLIDER_LABEL,
    LENGTH_SHORT_LABEL,
    LENGTH_LONG_LABEL,
    DIFFICULTY_PREF_LABEL,
    DIFFICULTY_SLIDER_LABEL,
    DIFFICULTY_EASY_LABEL,
    DIFFICULTY_HARD_LABEL,
    REMOVE_VR_LABEL,
    DEBUG_LENGTH_LABEL,
    DEBUG_DIFFICULTY_LABEL,
    DEBUG_TOTAL_SUM_LABEL,
    DEBUG_INFO_HEADER,
    DEBUG_SEMANTIC_LABEL,
    DEBUG_TAG_LABEL,
    DEBUG_QUALITY_LABEL,
    DEBUG_AGE_LABEL,
    DEBUG_POP_LABEL,
    RESET_BUTTON_LABEL,
    RANDOM_BUTTON_LABEL,
    UPDATING_RESULTS_TEXT,
    TOP_RECOMMENDATIONS_HEADER,
    APPID_LABEL,
    VIEW_ON_STEAM_LINK,
    RELEASE_DATE_LABEL,
    RELEASE_DATE_UNKNOWN_TEXT,
    ESTIMATED_LENGTH_LABEL,
    DIFFICULTY_SCORE_LABEL,
    QUALITY_SCORE_LABEL,
    SEMANTIC_SIMILARITY_LABEL,
    TAG_SIMILARITY_LABEL,
    GENRES_LABEL,
    TAGS_LABEL,
    USE_SEED_BUTTON,
    DATA_SOURCE_CAPTION,
    ERROR_LOADING_DATA,
    SEMANTIC_WEIGHT_HELP,
    TAG_WEIGHT_HELP,
    QUALITY_PREF_HELP,
    AGE_PREF_HELP,
    POP_PREF_HELP,
    DISC_PREF_HELP,
    LENGTH_PREF_HELP,
    DIFFICULTY_PREF_HELP,
    REMOVE_VR_HELP,
    ENGLISH_ONLY_LABEL,
    ENGLISH_ONLY_HELP,
    REMOVE_NSFW_LABEL,
    REMOVE_NSFW_HELP,
    REMOVE_UTILITIES_LABEL,
    REMOVE_UTILITIES_HELP,
    REMOVE_UNRELEASED_LABEL,
    REMOVE_UNRELEASED_HELP,
    GENRE_FILTER_LABEL,
    GENRE_FILTER_HELP,
    DEBUG_MODE_LABEL,
    DEBUG_MODE_HELP,
    TOP_K_LABEL,
    TOP_K_HELP,
    PROMPT_LABEL,
    PROMPT_PLACEHOLDER,
    PROMPT_HELP,
    SEED_LABEL,
    SEED_HELP,
    APP_IMAGE_TEXT,

    REMOVE_VR_DEFAULT,
    REMOVE_NSFW_DEFAULT,
    REMOVE_NON_ENGLISH_DEFAULT,
    REMOVE_UTILITIES_DEFAULT,
    REMOVE_UNRELEASED_DEFAULT,
    DEBUG_MODE_DEFAULT,

    TOP_K_DEFAULT,
    TOP_K_MAX,
    BACKEND_URL
)

# --- Configuration & Data Loading ---
st.set_page_config(page_title=APP_TITLE.replace("🎮 ", ""), layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def get_game_list():
    try:
        response = requests.get(f"{BACKEND_URL}/games", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_genre_list():
    try:
        response = requests.get(f"{BACKEND_URL}/genres", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

# --- UI Layout ---
st.title(APP_TITLE)

def add_seed(game_name):
    if "seed_multiselect" not in st.session_state:
        st.session_state.seed_multiselect = []
    
    current_seeds = list(st.session_state.seed_multiselect)
    if game_name not in current_seeds:
        current_seeds.append(game_name)
        st.session_state.seed_multiselect = current_seeds

def render_game_card(game, show_debug=False, alpha=0.0, beta=0.0, prompt="", selected_games=None, is_seed=False):
    score_display = f" (Score: {game['weighted_score']:.2f})" if 'weighted_score' in game else ""
    with st.expander(f"**{game['name']}**{score_display}"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(f"{APPID_LABEL} {game['appid']}")
            st.write(f"{VIEW_ON_STEAM_LINK}(https://store.steampowered.com/app/{game['appid']})")
            
            release_display = game['release_date']
            if not release_display or str(release_display).strip() == "" or str(release_display) == "NaT":
                release_display = RELEASE_DATE_UNKNOWN_TEXT
            st.write(f"{RELEASE_DATE_LABEL} {release_display}")

            if game.get('estimated_playtime'):
                hours = game['estimated_playtime'] / 60.0
                st.write(f"{ESTIMATED_LENGTH_LABEL} {hours:.1f} hours")
            
            if game.get('difficulty_predicted'):
                st.write(f"{DIFFICULTY_SCORE_LABEL} {game['difficulty_predicted']:.1f}")

            pos = int(game['positive'])
            neg = int(game['negative'])
            tot = pos + neg
            pct = (pos / tot * 100) if tot > 0 else 0
            
            st.write(f"{QUALITY_SCORE_LABEL} {game.get('rating', 0.0):.2f} ({pct:.1f}% of {tot:,} positive reviews)")
            
            if not is_seed:
                if prompt or selected_games:
                    st.write(f"{SEMANTIC_SIMILARITY_LABEL} {game.get('semantic_match', 0.0):.4f}")
                if selected_games:
                    st.write(f"{TAG_SIMILARITY_LABEL} {game.get('tag_match', 0.0):.4f}")
            
            if show_debug and not is_seed:
                st.divider()
                st.write(DEBUG_INFO_HEADER)
                
                st.write(f"{DEBUG_SEMANTIC_LABEL} {game['semantic_match']:.4f} (raw) → {game['z_semantic']:.3f} (z)")
                st.write(f"  - Weight: {alpha:.2f} * {SEMANTIC_WEIGHT_MULTIPLIER} = {game['w_semantic']:.2f}")
                st.write(f"  - Partial: {game['z_semantic']:.3f} * {game['w_semantic']:.2f} = **{game['z_semantic']*game['w_semantic']:.3f}**")
                
                st.write(f"{DEBUG_TAG_LABEL} {game['tag_match']:.4f} (raw) → {game['z_tag']:.3f} (z)")
                st.write(f"  - Weight: {beta:.2f} * {TAG_WEIGHT_MULTIPLIER} = {game['w_tag']:.2f}")
                st.write(f"  - Partial: {game['z_tag']:.3f} * {game['w_tag']:.2f} = **{game['z_tag']*game['w_tag']:.3f}**")
                
                st.write(f"{DEBUG_QUALITY_LABEL} {game['rating']:.3f} (raw z) → {game['z_spps']:.3f} (z)")
                st.write(f"  - Weight: {st.session_state.quality_pref * QUALITY_WEIGHT_MULTIPLIER:.2f} * {QUALITY_WEIGHT_MULTIPLIER} = {game['w_spps']:.2f}")
                st.write(f"  - Partial: {game['z_spps']:.3f} * {game['w_spps']:.2f} = **{game['z_spps']*game['w_spps']:.3f}**")
                
                st.write(f"{DEBUG_AGE_LABEL} {game['raw_date']:.0f} (year) → {game['z_date']:.3f} (z)")
                st.write(f"  - Weight: {st.session_state.age_pref:.2f} * {AGE_WEIGHT_MULTIPLIER} = {game['w_date']:.2f}")
                st.write(f"  - Partial: {game['z_date']:.3f} * {game['w_date']:.2f} = **{game['z_date']*game['w_date']:.3f}**")
                
                st.write(f"{DEBUG_POP_LABEL} {game['raw_pop']:,} (votes) → {game['z_pop']:.3f} (z)")
                st.write(f"  - Weight: {st.session_state.pop_pref:.2f} * {POPULARITY_WEIGHT_MULTIPLIER} = {game['w_pop']:.2f}")
                st.write(f"  - Partial: {game['z_pop']:.3f} * {game['w_pop']:.2f} = **{game['z_pop']*game['w_pop']:.3f}**")
                
                st.write(f"{DEBUG_LENGTH_LABEL} {game['raw_length']:.1f} (hrs) → {game['z_length']:.3f} (z)")
                st.write(f"  - Weight: {st.session_state.length_pref:.2f} * {LENGTH_WEIGHT_MULTIPLIER} = {game['w_length']:.2f}")
                st.write(f"  - Partial: {game['z_length']:.3f} * {game['w_length']:.2f} = **{game['z_length']*game['w_length']:.3f}**")

                st.write(f"{DEBUG_DIFFICULTY_LABEL} {game['raw_difficulty']:.1f} (1-5) → {game['z_difficulty']:.3f} (z)")
                st.write(f"  - Weight: {st.session_state.difficulty_pref:.2f} * {DIFFICULTY_WEIGHT_MULTIPLIER} = {game['w_difficulty']:.2f}")
                st.write(f"  - Partial: {game['z_difficulty']:.3f} * {game['w_difficulty']:.2f} = **{game['z_difficulty']*game['w_difficulty']:.3f}**")

                st.write(f"{DEBUG_TOTAL_SUM_LABEL} {game['weighted_score']:.3f}")

        with c2:
            genres = game['genres']
            if isinstance(genres, str):
                try:
                    genres = ast.literal_eval(genres)
                except:
                    pass
            if isinstance(genres, list):
                st.write(f"{GENRES_LABEL} {', '.join(genres)}")
            else:
                st.write(f"{GENRES_LABEL} {genres}")

            tags = game['tags']
            if isinstance(tags, str):
                try:
                    tags = ast.literal_eval(tags)
                except:
                    pass
            
            if isinstance(tags, dict):
                tag_df = pd.DataFrame(list(tags.items()), columns=['Tag', 'Count']).sort_values('Count', ascending=False)
                st.write(TAGS_LABEL)
                st.dataframe(tag_df, hide_index=True, width='stretch')
            else:
                st.write(f"{TAGS_LABEL} {tags}")
        
        if not is_seed:
            st.button(USE_SEED_BUTTON.format(game_name=game['name']), key=f"btn_{game['appid']}", on_click=add_seed, args=(game['name'],))

def reset_all_parameters():
    logger.info("Resetting all parameters to defaults")
    st.session_state.alpha = SEMANTIC_WEIGHT_MULTIPLIER / 2
    st.session_state.beta = TAG_WEIGHT_MULTIPLIER / 2
    st.session_state.quality_pref = 0.5
    st.session_state.age_pref = 0.0
    st.session_state.pop_pref = 0.0
    st.session_state.disc_pref = 0.0
    st.session_state.length_pref = 0.0
    st.session_state.difficulty_pref = 0.0
    st.session_state.remove_vr = REMOVE_VR_DEFAULT
    st.session_state.english_only = REMOVE_NON_ENGLISH_DEFAULT
    st.session_state.remove_nsfw = REMOVE_NSFW_DEFAULT
    st.session_state.remove_utilities = REMOVE_UTILITIES_DEFAULT
    st.session_state.remove_unreleased = REMOVE_UNRELEASED_DEFAULT
    st.session_state.debug_mode = DEBUG_MODE_DEFAULT
    st.session_state.top_k = TOP_K_DEFAULT
    st.session_state.prompt = ""
    st.session_state.seed_multiselect = []
    st.session_state.genres_multiselect = []
    logger.info(f"Parameters reset - alpha={st.session_state.alpha:.3f}, beta={st.session_state.beta:.3f}, quality_pref={st.session_state.quality_pref:.3f}")

def randomize_parameters():
    logger.info("Randomize button clicked")
    game_list = get_game_list()
    
    # Adjectives for random prompt
    adjectives = []
    adj_path = os.path.join(os.path.dirname(__file__), "..", "common", "common_adjectives.txt")
    if os.path.exists(adj_path):
        try:
            with open(adj_path, "r", encoding="utf-8") as f:
                adjectives = [line.strip() for line in f if line.strip()]
        except:
            pass
    
    if not adjectives:
        adjectives = ["exciting", "atmospheric", "challenging", "relaxing", "mysterious", "colorful"]

    st.session_state.alpha = random.uniform(0, SEMANTIC_WEIGHT_MULTIPLIER)
    st.session_state.beta = random.uniform(0, TAG_WEIGHT_MULTIPLIER)
    st.session_state.quality_pref = random.choice(AP_SLIDER_VALUES)
    st.session_state.age_pref = random.choice(AP_SLIDER_VALUES)
    st.session_state.pop_pref = random.choice(AP_SLIDER_VALUES)
    st.session_state.disc_pref = random.choice(AP_SLIDER_VALUES)
    st.session_state.length_pref = random.choice(AP_SLIDER_VALUES)
    st.session_state.difficulty_pref = random.choice(AP_SLIDER_VALUES)
    
    st.session_state.prompt = random.choice(adjectives)
    
    if game_list:
        st.session_state.seed_multiselect = [random.choice(game_list)]
    else:
        st.session_state.seed_multiselect = []
    
    logger.info(f"Randomized parameters: alpha={st.session_state.alpha:.3f}, beta={st.session_state.beta:.3f}, quality_pref={st.session_state.quality_pref:.3f}, "
                f"prompt='{st.session_state.prompt}', seed='{st.session_state.seed_multiselect}'")
    st.toast(f"Randomized! Try: {st.session_state.prompt}")

# Initialize session state defaults
if "alpha" not in st.session_state:
    reset_all_parameters()

# Methodology Loading
try:
    with open("methodology.md", "r", encoding="utf-8") as f:
        methodology_text = f.read()
except FileNotFoundError:
    methodology_text = METHODOLOGY_ERROR

tabs = st.tabs([RECOMMENDER_TAB, "Lists", METHODOLOGY_TAB])

with tabs[1]:
    render_lists_page()

with tabs[2]:
    parts = methodology_text.split("![")
    st.markdown(parts[0]) 
    
    for part in parts[1:]:
        label_end = part.find("](")
        path_end = part.find(")")
        
        label = part[:label_end]
        path = part[label_end+2:path_end]
        rest = part[path_end+1:]
        
        if os.path.exists(path):
            if "cosine" in path.lower():
                st.image(path, caption=label, width=400)
            else:
                st.image(path, caption=label)
        else:
            st.warning(f"Image not found: {path}")
            
        st.markdown(rest)

with tabs[0]:
    st.markdown(APP_HEADER)
    
    col_img, col_txt = st.columns([1, 4])
    with col_img:
        if os.path.exists("assets/jackalopeVR.jpg"):
            st.image("assets/jackalopeVR.jpg", use_container_width=True)
    with col_txt:
        st.write(APP_IMAGE_TEXT)

    with st.sidebar:
        st.header(SIDEBAR_HEADER)
        
        st.button(RESET_BUTTON_LABEL, on_click=reset_all_parameters, use_container_width=True)
        st.button(RANDOM_BUTTON_LABEL, on_click=randomize_parameters, use_container_width=True)
        st.divider()

        alpha = st.slider(SEMANTIC_WEIGHT_LABEL, 0.0, SEMANTIC_WEIGHT_MULTIPLIER, step=SEMANTIC_WEIGHT_MULTIPLIER/ABG_NOTCHES_ON_SLIDER, help=SEMANTIC_WEIGHT_HELP, key="alpha")
        logger.debug(f"Slider alpha changed: {alpha:.3f}")
        beta = st.slider(TAG_WEIGHT_LABEL, 0.0, TAG_WEIGHT_MULTIPLIER, step=TAG_WEIGHT_MULTIPLIER/ABG_NOTCHES_ON_SLIDER, help=TAG_WEIGHT_HELP, key="beta")
        logger.debug(f"Slider beta changed: {beta:.3f}")

        st.divider()

        st.write(QUALITY_PREF_LABEL)
        quality_pref = st.slider(
            QUALITY_SLIDER_LABEL,
            min_value=AP_SLIDER_MIN,
            max_value=AP_SLIDER_MAX,
            step=AP_SLIDER_STEP,
            label_visibility="collapsed",
            help=QUALITY_PREF_HELP,
            key="quality_pref"
        )
        logger.debug(f"Slider quality_pref changed: {quality_pref:.3f}")
        st.write(f"**Slider value: {quality_pref}**")
        col_hated, col_loved = st.columns(2)
        with col_hated: st.caption(QUALITY_HATED_LABEL)
        with col_loved: st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8rem;'>{QUALITY_LOVED_LABEL}</div>", unsafe_allow_html=True)

        st.write(AGE_PREF_LABEL)
        age_pref = st.slider(
            AGE_PREF_LABEL,
            min_value=AP_SLIDER_MIN,
            max_value=AP_SLIDER_MAX,
            step=AP_SLIDER_STEP,
            label_visibility="collapsed",
            help=AGE_PREF_HELP,
            key="age_pref"
        )
        logger.debug(f"Slider age_pref changed: {age_pref:.3f}")
        col_old, col_new = st.columns(2)
        with col_old: st.caption(AGE_OLD_LABEL)
        with col_new: st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8rem;'>{AGE_NEW_LABEL}</div>", unsafe_allow_html=True)

        st.write(POP_PREF_LABEL)
        pop_pref = st.slider(
            POP_SLIDER_LABEL,
            min_value=AP_SLIDER_MIN,
            max_value=AP_SLIDER_MAX,
            step=AP_SLIDER_STEP,
            label_visibility="collapsed",
            help=POP_PREF_HELP,
            key="pop_pref"
        )
        logger.debug(f"Slider pop_pref changed: {pop_pref:.3f}")
        col_niche, col_main = st.columns(2)
        with col_niche: st.caption(POP_NICHE_LABEL)
        with col_main: st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8rem;'>{POP_MAINSTREAM_LABEL}</div>", unsafe_allow_html=True)

        st.write(DISC_PREF_LABEL)
        disc_pref_raw = st.slider(
            DISC_SLIDER_LABEL,
            min_value=AP_SLIDER_MIN,
            max_value=AP_SLIDER_MAX,
            step=AP_SLIDER_STEP,
            label_visibility="collapsed",
            help=DISC_PREF_HELP,
            key="disc_pref"
        )
        disc_pref = -disc_pref_raw  # Negate so: left (-1) → +1 (Known Quantities), right (+1) → -1 (Wild Cards)
        logger.debug(f"Slider disc_pref changed: raw={disc_pref_raw:.3f}, final={disc_pref:.3f}")
        col_known, col_wild = st.columns(2)
        with col_known: st.caption(DISCOVERY_LABEL_LEFT)
        with col_wild: st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8rem;'>{DISCOVERY_LABEL_RIGHT}</div>", unsafe_allow_html=True)

        st.write(LENGTH_PREF_LABEL)
        length_pref = st.slider(
            LENGTH_SLIDER_LABEL,
            min_value=AP_SLIDER_MIN,
            max_value=AP_SLIDER_MAX,
            step=AP_SLIDER_STEP,
            label_visibility="collapsed",
            help=LENGTH_PREF_HELP,
            key="length_pref"
        )
        logger.debug(f"Slider length_pref changed: {length_pref:.3f}")
        col_short, col_long = st.columns(2)
        with col_short: st.caption(LENGTH_SHORT_LABEL)
        with col_long: st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8rem;'>{LENGTH_LONG_LABEL}</div>", unsafe_allow_html=True)

        st.write(DIFFICULTY_PREF_LABEL)
        difficulty_pref = st.slider(
            DIFFICULTY_SLIDER_LABEL,
            min_value=AP_SLIDER_MIN,
            max_value=AP_SLIDER_MAX,
            step=AP_SLIDER_STEP,
            label_visibility="collapsed",
            help=DIFFICULTY_PREF_HELP,
            key="difficulty_pref"
        )
        logger.debug(f"Slider difficulty_pref changed: {difficulty_pref:.3f}")
        col_easy, col_hard = st.columns(2)
        with col_easy: st.caption(DIFFICULTY_EASY_LABEL)
        with col_hard: st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8rem;'>{DIFFICULTY_HARD_LABEL}</div>", unsafe_allow_html=True)

        st.divider()
        remove_vr = st.checkbox(REMOVE_VR_LABEL, help=REMOVE_VR_HELP, key="remove_vr")
        english_only = st.checkbox(ENGLISH_ONLY_LABEL, help=ENGLISH_ONLY_HELP, key="english_only")
        remove_nsfw = st.checkbox(REMOVE_NSFW_LABEL, help=REMOVE_NSFW_HELP, key="remove_nsfw")
        remove_utilities = st.checkbox(REMOVE_UTILITIES_LABEL, help=REMOVE_UTILITIES_HELP, key="remove_utilities")
        remove_unreleased = st.checkbox(REMOVE_UNRELEASED_LABEL, help=REMOVE_UNRELEASED_HELP, key="remove_unreleased")
        
        st.divider()
        debug_mode = st.checkbox(DEBUG_MODE_LABEL, help=DEBUG_MODE_HELP, key="debug_mode")
        top_k = st.number_input(TOP_K_LABEL, 1, TOP_K_MAX, help=TOP_K_HELP, key="top_k")

    # Input section
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        prompt = st.text_input(PROMPT_LABEL, placeholder=PROMPT_PLACEHOLDER, help=PROMPT_HELP, key="prompt")

    with col2:
        game_list = get_game_list()
        if game_list is None:
            st.warning(f"Connecting to backend server at {BACKEND_URL}...")
            # If we couldn't get the list, don't show the multiselect yet or show it empty
            selected_games = st.multiselect(SEED_LABEL, options=[], disabled=True)
            # Clear cache so it retries on next interaction
            st.cache_data.clear()
        else:
            selected_games = st.multiselect(
                SEED_LABEL, 
                options=game_list, 
                key="seed_multiselect",
                help=SEED_HELP
            )

    with col3:
        genre_list = get_genre_list()
        selected_genres = st.multiselect(
            GENRE_FILTER_LABEL,
            options=genre_list,
            help=GENRE_FILTER_HELP,
            key="genres_multiselect"
        )

    # Search Logic
    with st.spinner(UPDATING_RESULTS_TEXT):
        payload = {
            "alpha": alpha,
            "beta": beta,
            "quality_pref": quality_pref,
            "age_pref": age_pref,
            "pop_pref": pop_pref,
            "disc_pref": disc_pref,
            "length_pref": length_pref,
            "difficulty_pref": difficulty_pref,
            "remove_vr": remove_vr,
            "english_only": english_only,
            "remove_nsfw": remove_nsfw,
            "remove_utilities": remove_utilities,
            "remove_unreleased": remove_unreleased,
            "top_k": top_k,
            "prompt": prompt,
            "seed_games": selected_games,
            "genres": selected_genres
        }
        
        logger.info(f"FRONTEND: Sending payload - quality_pref={quality_pref:.3f}, disc_pref={disc_pref:.3f}, alpha={alpha:.3f}, beta={beta:.3f}")
        logger.info(f"FRONTEND: Session state - st.session_state.quality_pref={st.session_state.get('quality_pref', 'NOT SET'):.3f}")

        try:
            response = requests.post(f"{BACKEND_URL}/recommend", json=payload)
            
            if response.status_code == 200:
                results = response.json()
                
                # Show seed game cards
                if selected_games:
                    st.subheader(f"Seed Games ({len(selected_games)})")
                    try:
                        seed_resp = requests.post(f"{BACKEND_URL}/metadata", json={"names": selected_games})
                        if seed_resp.status_code == 200:
                            seed_results = seed_resp.json()
                            for seed_game in seed_results:
                                render_game_card(seed_game, is_seed=True)
                        st.divider()
                    except:
                        pass

                st.subheader(TOP_RECOMMENDATIONS_HEADER.format(top_k=len(results)))
                
                for game in results:
                    render_game_card(game, show_debug=debug_mode, alpha=alpha, beta=beta, prompt=prompt, selected_games=selected_games)

            else:
                st.error(f"Error fetching recommendations. Status: {response.status_code}")
                try:
                     st.error(response.json()['detail'])
                except:
                    pass

        except requests.exceptions.ConnectionError:
             st.error(f"Cannot connect to backend server at {BACKEND_URL}.")

st.divider()
st.caption(DATA_SOURCE_CAPTION)

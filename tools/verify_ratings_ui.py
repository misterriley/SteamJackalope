import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import ROOT_DIR, QUALITY_TO_RATING_SLOPE, QUALITY_TO_RATING_INTERCEPT

st.set_page_config(page_title="Steam Jackalope - Verify Ratings", layout="wide")

st.title("🎯 Verify Your Predicted Ratings")
st.markdown("""
This tool helps us build your **Personalized Taste Vector**. 
We've predicted how much you liked these games based on your playtime and the global sentiment curves. 
**Please adjust the sliders to match your actual feelings.**
""")

# --- Load Data ---
@st.cache_data
def get_user_files():
    data_dir = os.path.join(ROOT_DIR, "data")
    files = [f for f in os.listdir(data_dir) if f.startswith("user_") and f.endswith("_soft_labels.csv")]
    return files

user_files = get_user_files()

if not user_files:
    st.error("No soft labels found. Please run the acquisition pipeline first.")
    st.stop()

selected_file = st.selectbox("Select User Profile", user_files)
steamid = selected_file.replace("user_", "").replace("_soft_labels.csv", "")

# Load the data
file_path = os.path.join(ROOT_DIR, "data", selected_file)
gt_path = os.path.join(ROOT_DIR, "data", f"user_{steamid}_ground_truth.csv")

if os.path.exists(gt_path):
    st.sidebar.success("Found existing Ground Truth data.")
    df = pd.read_csv(gt_path)
    # Ensure display_rating exists for fallback
    if 'actual_rating' in df.columns:
        df['display_rating'] = df['actual_rating'].fillna(df['predicted_rating'])
    else:
        df['display_rating'] = df['predicted_rating']
else:
    df = pd.read_csv(file_path)
    df['display_rating'] = df['predicted_rating']

# --- Persistent State Management ---
# We use 'master_ratings' and 'master_ignore' to keep data even if widgets are unrendered
if 'current_id' not in st.session_state or st.session_state.get('current_id') != steamid:
    st.session_state['current_id'] = steamid
    st.session_state['items_to_show'] = 50
    
    # Initialize Master Dictionaries from the loaded dataframe
    st.session_state['master_ratings'] = df.set_index('appid')['display_rating'].to_dict()
    if 'ignore' in df.columns:
        st.session_state['master_ignore'] = df.set_index('appid')['ignore'].to_dict()
    else:
        st.session_state['master_ignore'] = {aid: False for aid in df['appid']}

# Callback functions to update master state from widgets
def on_rating_change(appid):
    key = f"slider_{appid}"
    if key in st.session_state:
        st.session_state['master_ratings'][appid] = st.session_state[key]

def on_ignore_change(appid):
    key = f"ignore_{appid}"
    if key in st.session_state:
        st.session_state['master_ignore'][appid] = st.session_state[key]

# Add columns to df for sorting/filtering based on MASTER state
df['ignore'] = df['appid'].map(st.session_state['master_ignore'])
df['actual_rating'] = df['appid'].map(st.session_state['master_ratings'])

# --- Filter & Sort Sidebar ---
st.sidebar.header("Filter & Sort")
sort_options = {
    "Actual Rating (High to Low)": ('actual_rating', False),
    "Predicted Rating (High to Low)": ('predicted_rating', False),
    "Global Rating (High to Low)": ('global_q', False),
    "My Review (Thumbs Up First)": ('user_voted_up', False),
    "Playtime (High to Low)": ('playtime_forever', False),
    "Name": ('name', True),
    "Ignored": ('ignore', False)
}
sort_label = st.sidebar.selectbox("Sort By", list(sort_options.keys()), key="sort_selector")
search_q = st.sidebar.text_input("Search Games", key="search_input")

# Apply Filters
if search_q:
    display_df = df[df['name'].str.contains(search_q, case=False, na=False)]
else:
    display_df = df.copy()

# Apply Sorting
col, asc = sort_options[sort_label]
display_df = display_df.sort_values(col, ascending=asc)

# --- UI Header ---
col_widths = [1, 2, 2, 0.8, 0.8, 0.8, 1.5, 0.8]
cols = st.columns(col_widths)
cols[0].write("**Banner**")
cols[1].write("**Game Info**")
cols[2].write("**Your Review**")
cols[3].write("**Global**")
cols[4].write("**Pred.**")
cols[5].write("**Actual**")
cols[6].write("**Adjustment**")
cols[7].write("**Ignore**")
st.divider()

# --- Render Table ---
subset = display_df.head(st.session_state['items_to_show'])

for idx, row in subset.iterrows():
    appid = row['appid']
    name = row['name']
    playtime_hrs = row['playtime_forever'] / 60.0
    pred_rating = int(row['predicted_rating'])
    global_q = row['global_q']
    user_voted = row.get('user_voted_up', np.nan)
    user_review = row.get('user_review_text', "")
    
    # Calculate Global Rating on 0-10 scale
    global_rating = int(np.clip(np.round(QUALITY_TO_RATING_INTERCEPT + QUALITY_TO_RATING_SLOPE * global_q), 0, 10))
    
    # Master Values
    current_rating = int(st.session_state['master_ratings'].get(appid, pred_rating))
    current_ignore = st.session_state['master_ignore'].get(appid, False)

    with st.container():
        c = st.columns(col_widths)
        
        # Image
        img_url = f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"
        c[0].image(img_url, use_container_width=True)
        
        # Info
        c[1].markdown(f"**{name}**")
        c[1].caption(f"{playtime_hrs:.1f} hours played")
        
        # Review Text
        if pd.notna(user_voted):
            sentiment = "👍 Recommended" if user_voted else "👎 Not Recommended"
            c[2].markdown(f"**{sentiment}**")
            if isinstance(user_review, str) and user_review.strip():
                with c[2].expander("Read Review"):
                    st.write(user_review)
        else:
            c[2].write("")
            
        # Global
        c[3].subheader(f"{global_rating}")
            
        # Predicted
        c[4].subheader(f"{pred_rating}")
        
        # Adjustment Slider
        actual_val = c[6].slider(
            f"Rate {name}", 0, 10, 
            value=current_rating,
            key=f"slider_{appid}",
            label_visibility="collapsed",
            disabled=current_ignore,
            on_change=on_rating_change,
            args=(appid,)
        )
        
        # Actual Display
        if current_ignore:
            c[5].markdown("~~SKIP~~")
        else:
            c[5].subheader(f"**{actual_val}**")
            
        # Ignore Checkbox
        c[7].checkbox(
            "Ignore", 
            value=current_ignore,
            key=f"ignore_{appid}", 
            label_visibility="collapsed",
            on_change=on_ignore_change,
            args=(appid,)
        )
        
        st.divider()

# --- Load More ---
if st.session_state['items_to_show'] < len(display_df):
    if st.button("Load More Games...", use_container_width=True):
        st.session_state['items_to_show'] += 50
        st.rerun()

# --- Save Results ---
st.sidebar.divider()
if st.sidebar.button("💾 Save Ground Truth Ratings", type="primary"):
    final_df = df.copy()
    final_df['actual_rating'] = final_df['appid'].map(st.session_state['master_ratings'])
    final_df['ignore'] = final_df['appid'].map(st.session_state['master_ignore'])
    
    if 'display_rating' in final_df.columns:
        final_df.drop(columns=['display_rating'], inplace=True)
        
    output_path = os.path.join(ROOT_DIR, "data", f"user_{steamid}_ground_truth.csv")
    final_df.to_csv(output_path, index=False)
    st.sidebar.success(f"Saved {len(final_df)} ratings to {output_path}!")
    st.balloons()

st.sidebar.info(f"Showing top {min(st.session_state['items_to_show'], len(display_df))} of {len(display_df)} games")

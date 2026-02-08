import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import os
import sys
from scipy.stats import chi

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_TAG_FILE,
    TAG_VECTORS_FILE,
    METADATA_FILE,
    W_DESC_FILE,
    W_STRUCTURAL_FILE,
    MEAN_DESC_FILE,
    MEAN_STRUCTURAL_FILE,
    MODEL_NAME,
    DOT_PRODUCT_LAMBDA,
    EPSILON
)

st.set_page_config(page_title="Vector Interrogator", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    embeddings_desc = np.load(EMBEDDINGS_DESC_FILE)
    embeddings_structural = np.load(EMBEDDINGS_TAG_FILE)
    tag_vectors = np.load(TAG_VECTORS_FILE)
    metadata = pd.read_parquet(METADATA_FILE)
    
    # Load whitening
    w_desc = np.load(W_DESC_FILE) if os.path.exists(W_DESC_FILE) else None
    w_structural = np.load(W_STRUCTURAL_FILE) if os.path.exists(W_STRUCTURAL_FILE) else None
    mean_desc = np.load(MEAN_DESC_FILE) if os.path.exists(MEAN_DESC_FILE) else None
    mean_structural = np.load(MEAN_STRUCTURAL_FILE) if os.path.exists(MEAN_STRUCTURAL_FILE) else None
    
    return {
        "Descriptive": embeddings_desc,
        "Structural": embeddings_structural,
        "Tag": tag_vectors,
        "metadata": metadata,
        "w_desc": w_desc,
        "w_structural": w_structural,
        "mean_desc": mean_desc,
        "mean_structural": mean_structural
    }

def calculate_stats(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    mean_vec = np.mean(vectors, axis=0)
    # Covariance can be large, just get some summary stats
    # For efficiency, we don't always want full cov
    diag_cov = np.var(vectors, axis=0)
    
    return {
        "norms": norms,
        "mean_vec": mean_vec,
        "diag_cov": diag_cov
    }

def get_similarity_scores(vectors, query_vec, method='cosine', lambd=1.0):
    if method == 'cosine':
        # Normalize vectors and query
        v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        v_norms[v_norms == 0] = EPSILON
        v_normed = vectors / v_norms
        
        q_norm = np.linalg.norm(query_vec)
        q_normed = query_vec / (q_norm if q_norm > EPSILON else 1.0)
        
        return np.dot(v_normed, q_normed)
    elif method == 'regularized_cosine':
        dot_products = np.dot(vectors, query_vec)
        v_norms = np.linalg.norm(vectors, axis=1)
        q_norm = np.linalg.norm(query_vec)
        denom = (v_norms * q_norm) + lambd
        denom[denom == 0] = EPSILON
        return dot_products / denom
    return np.zeros(len(vectors))

st.title("Vector Interrogation Tool")

def render_game_card(row, similarity=None, show_tags=False):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**[{row['name']}](https://store.steampowered.com/app/{row['appid']})**")
            if show_tags and 'tags' in row and row['tags']:
                # Tags are often stored as a string representation of a dict or list
                tags = row['tags']
                if isinstance(tags, str):
                    try:
                        import ast
                        # If it's a dict-like string, try to get keys
                        if tags.startswith('{'):
                            tags_dict = ast.literal_eval(tags)
                            tags = list(tags_dict.keys())
                        elif tags.startswith('['):
                            tags = ast.literal_eval(tags)
                    except:
                        pass
                
                if isinstance(tags, list):
                    st.caption(f"Tags: {', '.join(tags[:15])}")
                else:
                    st.caption(f"Tags: {tags}")
        with col2:
            if similarity is not None:
                st.write(f"Sim: {similarity:.4f}")
            st.write(f"ID: {row['appid']}")

data = load_data()
metadata = data['metadata']
model = load_model()

# Sidebar for global settings
st.sidebar.header("Global Settings")
current_lambda = st.sidebar.slider("DOT_PRODUCT_LAMBDA", 0.0, 20.0, float(DOT_PRODUCT_LAMBDA), 0.1)

tabs = st.tabs(["Distribution Stats", "Similarity Search", "Prompt Interrogation"])

with tabs[0]:
    st.header("Overall Distribution Statistics")
    category = st.selectbox("Select Vector Category", ["Descriptive", "Structural", "Tag"])
    vectors = data[category]
    
    stats = calculate_stats(vectors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Length (Norm) Distribution")
        fig, ax = plt.subplots()
        sns.histplot(stats['norms'], kde=True, ax=ax)
        ax.set_title(f"Distribution of {category} Vector Norms")
        st.pyplot(fig)
        
        st.write(f"**Mean Norm:** {np.mean(stats['norms']):.4f}")
        st.write(f"**Median Norm:** {np.median(stats['norms']):.4f}")
        st.write(f"**Std Norm:** {np.std(stats['norms']):.4f}")

    with col2:
        st.subheader("Dot Product Distribution (Sampled)")
        # Sample pairs for dot product distribution
        sample_size = min(2000, len(vectors))
        indices = np.random.choice(len(vectors), sample_size, replace=False)
        sample_vectors = vectors[indices]
        
        # Normalize for standard cosine similarity distribution if not Tag
        if category != "Tag":
            s_norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
            s_norms[s_norms == 0] = EPSILON
            sample_vectors = sample_vectors / s_norms
        
        # Compute subset of dot products (pairwise)
        # To avoid N^2, we'll just do a few thousand random pairs or a small matrix
        dots = np.dot(sample_vectors, sample_vectors.T)
        # Get upper triangle only, excluding diagonal
        dots_flat = dots[np.triu_indices(sample_size, k=1)]
        
        fig, ax = plt.subplots()
        sns.histplot(dots_flat, kde=True, ax=ax)
        ax.set_title(f"Distribution of {category} Pairwise Dot Products")
        st.pyplot(fig)
        
        st.write(f"**Mean Dot Product:** {np.mean(dots_flat):.4f}")
        st.write(f"**Std Dot Product:** {np.std(dots_flat):.4f}")

    st.divider()
    st.subheader("Mean and Variance Statistics")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Mean Vector Stats:**")
        st.write(f"- Min: {np.min(stats['mean_vec']):.6f}")
        st.write(f"- Max: {np.max(stats['mean_vec']):.6f}")
        st.write(f"- Mean: {np.mean(stats['mean_vec']):.6f}")
        st.write(f"- Abs Mean: {np.mean(np.abs(stats['mean_vec'])):.6f}")
    
    with c2:
        st.write("**Diagonal Covariance (Variance) Stats:**")
        st.write(f"- Min: {np.min(stats['diag_cov']):.6f}")
        st.write(f"- Max: {np.max(stats['diag_cov']):.6f}")
        st.write(f"- Mean: {np.mean(stats['diag_cov']):.6f}")
        st.write(f"- Total Variance (Trace): {np.sum(stats['diag_cov']):.4f}")

with tabs[1]:
    st.header("Game-to-Game Similarity")
    
    target_game_name = st.selectbox("Select Target Game", metadata['name'].tolist())
    target_idx = metadata[metadata['name'] == target_game_name].index[0]
    
    st.write(f"Interrogating similarities for: **{target_game_name}** (AppID: {metadata.loc[target_idx, 'appid']})")
    
    # Calculate similarities for all categories
    sims_desc = get_similarity_scores(data['Descriptive'], data['Descriptive'][target_idx], method='cosine')
    sims_struct = get_similarity_scores(data['Structural'], data['Structural'][target_idx], method='cosine')
    sims_tag = get_similarity_scores(data['Tag'], data['Tag'][target_idx], method='regularized_cosine', lambd=current_lambda)
    
    # Combined scores
    sims_semantic = (sims_desc + sims_struct) / 2.0
    sims_all = (sims_desc + sims_struct + sims_tag) / 3.0
    
    category_to_show = st.radio("Show results for:", ["Descriptive", "Structural", "Tag", "Semantic Sum", "All Categories Sum"], horizontal=True)
    
    score_map = {
        "Descriptive": sims_desc,
        "Structural": sims_struct,
        "Tag": sims_tag,
        "Semantic Sum": sims_semantic,
        "All Categories Sum": sims_all
    }
    
    selected_scores = score_map[category_to_show]
    
    # Exclude the game itself from most similar
    display_scores = selected_scores.copy()
    display_scores[target_idx] = -1e12
    
    most_sim_indices = np.argsort(-display_scores)[:10]
    least_sim_indices = np.argsort(selected_scores)[:10]
    
    col_most, col_least = st.columns(2)
    
    show_tags = category_to_show in ["Tag", "Structural", "All Categories Sum"]
    
    with col_most:
        st.subheader("Most Similar Games")
        for i, idx in enumerate(most_sim_indices):
            render_game_card(metadata.iloc[idx], similarity=selected_scores[idx], show_tags=show_tags)
        
    with col_least:
        st.subheader("Least Similar Games")
        for i, idx in enumerate(least_sim_indices):
            render_game_card(metadata.iloc[idx], similarity=selected_scores[idx], show_tags=show_tags)

with tabs[2]:
    st.header("Prompt Interrogation")
    test_prompt = st.text_input("Enter a test prompt (simulates website search)", "")
    
    if test_prompt:
        with st.spinner("Encoding and whitening..."):
            prompt_vec = model.encode([test_prompt])[0]
            
            # Apply whitening
            p_desc_centered = (prompt_vec - data['mean_desc']) if data['mean_desc'] is not None else prompt_vec
            p_struct_centered = (prompt_vec - data['mean_structural']) if data['mean_structural'] is not None else prompt_vec
            
            p_desc_white = np.dot(p_desc_centered, data['w_desc']) if data['w_desc'] is not None else p_desc_centered
            p_struct_white = np.dot(p_struct_centered, data['w_structural']) if data['w_structural'] is not None else p_struct_centered
            
            # Similarity scores
            p_sims_desc = get_similarity_scores(data['Descriptive'], p_desc_white, method='cosine')
            p_sims_struct = get_similarity_scores(data['Structural'], p_struct_white, method='cosine')
            p_sims_combined = (p_sims_desc + p_sims_struct) / 2.0
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("Descriptive Matches")
                d_idx = np.argsort(-p_sims_desc)[:10]
                for idx in d_idx:
                    render_game_card(metadata.iloc[idx], similarity=p_sims_desc[idx], show_tags=False)
                
            with c2:
                st.subheader("Structural Matches")
                s_idx = np.argsort(-p_sims_struct)[:10]
                for idx in s_idx:
                    render_game_card(metadata.iloc[idx], similarity=p_sims_struct[idx], show_tags=True)
                
            with c3:
                st.subheader("Combined Matches")
                c_idx = np.argsort(-p_sims_combined)[:10]
                for idx in c_idx:
                    render_game_card(metadata.iloc[idx], similarity=p_sims_combined[idx], show_tags=True)
    else:
        st.info("Enter a prompt to see matching games.")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sys
import logging

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    BACKEND_URL,
    SIMILARITY_LISTS_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def make_md_link(name, appid):
    """Generates a markdown link to a Steam app page, escaping special characters."""
    # Escape brackets and pipes to avoid breaking markdown table link syntax
    clean_name = str(name).replace('[', '').replace(']', '').replace('|', '&#124;')
    return f"[{clean_name}](https://store.steampowered.com/app/{appid})"

def render_lists_page():
    """
    Renders the 'Lists' tab in the Streamlit app.
    Fetches curated game lists from the FastAPI backend.
    """
    logger.info("Rendering Lists page")
    st.header("Project Insights & Rankings")
    st.write("Explore the extremes of the Steam library through curated lists.")

    list_tabs = st.tabs(["Quality", "Length", "Popularity", "Age", "Difficulty", "Similarity"])

    with list_tabs[0]:
        st.subheader("Top & Bottom Games by Quality")
        st.write("Quality is calculated using a Bayesian Probit model. Discovery level affects how much we trust games with fewer reviews.")
        
        discovery_levels = {
            "Low Discovery (Known Quantities)": 1.0,
            "Medium Discovery (Balanced)": 0.0,
            "High Discovery (Wild Cards)": -1.0
        }
        
        disc_choice = st.selectbox("Select Discovery Level", list(discovery_levels.keys()))
        disc_pref = discovery_levels[disc_choice]
        
        try:
            logger.info(f"Fetching quality list with discovery_pref={disc_pref}")
            response = requests.get(f"{BACKEND_URL}/lists/quality?discovery_pref={disc_pref}", timeout=10)
            logger.info(f"Quality list response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Quality list data: {len(data.get('top', []))} top, {len(data.get('bottom', []))} bottom")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Top 50 Highest Quality")
                    top_df = pd.DataFrame(data['top'])
                    if not top_df.empty:
                        top_df['Link'] = top_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        top_df['Quality Score'] = top_df['quality_score'].round(2)
                        st.write(top_df[['Link', 'Quality Score']].to_markdown(index=False))
                    else:
                        st.warning("No quality data available")
                    
                with col2:
                    st.write("### Top 50 Lowest Quality")
                    bottom_df = pd.DataFrame(data['bottom'])
                    if not bottom_df.empty:
                        bottom_df['Link'] = bottom_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        bottom_df['Quality Score'] = bottom_df['quality_score'].round(2)
                        st.write(bottom_df[['Link', 'Quality Score']].to_markdown(index=False))
                    else:
                        st.warning("No quality data available")
            else:
                logger.error(f"Quality list fetch failed: {response.status_code} - {response.text}")
                st.error(f"Failed to fetch quality list. Status: {response.status_code}")
        except Exception as e:
            logger.exception(f"Error fetching quality list: {e}")
            st.error(f"Error: {e}")

    with list_tabs[1]:
        st.subheader("Top 50 Longest & Shortest Games")
        st.write("Based on estimated average playtime.")
        
        try:
            logger.info("Fetching length list")
            response = requests.get(f"{BACKEND_URL}/lists/length", timeout=10)
            logger.info(f"Length list response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Top 50 Longest")
                    top_df = pd.DataFrame(data['top'])
                    if not top_df.empty:
                        top_df['Hours'] = (top_df['playtime'] / 60.0).round(1)
                        top_df['Link'] = top_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        st.write(top_df[['Link', 'Hours']].to_markdown(index=False))
                    else:
                        st.warning("No length data available")
                    
                with col2:
                    st.write("### Top 50 Shortest")
                    bottom_df = pd.DataFrame(data['bottom'])
                    if not bottom_df.empty:
                        bottom_df['Hours'] = (bottom_df['playtime'] / 60.0).round(1)
                        bottom_df['Link'] = bottom_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        st.write(bottom_df[['Link', 'Hours']].to_markdown(index=False))
                    else:
                        st.warning("No length data available")
            else:
                logger.error(f"Length list fetch failed: {response.status_code}")
                st.error("Failed to fetch length list.")
        except Exception as e:
            logger.exception(f"Error fetching length list: {e}")
            st.error(f"Error: {e}")

    with list_tabs[2]:
        st.subheader("Top 50 Most & Least Popular Games")
        st.write("Popularity is measured by the total number of reviews. Least popular games must have at least 1 review.")
        
        try:
            logger.info("Fetching popularity list")
            response = requests.get(f"{BACKEND_URL}/lists/popularity", timeout=10)
            logger.info(f"Popularity list response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Top 50 Most Popular")
                    top_df = pd.DataFrame(data['top'])
                    if not top_df.empty:
                        top_df['Total Reviews'] = top_df['total_reviews'].astype(int)
                        top_df['Link'] = top_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        st.write(top_df[['Link', 'Total Reviews']].to_markdown(index=False))
                    else:
                        st.warning("No popularity data available")
                    
                with col2:
                    st.write("### Top 50 Least Popular (min 1 review)")
                    bottom_df = pd.DataFrame(data['bottom'])
                    if not bottom_df.empty:
                        bottom_df['Total Reviews'] = bottom_df['total_reviews'].astype(int)
                        bottom_df['Link'] = bottom_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        st.write(bottom_df[['Link', 'Total Reviews']].to_markdown(index=False))
                    else:
                        st.warning("No popularity data available")
            else:
                logger.error(f"Popularity list fetch failed: {response.status_code}")
                st.error("Failed to fetch popularity list.")
        except Exception as e:
            logger.exception(f"Error fetching popularity list: {e}")
            st.error(f"Error: {e}")

    with list_tabs[3]:
        st.subheader("Top 50 Oldest & Newest Games")
        st.write("Excluding games with future release dates.")
        
        try:
            logger.info("Fetching age list")
            response = requests.get(f"{BACKEND_URL}/lists/age", timeout=10)
            logger.info(f"Age list response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Top 50 Newest")
                    top_df = pd.DataFrame(data['top'])
                    if not top_df.empty:
                        top_df['Release Date'] = top_df['release_date']
                        top_df['Link'] = top_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        st.write(top_df[['Link', 'Release Date']].to_markdown(index=False))
                    else:
                        st.warning("No age data available")
                    
                with col2:
                    st.write("### Top 50 Oldest")
                    bottom_df = pd.DataFrame(data['bottom'])
                    if not bottom_df.empty:
                        bottom_df['Release Date'] = bottom_df['release_date']
                        bottom_df['Link'] = bottom_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                        st.write(bottom_df[['Link', 'Release Date']].to_markdown(index=False))
                    else:
                        st.warning("No age data available")
            else:
                logger.error(f"Age list fetch failed: {response.status_code}")
                st.error("Failed to fetch age list.")
        except Exception as e:
            logger.exception(f"Error fetching age list: {e}")
            st.error(f"Error: {e}")

    with list_tabs[4]:
        st.subheader("Difficulty Analysis")
        st.write("Difficulty is predicted using a model trained on GameFAQs ratings and Steam tags.")
        
        try:
            logger.info("Fetching difficulty list")
            response = requests.get(f"{BACKEND_URL}/lists/difficulty", timeout=10)
            logger.info(f"Difficulty list response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                diff_view = st.selectbox("Select View", ["Games", "Tags"])
                
                if diff_view == "Games":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Top 50 Most Difficult")
                        top_df = pd.DataFrame(data['top'])
                        if not top_df.empty:
                            top_df['Difficulty Score'] = top_df['difficulty_predicted'].round(1)
                            top_df['Link'] = top_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                            st.write(top_df[['Link', 'Difficulty Score']].to_markdown(index=False))
                        else:
                            st.warning("No difficulty game data available")
                        
                    with col2:
                        st.write("### Top 50 Easiest")
                        bottom_df = pd.DataFrame(data['bottom'])
                        if not bottom_df.empty:
                            bottom_df['Difficulty Score'] = bottom_df['difficulty_predicted'].round(1)
                            bottom_df['Link'] = bottom_df.apply(lambda row: make_md_link(row['name'], row['appid']), axis=1)
                            st.write(bottom_df[['Link', 'Difficulty Score']].to_markdown(index=False))
                        else:
                            st.warning("No difficulty game data available")
                
                else:  # Tags view
                    st.write("### Difficulty Predictors (Tags)")
                    st.write("These tags are the strongest positive and negative predictors of game difficulty in our model.")
                    
                    tag_impacts = data.get('tag_impacts', [])
                    logger.info(f"Difficulty tag impacts: {len(tag_impacts)} tags")
                    
                    impact_df = pd.DataFrame(tag_impacts)
                    if not impact_df.empty:
                        impact_df.rename(columns={'tag': 'Tag', 'impact': 'Average Impact'}, inplace=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("#### Strongest Difficulty Increase")
                            st.dataframe(impact_df.head(20), hide_index=True, use_container_width=True)
                        with c2:
                            st.write("#### Strongest Difficulty Decrease")
                            st.dataframe(impact_df.tail(20).sort_values('Average Impact', ascending=True), hide_index=True, use_container_width=True)
                    else:
                        st.info("Difficulty prediction data not found on server.")
                        logger.warning("No tag impacts returned from difficulty endpoint")
            else:
                logger.error(f"Difficulty list fetch failed: {response.status_code}")
                st.error("Failed to fetch difficulty list.")
        except Exception as e:
            logger.exception(f"Error fetching difficulty list: {e}")
            st.error(f"Error: {e}")

    with list_tabs[5]:
        st.subheader("Similarity Analysis")
        st.write("Find the most similar games for 20 popular but diverse seeds.")
        
        sim_type = st.selectbox("Select Similarity Type", ["Tags", "Semantic"])
        
        try:
            logger.info(f"Loading similarity data for type: {sim_type}")
            import json
            data_path = SIMILARITY_LISTS_FILE
            logger.debug(f"Looking for similarity data at: {data_path}")
            
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    sim_data = json.load(f)
                logger.info(f"Similarity data loaded: keys={list(sim_data.keys())}")
                
                display_data = sim_data.get('tags', []) if sim_type == "Tags" else sim_data.get('semantic', [])
                logger.info(f"Display data count: {len(display_data)} entries")
                
                # Build table
                rows = []
                for entry in display_data:
                    seed_link = make_md_link(entry['seed_name'], entry['seed_appid'])
                    similar_links = [make_md_link(s['name'], s['appid']) for s in entry['similar']]
                    rows.append([seed_link] + similar_links)
                
                columns = ["Seed Game", "1st Match", "2nd Match", "3rd Match", "4th Match", "5th Match"]
                df = pd.DataFrame(rows, columns=columns)
                
                # Custom CSS for visual demarcation
                st.markdown("""
                <style>
                .similarity-table table th:first-child {
                    border-right: 3px solid #464855 !important;
                }
                .similarity-table table td:first-child {
                    font-weight: bold;
                    border-right: 3px solid #464855 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="similarity-table">', unsafe_allow_html=True)
                st.write(df.to_markdown(index=False))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                logger.warning(f"Similarity data file not found at {data_path}")
                st.info("Similarity data not found. Please run the precalculation script.")
        except Exception as e:
            logger.exception(f"Error loading similarity data: {e}")
            st.error(f"Error loading similarity data: {e}")
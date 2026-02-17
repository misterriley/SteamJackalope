from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import sys
import re
import ast
import gc
import logging
import subprocess
import asyncio
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Memory Optimization: Limit threads to reduce buffer overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_TAG_FILE,
    QUALITY_GRID_FILE,
    W_DESC_FILE,
    W_STRUCTURAL_FILE,
    MEAN_DESC_FILE,
    MEAN_STRUCTURAL_FILE,
    TAG_VECTORS_FILE,
    TAG_NORMS_FILE,
    METADATA_FILE,
    TRENDING_APPIDS_FILE,
    DOT_PRODUCT_LAMBDA,
    EPSILON,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    TOP_K_SORT_MULTIPLIER,
    SEMANTIC_PROMPT_SEED_BLEND,
    MODEL_NAME,
    SENTENCE_TRANSFORMER_BACKEND,
    SENTENCE_TRANSFORMER_MODEL_KWARGS,
    AP_SLIDER_VALUES,
    AP_SLIDER_STEP,
    AP_SLIDER_MIN,
    SEMANTIC_WEIGHT_MULTIPLIER,
    TAG_WEIGHT_MULTIPLIER,
    QUALITY_WEIGHT_MULTIPLIER,
    AGE_WEIGHT_MULTIPLIER,
    POPULARITY_WEIGHT_MULTIPLIER,
    LENGTH_WEIGHT_MULTIPLIER,
    DIFFICULTY_WEIGHT_MULTIPLIER,
    NSFW_TAGS,
    NSFW_NAME_PATTERNS,
    DIFFICULTY_PREDICTIONS_FILE,
    ROOT_DIR,
    TAG_GLOBAL_SCALING_FACTOR
)
from common.utils import to_z, calculate_hybrid_score, calculate_linear_scores

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading ---

class DataManager:
    def __init__(self):
        self.embeddings_desc_norm = None
        self.embeddings_structural_norm = None
        self.metadata = None
        self.tag_vectors = None
        self.quality_grid = None
        self.tag_vectors_norms = None
        self.w_desc = None
        self.w_structural = None
        self.mean_desc = None
        self.mean_structural = None
        self.model = None
        self.all_genres = []
        self.trending_names = []
        self.term_links = {}
        self.lists_cache = {}

    def filter_dead_tags(self, tags_str):
        """Filters out tags that are not in the term_links mapping (dead tags)."""
        if not tags_str or not self.term_links:
            return tags_str
            
        try:
            # Handle dictionary-like format: {'Tag': count, ...}
            if tags_str.startswith('{') and tags_str.endswith('}'):
                data = ast.literal_eval(tags_str)
                if isinstance(data, dict):
                    # Only keep tags that have a validated URL (value is not None)
                    filtered = {k: v for k, v in data.items() if self.term_links.get(k) is not None}
                    return str(filtered)
            return tags_str
        except:
            return tags_str

    def clean_release_date(self, date_str):
        if pd.isna(date_str) or date_str == "":
            return pd.NaT
        
        s = str(date_str).strip()
        
        # Match "YYYY-MM-DD"
        if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
            return pd.to_datetime(s, errors='coerce')
        
        # Match "YYYY" -> YYYY-07-01
        if re.match(r'^\d{4}$', s):
            return pd.to_datetime(s + "-07-01", errors='coerce')
        
        # Match "Month YYYY" or "YYYY Month" (assume 15th)
        m1 = re.match(r'^([A-Za-z]+)\s+(\d{4})$', s)
        if m1:
            return pd.to_datetime(f"{m1.group(1)} 15, {m1.group(2)}", errors='coerce')
        
        m2 = re.match(r'^(\d{4})\s+([A-Za-z]+)$', s)
        if m2:
            return pd.to_datetime(f"{m2.group(2)} 15, {m2.group(1)}", errors='coerce')

        # Default fallback for "DD Mon, YYYY" etc.
        return pd.to_datetime(s, errors='coerce')

    def load_data(self):
        logger.info("Starting data load...")
        import pyarrow.parquet as pq
        
        # 1. Use Memory Mapping for large NumPy arrays
        logger.info(f"Loading embeddings_desc_norm from {EMBEDDINGS_DESC_FILE}")
        self.embeddings_desc_norm = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        logger.info(f"Embeddings_desc_norm: shape={self.embeddings_desc_norm.shape}, dtype={self.embeddings_desc_norm.dtype}")
        
        logger.info(f"Loading embeddings_structural_norm from {EMBEDDINGS_TAG_FILE}")
        self.embeddings_structural_norm = np.load(EMBEDDINGS_TAG_FILE, mmap_mode='r')
        logger.info(f"Embeddings_structural_norm: shape={self.embeddings_structural_norm.shape}, dtype={self.embeddings_structural_norm.dtype}")

        # 2. Memory Map Tag Vectors and Quality Grid
        logger.info(f"Loading tag_vectors from {TAG_VECTORS_FILE}")
        self.tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
        logger.info(f"Tag vectors: shape={self.tag_vectors.shape}, dtype={self.tag_vectors.dtype}")
        
        logger.info(f"Loading quality_grid from {QUALITY_GRID_FILE}")
        self.quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
        logger.info(f"Quality grid: shape={self.quality_grid.shape}, dtype={self.quality_grid.dtype}")
        
        # 3. Load Metadata
        logger.info(f"Loading metadata from {METADATA_FILE}")
        # Determine which columns are actually available using pyarrow for speed
        try:
            schema = pq.read_schema(METADATA_FILE)
            available_cols = schema.names
        except:
            available_cols = []

        needed_cols = [
            'appid', 'name', 'release_date', 'positive', 'negative', 
            'genres', 'tags', 'categories', 'supported_languages',
            'mature_content', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z',
            'estimated_playtime', 'difficulty_predicted'
        ]
        # Only request short_description if it exists
        if 'short_description' in available_cols:
            needed_cols.insert(2, 'short_description')  # Insert after name
            logger.info("Found short_description in metadata")
        else:
            logger.warning("short_description NOT found in metadata")

        if 'release_year' in available_cols:
            needed_cols.append('release_year')

        try:
            self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols, dtype_backend='pyarrow')
            logger.info(f"Metadata loaded with pyarrow backend. Columns: {self.metadata.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Failed to load with pyarrow backend: {e}. Falling back to standard.")
            self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols)
        
        logger.info("Extracting boolean features...")
        self.metadata['is_vr_only'] = self.metadata['categories'].fillna('').astype(str).str.contains('VR Only', case=False).astype(bool)
        self.metadata['is_english'] = self.metadata['supported_languages'].fillna('').astype(str).str.contains('English', case=False).astype(bool)
        self.metadata['is_utility'] = self.metadata['tags'].fillna('').astype(str).str.contains('Utilities', case=False).astype(bool)
        
        logger.info("Identifying NSFW games...")
        # Start with mature_content flag
        self.metadata['is_nsfw'] = (self.metadata['mature_content'] > 0).values
        
        # Check tags
        tags_lower = self.metadata['tags'].fillna('').astype(str).str.lower()
        nsfw_tag_mask = tags_lower.apply(lambda x: any(tag in x for tag in NSFW_TAGS))
        self.metadata['is_nsfw'] |= nsfw_tag_mask.values
        
        # Check names
        nsfw_name_mask = self.metadata['name'].fillna('').str.contains('|'.join(NSFW_NAME_PATTERNS), case=False, na=False)
        self.metadata['is_nsfw'] |= nsfw_name_mask.values
        
        self.metadata.drop(columns=['categories', 'supported_languages'], inplace=True)
        logger.info(f"Boolean features extracted. Metadata shape: {self.metadata.shape}")
        
        # Optimize metadata types
        logger.info("Optimizing dtypes...")
        self.metadata['appid'] = self.metadata['appid'].astype(np.int32)
        self.metadata['positive'] = self.metadata['positive'].astype(np.int32)
        self.metadata['negative'] = self.metadata['negative'].astype(np.int32)
        self.metadata['date_z'] = self.metadata['date_z'].astype(np.float16)
        self.metadata['pop_z'] = self.metadata['pop_z'].astype(np.float16)
        self.metadata['playtime_z'] = self.metadata['playtime_z'].astype(np.float16)
        self.metadata['difficulty_z'] = self.metadata['difficulty_z'].astype(np.float16)
        self.metadata['estimated_playtime'] = self.metadata['estimated_playtime'].astype(np.float32)
        self.metadata['difficulty_predicted'] = self.metadata['difficulty_predicted'].astype(np.float32)
        self.metadata['mature_content'] = self.metadata['mature_content'].fillna(0).astype(np.int8)

        # Parse dates
        logger.info("Parsing release dates...")
        self.metadata['parsed_date'] = self.metadata['release_date'].apply(self.clean_release_date)
        
        logger.info(f"Metadata loaded: {len(self.metadata)} rows, {len(self.metadata.columns)} columns")
        
        # 4. Load weights and means
        logger.info("Loading transformation matrices...")
        self.w_desc = np.load(W_DESC_FILE).astype(np.float16) if os.path.exists(W_DESC_FILE) else None
        if self.w_desc is not None:
            logger.info(f"W_desc: shape={self.w_desc.shape}")
        self.w_structural = np.load(W_STRUCTURAL_FILE).astype(np.float16) if os.path.exists(W_STRUCTURAL_FILE) else None
        if self.w_structural is not None:
            logger.info(f"W_structural: shape={self.w_structural.shape}")
        self.mean_desc = np.load(MEAN_DESC_FILE).astype(np.float16) if os.path.exists(MEAN_DESC_FILE) else None
        if self.mean_desc is not None:
            logger.info(f"Mean_desc: shape={self.mean_desc.shape}")
        self.mean_structural = np.load(MEAN_STRUCTURAL_FILE).astype(np.float16) if os.path.exists(MEAN_STRUCTURAL_FILE) else None
        if self.mean_structural is not None:
            logger.info(f"Mean_structural: shape={self.mean_structural.shape}")

        # Release year
        if 'release_year' not in self.metadata.columns:
            self.metadata['release_year'] = self.metadata['parsed_date'].dt.year
            mean_year = self.metadata['release_year'].mean()
            self.metadata['release_year'] = self.metadata['release_year'].fillna(mean_year).astype(np.int16)
        else:
            self.metadata['release_year'] = self.metadata['release_year'].fillna(0).astype(np.int16)
        
        # Load pre-calculated tag vector norms
        loaded_norms = False
        if os.path.exists(TAG_NORMS_FILE):
             logger.info(f"Loading pre-calculated tag norms from {TAG_NORMS_FILE}...")
             self.tag_vectors_norms = np.load(TAG_NORMS_FILE)
             if len(self.tag_vectors_norms) == len(self.tag_vectors):
                 loaded_norms = True
                 logger.info(f"Tag norms loaded: shape={self.tag_vectors_norms.shape}")
             else:
                 logger.warning(f"Tag norms shape mismatch: {len(self.tag_vectors_norms)} vs {len(self.tag_vectors)}")
        
        if not loaded_norms:
             logger.warning("Computing tag norms on the fly (may cause RAM spike)...")
             self.tag_vectors_norms = np.linalg.norm(self.tag_vectors.astype(np.float32), axis=1).astype(np.float16)
        
        # Extract genres
        logger.info("Extracting unique genres...")
        def parse_genres_safe(x):
            if pd.isna(x): return []
            x = str(x)
            if x.startswith('[') and x.endswith(']'):
                try:
                    return ast.literal_eval(x)
                except:
                    pass
            return [g.strip() for g in x.split(',') if g.strip()]

        all_genres_set = set()
        for g_str in self.metadata['genres'].dropna().unique():
            all_genres_set.update(parse_genres_safe(g_str))
        self.all_genres = sorted(list(all_genres_set))
        logger.info(f"Extracted {len(self.all_genres)} unique genres")
        
        # Load trending games
        if os.path.exists(TRENDING_APPIDS_FILE):
            logger.info(f"Loading trending AppIDs from {TRENDING_APPIDS_FILE}...")
            try:
                import json
                with open(TRENDING_APPIDS_FILE, 'r') as f:
                    trending_appids = json.load(f)
                
                # Resolve AppIDs to names using metadata
                mask = self.metadata['appid'].isin(trending_appids)
                self.trending_names = self.metadata[mask]['name'].tolist()
                logger.info(f"Resolved {len(self.trending_names)} trending games from {len(trending_appids)} AppIDs")
            except Exception as e:
                logger.error(f"Failed to load trending games: {e}")
        else:
            logger.warning(f"Trending AppIDs file NOT FOUND at {TRENDING_APPIDS_FILE}")

        # Load validated Steam links
        links_path = os.path.join(ROOT_DIR, "data", "validated_steam_links.json")
        if os.path.exists(links_path):
            logger.info(f"Loading validated Steam links from {links_path}...")
            try:
                with open(links_path, 'r') as f:
                    self.term_links = json.load(f)
                logger.info(f"Loaded {len(self.term_links)} term links")
                
                # Filter tags in metadata based on validated links
                logger.info("Filtering dead tags from metadata...")
                self.metadata['tags'] = self.metadata['tags'].apply(self.filter_dead_tags)
            except Exception as e:
                logger.error(f"Failed to load term links: {e}")
        else:
            logger.warning(f"Validated Steam links file NOT FOUND at {links_path}")

        # 5. Load SentenceTransformer model
        logger.info("Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            MODEL_NAME,
            backend=SENTENCE_TRANSFORMER_BACKEND,
            model_kwargs=SENTENCE_TRANSFORMER_MODEL_KWARGS
        )
        logger.info("SentenceTransformer model loaded successfully")

        mem_usage = self.metadata.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Metadata RAM size: {mem_usage:.2f} MB")
        logger.info("Data loading complete.")

data_manager = DataManager()

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup event triggered")
    data_manager.load_data()
    logger.info("DataManager initialization complete")

# --- API Models ---

class MetadataRequest(BaseModel):
    names: List[str]

class UserFetchRequest(BaseModel):
    steam_id: str
    review_html: Optional[str] = None

class UserVerifyUpdate(BaseModel):
    steam_id: str
    appid: int
    actual_rating: float
    ignore: bool

class RecommendationRequest(BaseModel):
    alpha: float
    beta: float
    quality_pref: float
    age_pref: float
    pop_pref: float
    disc_pref: float
    length_pref: float
    difficulty_pref: float
    remove_vr: bool
    english_only: bool
    remove_nsfw: bool
    remove_utilities: bool
    remove_unreleased: bool
    top_k: int
    prompt: str
    seed_games: List[str]
    genres: Optional[List[str]] = []
    
    # Taste DNA Extensions
    vibe_vector: Optional[List[float]] = None
    intercept: Optional[float] = None
    metadata_weights: Optional[Dict[str, float]] = None
    
    # Profile Exclusion
    profile_filter: Optional[str] = "none" # "none", "rated", "all"
    library_appids: Optional[List[int]] = []
    rated_appids: Optional[List[int]] = []

# --- Endpoints ---

@app.get("/genres")
def get_genres():
    """Returns a list of all unique genres available in the dataset, filtered for validity."""
    logger.info("GET /genres called")
    # Filter out genres that don't have a validated link
    genres = [g for g in data_manager.all_genres if data_manager.term_links.get(g) is not None]
    logger.debug(f"Returning {len(genres)} filtered genres")
    return genres

@app.get("/term_links")
def get_term_links():
    """Returns a mapping of tags/genres to their validated Steam store links."""
    logger.info("GET /term_links called")
    return data_manager.term_links

@app.get("/games")
def get_games():
    """Returns a list of all available game names for the seed selector."""
    logger.info("GET /games called")
    if data_manager.metadata is None:
        logger.warning("Data not loaded yet when /games called")
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    # Limiting to 1000 for safety, though frontend should use /search
    games = sorted(data_manager.metadata['name'].fillna("Unknown").head(1000).tolist())
    return games

@app.get("/games/search")
def search_games(q: str = "", limit: int = 50):
    """Returns game names matching the query string."""
    if data_manager.metadata is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    if not q or len(q) < 2:
        # Return empty or top games if query is too short
        return []
        
    mask = data_manager.metadata['name'].str.contains(re.escape(q), case=False, na=False)
    matches = data_manager.metadata[mask]['name'].head(limit).tolist()
    return sorted(matches)

@app.get("/games/random")
def get_random_game():
    """Returns a random game name from the dataset."""
    if data_manager.metadata is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    random_game = data_manager.metadata.sample(n=1).iloc[0]['name']
    return str(random_game)

@app.get("/games/trending/random")
def get_random_trending_game():
    """Returns a random game name from the trending list."""
    if not data_manager.trending_names:
        # Fallback to general random if trending is not loaded
        return get_random_game()
    
    random_game = np.random.choice(data_manager.trending_names)
    return str(random_game)

@app.get("/changelog")
def get_changelog():
    """Returns the content of CHANGELOG.md."""
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Changelog file not found")

# --- Personalization Pipeline Endpoints ---

@app.post("/user/fetch")
async def fetch_user_data(request: UserFetchRequest, background_tasks: BackgroundTasks):
    """Triggers the scraping and soft-labeling process for a user."""
    input_val = request.steam_id.strip()
    if not input_val:
        raise HTTPException(status_code=400, detail="SteamID is required")
    
    # Resolve SteamID from URL or vanity name synchronously
    # This ensures the frontend and background task use the SAME ID for files/polling.
    steam_id_clean = input_val.rstrip('/')
    if "steamcommunity.com" in input_val:
        profile_match = re.search(r'profiles/(\d+)', input_val)
        if profile_match:
            steam_id_clean = profile_match.group(1)
        else:
            vanity_match = re.search(r'id/([^/]+)', input_val)
            if vanity_match:
                steam_id_clean = vanity_match.group(1).rstrip('/')
    
    # If not numeric, try to resolve as vanity name
    if not (steam_id_clean.isdigit() and (len(steam_id_clean) == 17 or steam_id_clean.startswith('76'))):
        from scraping.get_user_stats import resolve_vanity_url
        logger.info(f"Resolving vanity name: {steam_id_clean}")
        resolved = resolve_vanity_url(steam_id_clean)
        if resolved:
            logger.info(f"Resolved '{steam_id_clean}' to '{resolved}'")
            steam_id_clean = resolved
        else:
            logger.warning(f"Failed to resolve vanity name: {steam_id_clean}")
            # We'll still proceed and let the script try one more time, 
            # but usually this means the user entered junk.
    
    # Save review HTML to temp file if provided
    if request.review_html:
        html_path = f"data/user_{steam_id_clean}_reviews.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(request.review_html)
    
    def run_pipeline(sid):
        try:
            # 1. Scraping
            logger.info(f"Background: Scraping for {sid}")
            cmd_scrape = ["python", "scraping/get_user_stats.py", sid]
            if request.review_html:
                cmd_scrape.append(f"data/user_{sid}_reviews.html")
            
            # Use subprocess and capture output for debugging
            result = subprocess.run(cmd_scrape, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Scraping failed for {sid}: {result.stderr}")
                return

            # Extract the actual resolved SteamID from output just in case
            resolved_id = sid
            match = re.search(r'Fetching library for SteamID: (\d+)', result.stdout)
            if match:
                resolved_id = match.group(1)
            
            # 2. Soft Labeling
            logger.info(f"Background: Labeling for {resolved_id}")
            subprocess.run(["python", "pipeline/generate_user_soft_labels.py", f"data/user_{resolved_id}_library.csv"], check=True)
            logger.info(f"Background: Pipeline complete for {resolved_id}")
        except Exception as e:
            logger.error(f"Background Pipeline Failed: {e}")

    background_tasks.add_task(run_pipeline, steam_id_clean)
    return {"message": "Data acquisition started.", "resolved_as": steam_id_clean}

@app.get("/user/status/{steam_id}")
def get_user_status(steam_id: str):
    """Checks the progress of data acquisition."""
    library_exists = os.path.exists(f"data/user_{steam_id}_library.csv")
    soft_labels_exists = os.path.exists(f"data/user_{steam_id}_soft_labels.csv")
    ground_truth_exists = os.path.exists(f"data/user_{steam_id}_ground_truth.csv")
    profile_exists = os.path.exists(f"data/user_{steam_id}_taste_profile.json")
    
    return {
        "steam_id": steam_id,
        "has_library": library_exists,
        "has_soft_labels": soft_labels_exists,
        "has_ground_truth": ground_truth_exists,
        "has_profile": profile_exists
    }

@app.get("/user/verify/{steam_id}")
def get_verification_data(steam_id: str):
    """Returns the combined soft labels and ground truth for the verification UI."""
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    sl_path = f"data/user_{steam_id}_soft_labels.csv"
    
    if not os.path.exists(sl_path):
        raise HTTPException(status_code=404, detail="User data not found. Please fetch first.")
    
    df_sl = pd.read_csv(sl_path)
    
    if os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)
        # 1. Identify games in GT that are NOT in SL (Manual Additions)
        manual_appids = df_gt[~df_gt['appid'].isin(df_sl['appid'])]['appid'].tolist()
        
        # 2. Get metadata for manual games
        manual_rows = []
        if manual_appids:
            metadata = data_manager.metadata
            m_data = metadata[metadata['appid'].isin(manual_appids)]
            for _, row in m_data.iterrows():
                gt_row = df_gt[df_gt['appid'] == row['appid']].iloc[0]
                manual_rows.append({
                    'appid': int(row['appid']),
                    'name': row['name'],
                    'playtime_forever': 0,
                    'predicted_rating': 5, # Default for manual
                    'actual_rating': gt_row['actual_rating'],
                    'ignore': bool(gt_row['ignore']),
                    'is_manual': True,
                    'is_nsfw': bool(row.get('is_nsfw', False))
                })
        
        # 3. Merge library games
        # Also need is_nsfw for library games
        metadata = data_manager.metadata
        df_sl = df_sl.merge(metadata[['appid', 'is_nsfw']], on='appid', how='left')
        
        # FILTER: Only show games with playtime in the verification UI
        df_sl_visible = df_sl[df_sl['has_playtime'] == True].copy()
        
        df = df_sl_visible.merge(df_gt[['appid', 'actual_rating', 'ignore']], on='appid', how='left')
        df['actual_rating'] = df['actual_rating'].fillna(df['predicted_rating'])
        df['ignore'] = df['ignore'].fillna(False)
        df['is_manual'] = False
        
        # 4. Combine
        final_list = df.to_dict(orient='records') + manual_rows
    else:
        # Initial fetch (no ground truth yet)
        # Add is_nsfw from metadata
        metadata = data_manager.metadata
        df = df_sl.merge(metadata[['appid', 'is_nsfw']], on='appid', how='left')
        
        # FILTER: Only show games with playtime in the verification UI
        df_visible = df[df['has_playtime'] == True].copy()
        
        df_visible['actual_rating'] = df_visible['predicted_rating']
        df_visible['ignore'] = False
        df_visible['is_manual'] = False
        final_list = df_visible.to_dict(orient='records')
        
    # JSON cannot handle NaN values, replace with None (null)
    # We do this on the list of dicts for safety
    def clean_dict(d):
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}
    
    return [clean_dict(item) for item in final_list]

@app.post("/user/verify")
def update_verification_data(updates: List[UserVerifyUpdate]):
    """Saves the user's manual ratings and ignore status."""
    if not updates: return {"message": "No updates provided"}
    
    steam_id = updates[0].steam_id
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    
    # Load existing or create new
    if os.path.exists(gt_path):
        df = pd.read_csv(gt_path)
    else:
        df = pd.DataFrame(columns=['appid', 'actual_rating', 'ignore'])
        
    for up in updates:
        # Update or add
        mask = df['appid'] == up.appid
        if mask.any():
            df.loc[mask, 'actual_rating'] = up.actual_rating
            df.loc[mask, 'ignore'] = up.ignore
        else:
            new_row = pd.DataFrame([{'appid': up.appid, 'actual_rating': up.actual_rating, 'ignore': up.ignore}])
            df = pd.concat([df, new_row], ignore_index=True)
            
    df.to_csv(gt_path, index=False)
    return {"message": "Verification data saved successfully"}

@app.post("/user/solve/{steam_id}")
async def solve_taste(steam_id: str):
    """Runs the regression solver to generate the taste profile."""
    try:
        logger.info(f"Solving Taste DNA for {steam_id}...")
        process = await asyncio.create_subprocess_exec(
            "python", "pipeline/solve_user_taste.py", steam_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            err_msg = stderr.decode().strip()
            logger.error(f"Solver failed for {steam_id}: {err_msg}")
            raise HTTPException(status_code=500, detail=f"Solver failed: {err_msg}")
            
        logger.info(f"Taste DNA solved for {steam_id}")
        return {"message": "Taste profile solved successfully"}
    except Exception as e:
        logger.exception(f"Unexpected error in solve_taste for {steam_id}")
        raise HTTPException(status_code=500, detail=f"Solver failed: {e}")

@app.get("/user/insights/{steam_id}")
def get_user_insights(steam_id: str):
    """Returns the solved weights and top/bottom tags for the insights view."""
    profile_path = f"data/user_{steam_id}_taste_profile.json"
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail="Taste profile not found.")
        
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    return profile

@app.get("/lists/{category}")
def get_list(category: str, discovery_pref: float = 0.0):
    """
    Returns top/bottom game lists for various categories (quality, length, popularity, age, difficulty).
    Used by the 'Lists' tab in the frontend.
    """
    logger.info(f"GET /lists/{category} called with discovery_pref={discovery_pref}")
    if data_manager.metadata is None:
        logger.warning("Data not loaded when /lists endpoint called")
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    # Check cache first
    cache_key = (category, discovery_pref)
    if cache_key in data_manager.lists_cache:
        logger.info(f"GET /lists/{category} cache HIT (discovery_pref={discovery_pref})")
        return data_manager.lists_cache[cache_key]
    
    logger.info(f"GET /lists/{category} cache MISS (discovery_pref={discovery_pref})")
    metadata = data_manager.metadata
    
    if category == "quality":
        logger.debug(f"Quality list request: quality_grid shape={data_manager.quality_grid.shape}")
        num_grid_rows = data_manager.quality_grid.shape[0]
        grid_index = int(round(((discovery_pref - (-1.0)) / 2.0) * (num_grid_rows - 1)))
        grid_index = max(0, min(num_grid_rows - 1, grid_index))
        scores = data_manager.quality_grid[grid_index]
        
        top_indices = np.argsort(-scores)[:50]
        bottom_indices = np.argsort(scores)[:50]
        
        def format_quality_list(indices):
            df = metadata.iloc[indices].copy()
            df['quality_score'] = scores[indices]
            return df[['appid', 'name', 'quality_score']].to_dict(orient='records')

        logger.info(f"Quality list: {len(top_indices)} top, {len(bottom_indices)} bottom")
        result = {
            "top": format_quality_list(top_indices),
            "bottom": format_quality_list(bottom_indices)
        }
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "length":
        logger.debug("Length list request")
        playtime_col = 'estimated_playtime'
        valid = metadata[metadata[playtime_col] > 0].copy()
        logger.info(f"Games with valid playtime: {len(valid)}")
        
        longest = valid.sort_values(playtime_col, ascending=False).head(50)
        shortest = valid.sort_values(playtime_col, ascending=True).head(50)
        
        if not longest.empty:
            logger.info(f"Longest: {longest.iloc[0]['name']} ({longest.iloc[0][playtime_col]:.1f} minutes)")
        if not shortest.empty:
            logger.info(f"Shortest: {shortest.iloc[0]['name']} ({shortest.iloc[0][playtime_col]:.1f} minutes)")
        
        result = {
            "top": longest[['appid', 'name', playtime_col]].rename(columns={playtime_col: 'playtime'}).to_dict(orient='records'),
            "bottom": shortest[['appid', 'name', playtime_col]].rename(columns={playtime_col: 'playtime'}).to_dict(orient='records')
        }
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "popularity":
        logger.debug("Popularity list request")
        if 'total_reviews' not in metadata.columns:
            metadata = metadata.copy()
            metadata['total_reviews'] = metadata['positive'] + metadata['negative']
        
        popular_df = metadata[metadata['total_reviews'] >= 1]
        logger.info(f"Games with >=1 review: {len(popular_df)}")
        most_pop = popular_df.sort_values('total_reviews', ascending=False).head(50)
        least_pop = popular_df.sort_values('total_reviews', ascending=True).head(50)
        
        if not most_pop.empty:
            logger.info(f"Most popular: {most_pop.iloc[0]['name']} ({most_pop.iloc[0]['total_reviews']:,} reviews)")
        if not least_pop.empty:
            logger.info(f"Least popular: {least_pop.iloc[0]['name']} ({least_pop.iloc[0]['total_reviews']:,} reviews)")
        
        result = {
            "top": most_pop[['appid', 'name', 'total_reviews']].to_dict(orient='records'),
            "bottom": least_pop[['appid', 'name', 'total_reviews']].to_dict(orient='records')
        }
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "age":
        logger.debug("Age list request")
        if os.path.exists(METADATA_FILE):
            build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s')
        else:
            build_time = pd.Timestamp.now()
            
        valid_dates = metadata[metadata['parsed_date'] <= build_time].copy()
        logger.info(f"Games released before {build_time}: {len(valid_dates)}")
        oldest = valid_dates.sort_values('parsed_date', ascending=True).head(50)
        newest = valid_dates.sort_values('parsed_date', ascending=False).head(50)
        
        result = {
            "top": newest[['appid', 'name', 'release_date']].to_dict(orient='records'),
            "bottom": oldest[['appid', 'name', 'release_date']].to_dict(orient='records')
        }
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "difficulty":
        logger.debug("Difficulty list request")
        hardest = metadata.sort_values('difficulty_predicted', ascending=False).head(50)
        easiest = metadata.sort_values('difficulty_predicted', ascending=True).head(50)
        
        logger.info(f"Difficulty range: hardest={hardest.iloc[0]['difficulty_predicted'] if not hardest.empty else 'N/A'}, easiest={easiest.iloc[0]['difficulty_predicted'] if not easiest.empty else 'N/A'}")
        
        # Tag predictors
        tag_impacts = []
        pred_file = DIFFICULTY_PREDICTIONS_FILE
        logger.info(f"Looking for difficulty predictions at: {pred_file}")
        if os.path.exists(pred_file):
            logger.info(f"Found predictions file. Loading...")
            try:
                pred_df = pd.read_csv(pred_file)
                contrib_cols = [c for c in pred_df.columns if c.startswith('contrib_')]
                logger.info(f"Found {len(contrib_cols)} contribution columns")
                if contrib_cols:
                    for col in contrib_cols:
                        tag_name = col.replace('contrib_', '').replace('_', ' ').title()
                        impact = pred_df[pred_df[col] != 0][col].mean()
                        if not np.isnan(impact):
                            tag_impacts.append({'tag': tag_name, 'impact': float(impact)})
            except Exception as e:
                logger.error(f"Error reading difficulty predictions: {e}")
        else:
            logger.warning(f"Difficulty predictions file NOT FOUND at {pred_file}")
        
        logger.info(f"Returning {len(tag_impacts)} tag impacts")
        result = {
            "top": hardest[['appid', 'name', 'difficulty_predicted']].to_dict(orient='records'),
            "bottom": easiest[['appid', 'name', 'difficulty_predicted']].to_dict(orient='records'),
            "tag_impacts": sorted(tag_impacts, key=lambda x: x['impact'], reverse=True)
        }
        data_manager.lists_cache[cache_key] = result
        return result

    raise HTTPException(status_code=400, detail="Invalid category")

@app.post("/metadata")
def get_metadata(request: MetadataRequest):
    """Returns metadata for a specific list of game names."""
    logger.info(f"POST /metadata called with {len(request.names)} game names")
    if data_manager.metadata is None:
        logger.error("Data not loaded when /metadata called")
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    logger.debug(f"Requested names: {request.names}")
    metadata = data_manager.metadata
    matches = metadata[metadata['name'].isin(request.names)]
    logger.info(f"Found {len(matches)} matches out of {len(request.names)} requested")
    
    response_items = []
    for _, game_meta in matches.iterrows():
        raw_pop = game_meta['positive'] + game_meta['negative']
        raw_length = game_meta['estimated_playtime'] / 60.0
        
        item = {
            "appid": int(game_meta['appid']),
            "name": str(game_meta['name']),
            "release_date": str(game_meta['release_date']),
            "short_description": str(game_meta['short_description']) if 'short_description' in game_meta and pd.notna(game_meta['short_description']) else "",
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'],
            "tags": game_meta['tags'],
            "is_nsfw": bool(game_meta['is_nsfw']),
            "raw_pop": int(raw_pop),
            "raw_length": float(raw_length)
        }
        response_items.append(item)
    logger.info(f"/metadata returning {len(response_items)} items")
    return response_items

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    logger.info("POST /recommend called")
    if data_manager.metadata is None:
        logger.error("Data not loaded when /recommend called")
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    logger.debug(f"Request params: alpha={request.alpha:.3f}, beta={request.beta:.3f}, quality_pref={request.quality_pref:.3f}, "
                 f"prompt='{request.prompt}', seed_games={request.seed_games}, genres={request.genres}")
    
    metadata = data_manager.metadata
    
    # Identify seeds
    seed_indices = np.where(metadata['name'].isin(request.seed_games))[0]
    logger.info(f"Seed games: requested={len(request.seed_games)}, found={len(seed_indices)}")
    seed_appids = metadata.iloc[seed_indices]['appid'].tolist()
    
    # 1. Filtering
    mask = np.ones(len(metadata), dtype=bool)
    initial_count = np.sum(mask)
    
    if request.remove_vr:
        mask &= ~metadata['is_vr_only'].values
        vr_removed = initial_count - np.sum(mask)
        logger.debug(f"VR filter removed {vr_removed} games")
        
    if request.english_only:
        mask &= metadata['is_english'].values
        eng_removed = initial_count - np.sum(mask) if request.remove_vr else np.sum(~mask)
        logger.debug(f"English filter removed {eng_removed} games")
    
    if request.remove_utilities:
        mask &= ~metadata['is_utility'].values
        util_removed = initial_count - np.sum(mask)
        logger.debug(f"Utilities filter removed {util_removed} games")

    if request.remove_unreleased:
        if os.path.exists(METADATA_FILE):
            build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s')
            future_mask = (metadata['parsed_date'] > build_time).fillna(False)
            mask &= ~future_mask.values
            unreleased_removed = initial_count - np.sum(mask)
            logger.debug(f"Unreleased filter removed {unreleased_removed} games")

    if request.genres:
        genre_mask = np.zeros(len(metadata), dtype=bool)
        for genre in request.genres:
            escaped_genre = re.escape(genre)
            genre_mask |= metadata['genres'].fillna('').astype(str).str.contains(escaped_genre, regex=True, case=False).values
        mask &= genre_mask
        genre_kept = np.sum(mask)
        logger.debug(f"Genre filter ({request.genres}): {genre_kept} games remaining")

    keep_indices = np.where(mask)[0]
    logger.info(f"After filtering: {len(keep_indices)} games remaining (from {len(metadata)} total)")
    
    if len(keep_indices) == 0:
        logger.warning("No games passed filters!")
        return []

    # 2. Semantic Component
    all_semantic_sims = np.zeros(len(metadata))
    
    if request.prompt or seed_appids:
        if request.prompt:
            logger.debug(f"Encoding prompt: '{request.prompt}'")
            prompt_vec = data_manager.model.encode([request.prompt])[0]
            logger.debug(f"Prompt vector shape: {prompt_vec.shape}")
            
            p_desc_centered = (prompt_vec - data_manager.mean_desc) if data_manager.mean_desc is not None else prompt_vec
            p_struct_centered = (prompt_vec - data_manager.mean_structural) if data_manager.mean_structural is not None else prompt_vec
            
            p_desc = np.dot(p_desc_centered, data_manager.w_desc) if data_manager.w_desc is not None else p_desc_centered
            p_struct = np.dot(p_struct_centered, data_manager.w_structural) if data_manager.w_structural is not None else p_struct_centered
            
            def norm_vec(v):
                mag = np.linalg.norm(v)
                return v / (mag if mag > EPSILON else 1.0)
            
            p_desc_norm = norm_vec(p_desc)
            p_struct_norm = norm_vec(p_struct)
            
            logger.debug("Computing prompt similarities...")
            prompt_desc_sims = np.dot(data_manager.embeddings_desc_norm, p_desc_norm)
            prompt_structural_sims = np.dot(data_manager.embeddings_structural_norm, p_struct_norm)
            
            all_semantic_sims = (prompt_desc_sims + prompt_structural_sims) * SEMANTIC_PROMPT_SEED_BLEND
            logger.debug(f"Prompt similarities computed. Desc range: [{prompt_desc_sims.min():.4f}, {prompt_desc_sims.max():.4f}], "
                         f"Structural range: [{prompt_structural_sims.min():.4f}, {prompt_structural_sims.max():.4f}]")

        if seed_indices.size > 0:
            logger.info(f"Computing seed-based semantic similarity with {len(seed_indices)} seed(s)")
            seed_desc_vecs = data_manager.embeddings_desc_norm[seed_indices]
            avg_seed_desc = np.mean(seed_desc_vecs, axis=0)
            sd_mag = np.linalg.norm(avg_seed_desc)
            sd_norm = avg_seed_desc / (sd_mag if sd_mag > EPSILON else 1.0)
            seed_desc_sims = np.dot(data_manager.embeddings_desc_norm, sd_norm)
            
            seed_structural_vecs = data_manager.embeddings_structural_norm[seed_indices]
            avg_seed_structural = np.mean(seed_structural_vecs, axis=0)
            ss_mag = np.linalg.norm(avg_seed_structural)
            ss_norm = avg_seed_structural / (ss_mag if ss_mag > EPSILON else 1.0)
            seed_structural_sims = np.dot(data_manager.embeddings_structural_norm, ss_norm)
            
            seed_combined_sims = (seed_desc_sims + seed_structural_sims) * SEMANTIC_PROMPT_SEED_BLEND
            
            if request.prompt:
                all_semantic_sims = (all_semantic_sims + seed_combined_sims) * SEMANTIC_PROMPT_SEED_BLEND
                logger.debug("Combined prompt and seed similarities")
            else:
                all_semantic_sims = seed_combined_sims
                logger.debug("Used seed-only similarities")

    all_tag_sims = np.zeros(len(metadata))
    if seed_indices.size > 0 or request.vibe_vector:
        logger.debug("Computing unified tag scoring (Linear Mode)...")
        
        # 1. Calculate Seed Beta (if any)
        # beta_seed = mean( V / (||V|| + lambda) )
        if seed_indices.size > 0:
            tag_seed_vectors = data_manager.tag_vectors[seed_indices].astype(np.float32)
            seed_norms = data_manager.tag_vectors_norms[seed_indices].astype(np.float32)
            
            # Penalized normalization for each seed
            penalized_seeds = tag_seed_vectors / (seed_norms[:, None] + DOT_PRODUCT_LAMBDA)
            beta_seed_unit = np.mean(penalized_seeds, axis=0)
            # Apply the user's manual Tag Match slider as the absolute weight for seeds
            beta_seed = beta_seed_unit * request.beta
        else:
            beta_seed = None

        # 2. Calculate DNA Beta (if any)
        # request.vibe_vector contains the UNIT coefficients.
        # We restore the absolute coefficients by multiplying by 'tag_match' (the norm).
        if request.vibe_vector:
            beta_dna_unit = np.array(request.vibe_vector, dtype=np.float32)
            # Use the absolute weight from metadata if available (the norm of the solver's tag coeffs)
            w_tag_dna = request.metadata_weights.get('tag_match', 1.0) if request.metadata_weights else 1.0
            beta_dna = beta_dna_unit * w_tag_dna
            logger.info(f"Loaded DNA Beta: Unit norm={np.linalg.norm(beta_dna_unit):.4f}, Scale={w_tag_dna:.4f}")
        else:
            beta_dna = None

        # 3. Select or Blend Betas
        if beta_seed is not None and beta_dna is not None:
            # If both exist, we blend them. 
            combined_beta = (beta_seed + beta_dna) / 2.0
            logger.info("Blended Seed Beta and DNA Beta")
        elif beta_seed is not None:
            combined_beta = beta_seed
            logger.info("Using Seed Beta only")
        else:
            combined_beta = beta_dna
            logger.info("Using DNA Beta only")

        # 4. Final Scoring: dot(U_scaled, combined_beta)
        # U_scaled = (U / (||U|| + lambda)) * TAG_GLOBAL_SCALING_FACTOR
        dot_products = np.dot(data_manager.tag_vectors, combined_beta)
        denom = data_manager.tag_vectors_norms + DOT_PRODUCT_LAMBDA
        
        # Note: We multiply by the scaling factor to match the solver's feature space
        all_tag_sims = (dot_products / denom) * TAG_GLOBAL_SCALING_FACTOR
        logger.info(f"Unified Tag Scorer: Range=[{all_tag_sims.min():.4f}, {all_tag_sims.max():.4f}]")

    semantic_sims = all_semantic_sims[keep_indices]
    tag_sims = all_tag_sims[keep_indices]

    # 4. Rating Component
    num_grid_rows = data_manager.quality_grid.shape[0]
    grid_index = int(round(((request.disc_pref - (-1.0)) / 2.0) * (num_grid_rows - 1)))
    grid_index = max(0, min(num_grid_rows - 1, grid_index))
    logger.debug(f"Quality grid index: {grid_index} (discovery={request.disc_pref:.3f})")
    
    z_spps = np.clip(data_manager.quality_grid[grid_index][keep_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    z_date = np.clip(metadata.iloc[keep_indices]['date_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_pop = np.clip(metadata.iloc[keep_indices]['pop_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_length = np.clip(metadata.iloc[keep_indices]['playtime_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_difficulty = np.clip(metadata.iloc[keep_indices]['difficulty_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

    # Initialize variables for the response to prevent UnboundLocalError
    z_semantic = np.zeros(len(keep_indices))
    z_tag = tag_sims # In linear mode, tag_sims IS the feature. In recommender mode, it's the dot product.

    # 5. Unified Linear Scoring
    # All weights (alpha, beta, quality_pref, etc.) are now ABSOLUTE contributions 
    # (rating points per SD of the feature) to ensure consistent scale between modes.
    
    # 5.1 Determine Mode
    # Pure Linear Mode: Used when vibe_vector is provided (Taste DNA import)
    # Recommender Mode: Used for manual slider tuning
    expected_tag_dim = data_manager.tag_vectors.shape[1]
    is_vibe_present = request.vibe_vector is not None and len(request.vibe_vector) > 0
    is_vibe_valid = is_vibe_present and (len(request.vibe_vector) == expected_tag_dim)
    
    is_linear_mode = is_vibe_valid
    
    if is_vibe_present and not is_vibe_valid:
        logger.warning(f"Vibe vector dimension mismatch: received {len(request.vibe_vector)}, expected {expected_tag_dim}. Profile may be outdated. Falling back to manual mode.")
    
    if is_linear_mode:
        # PURE LINEAR MODE: Match solve_user_taste.py exactly using unified utility
        logger.info("Using Pure Linear Mode (Taste DNA) via unified scoring path")
        
        # Use the pre-calculated combined_beta (which already blends DNA and Seeds if present)
        # Note: combined_beta was calculated at line ~995
        beta_to_use = combined_beta
        
        # Use unified utility for scoring
        final_scores_all = calculate_linear_scores(
            z_quality=data_manager.quality_grid[grid_index],
            z_date=metadata['date_z'].values,
            z_pop=metadata['pop_z'].values,
            z_playtime=metadata['playtime_z'].values,
            z_difficulty=metadata['difficulty_z'].values,
            tag_vectors=data_manager.tag_vectors,
            tag_norms=data_manager.tag_vectors_norms,
            beta_tag=beta_to_use,
            weights=request.metadata_weights or {},
            intercept=request.intercept or 0.0,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
            dot_product_lambda=DOT_PRODUCT_LAMBDA,
            z_clamp_min=Z_SCORE_CLAMP_MIN,
            z_clamp_max=Z_SCORE_CLAMP_MAX
        )
        
        # Filter to keep_indices
        final_scores = final_scores_all[keep_indices]
        
        # Set weights for total_weight/sorting logic
        # If seeds are present, we use the average of the profile weight and the user's manual tag weight
        w_tag_dna = request.metadata_weights.get('tag_match', 1.0) if request.metadata_weights else 1.0
        w_tag = (w_tag_dna + request.beta) / 2.0 if seed_indices.size > 0 else w_tag_dna
        
        w_quality = request.metadata_weights.get('quality', 0.0) if request.metadata_weights else 0.0
        w_date = request.metadata_weights.get('age', 0.0) if request.metadata_weights else 0.0
        w_pop = request.metadata_weights.get('popularity', 0.0) if request.metadata_weights else 0.0
        w_length = request.metadata_weights.get('length', 0.0) if request.metadata_weights else 0.0
        w_difficulty = request.metadata_weights.get('difficulty', 0.0) if request.metadata_weights else 0.0
        w_semantic = request.alpha
        
        # If there's a prompt or seeds, we treat it as a temporary z-scored offset
        if request.prompt or seed_appids:
            z_semantic = np.clip(to_z(semantic_sims), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
            final_scores += (z_semantic * request.alpha)
            
    else:
        # RECOMMENDER MODE: Similarity-based weighting
        w_semantic = request.alpha
        w_tag = request.beta
        w_quality = request.quality_pref
        w_date = request.age_pref
        w_pop = request.pop_pref
        w_length = request.length_pref
        w_difficulty = request.difficulty_pref

        logger.debug(f"Weights (Absolute): semantic={w_semantic:.2f}, tag={w_tag:.2f}, quality={w_quality:.2f}, "
                    f"age={w_date:.2f}, pop={w_pop:.2f}, length={w_length:.2f}, difficulty={w_difficulty:.2f}")

        # Process Semantic Component
        if request.prompt or seed_appids:
            z_semantic = np.clip(to_z(semantic_sims), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
            
        # Process Tag Component
        z_tag = np.clip(to_z(tag_sims), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

        final_scores = (
            (z_semantic * w_semantic) +
            (z_tag * w_tag) +
            (z_spps * w_quality) +
            (z_date * w_date) +
            (z_pop * w_pop) +
            (z_length * w_length) +
            (z_difficulty * w_difficulty)
        )

    logger.debug(f"Final scores: min={final_scores.min():.3f}, max={final_scores.max():.3f}, mean={final_scores.mean():.3f}")

    # Exclude seeds
    meta_filt = metadata.iloc[keep_indices].copy()
    if seed_appids:
        seed_mask = meta_filt['appid'].isin(seed_appids)
        seeds_excluded = np.sum(seed_mask)
        final_scores[seed_mask] = -1e12
        logger.debug(f"Excluded {seeds_excluded} seed games from results")

    # Exclude profile games (Library/Rated)
    if request.profile_filter != "none":
        exclude_appids = []
        if request.profile_filter == "all":
            exclude_appids = request.library_appids or []
            logger.info(f"Filtering ALL profile games ({len(exclude_appids)})")
        elif request.profile_filter == "rated":
            exclude_appids = request.rated_appids or []
            logger.info(f"Filtering RATED profile games ({len(exclude_appids)})")
            
        if exclude_appids:
            profile_mask = meta_filt['appid'].isin(exclude_appids).values
            profile_excluded = np.sum(profile_mask)
            if profile_excluded > 0:
                # Set scores for excluded games to a very low value
                final_scores[profile_mask] = -1e12
                
                # Log a few examples of excluded games for debugging
                excluded_names = meta_filt[profile_mask]['name'].head(5).tolist()
                logger.info(f"Excluded {profile_excluded} profile games (Mode: {request.profile_filter}). Examples: {excluded_names}")
            else:
                logger.debug(f"Profile Filter ({request.profile_filter}): No matches found in current results.")

    # 6. Sorting and Result Formatting
    total_weight = abs(w_semantic) + abs(w_tag) + abs(w_quality) + abs(w_date) + abs(w_pop) + abs(w_length) + abs(w_difficulty)
    
    if total_weight < EPSILON:
        # All weights zero: alphabetical fallback
        if seed_appids:
            non_seed_mask = ~seed_mask
        else:
            non_seed_mask = np.ones(len(meta_filt), dtype=bool)
        if np.any(non_seed_mask):
            positions = np.where(non_seed_mask)[0]
            non_seed_names = meta_filt['name'].fillna("").values[non_seed_mask]
            sorted_name_positions = positions[np.argsort(non_seed_names)]
            top_indices = sorted_name_positions[:request.top_k]
        else:
            top_indices = []
    else:
        num_to_extract = min(len(final_scores), request.top_k * TOP_K_SORT_MULTIPLIER)
        if num_to_extract > 0:
            partitioned_indices = np.argpartition(-final_scores, num_to_extract-1)[:num_to_extract]
            subset_scores = final_scores[partitioned_indices]
            subset_names = meta_filt['name'].fillna("").values[partitioned_indices]
            subset_sorted_indices = np.lexsort((subset_names, -subset_scores))
            top_indices = partitioned_indices[subset_sorted_indices[:request.top_k]]
        else:
            top_indices = []

    results = meta_filt.iloc[top_indices].copy()
    
    response_items = []
    
    for i, idx in enumerate(top_indices):
        game_meta = results.iloc[i]
        
        raw_pop = game_meta['positive'] + game_meta['negative']
        raw_length = game_meta['estimated_playtime'] / 60.0
        
        desc = ""
        if 'short_description' in game_meta and pd.notna(game_meta['short_description']):
            desc = str(game_meta['short_description'])
        
        item = {
            "appid": int(game_meta['appid']),
            "name": str(game_meta['name']),
            "release_date": str(game_meta['release_date']),
            "short_description": desc,
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'], 
            "tags": game_meta['tags'],     
            "is_nsfw": bool(game_meta['is_nsfw']),
            
            "weighted_score": float(final_scores[idx]),
            "semantic_match": float(semantic_sims[idx]),
            "tag_match": float(tag_sims[idx]),
            "rating": float(z_spps[idx]), 
            
            "z_semantic": float(z_semantic[idx]),
            "w_semantic": float(w_semantic),
            "z_tag": float(tag_sims[idx]), # tag_sims IS the linear feature
            "w_tag": float(w_tag),
            "z_spps": float(z_spps[idx]),
            "w_spps": float(w_quality),
            "z_date": float(z_date[idx]),
            "w_date": float(w_date),
            "z_pop": float(z_pop[idx]),
            "w_pop": float(w_pop),
            "z_length": float(z_length[idx]),
            "w_length": float(w_length),
            "z_difficulty": float(z_difficulty[idx]),
            "w_difficulty": float(w_difficulty),
            
            "raw_date": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "raw_pop": int(raw_pop),
            "raw_length": float(raw_length),
            "raw_difficulty": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0
        }
        response_items.append(item)
        
    logger.info(f"/recommend returning {len(response_items)} results")
    return response_items

    results = meta_filt.iloc[top_indices].copy()
    
    response_items = []
    
    for i, idx in enumerate(top_indices):
        game_meta = results.iloc[i]
        
        raw_pop = game_meta['positive'] + game_meta['negative']
        raw_length = game_meta['estimated_playtime'] / 60.0
        
        desc = ""
        if 'short_description' in game_meta and pd.notna(game_meta['short_description']):
            desc = str(game_meta['short_description'])
        
        item = {
            "appid": int(game_meta['appid']),
            "name": str(game_meta['name']),
            "release_date": str(game_meta['release_date']),
            "short_description": desc,
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'], 
            "tags": game_meta['tags'],     
            "is_nsfw": bool(game_meta['is_nsfw']),
            
            "weighted_score": float(final_scores[idx]),
            "semantic_match": float(semantic_sims[idx]),
            "tag_match": float(tag_sims[idx]),
            "rating": float(z_spps[idx]), 
            
            "z_semantic": float(z_semantic[idx]),
            "w_semantic": float(w_semantic),
            "z_tag": float(z_tag[idx]),
            "w_tag": float(w_tag),
            "z_spps": float(z_spps[idx]),
            "w_spps": float(w_spps),
            "z_date": float(z_date[idx]),
            "w_date": float(w_date),
            "z_pop": float(z_pop[idx]),
            "w_pop": float(w_pop),
            "z_length": float(z_length[idx]),
            "w_length": float(w_length),
            "z_difficulty": float(z_difficulty[idx]),
            "w_difficulty": float(w_difficulty),
            
            "raw_date": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "raw_pop": int(raw_pop),
            "raw_length": float(raw_length),
            "raw_difficulty": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0
        }
        response_items.append(item)
        
    logger.info(f"/recommend returning {len(response_items)} results")
    return response_items

# --- Static File Serving ---
# Serve the React frontend build files if they exist
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist'))
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # If the path looks like a file (has an extension), try to serve it from dist
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Otherwise, serve index.html for SPA routing (React Router etc.)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    logger.warning(f"Frontend dist directory not found at {frontend_dist}. API only mode.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
            
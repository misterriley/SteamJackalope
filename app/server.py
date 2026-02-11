from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer # Moved to lazy load
import os
import sys
import re
import ast
import gc
import logging

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
    NSFW_NAME_PATTERNS
)
from common.utils import to_z, calculate_hybrid_score

app = FastAPI()

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
        self._model = None
        self.all_genres = []
        self.lists_cache = {}

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading SentenceTransformer model (Lazy Load)...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                MODEL_NAME,
                backend=SENTENCE_TRANSFORMER_BACKEND,
                model_kwargs=SENTENCE_TRANSFORMER_MODEL_KWARGS
            )
            logger.info("SentenceTransformer model loaded successfully")
        return self._model

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
        needed_cols = [
            'appid', 'name', 'release_date', 'positive', 'negative', 
            'genres', 'tags', 'categories', 'supported_languages',
            'mature_content', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z',
            'estimated_playtime', 'difficulty_predicted'
        ]
        if 'release_year' in pd.read_parquet(METADATA_FILE, columns=[]).columns:
            needed_cols.append('release_year')

        try:
            self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols, dtype_backend='pyarrow')
            logger.info("Metadata loaded with pyarrow backend")
        except Exception as e:
            logger.warning(f"Failed to load with pyarrow backend: {e}. Falling back to standard.")
            self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols)
        
        logger.info("Extracting boolean features...")
        self.metadata['is_vr_only'] = self.metadata['categories'].fillna('').astype(str).str.contains('VR Only', case=False).astype(bool)
        self.metadata['is_english'] = self.metadata['supported_languages'].fillna('').astype(str).str.contains('English', case=False).astype(bool)
        self.metadata['is_utility'] = self.metadata['tags'].fillna('').astype(str).str.contains('Utilities', case=False).astype(bool)
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

# --- Endpoints ---

@app.get("/genres")
def get_genres():
    """Returns a list of all unique genres available in the dataset."""
    logger.info("GET /genres called")
    genres = data_manager.all_genres
    logger.debug(f"Returning {len(genres)} genres")
    return genres

@app.get("/games")
def get_games():
    """Returns a list of all available game names for the seed selector."""
    logger.info("GET /games called")
    if data_manager.metadata is None:
        logger.warning("Data not loaded yet when /games called")
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    games = sorted(data_manager.metadata['name'].fillna("Unknown").tolist())
    logger.debug(f"Returning {len(games)} game names")
    return games

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
        pred_file = "data/difficulty_predictions.csv"
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
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'],
            "tags": game_meta['tags'],
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
    
    if request.remove_nsfw:
        mask &= (metadata['mature_content'] == 0).values
        nsfw_removed = initial_count - np.sum(mask)
        logger.debug(f"NSFW filter removed {nsfw_removed} games")
        
        tags = metadata['tags'].fillna('')
        nsfw_tag_mask = tags.apply(lambda x: any(tag in str(x).lower() for tag in NSFW_TAGS) if x else False).values
        mask &= ~nsfw_tag_mask
        nsfw_tag_removed = initial_count - np.sum(mask)
        logger.debug(f"NSFW tag filter removed additional {nsfw_tag_removed - nsfw_removed} games")
        
        nsfw_name_mask = metadata['name'].fillna('').str.contains('|'.join(NSFW_NAME_PATTERNS), case=False, na=False).values
        mask &= ~nsfw_name_mask
        nsfw_name_removed = initial_count - np.sum(mask)
        logger.debug(f"NSFW name filter removed additional {nsfw_name_removed - nsfw_tag_removed} games")
    
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
    if seed_indices.size > 0:
        logger.debug("Computing tag similarity...")
        tag_seed_vectors = data_manager.tag_vectors[seed_indices]
        combined_tag_query = np.mean(tag_seed_vectors, axis=0)
        tag_q_mag = np.linalg.norm(combined_tag_query)
        
        dot_products = np.dot(data_manager.tag_vectors, combined_tag_query)
        denom = (data_manager.tag_vectors_norms * tag_q_mag) + DOT_PRODUCT_LAMBDA
        denom[denom == 0] = EPSILON
        
        all_tag_sims = dot_products / denom
        logger.info(f"Tag similarities computed: min={all_tag_sims.min():.4f}, max={all_tag_sims.max():.4f}, mean={all_tag_sims.mean():.4f}")

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

    # 5. Hybrid Scoring
    gamma = QUALITY_WEIGHT_MULTIPLIER * request.quality_pref
    
    w_semantic = request.alpha if (request.prompt or seed_appids) else 0.0
    w_tag = request.beta if seed_appids else 0.0
    w_spps = gamma
    w_date = AGE_WEIGHT_MULTIPLIER * request.age_pref
    w_pop = POPULARITY_WEIGHT_MULTIPLIER * request.pop_pref
    w_length = LENGTH_WEIGHT_MULTIPLIER * request.length_pref
    w_difficulty = DIFFICULTY_WEIGHT_MULTIPLIER * request.difficulty_pref

    logger.debug(f"Weights: semantic={w_semantic:.2f}, tag={w_tag:.2f}, quality={w_spps:.2f}, "
                 f"age={w_date:.2f}, pop={w_pop:.2f}, length={w_length:.2f}, difficulty={w_difficulty:.2f}")

    z_semantic = np.clip(to_z(semantic_sims), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) if (request.prompt or seed_appids) else np.zeros(len(keep_indices))
    z_tag = np.clip(to_z(tag_sims, ignore_zeros=True), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) if seed_appids else np.zeros(len(keep_indices))

    logger.debug(f"Z-scores before hybrid: semantic mean={z_semantic.mean():.3f} (nz={np.sum(z_semantic != 0)}), "
                 f"tag mean={z_tag.mean():.3f} (nz={np.sum(z_tag != 0)}), "
                 f"quality mean={z_spps.mean():.3f}")
    
    # NEW: Log quality weight and z-score distribution to diagnose slider issues
    # Use float64 to avoid overflow in mean/std with float16 arrays
    mean_f64 = np.mean(z_spps, dtype=np.float64)
    std_f64 = np.std(z_spps, dtype=np.float64)
    logger.info(f"QUALITY ANALYSIS: quality_pref={request.quality_pref:.3f}, w_spps={w_spps:.3f}, "
                f"z_spps range=[{z_spps.min():.3f}, {z_spps.max():.3f}], mean={mean_f64:.3f}, "
                f"std={std_f64:.3f}")
    # Log the partial contributions for top 10 candidates to see if quality is making a difference
    if len(keep_indices) > 0 and request.top_k > 0:
        num_to_show = min(10, len(keep_indices))
        logger.info(f"Top {num_to_show} candidates quality contribution (z_spps * w_spps):")
        for i in range(num_to_show):
            idx = i  # keep_indices are already in order, we'll sort later
            q_contrib = z_spps[idx] * w_spps
            logger.info(f"  Candidate {idx}: z_spps={z_spps[idx]:.3f}, w_spps={w_spps:.3f} → contrib={q_contrib:.3f}")

    final_scores = calculate_hybrid_score(
        z_semantic, w_semantic,
        z_tag, w_tag,
        z_spps, w_spps,
        z_date, w_date,
        z_pop, w_pop,
        z_length, w_length,
        z_difficulty, w_difficulty
    )

    logger.debug(f"Final scores: min={final_scores.min():.3f}, max={final_scores.max():.3f}, mean={final_scores.mean():.3f}")

    # Exclude seeds
    meta_filt = metadata.iloc[keep_indices].copy()
    if seed_appids:
        seed_mask = meta_filt['appid'].isin(seed_appids)
        seeds_excluded = np.sum(seed_mask)
        final_scores[seed_mask] = -1e12
        logger.debug(f"Excluded {seeds_excluded} seed games from results")

    # 6. Sorting and Result Formatting
    num_to_extract = min(len(final_scores), request.top_k * TOP_K_SORT_MULTIPLIER)
    logger.info(f"Extracting top {request.top_k} from {num_to_extract} candidates")
    
    if num_to_extract > 0:
        partitioned_indices = np.argpartition(-final_scores, num_to_extract-1)[:num_to_extract]
        subset_scores = final_scores[partitioned_indices]
        subset_names = meta_filt['name'].fillna("").values[partitioned_indices]
        subset_sorted_indices = np.lexsort((subset_names, -subset_scores))
        top_indices = partitioned_indices[subset_sorted_indices[:request.top_k]]
        
        if len(top_indices) > 0:
            top_game = meta_filt.iloc[top_indices[0]]
            logger.info(f"Top recommendation: {top_game['name']} (score={final_scores[top_indices[0]]:.3f})")
    else:
        top_indices = []
        logger.warning("No candidates to return")

    results = meta_filt.iloc[top_indices].copy()
    
    response_items = []
    
    for i, idx in enumerate(top_indices):
        game_meta = results.iloc[i]
        
        raw_pop = game_meta['positive'] + game_meta['negative']
        raw_length = game_meta['estimated_playtime'] / 60.0
        
        item = {
            "appid": int(game_meta['appid']),
            "name": str(game_meta['name']),
            "release_date": str(game_meta['release_date']),
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'], 
            "tags": game_meta['tags'],     
            
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
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
import unicodedata
from scipy.stats import norm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Windows Compatibility: ProactorEventLoop is required for subprocesses
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
    EMBEDDINGS_DESC_NORMS_FILE,
    QUALITY_GRID_FILE,
    W_DESC_FILE,
    MEAN_DESC_FILE,
    TAG_VECTORS_FILE,
    DIFFUSED_VERB_PROFILES_FILE,
    TAG_NORMS_FILE,
    METADATA_FILE,
    TRENDING_APPIDS_FILE,
    DOT_PRODUCT_LAMBDA,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
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
    DIFFICULTY_PREDICTIONS_FILE,
    DNA_UI_SCALING_FACTOR,
    ROOT_DIR,
    PRODUCTION_DATA_DIR,
    TAG_GLOBAL_SCALING_FACTOR,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    SOFTMIN_TEMPERATURE,
    SEMANTIC_SIMILARITY_MEAN,
    SEMANTIC_SIMILARITY_STD,
    TOPIC_DISTRIBUTIONS_FILE,
    TOPIC_MODEL_FILE,
    TOPIC_GLOBAL_SCALING_FACTOR,
    TOPIC_DOT_PRODUCT_LAMBDA
)
from common.utils import (
    to_z, calculate_hybrid_score, calculate_linear_scores, calculate_personalized_quality, 
    softmin_blend, fast_jsd_similarity, calculate_jackalope_kernel, normalize_string, 
    clean_release_date, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS
)

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
        self.metadata = None
        self.tag_vectors = None
        self.verb_profiles = None
        self.quality_grid = None
        self.tag_vectors_norms = None
        self.topic_distributions = None
        self.topic_means = None
        self.topic_stds = None
        self.topic_model = None
        self.embeddings_graph = None
        self.w_desc = None
        self.mean_desc = None
        self.model = None
        self.all_genres = []
        self.all_tags = []
        self.trending_names = []
        self.term_links = {}
        self.tag_dimension_descriptions = {}
        self.lists_cache = {}
        self.appid_to_idx = {}

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

    def load_data(self):
        logger.info("Starting data load...")
        import pyarrow.parquet as pq
        
        # 1. Use Memory Mapping for large NumPy arrays
        logger.info(f"Loading embeddings_desc_norm from {EMBEDDINGS_DESC_FILE}")
        self.embeddings_desc_norm = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        logger.info(f"Embeddings_desc_norm: shape={self.embeddings_desc_norm.shape}, dtype={self.embeddings_desc_norm.dtype}")

        # 2. Memory Map Tag Vectors and Quality Grid
        logger.info(f"Loading tag_vectors from {TAG_VECTORS_FILE}")
        self.tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
        logger.info(f"Tag vectors: shape={self.tag_vectors.shape}, dtype={self.tag_vectors.dtype}")
        
        logger.info(f"Loading verb_profiles from {DIFFUSED_VERB_PROFILES_FILE}")
        self.verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
        logger.info(f"Verb profiles: shape={self.verb_profiles.shape}, dtype={self.verb_profiles.dtype}")
        
        logger.info(f"Loading quality_grid from {QUALITY_GRID_FILE}")
        self.quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
        logger.info(f"Quality grid: shape={self.quality_grid.shape}, dtype={self.quality_grid.dtype}")
        
        # 2.5 Load Topic Modeling Artifacts
        if os.path.exists(TOPIC_DISTRIBUTIONS_FILE):
            logger.info(f"Loading topic_distributions from {TOPIC_DISTRIBUTIONS_FILE}")
            self.topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
            logger.info(f"Topic distributions: shape={self.topic_distributions.shape}")
        else:
            logger.warning(f"Topic distributions file NOT FOUND at {TOPIC_DISTRIBUTIONS_FILE}")

        # Load Topic Means and Stds
        means_path = os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")
        stds_path = os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")
        if os.path.exists(means_path) and os.path.exists(stds_path):
            logger.info(f"Loading topic means/stds from {PRODUCTION_DATA_DIR}")
            self.topic_means = np.load(means_path).astype(np.float32)
            self.topic_stds = np.load(stds_path).astype(np.float32)
        else:
            logger.warning("Topic means/stds NOT FOUND. Falling back to zeros/ones.")
            
        # Load Tone Z-scores
        tone_path = os.path.join(PRODUCTION_DATA_DIR, "tone_z.npy")
        if os.path.exists(tone_path):
            logger.info(f"Loading tone_z from {tone_path}")
            self.tone_z = np.load(tone_path, mmap_mode='r')
        else:
            logger.warning("Tone Z-scores file NOT FOUND. Will attempt to pull from metadata.")
            self.tone_z = None
            # We'll initialize these later if needed during recommend, but safer to warn now.
            
        if os.path.exists(TOPIC_MODEL_FILE):
            logger.info(f"Loading topic_model from {TOPIC_MODEL_FILE}")
            import pickle
            with open(TOPIC_MODEL_FILE, "rb") as f:
                self.topic_model = pickle.load(f)
            logger.info("Topic model loaded successfully")
        else:
            logger.warning(f"Topic model file NOT FOUND at {TOPIC_MODEL_FILE}")

        # 2.6 Load Behavioral Graph Embeddings (v7.0)
        graph_path = os.path.join(PRODUCTION_DATA_DIR, "embeddings_graph.npy")
        if os.path.exists(graph_path):
            logger.info(f"Loading embeddings_graph from {graph_path}")
            self.embeddings_graph = np.load(graph_path, mmap_mode='r')
            logger.info(f"Embeddings_graph: shape={self.embeddings_graph.shape}, dtype={self.embeddings_graph.dtype}")
        else:
            logger.warning(f"Graph embeddings NOT FOUND at {graph_path}")
            self.embeddings_graph = None


        # 3. Load Metadata
        logger.info(f"Loading metadata from {METADATA_FILE}")
        # Determine which columns are actually available using pyarrow for speed
        try:
            schema = pq.read_schema(METADATA_FILE)
            available_cols = schema.names
        except:
            available_cols = []

        needed_cols = [
            'appid', 'name', 'release_date', 'parsed_date', 'positive', 'negative',
            'genres', 'tags', 'categories', 'supported_languages',
            'mature_content', 'price', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z', 
            'estimated_playtime', 'difficulty_predicted',
            'is_vr_only', 'is_english', 'is_utility', 'is_nsfw', 'is_delisted', 'is_hollow'
        ]

        # Dynamically append columns that might not exist in all versions of the metadata
        if 'short_description' in available_cols:
            needed_cols.insert(2, 'short_description')  # Insert after name
            logger.info("Found short_description in metadata")
        else:
            logger.warning("short_description NOT found in metadata")

        if 'price_z' in available_cols:
            needed_cols.append('price_z')
            logger.info("Found price_z in metadata")
        else:
            logger.warning("price_z NOT found in metadata")

        if 'tone_z' in available_cols:
            needed_cols.append('tone_z')
            logger.info("Found tone_z in metadata")
        else:
            logger.warning("tone_z NOT found in metadata")

        if 'release_year' in available_cols:
            needed_cols.append('release_year')

        try:
            self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols, dtype_backend='pyarrow')
            logger.info(f"Metadata loaded with pyarrow backend. Columns: {self.metadata.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Failed to load with pyarrow backend: {e}. Falling back to standard.")
            self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols)
        
        # Ensure string columns are actually strings (prevents Arrow errors with empty columns)
        string_cols = ['name', 'short_description', 'genres', 'tags', 'price', 'release_date']
        for col in string_cols:
            if col in self.metadata.columns:
                self.metadata[col] = self.metadata[col].fillna('').astype(str)

        # Boolean features are now pre-calculated in the pipeline
        logger.info("Verifying pre-calculated boolean features...")
        bool_cols = ['is_vr_only', 'is_english', 'is_utility', 'is_nsfw', 'is_delisted', 'is_hollow']
        for col in bool_cols:
            if col not in self.metadata.columns:
                logger.warning(f"Feature {col} missing from metadata! Re-calculating...")
                if col == 'is_vr_only':
                    self.metadata['is_vr_only'] = self.metadata['categories'].fillna('').astype(str).str.contains('VR Only', case=False).astype(bool)
                elif col == 'is_english':
                    langs = self.metadata['supported_languages'].fillna('').astype(str)
                    self.metadata['is_english'] = langs.str.contains('English', case=False).values
                elif col == 'is_utility':
                    self.metadata['is_utility'] = self.metadata['tags'].fillna('').astype(str).str.contains('Utilities', case=False).astype(bool)
                elif col == 'is_nsfw':
                    nsfw_tags_pattern = r"'Hentai':"
                    self.metadata['is_nsfw'] = (
                        (self.metadata['mature_content'] > 0) | 
                        (self.metadata['tags'].fillna('').astype(str).str.contains(nsfw_tags_pattern, regex=True, case=False))
                    ).values
                elif col == 'is_delisted':
                    self.metadata['is_delisted'] = (
                        self.metadata['price'].fillna('').astype(str).str.contains('delisted', case=False) |
                        self.metadata['name'].fillna('').astype(str).str.contains('DELISTED', case=False)
                    ).values
                elif col == 'is_hollow':
                    self.metadata['is_hollow'] = (
                        (self.metadata['short_description'].fillna('').str.len() < 10) & 
                        (self.metadata['tags'].fillna('') == '{}') &
                        (self.metadata['genres'].fillna('') == '')
                    ).values
            else:
                self.metadata[col] = self.metadata[col].astype(bool)

        logger.info(f"NSFW games identified: {np.sum(self.metadata['is_nsfw'])}")
        logger.info(f"Hollow games identified: {np.sum(self.metadata['is_hollow'])}")
        
        # Categories and supported_languages are only needed for re-calculating if missing
        if 'categories' in self.metadata.columns:
            self.metadata.drop(columns=['categories'], inplace=True)
        if 'supported_languages' in self.metadata.columns:
            self.metadata.drop(columns=['supported_languages'], inplace=True)
            
        logger.info(f"Boolean features ready. Metadata shape: {self.metadata.shape}")
        
        # Optimize metadata types
        logger.info("Optimizing dtypes...")
        self.metadata['appid'] = self.metadata['appid'].astype(np.int32)
        self.metadata['positive'] = self.metadata['positive'].astype(np.int32)
        self.metadata['negative'] = self.metadata['negative'].astype(np.int32)
        self.metadata['date_z'] = self.metadata['date_z'].astype(np.float16)
        self.metadata['pop_z'] = self.metadata['pop_z'].astype(np.float16)
        self.metadata['playtime_z'] = self.metadata['playtime_z'].astype(np.float16)
        self.metadata['difficulty_z'] = self.metadata['difficulty_z'].astype(np.float16)
        if 'price_z' in self.metadata.columns:
            self.metadata['price_z'] = self.metadata['price_z'].astype(np.float16)
        else:
            self.metadata['price_z'] = np.float16(0.0)
        self.metadata['estimated_playtime'] = self.metadata['estimated_playtime'].astype(np.float32)
        self.metadata['difficulty_predicted'] = self.metadata['difficulty_predicted'].astype(np.float32)
        self.metadata['mature_content'] = self.metadata['mature_content'].fillna(0).astype(np.int8)

        # Release dates are now pre-parsed in the pipeline
        if 'parsed_date' not in self.metadata.columns:
            logger.warning("parsed_date missing from metadata! Re-parsing...")
            self.metadata['parsed_date'] = self.metadata['release_date'].apply(clean_release_date)
        else:
            self.metadata['parsed_date'] = pd.to_datetime(self.metadata['parsed_date'])

        # Pre-calculate normalized names for fuzzy search
        logger.info("Pre-calculating normalized names for search...")
        self.metadata['normalized_name'] = self.metadata['name'].apply(normalize_string)
        
        # Mapping AppID to Index for fast lookups
        self.appid_to_idx = {appid: i for i, appid in enumerate(self.metadata['appid'])}
        
        logger.info(f"Metadata loaded: {len(self.metadata)} rows, {len(self.metadata.columns)} columns")
        
        # 4. Load weights and means
        logger.info("Loading transformation matrices...")
        self.w_desc = np.load(W_DESC_FILE).astype(np.float16) if os.path.exists(W_DESC_FILE) else None
        if self.w_desc is not None:
            logger.info(f"W_desc: shape={self.w_desc.shape}")
        self.mean_desc = np.load(MEAN_DESC_FILE).astype(np.float16) if os.path.exists(MEAN_DESC_FILE) else None
        if self.mean_desc is not None:
            logger.info(f"Mean_desc: shape={self.mean_desc.shape}")

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

        # Load semantic norms
        logger.info("Loading semantic norms...")
        self.embeddings_desc_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE).astype(np.float16) if os.path.exists(EMBEDDINGS_DESC_NORMS_FILE) else None
        
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

        # Extract unique tags for the UI
        logger.info("Extracting unique tags...")
        all_tags_set = set()
        for t_str in self.metadata['tags'].dropna().unique():
            if t_str.startswith('{') and t_str.endswith('}'):
                try:
                    # Using regex for speed over ast.literal_eval for just keys
                    tag_keys = re.findall(r"'([^']+)':", t_str)
                    all_tags_set.update(tag_keys)
                except:
                    pass
        self.all_tags = sorted(list(all_tags_set))
        logger.info(f"Extracted {len(self.all_tags)} unique tags")
        
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

        # Load tag dimension descriptions
        desc_path = os.path.join(ROOT_DIR, "data", "production", "tag_dimension_descriptions.json")
        if os.path.exists(desc_path):
            logger.info(f"Loading tag dimension descriptions from {desc_path}...")
            try:
                with open(desc_path, 'r') as f:
                    self.tag_dimension_descriptions = json.load(f)
                logger.info(f"Loaded {len(self.tag_dimension_descriptions)} dimension descriptions")
            except Exception as e:
                logger.error(f"Failed to load tag dimension descriptions: {e}")
        else:
            logger.warning(f"Tag dimension descriptions file NOT FOUND at {desc_path}")

        # Pre-calculate anchor masks for MIGs and Narrative tags
        logger.info("Pre-calculating anchor masks...")
        self.anchor_masks = {}
        
        # Collect all tags needed for masks
        all_anchor_tags = set()
        for tags in MIGS.values(): all_anchor_tags.update(tags)
        all_anchor_tags.update(NARRATIVE_TAGS)
        all_anchor_tags.update(HORROR_MARKERS)
        all_anchor_tags.update(HARD_ANCHORS)
        all_anchor_tags.add("Isometric")
        all_anchor_tags.add("CRPG")
        
        tag_series = self.metadata['tags'].fillna('').astype(str)
        for tag in all_anchor_tags:
            pattern = rf"'{re.escape(tag)}':"
            self.anchor_masks[tag] = tag_series.str.contains(pattern, regex=True).values
        
        # Also pre-calculate title keyword masks for hijacking detection
        self.title_keyword_masks = {} # Will be populated per-request
        
        logger.info(f"Pre-calculated {len(self.anchor_masks)} anchor masks")

        # 5. Load SentenceTransformer model
        logger.info("Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
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
    status: Optional[str] = None
    notes: Optional[str] = None

class RecommendationRequest(BaseModel):
    alpha: float = 0.5
    beta: float = 0.5
    gamma_topic: float = 0.5 # Topic Match weight
    quality_pref: float = 1.0
    age_pref: float = 0.0
    pop_pref: float = 0.0
    disc_pref: float = 0.0
    length_pref: float = 0.0
    difficulty_pref: float = 0.0
    difficulty_sim_weight: float = 0.0 # Weight for matching seed difficulty
    tone_sim_weight: float = 0.0 # New: Weight for matching seed tone (Serious-Bizarre)
    price_pref: float = 0.0
    remove_vr: bool = True
    english_only: bool = False
    remove_nsfw: bool = True
    remove_utilities: bool = True
    remove_unreleased: bool = True
    remove_delisted: bool = True
    top_k: int = 10
    prompt: str = ""
    seed_games: List[str] = []
    genres: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    
    # Taste DNA Extensions
    vibe_vector: Optional[List[float]] = None
    semantic_vibe_vector: Optional[List[float]] = None
    topic_vibe_vector: Optional[List[float]] = None
    metadata_weights: Optional[Dict[str, float]] = None
    intercept: Optional[float] = 0.0
    scaling_factor: Optional[float] = 1.0
    
    # Profile Exclusion
    profile_filter: Optional[str] = "none" # "none", "rated", "all"
    library_appids: Optional[List[int]] = []
    rated_appids: Optional[List[int]] = []
    ignored_appids: Optional[List[int]] = []
    library_details: Optional[Dict[int, Dict[str, float]]] = {}

# --- Endpoints ---

@app.get("/genres")
def get_genres():
    """Returns a list of all unique genres available in the dataset, filtered for validity."""
    logger.info("GET /genres called")
    # Filter out genres that don't have a validated link
    genres = [g for g in data_manager.all_genres if data_manager.term_links.get(g) is not None]
    logger.debug(f"Returning {len(genres)} filtered genres")
    return genres

@app.get("/tags")
def get_tags():
    """Returns a list of all unique tags available in the dataset, filtered for validity."""
    logger.info("GET /tags called")
    # Tags are already extracted from metadata which was filtered for dead tags
    return data_manager.all_tags

@app.get("/tag_dimensions")
def get_tag_dimensions():
    """Returns the pre-calculated tag dimension descriptions."""
    logger.info("GET /tag_dimensions called")
    return data_manager.tag_dimension_descriptions

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
    """Returns game names matching the query string, prioritized by relevance and popularity."""
    if data_manager.metadata is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    if not q or len(q) < 2:
        return []
        
    query_norm = normalize_string(q)
    # 1. Get all matches using normalized names
    mask = data_manager.metadata['normalized_name'].str.contains(re.escape(query_norm), case=False, na=False)
    matches_df = data_manager.metadata[mask][['name', 'normalized_name', 'pop_z']].copy()
    
    if matches_df.empty:
        return []
        
    # 2. Add scoring columns for sorting
    # Priority 1: Exact match (ignoring case/symbols)
    matches_df['exact'] = matches_df['normalized_name'] == query_norm
    # Priority 2: Starts with query
    matches_df['starts_with'] = matches_df['normalized_name'].str.startswith(query_norm)
    
    # 3. Sort by: Exact Match > Starts With > Popularity (pop_z)
    # We use popularity to choose WHICH games make the top 50.
    # Cast pop_z to float32 for sorting compatibility.
    matches_df['pop_z_f32'] = matches_df['pop_z'].astype(np.float32)
    matches_df = matches_df.sort_values(
        by=['exact', 'starts_with', 'pop_z_f32'],
        ascending=[False, False, False]
    )
    
    # 4. Take the top results
    top_matches = matches_df.head(limit).copy()
    
    # 5. Re-sort the final selection alphabetically (case-insensitive) for UI predictability
    top_matches['names_lower'] = top_matches['name'].str.lower()
    top_matches = top_matches.sort_values(by='names_lower', ascending=True)
    
    return top_matches['name'].tolist()

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

def ensure_python_types(obj):
    """
    Recursively converts NumPy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: ensure_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_python_types(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

@app.get("/changelog")
def get_changelog():
    """Returns the content of CHANGELOG.md."""
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Changelog file not found")

@app.get("/methodology")
def get_methodology():
    """Returns the content of methodology.md."""
    try:
        with open("methodology.md", "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Methodology file not found")

@app.get("/about")
def get_about():
    """Returns the content of about.md."""
    try:
        with open("about.md", "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="About file not found")

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
            cmd_scrape = [sys.executable, "scraping/get_user_stats.py", sid]
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
            subprocess.run([sys.executable, "pipeline/generate_user_soft_labels.py", f"data/user_{resolved_id}_library.csv"], check=True)
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
        # Ensure required columns exist for backward compatibility
        if 'status' not in df_gt.columns:
            df_gt['status'] = np.nan
        if 'actual_rating' not in df_gt.columns:
            df_gt['actual_rating'] = np.nan
        if 'ignore' not in df_gt.columns:
            df_gt['ignore'] = False

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
                    'status': gt_row.get('status', 'ignored' if gt_row['ignore'] else 'rated'),
                    'is_manual': True,
                    'is_nsfw': bool(row.get('is_nsfw', False))
                })
        
        # 3. Merge library games
        # Also need is_nsfw, parsed_date, and release_date for library games
        metadata = data_manager.metadata
        df_sl = df_sl.merge(metadata[['appid', 'is_nsfw', 'parsed_date', 'release_date']], on='appid', how='left')
        
        # FILTER: Only show games with playtime in the verification UI and that are already released
        build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s') if os.path.exists(METADATA_FILE) else pd.Timestamp.now()
        placeholders = ['coming soon', 'to be announced', 'maybe', 'tbd']
        
        # Better: use the merged metadata columns
        df_sl_visible = df_sl[(df_sl['has_playtime'] == True) & (df_sl['parsed_date'] <= build_time)].copy()
        # Add placeholder check to merged df
        is_placeholder_merged = df_sl_visible['release_date'].fillna('').astype(str).str.lower().str.contains('|'.join(placeholders), regex=True)
        df_sl_visible = df_sl_visible[~is_placeholder_merged].copy()
        
        df = df_sl_visible.merge(df_gt[['appid', 'actual_rating', 'ignore', 'status']], on='appid', how='left')
        df['actual_rating'] = df['actual_rating'].fillna(df['predicted_rating'])
        df['ignore'] = df['ignore'].fillna(False)
        df['is_manual'] = False
        
        # Define default status
        def get_default_status(row):
            if not pd.isna(row.get('status')): return row['status']
            if row['ignore']: return 'ignored'
            return 'played' # Default for verify view visible games
        
        df['status'] = df.apply(get_default_status, axis=1)
        
        # Build 55: Filter out ignored games from the Verify view
        df = df[df['status'] != 'ignored'].copy()
        manual_rows = [m for m in manual_rows if m.get('status') != 'ignored']
        
        # 4. Combine
        final_list = df.to_dict(orient='records') + manual_rows
    else:
        # Initial fetch (no ground truth yet)
        # Add is_nsfw, parsed_date, and release_date from metadata
        metadata = data_manager.metadata
        df = df_sl.merge(metadata[['appid', 'is_nsfw', 'parsed_date', 'release_date']], on='appid', how='left')
        
        # FILTER: Only show games with playtime in the verification UI and that are already released
        build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s') if os.path.exists(METADATA_FILE) else pd.Timestamp.now()
        df_visible = df[(df['has_playtime'] == True) & (df['parsed_date'] <= build_time)].copy()
        
        # Explicitly check for placeholders in the raw string as well
        placeholders = ['coming soon', 'to be announced', 'maybe', 'tbd']
        is_placeholder = df_visible['release_date'].fillna('').astype(str).str.lower().str.contains('|'.join(placeholders), regex=True)
        df_visible = df_visible[~is_placeholder].copy()
        
        df_visible['actual_rating'] = df_visible['predicted_rating']
        df_visible['ignore'] = False
        df_visible['is_manual'] = False
        df_visible['status'] = 'played'
        final_list = df_visible.to_dict(orient='records')
        
    # JSON cannot handle NaN values, and NumPy types cause errors.
    return ensure_python_types(final_list)

@app.get("/user/catalogue/{steam_id}")
def get_catalogue_data(steam_id: str):
    """Returns all games in library plus manual additions for the Catalogue UI."""
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    sl_path = f"data/user_{steam_id}_soft_labels.csv"
    
    if not os.path.exists(sl_path):
        raise HTTPException(status_code=404, detail="User data not found. Please fetch first.")
    
    df_sl = pd.read_csv(sl_path)
    
    # Merge with metadata for is_nsfw, etc.
    metadata = data_manager.metadata
    df_sl = df_sl.merge(metadata[['appid', 'is_nsfw', 'parsed_date', 'release_date']], on='appid', how='left')
    
    if os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)
        # Ensure required columns exist for backward compatibility
        if 'status' not in df_gt.columns:
            df_gt['status'] = np.nan
        if 'actual_rating' not in df_gt.columns:
            df_gt['actual_rating'] = np.nan
        if 'ignore' not in df_gt.columns:
            df_gt['ignore'] = False
        if 'notes' not in df_gt.columns:
            df_gt['notes'] = ""

        # Identify manual additions
        manual_appids = df_gt[~df_gt['appid'].isin(df_sl['appid'])]['appid'].tolist()
        
        manual_rows = []
        if manual_appids:
            m_data = metadata[metadata['appid'].isin(manual_appids)]
            for _, row in m_data.iterrows():
                gt_row = df_gt[df_gt['appid'] == row['appid']].iloc[0]
                status = gt_row.get('status')
                if pd.isna(status):
                    status = 'ignored' if gt_row['ignore'] else 'rated'
                
                manual_rows.append({
                    'appid': int(row['appid']),
                    'name': row['name'],
                    'playtime_forever': 0,
                    'predicted_rating': 5,
                    'actual_rating': gt_row['actual_rating'],
                    'ignore': bool(gt_row['ignore']),
                    'status': status,
                    'notes': str(gt_row.get('notes', "")),
                    'is_manual': True,
                    'is_nsfw': bool(row.get('is_nsfw', False))
                })
        
        # Merge library games
        df = df_sl.merge(df_gt[['appid', 'actual_rating', 'ignore', 'status', 'notes']], on='appid', how='left')
        df['actual_rating'] = df['actual_rating'].fillna(df['predicted_rating'])
        df['ignore'] = df['ignore'].fillna(False)
        df['notes'] = df['notes'].fillna("")
        df['is_manual'] = False
        
        # Define default status if missing
        def get_default_status(row):
            if not pd.isna(row.get('status')):
                return row['status']
            if row['ignore']:
                return 'ignored'
            if row['has_playtime']:
                return 'played'
            return 'backlog'
            
        df['status'] = df.apply(get_default_status, axis=1)
        
        final_list = df.to_dict(orient='records') + manual_rows
    else:
        # Initial fetch
        df = df_sl.copy()
        df['actual_rating'] = df['predicted_rating']
        df['ignore'] = False
        df['is_manual'] = False
        df['status'] = df.apply(lambda r: 'played' if r['has_playtime'] else 'backlog', axis=1)
        final_list = df.to_dict(orient='records')
        
    return ensure_python_types(final_list)

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
        df = pd.DataFrame(columns=['appid', 'actual_rating', 'ignore', 'status', 'notes'])
        
    for up in updates:
        # Update or add
        mask = df['appid'] == up.appid
        if mask.any():
            df.loc[mask, 'actual_rating'] = up.actual_rating
            df.loc[mask, 'ignore'] = up.ignore
            if up.status:
                df.loc[mask, 'status'] = up.status
            if up.notes is not None:
                df.loc[mask, 'notes'] = up.notes
        else:
            new_row = pd.DataFrame([{
                'appid': up.appid, 
                'actual_rating': up.actual_rating, 
                'ignore': up.ignore,
                'status': up.status,
                'notes': up.notes or ""
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            
    df.to_csv(gt_path, index=False)
    return {"message": "Verification data saved successfully"}

def run_taste_solver(steam_id: str):
    """Sync wrapper for the taste solver to run in a thread."""
    cmd = [sys.executable, "pipeline/solve_user_taste.py", steam_id]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

@app.post("/user/solve/{steam_id}")
async def solve_taste(steam_id: str):
    """Runs the regression solver to generate the taste profile."""
    try:
        logger.info(f"Solving Taste DNA for {steam_id}...")
        # Use a thread to avoid asyncio's subprocess transport limitations on Windows
        returncode, stdout_str, stderr_str = await asyncio.to_thread(run_taste_solver, steam_id)
        
        if returncode != 0:
            err_msg = stderr_str.strip()
            logger.error(f"Solver failed for {steam_id}: {err_msg}")
            raise HTTPException(status_code=500, detail=f"Solver failed: {err_msg}")
            
        logger.info(f"Taste DNA solved for {steam_id}")
        return {"message": "Taste profile solved successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
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
    
    return ensure_python_types(profile)

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
    
    # Filter Mask: Remove delisted, hollow, and utility games by default for all lists
    # This prevents junk from cluttering up the category views
    base_mask = (~metadata['is_delisted']) & (~metadata['is_hollow']) & (~metadata['is_utility'])
    
    if category == "quality":
        logger.debug(f"Quality list request: quality_grid shape={data_manager.quality_grid.shape}")
        num_grid_rows = data_manager.quality_grid.shape[0]
        grid_index = int(round(((discovery_pref - (-1.0)) / 2.0) * (num_grid_rows - 1)))
        grid_index = max(0, min(num_grid_rows - 1, grid_index))
        scores = data_manager.quality_grid[grid_index].copy()
        
        # Apply mask using float16-safe values (min/max ~65504)
        scores[~base_mask.values] = -60000.0
        
        top_indices = np.argsort(-scores)[:50]
        # For bottom, we need a different mask because -60000.0 would be the bottom!
        bottom_scores = data_manager.quality_grid[grid_index].copy()
        bottom_scores[~base_mask.values] = 60000.0
        bottom_indices = np.argsort(bottom_scores)[:50]
        
        def format_quality_list(indices, scores_array):
            df = metadata.iloc[indices].copy()
            df['quality_score'] = scores_array[indices]
            return df[['appid', 'name', 'quality_score']].to_dict(orient='records')

        logger.info(f"Quality list: {len(top_indices)} top, {len(bottom_indices)} bottom")
        result = ensure_python_types({
            "top": format_quality_list(top_indices, data_manager.quality_grid[grid_index]),
            "bottom": format_quality_list(bottom_indices, data_manager.quality_grid[grid_index])
        })
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "length":
        logger.debug("Length list request")
        playtime_col = 'estimated_playtime'
        valid = metadata[base_mask & (metadata[playtime_col] > 0)].copy()
        logger.info(f"Games with valid playtime: {len(valid)}")
        
        longest = valid.sort_values(playtime_col, ascending=False).head(50)
        shortest = valid.sort_values(playtime_col, ascending=True).head(50)
        
        if not longest.empty:
            logger.info(f"Longest: {longest.iloc[0]['name']} ({longest.iloc[0][playtime_col]:.1f} minutes)")
        if not shortest.empty:
            logger.info(f"Shortest: {shortest.iloc[0]['name']} ({shortest.iloc[0][playtime_col]:.1f} minutes)")
        
        result = ensure_python_types({
            "top": longest[['appid', 'name', playtime_col]].rename(columns={playtime_col: 'playtime'}).to_dict(orient='records'),
            "bottom": shortest[['appid', 'name', playtime_col]].rename(columns={playtime_col: 'playtime'}).to_dict(orient='records')
        })
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "popularity":
        logger.debug("Popularity list request")
        if 'total_reviews' not in metadata.columns:
            metadata = metadata.copy()
            metadata['total_reviews'] = metadata['positive'] + metadata['negative']
        
        popular_df = metadata[base_mask & (metadata['total_reviews'] >= 1)]
        logger.info(f"Games with >=1 review: {len(popular_df)}")
        most_pop = popular_df.sort_values('total_reviews', ascending=False).head(50)
        least_pop = popular_df.sort_values('total_reviews', ascending=True).head(50)
        
        if not most_pop.empty:
            logger.info(f"Most popular: {most_pop.iloc[0]['name']} ({most_pop.iloc[0]['total_reviews']:,} reviews)")
        if not least_pop.empty:
            logger.info(f"Least popular: {least_pop.iloc[0]['name']} ({least_pop.iloc[0]['total_reviews']:,} reviews)")
        
        result = ensure_python_types({
            "top": most_pop[['appid', 'name', 'total_reviews']].to_dict(orient='records'),
            "bottom": least_pop[['appid', 'name', 'total_reviews']].to_dict(orient='records')
        })
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "age":
        logger.debug("Age list request")
        if os.path.exists(METADATA_FILE):
            build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s')
        else:
            build_time = pd.Timestamp.now()
            
        valid_dates = metadata[base_mask & (metadata['parsed_date'] <= build_time)].copy()
        logger.info(f"Games released before {build_time}: {len(valid_dates)}")
        oldest = valid_dates.sort_values('parsed_date', ascending=True).head(50)
        newest = valid_dates.sort_values('parsed_date', ascending=False).head(50)
        
        result = ensure_python_types({
            "top": newest[['appid', 'name', 'release_date']].to_dict(orient='records'),
            "bottom": oldest[['appid', 'name', 'release_date']].to_dict(orient='records')
        })
        data_manager.lists_cache[cache_key] = result
        return result

    elif category == "difficulty":
        logger.debug("Difficulty list request")
        valid_diff = metadata[base_mask].copy()
        hardest = valid_diff.sort_values('difficulty_predicted', ascending=False).head(50)
        easiest = valid_diff.sort_values('difficulty_predicted', ascending=True).head(50)
        
        logger.info(f"Difficulty range: hardest={hardest.iloc[0]['difficulty_predicted'] if not hardest.empty else 'N/A'}, easiest={easiest.iloc[0]['difficulty_predicted'] if not easiest.empty else 'N/A'}")
        
        from common.constants import DIFFICULTY_COEFFICIENTS_FILE, TOPIC_DESCRIPTIONS_FILE
        
        # Difficulty predictors (Tags + Topics)
        feature_impacts = []
        logger.info(f"Looking for difficulty coefficients at: {DIFFICULTY_COEFFICIENTS_FILE}")
        
        if os.path.exists(DIFFICULTY_COEFFICIENTS_FILE):
            try:
                with open(DIFFICULTY_COEFFICIENTS_FILE, "r") as f:
                    coeffs = json.load(f)
                
                # Load topic labels if they exist
                topic_labels = {}
                if os.path.exists(TOPIC_DESCRIPTIONS_FILE):
                    with open(TOPIC_DESCRIPTIONS_FILE, "r") as f:
                        topic_labels = json.load(f)

                for item in coeffs:
                    feature = item['feature']
                    impact = item['coefficient']
                    
                    if feature.startswith('topic_'):
                        topic_id = feature.replace('topic_', '')
                        label = topic_labels.get(topic_id, "Unknown Topic")
                        feature_impacts.append({
                            'tag': f"Topic {topic_id}: {label}", 
                            'impact': float(impact),
                            'type': 'topic'
                        })
                    else:
                        feature_impacts.append({
                            'tag': feature, 
                            'impact': float(impact),
                            'type': 'tag'
                        })
            except Exception as e:
                logger.error(f"Error reading difficulty coefficients: {e}")
        else:
            logger.warning(f"Difficulty coefficients file NOT FOUND at {DIFFICULTY_COEFFICIENTS_FILE}")
        
        logger.info(f"Returning {len(feature_impacts)} feature impacts")
        result = ensure_python_types({
            "top": hardest[['appid', 'name', 'difficulty_predicted']].to_dict(orient='records'),
            "bottom": easiest[['appid', 'name', 'difficulty_predicted']].to_dict(orient='records'),
            "tag_impacts": sorted(feature_impacts, key=lambda x: x['impact'], reverse=True)
        })
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
            "appid": game_meta['appid'],
            "name": game_meta['name'],
            "release_date": game_meta['release_date'],
            "short_description": game_meta['short_description'] if 'short_description' in game_meta and pd.notna(game_meta['short_description']) else "",
            "release_year": game_meta['release_year'],
            "estimated_playtime": game_meta['estimated_playtime'],
            "difficulty_predicted": game_meta['difficulty_predicted'],
            "positive": game_meta['positive'],
            "negative": game_meta['negative'],
            "genres": game_meta['genres'],
            "tags": game_meta['tags'],
            "price": game_meta['price'] if pd.notna(game_meta['price']) else "Free",
            "is_nsfw": game_meta['is_nsfw'],
            "is_delisted": game_meta['is_delisted'],
            "raw_pop": raw_pop,
            "raw_length": raw_length
        }
        response_items.append(item)
    
    cleaned_response = ensure_python_types(response_items)
    logger.info(f"/metadata returning {len(cleaned_response)} items")
    return cleaned_response

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    logger.info("POST /recommend called")
    if data_manager.metadata is None:
        logger.error("Data not loaded when /recommend called")
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    # Determine Mode early
    expected_tag_dim = data_manager.tag_vectors.shape[1]
    is_vibe_present = request.vibe_vector is not None and len(request.vibe_vector) > 0 
    is_vibe_valid = is_vibe_present and (len(request.vibe_vector) == expected_tag_dim) 
    is_linear_mode = is_vibe_valid

    logger.info(f"Mode Analysis: vibe_present={is_vibe_present}, vibe_len={len(request.vibe_vector) if request.vibe_vector else 0}, expected={expected_tag_dim}, is_linear={is_linear_mode}")

    metadata = data_manager.metadata

    # Identify seeds
    seed_indices = np.where(metadata['name'].isin(request.seed_games))[0]
    logger.info(f"Seed games: requested={len(request.seed_games)}, found={len(seed_indices)}")
    seed_appids = metadata.iloc[seed_indices]['appid'].tolist()

    # 1. Base Filtering via unified helper
    from common.utils import get_base_filter_mask
    mask = get_base_filter_mask(
        metadata,
        english_only=request.english_only or is_linear_mode,
        remove_vr=request.remove_vr or is_linear_mode,
        remove_utilities=request.remove_utilities or is_linear_mode,
        remove_delisted=request.remove_delisted or is_linear_mode,
        remove_hollow=is_linear_mode, # Linear mode always removes hollow
        remove_unreleased=request.remove_unreleased or is_linear_mode
    )

    # 7. No Games with Zero Reviews (Parity with Solver)
    total_reviews = metadata['positive'].fillna(0) + metadata['negative'].fillna(0)
    mask &= (total_reviews > 0).values

    if request.genres:
        genre_mask = np.zeros(len(metadata), dtype=bool)
        for genre in request.genres:
            escaped_genre = re.escape(genre)
            genre_mask |= metadata['genres'].fillna('').astype(str).str.contains(escaped_genre, regex=True, case=False).values
        mask &= genre_mask
    if request.tags:
        tag_mask = np.ones(len(metadata), dtype=bool)
        for tag in request.tags:
            escaped_tag = re.escape(tag)
            pattern = rf"'{escaped_tag}':"
            tag_mask &= metadata['tags'].fillna('').astype(str).str.contains(pattern, regex=True).values
        mask &= tag_mask

    keep_indices = np.where(mask)[0]
    logger.info(f"Final filtered pool: {len(keep_indices)} games")

    if len(keep_indices) == 0:
        logger.warning("All games were filtered out!")
        return []

    # 2. Jackalope Kernel (High-Fidelity Seed Similarity)
    jackalope_sims = None
    jackalope_components = None
    if seed_indices.size > 0:
        logger.info("Calculating Jackalope Kernel for seeds...")
        
        # Prepare Seed Vectors
        seed_verb_profile = np.mean(data_manager.verb_profiles[seed_indices], axis=0)
        seed_sem_vec = np.mean(data_manager.embeddings_desc_norm[seed_indices], axis=0)
        seed_sem_norm = np.linalg.norm(seed_sem_vec)
        seed_sem_vec /= (seed_sem_norm + EPSILON)
        
        seed_topic_dist = np.mean(data_manager.topic_distributions[seed_indices], axis=0)
        seed_topic_dist /= (np.sum(seed_topic_dist) + EPSILON)
        
        # Prepare Metadata via unified helper
        from common.utils import extract_seed_metadata, calculate_title_hijack_mask
        seed_meta = extract_seed_metadata(seed_indices, metadata)
        title_hijack_mask = calculate_title_hijack_mask(request.seed_games, metadata)

        jackalope_sims, jackalope_components = calculate_jackalope_kernel(
            verb_profiles=data_manager.verb_profiles,
            seed_verb_profile=seed_verb_profile,
            sem_vectors=data_manager.embeddings_desc_norm,
            sem_norms=data_manager.embeddings_desc_norms,
            seed_sem_vec=seed_sem_vec,
            seed_sem_norm=seed_sem_norm,
            topic_distributions=data_manager.topic_distributions,
            seed_topic_dist=seed_topic_dist,
            topic_means=data_manager.topic_means,
            topic_stds=data_manager.topic_stds,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
            dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR,
            sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            mature_content_flags=metadata['mature_content'].values > 0,
            seed_mature_content=bool(np.any(seed_meta['mature_flags'])),
            seed_migs=seed_meta['migs_list'][0] if len(seed_meta['migs_list']) == 1 else set().union(*seed_meta['migs_list']),
            seed_tags=seed_meta['all_soul_tags'],
            candidate_anchor_masks=data_manager.anchor_masks,
            active_narrative_seed=seed_meta['active_narrative'],
            is_cinematic_seed=seed_meta['is_cinematic'],
            precalculated_masks={"title_hijack": title_hijack_mask},
            difficulty_z=metadata['difficulty_z'].values,
            seed_difficulty_z=np.mean(metadata.iloc[seed_indices]['difficulty_z']),
            tone_z=metadata['tone_z'].values if 'tone_z' in metadata.columns else None,
            seed_tone_z=np.mean(metadata.iloc[seed_indices]['tone_z']) if 'tone_z' in metadata.columns else None,
            return_components=True,
            graph_embeddings=data_manager.embeddings_graph,
            seed_graph_vec=np.mean(data_manager.embeddings_graph[seed_indices], axis=0) if data_manager.embeddings_graph is not None else None
        )
    # 3. Component Blending (Prompt + DNA + Jackalope)
    all_semantic_sims = np.zeros(len(metadata))
    all_tag_sims = np.zeros(len(metadata))
    all_topic_sims = np.zeros(len(metadata))
    all_diff_sims = np.zeros(len(metadata))
    all_tone_sims = np.zeros(len(metadata))

    try:
        # --- TOPIC SIGNAL ---
        topic_signals = []
        if jackalope_components: topic_signals.append(jackalope_components['cluster'])
        
        if request.prompt:
            from common.constants import TOPIC_SIMILARITY_MEAN, TOPIC_SIMILARITY_STD
            clean_prompt = request.prompt.lower()
            prompt_vec = data_manager.model.encode([clean_prompt])[0].astype(np.float32)
            topic_embeddings = data_manager.topic_model.topic_embeddings_[1:]
            topic_sims_raw = np.dot(topic_embeddings, prompt_vec)
            topic_sims_raw = topic_sims_raw - np.max(topic_sims_raw)
            exp_sim = np.exp(topic_sims_raw / 0.05)
            prompt_topic_p = exp_sim / np.sum(exp_sim)
            prompt_topic_sims = fast_jsd_similarity(prompt_topic_p, data_manager.topic_distributions, mean=TOPIC_SIMILARITY_MEAN, std=TOPIC_SIMILARITY_STD)
            topic_signals.append(prompt_topic_sims)
            
        if request.topic_vibe_vector:
            topic_vibe_unit = np.array(request.topic_vibe_vector, dtype=np.float32)
            dna_topic_sims = np.dot(data_manager.topic_distributions.astype(np.float32) * TOPIC_GLOBAL_SCALING_FACTOR, topic_vibe_unit)
            topic_signals.append(dna_topic_sims)
            
        if topic_signals:
            all_topic_sims = softmin_blend(topic_signals, temperature=SOFTMIN_TEMPERATURE)

        # --- SEMANTIC SIGNAL ---
        sem_signals = []
        if jackalope_components: sem_signals.append(jackalope_components['theme'])
        
        if request.prompt:
            def norm_vec(v):
                mag = np.linalg.norm(v)
                return v / (mag if mag > EPSILON else 1.0)
            clean_prompt = request.prompt.lower()
            prompt_vec = data_manager.model.encode([clean_prompt])[0]
            p_desc = np.dot(prompt_vec, data_manager.w_desc) if data_manager.w_desc is not None else prompt_vec
            p_desc_norm = norm_vec(p_desc)
            prompt_desc_sims = np.dot(data_manager.embeddings_desc_norm, p_desc_norm)
            if data_manager.embeddings_desc_norms is not None:
                denom_desc = data_manager.embeddings_desc_norms + SEMANTIC_DOT_PRODUCT_LAMBDA
                prompt_desc_sims = (prompt_desc_sims / denom_desc) * SEMANTIC_GLOBAL_SCALING_FACTOR
            sem_signals.append(prompt_desc_sims)
            
        if request.semantic_vibe_vector:
            sem_vibe = np.array(request.semantic_vibe_vector, dtype=np.float32)
            dot_products = np.dot(data_manager.embeddings_desc_norm, sem_vibe)
            denom = data_manager.embeddings_desc_norms + SEMANTIC_DOT_PRODUCT_LAMBDA
            dna_sem_sims = (dot_products / denom) * SEMANTIC_GLOBAL_SCALING_FACTOR
            sem_signals.append(dna_sem_sims)

        if sem_signals:
            all_semantic_sims = softmin_blend(sem_signals, temperature=SOFTMIN_TEMPERATURE)
            all_semantic_sims[data_manager.metadata['is_hollow'].values] = 0.0

        # --- TAG SIGNAL ---
        tag_signals = []
        if jackalope_components: tag_signals.append(jackalope_components['vibe'])
        
        if request.vibe_vector:
            beta_dna_unit = np.array(request.vibe_vector, dtype=np.float32)
            dot_products_dna = np.dot(data_manager.tag_vectors, beta_dna_unit)
            denom = data_manager.tag_vectors_norms + DOT_PRODUCT_LAMBDA
            dna_tag_sims = (dot_products_dna / denom) * TAG_GLOBAL_SCALING_FACTOR
            tag_signals.append(dna_tag_sims)
            
        if tag_signals:
            all_tag_sims = softmin_blend(tag_signals, temperature=SOFTMIN_TEMPERATURE)

        # --- SIMILARITY DIMENSIONS ---
        if jackalope_components:
            all_diff_sims = jackalope_components['difficulty']
            all_tone_sims = jackalope_components['tone']

    except Exception as e:
        logger.exception(f"Signal blending failed: {e}")

    semantic_sims = all_semantic_sims[keep_indices]
    tag_sims = all_tag_sims[keep_indices]
    topic_sims = all_topic_sims[keep_indices]
    diff_sims = all_diff_sims[keep_indices]
    tone_sims = all_tone_sims[keep_indices]

    # 4. Rating Component (Grid lookup)
    num_grid_rows = data_manager.quality_grid.shape[0]
    grid_index = int(round(((request.disc_pref - (-1.0)) / 2.0) * (num_grid_rows - 1)))
    grid_index = max(0, min(num_grid_rows - 1, grid_index))
    
    q_grid_working = data_manager.quality_grid[grid_index].copy()
    if request.library_details:
        for appid_str, details in request.library_details.items():
            appid = int(appid_str)
            if appid in data_manager.appid_to_idx:
                idx = data_manager.appid_to_idx[appid]
                p_plus_t = details.get('p_plus_t')
                if p_plus_t is not None:
                    q_global = q_grid_working[idx]
                    q_grid_working[idx] = calculate_personalized_quality(np.array([q_global]), np.array([p_plus_t]))[0]

    z_spps = to_z(q_grid_working[keep_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    z_date = to_z(metadata.iloc[keep_indices]['date_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    z_pop = to_z(metadata.iloc[keep_indices]['pop_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    z_length = to_z(metadata.iloc[keep_indices]['playtime_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    z_difficulty = to_z(metadata.iloc[keep_indices]['difficulty_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    z_price = to_z(metadata.iloc[keep_indices]['price_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))

    # 5. Final Scoring
    if is_linear_mode:
        logger.info("Using Linear Mode (Taste DNA)")
        w_tag = request.beta
        w_semantic = request.alpha
        
        effective_weights = {
            'quality': request.quality_pref,
            'age': request.age_pref,
            'popularity': request.pop_pref,
            'length': request.length_pref,
            'difficulty': request.difficulty_pref,
            'price': request.price_pref,
            'tag_match': w_tag,
            'topic_match': request.gamma_topic,
            'kernel_match': request.metadata_weights.get('kernel_match', 0.0) if request.metadata_weights else 0.0,
            'graph_match': request.metadata_weights.get('graph_match', 0.0) if request.metadata_weights else 0.0
        }
        
        # Add MIG weights from profile
        if request.metadata_weights:
            for group in MIGS.keys():
                feat_key = f"MIG_{group}"
                for anchor in request.metadata_weights.get('kernel_anchors', []):
                    if anchor['group'] == group:
                        effective_weights[feat_key] = anchor['impact']

        # --- ZERO-DRIFT STRUCTURAL FEATURES ---
        # 1. MIG Features for Candidates
        candidate_mig_features_list = []
        for group in MIGS.keys():
            m = np.zeros(len(metadata), dtype=bool)
            for t in MIGS[group]:
                if t in data_manager.anchor_masks: m |= data_manager.anchor_masks[t]
            candidate_mig_features_list.append(m.astype(float))
        
        # [N_library, 38]
        mig_matrix = np.column_stack(candidate_mig_features_list)

        # 2. NW Kernel Feature (Relative to Rated Games)
        x_kernel_all = np.zeros(len(metadata), dtype=np.float32)
        x_graph_all = np.zeros(len(metadata), dtype=np.float32)
        
        if request.rated_appids:
            rated_indices = [data_manager.appid_to_idx[aid] for aid in request.rated_appids if aid in data_manager.appid_to_idx]
            if rated_indices:
                # Need seed tags/migs for constraints
                rated_seed_tags = []
                rated_seed_migs = []
                for idx in rated_indices:
                    t_str = metadata.iloc[idx]['tags']
                    t_dict = ast.literal_eval(t_str)
                    max_v = max(t_dict.values()) if t_dict else 1.0
                    s_tags_soul = {t for t, v in t_dict.items() if v / max_v > 0.15}
                    rated_seed_tags.append(s_tags_soul)
                    s_migs = {group for group, tags in MIGS.items() if any(t in s_tags_soul for t in tags)}
                    rated_seed_migs.append(s_migs)
                
                # Pre-calculate candidate MIG mask array
                mig_mask_array = np.zeros((len(metadata), len(MIGS)), dtype=bool)
                for j, group in enumerate(MIGS.keys()):
                    for t in MIGS[group]:
                        if t in data_manager.anchor_masks: mig_mask_array[:, j] |= data_manager.anchor_masks[t]
                
                # Use synchronized 2D kernel
                from common.utils import calculate_jackalope_kernel_2d
                k_mat = calculate_jackalope_kernel_2d(
                    verb_profiles=data_manager.verb_profiles,
                    seed_verb_profiles=data_manager.verb_profiles[rated_indices],
                    sem_vectors=data_manager.embeddings_desc_norm,
                    sem_norms=data_manager.embeddings_desc_norms,
                    seed_sem_vecs=data_manager.embeddings_desc_norm[rated_indices],
                    seed_sem_norms=data_manager.embeddings_desc_norms[rated_indices],
                    topic_distributions=data_manager.topic_distributions,
                    seed_topic_dists=data_manager.topic_distributions[rated_indices],
                    topic_means=data_manager.topic_means,
                    topic_stds=data_manager.topic_stds,
                    candidate_mig_masks=mig_mask_array,
                    seed_mig_masks=mig_mask_array[rated_indices],
                    difficulty_z=metadata['difficulty_z'].values,
                    seed_difficulty_z=metadata['difficulty_z'].values[rated_indices],
                    tone_z=data_manager.tone_z if data_manager.tone_z is not None else metadata['tone_z'].values,
                    seed_tone_z=data_manager.tone_z[rated_indices] if data_manager.tone_z is not None else metadata['tone_z'].values[rated_indices],
                    seed_tags=rated_seed_tags,
                    seed_migs=rated_seed_migs,
                    mature_content_flags=metadata['mature_content'].values > 0,
                    seed_mature_content_flags=metadata['mature_content'].values[rated_indices] > 0,
                    graph_embeddings=data_manager.embeddings_graph,
                    seed_graph_vecs=data_manager.embeddings_graph[rated_indices] if data_manager.embeddings_graph is not None else None
                )
                
                # Nadaraya-Watson aggregation
                k_exp = np.exp(k_mat * 10.0) # Corrected sharpening
                # Prevent games from predicting themselves
                for c, r_idx in enumerate(rated_indices):
                    k_exp[r_idx, c] = 0.0
                
                y_rated = np.array([request.library_details.get(str(aid), {}).get('actual_rating', 5.0) for aid in request.rated_appids])
                y_dev = y_rated - 5.0
                sum_weights = np.sum(k_exp, axis=1)
                x_kernel_all = np.sum(k_exp * y_dev, axis=1) / (sum_weights + 1e-9)

                # Calculate full library Graph Similarity feature (v6.0 Restoration)
                if data_manager.embeddings_graph is not None:
                    user_graph_vecs = data_manager.embeddings_graph[rated_indices]
                    all_graph_vectors = data_manager.embeddings_graph
                    # (N_library, M_seeds)
                    graph_sim_matrix = np.dot(all_graph_vectors.astype(np.float32), user_graph_vecs.astype(np.float32).T)
                    g_norms = np.linalg.norm(all_graph_vectors.astype(np.float32), axis=1)
                    s_norms = np.linalg.norm(user_graph_vecs.astype(np.float32), axis=1)
                    graph_sim_matrix /= (g_norms[:, None] * s_norms[None, :] + 1e-9)
                    graph_sim_matrix = np.maximum(0, graph_sim_matrix)
                    # Zero out identity matches
                    for c, f_idx in enumerate(rated_indices): graph_sim_matrix[f_idx, c] = 0.0
                    x_graph_all = (np.dot(graph_sim_matrix, y_dev) / (np.sum(graph_sim_matrix, axis=1) + 1e-9))

        # --- ASSEMBLE X AND APPLY LEARNED COEFFS ---
        x_q = to_z(q_grid_working, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        x_date = to_z(metadata['date_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        x_pop = to_z(metadata['pop_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        x_playtime = to_z(metadata['playtime_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        x_diff = to_z(metadata['difficulty_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        x_price = to_z(metadata['price_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        x_tone = to_z(data_manager.tone_z if data_manager.tone_z is not None else metadata['tone_z'].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
        
        # X order must match solver: [Kernel, Graph, Q, Date, Pop, Playtime, Diff, Price, Tone, MIGs, Topics]
        X_full = np.column_stack([
            x_kernel_all, x_graph_all, x_q, 
            x_date, x_pop, x_playtime, x_diff, x_price, x_tone,
            mig_matrix,
            data_manager.topic_distributions
        ])
        
        # Linear combination using profile weights
        learned_coeffs = []
        learned_coeffs.append(request.metadata_weights.get('kernel_match', 0.0))
        learned_coeffs.append(request.metadata_weights.get('graph_match', 0.0))
        learned_coeffs.append(request.metadata_weights.get('quality', 0.0))
        learned_coeffs.append(request.metadata_weights.get('age', 0.0))
        learned_coeffs.append(request.metadata_weights.get('popularity', 0.0))
        learned_coeffs.append(request.metadata_weights.get('length', 0.0))
        learned_coeffs.append(request.metadata_weights.get('difficulty', 0.0))
        learned_coeffs.append(request.metadata_weights.get('price', 0.0))
        learned_coeffs.append(request.metadata_weights.get('tone', 0.0))
        
        mig_weights = {a['group']: a['impact'] for a in request.metadata_weights.get('kernel_anchors', [])}
        for group in MIGS.keys():
            learned_coeffs.append(mig_weights.get(group, 0.0))
            
        # Add Topic weights from profile if they exist
        topic_weights = request.metadata_weights.get('topic_coeffs', [0.0] * 249)
        learned_coeffs.extend(topic_weights)
        
        final_scores_all = np.dot(X_full, np.array(learned_coeffs)) + (request.intercept or 5.0)
        final_scores = final_scores_all[keep_indices]
        
        # Add Dimension Similarities to linear scores if requested
        if seed_indices.size > 0:
            if request.difficulty_sim_weight > 0:
                final_scores += diff_sims * request.difficulty_sim_weight
            if request.tone_sim_weight > 0:
                final_scores += tone_sims * request.tone_sim_weight

        w_semantic_active = w_semantic
        w_tag_active = w_tag
        w_topic_active = request.gamma_topic
        w_quality_active, w_date_active, w_pop_active, w_length_active, w_difficulty_active, w_price_active = [effective_weights[k] for k in ['quality', 'age', 'popularity', 'length', 'difficulty', 'price']]
    else:
        logger.info("Using Manual Mode")
        w_semantic_active, w_tag_active, w_topic_active, w_quality_active, w_date_active, w_pop_active, w_length_active, w_difficulty_active, w_price_active = [
            request.alpha, request.beta, request.gamma_topic, request.quality_pref, request.age_pref, request.pop_pref, request.length_pref, request.difficulty_pref, request.price_pref
        ]
        
        z_tag_hybrid = np.clip(tag_sims, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        z_topic_hybrid = np.clip(topic_sims, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        
        final_scores = (
            (semantic_sims * w_semantic_active) +
            (z_tag_hybrid * w_tag_active) +
            (z_topic_hybrid * w_topic_active) +
            (z_spps * w_quality_active) +
            (z_date * w_date_active) +
            (z_pop * w_pop_active) +
            (z_length * w_length_active) +
            (z_difficulty * w_difficulty_active) +
            (z_price * w_price_active) +
            (diff_sims * request.difficulty_sim_weight) + # Add difficulty similarity
            (tone_sims * request.tone_sim_weight) + # Add tone similarity
            5.0
        )
        
        # If Jackalope kernel was calculated, apply its global vetoes/consensus floor
        if jackalope_sims is not None:
            # We apply the jackalope's consensus-based floor to the manual scores
            # If the high-fidelity kernel says 0, we heavily penalize the manual score
            j_mask = jackalope_sims[keep_indices] < 1e-6
            final_scores[j_mask] -= 20.0 # Heavy penalty for mechanical/vibe clash

    # Filter Seeds and Profile
    meta_filt = metadata.iloc[keep_indices].copy()
    if seed_appids:
        final_scores[meta_filt['appid'].isin(seed_appids)] = -1e12
    
    # Global Ignore (Build 55)
    if request.ignored_appids:
        final_scores[meta_filt['appid'].isin(request.ignored_appids)] = -1e12

    if request.profile_filter != "none":
        if request.profile_filter == "all":
            exclude_appids = request.library_appids
        elif request.profile_filter == "rated":
            exclude_appids = request.rated_appids
        elif request.profile_filter == "completed":
            # Exclude only rated or played games
            exclude_appids = []
            if request.library_appids:
                rated_set = set(request.rated_appids or [])
                for aid in request.library_appids:
                    details = (request.library_details or {}).get(str(aid), {})
                    playtime = details.get('playtime', 0)
                    if aid in rated_set or playtime > 0:
                        exclude_appids.append(aid)
        else:
            exclude_appids = []
            
        if exclude_appids:
            final_scores[meta_filt['appid'].isin(exclude_appids)] = -1e12

    # Sorting
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
    ui_w_semantic = w_semantic_active if (request.prompt or seed_appids or request.semantic_vibe_vector) else 0.0
    ui_w_tag = w_tag_active if (seed_appids or is_vibe_present) else 0.0

    # Inverse Probability Mapping
    from common.constants import GLOBAL_POSITIVE_RATE, QUALITY_SCORE_S_CONST, QUALITY_SCORE_S_BASE, SEMANTIC_SIMILARITY_MEAN
    
    s_discovery = QUALITY_SCORE_S_CONST * (QUALITY_SCORE_S_BASE ** (-request.disc_pref))
    all_pos, all_neg = metadata['positive'].values.astype(float), metadata['negative'].values.astype(float)
    all_prob_discovery = (all_pos + s_discovery * GLOBAL_POSITIVE_RATE) / (all_pos + all_neg + s_discovery)
    all_q_natural = norm.ppf(np.clip(all_prob_discovery, 1e-6, 1-1e-6))
    all_q_grid = data_manager.quality_grid[grid_index]
    sort_idx = np.argsort(all_q_grid)
    ref_q_grid, ref_q_natural = all_q_grid[sort_idx], all_q_natural[sort_idx]
    
    mean_semantic_contrib = SEMANTIC_SIMILARITY_MEAN * w_semantic_active if ui_w_semantic > 0 else 0.0
    mean_tag_contrib = all_tag_sims.mean() * w_tag_active if ui_w_tag > 0 else 0.0
    expected_neutral_score = 5.0 + mean_semantic_contrib + mean_tag_contrib
    
    response_items = []
    for i, idx in enumerate(top_indices):
        game_meta = results.iloc[i]
        current_score = final_scores[idx]
        w_q_for_map = abs(w_quality_active) if abs(w_quality_active) > 1e-3 else 1.0
        q_equiv_grid = (current_score - expected_neutral_score) / w_q_for_map
        q_final_probit = np.interp(q_equiv_grid, ref_q_grid, ref_q_natural)
        match_percent = float(norm.cdf(q_final_probit) * 100.0)
        
        lib_details = request.library_details.get(str(game_meta['appid'])) if request.library_details else None
        is_pers = lib_details is not None and lib_details.get('p_plus_t') is not None
        
        item = {
            "appid": int(game_meta['appid']), "name": str(game_meta['name']), "release_date": str(game_meta['release_date']),
            "short_description": str(game_meta.get('short_description', '')), "release_year": int(game_meta['release_year']),
            "estimated_playtime": float(game_meta['estimated_playtime']), "difficulty_predicted": float(game_meta['difficulty_predicted']),
            "positive": int(game_meta['positive']), "negative": int(game_meta['negative']), "genres": str(game_meta['genres']),
            "tags": str(game_meta['tags']), "price": str(game_meta['price'] if pd.notna(game_meta['price']) else "Free"),
            "is_nsfw": bool(game_meta['is_nsfw']), "is_delisted": bool(game_meta['is_delisted']),
            "match_percent": match_percent, "is_personalized": is_pers, "weighted_score": float(np.clip(current_score, 0, 10)),
            "z_semantic": float(semantic_sims[idx]), "w_semantic": float(ui_w_semantic),
            "z_tag": float(tag_sims[idx]), "w_tag": float(ui_w_tag),
            "z_spps": float(z_spps[idx]), "w_spps": float(w_quality_active),
            "z_date": float(z_date[idx]), "w_date": float(w_date_active),
            "z_pop": float(z_pop[idx]), "w_pop": float(w_pop_active),
            "z_length": float(z_length[idx]), "w_length": float(w_length_active),
            "z_difficulty": float(z_difficulty[idx]), "w_difficulty": float(w_difficulty_active),
            "z_price": float(z_price[idx]), "w_price": float(w_price_active),
            "z_tone": float(tone_sims[idx]), "w_tone": float(request.tone_sim_weight),
            "raw_pop": int(game_meta['positive'] + game_meta['negative']), "raw_length": float(game_meta['estimated_playtime'] / 60.0)
        }
        response_items.append(item)

    cleaned_response = ensure_python_types(response_items)
    logger.info(f"/recommend returning {len(cleaned_response)} results")
    return cleaned_response

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

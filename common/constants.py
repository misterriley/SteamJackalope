import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Build Versioning
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD_VERSION_BASE = "0.0.1-pre-alpha"
BUILD_COUNT_FILE = os.path.join(ROOT_DIR, "build_count.json")
try:
    with open(BUILD_COUNT_FILE, "r") as f:
        _bc_data = json.load(f)
        BUILD_COUNT = _bc_data.get("build_count", 0)
except:
    BUILD_COUNT = 0
BUILD_VERSION = f"{BUILD_VERSION_BASE}+build.{BUILD_COUNT}"

# Weights & Multipliers
ABG_NOTCHES_ON_SLIDER = 200
SEMANTIC_WEIGHT_MULTIPLIER = 2.0
TAG_WEIGHT_MULTIPLIER = 2.0
QUALITY_WEIGHT_MULTIPLIER = 4.0
AGE_WEIGHT_MULTIPLIER = 1.4
POPULARITY_WEIGHT_MULTIPLIER = 1.0
PRICE_WEIGHT_MULTIPLIER = 1.0
LENGTH_WEIGHT_MULTIPLIER = 0.25
DIFFICULTY_WEIGHT_MULTIPLIER = 1.3

PRODUCTION_DATA_DIR = os.path.join(ROOT_DIR, "data", "production")

# Environment variable overrides for test isolation and flexibility
EMBEDDINGS_DESC_FILE = os.getenv("STEAM_EMBEDDINGS_DESC_FILE", os.path.join(PRODUCTION_DATA_DIR, "embeddings_desc.npy"))
EMBEDDINGS_DESC_RAW_FILE = os.getenv("STEAM_EMBEDDINGS_DESC_RAW_FILE", os.path.join(PRODUCTION_DATA_DIR, "embeddings_desc_raw.npy"))
EMBEDDINGS_DESC_NORMS_FILE = os.getenv("STEAM_EMBEDDINGS_DESC_NORMS_FILE", os.path.join(PRODUCTION_DATA_DIR, "embeddings_desc_norms.npy"))
TAG_VECTORS_FILE = os.getenv("STEAM_TAG_VECTORS_FILE", os.path.join(PRODUCTION_DATA_DIR, "steam_tag_vectors.npy"))
DIFFUSED_VERB_PROFILES_FILE = os.getenv("STEAM_DIFFUSED_VERB_PROFILES_FILE", os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"))
TAG_NORMS_FILE = os.getenv("STEAM_TAG_NORMS_FILE", os.path.join(PRODUCTION_DATA_DIR, "tag_vectors_norms.npy"))
QUALITY_GRID_FILE = os.getenv("STEAM_QUALITY_GRID_FILE", os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"))
METADATA_FILE = os.getenv("STEAM_METADATA_FILE", os.path.join(PRODUCTION_DATA_DIR, "metadata.parquet"))
DIFFICULTY_PREDICTIONS_FILE = os.getenv("STEAM_DIFFICULTY_PREDICTIONS_FILE", os.path.join(PRODUCTION_DATA_DIR, "difficulty_predictions.csv"))
DIFFICULTY_COEFFICIENTS_FILE = os.getenv("STEAM_DIFFICULTY_COEFFICIENTS_FILE", os.path.join(PRODUCTION_DATA_DIR, "difficulty_coefficients.json"))
SIMILARITY_LISTS_FILE = os.getenv("STEAM_SIMILARITY_LISTS_FILE", os.path.join(PRODUCTION_DATA_DIR, "similarity_lists.json"))
TAG_NAMES_FILE = os.getenv("STEAM_TAG_NAMES_FILE", os.path.join(PRODUCTION_DATA_DIR, "tag_names.json"))
TAG_PRIOR_COUNTS_FILE = os.getenv("STEAM_TAG_PRIOR_COUNTS_FILE", os.path.join(PRODUCTION_DATA_DIR, "tag_prior_counts.npy"))
TAG_PRIOR_TRANSFORMED_FILE = os.getenv("STEAM_TAG_PRIOR_TRANSFORMED_FILE", os.path.join(PRODUCTION_DATA_DIR, "tag_prior_transformed.npy"))
W_DESC_FILE = os.getenv("STEAM_W_DESC_FILE", os.path.join(PRODUCTION_DATA_DIR, "w_desc.npy"))
MEAN_DESC_FILE = os.getenv("STEAM_MEAN_DESC_FILE", os.path.join(PRODUCTION_DATA_DIR, "mean_desc.npy"))
W_TAG_FILE = os.getenv("STEAM_W_TAG_FILE", os.path.join(PRODUCTION_DATA_DIR, "w_tag.npy"))

# Topic Modeling Artifacts
TOPIC_DISTRIBUTIONS_FILE = os.getenv("STEAM_TOPIC_DISTRIBUTIONS_FILE", os.path.join(PRODUCTION_DATA_DIR, "topic_distributions.npy"))
TOPIC_MODEL_FILE = os.getenv("STEAM_TOPIC_MODEL_FILE", os.path.join(PRODUCTION_DATA_DIR, "topic_model.pkl"))
TOPIC_DESCRIPTIONS_FILE = os.getenv("STEAM_TOPIC_DESCRIPTIONS_FILE", os.path.join(PRODUCTION_DATA_DIR, "topic_descriptions.json"))

# Globally optimized playtime sentiment parameters - Initial Defaults
_DEFAULT_PLAYTIME_SENTIMENT_GAMMA = 0.688395
_DEFAULT_PLAYTIME_SENTIMENT_S = 1.452654

PLAYTIME_SENTIMENT_GAMMA = _DEFAULT_PLAYTIME_SENTIMENT_GAMMA
PLAYTIME_SENTIMENT_S = _DEFAULT_PLAYTIME_SENTIMENT_S

# Blending
SOFTMIN_TEMPERATURE = 3.0
SIMILARITY_THRESHOLD_FAVORITES = 0.35

# Regularization & Constants
REGULARIZATION_FILE = os.getenv("STEAM_REGULARIZATION_JSON", os.path.join(PRODUCTION_DATA_DIR, "regularization_constants.json"))
if os.path.exists(REGULARIZATION_FILE):
    try:
        with open(REGULARIZATION_FILE, "r") as f:
            reg_data = json.load(f)
            TAG_VECTOR_K = reg_data.get("TAG_VECTOR_K", 100.0)
            GLOBAL_POSITIVE_RATE = reg_data.get("GLOBAL_POSITIVE_RATE", 0.8)
            DOT_PRODUCT_LAMBDA = reg_data.get("DOT_PRODUCT_LAMBDA", 0.1)
            SEMANTIC_DOT_PRODUCT_LAMBDA = reg_data.get("SEMANTIC_DOT_PRODUCT_LAMBDA", 0.1)
            TOPIC_DOT_PRODUCT_LAMBDA = reg_data.get("TOPIC_DOT_PRODUCT_LAMBDA", 0.1)
            PLAYTIME_REGULARIZATION_C = reg_data.get("PLAYTIME_REGULARIZATION_C", 0.781171)
            TAG_GLOBAL_SCALING_FACTOR = reg_data.get("TAG_GLOBAL_SCALING_FACTOR", 1.0)
            SEMANTIC_GLOBAL_SCALING_FACTOR = reg_data.get("SEMANTIC_GLOBAL_SCALING_FACTOR", 2.0)
            TOPIC_GLOBAL_SCALING_FACTOR = reg_data.get("TOPIC_GLOBAL_SCALING_FACTOR", 26.5)
            SOFTMIN_TEMPERATURE = reg_data.get("SOFTMIN_TEMPERATURE", 3.0)
            QUALITY_TO_RATING_SLOPE = reg_data.get("QUALITY_TO_RATING_SLOPE", 3.008809)
            QUALITY_TO_RATING_INTERCEPT = reg_data.get("QUALITY_TO_RATING_INTERCEPT", 3.277190)
            PLAYTIME_SENTIMENT_GAMMA = reg_data.get("PLAYTIME_SENTIMENT_GAMMA", _DEFAULT_PLAYTIME_SENTIMENT_GAMMA)
            PLAYTIME_SENTIMENT_S = reg_data.get("PLAYTIME_SENTIMENT_S", _DEFAULT_PLAYTIME_SENTIMENT_S)
            SEMANTIC_SIMILARITY_MEAN = reg_data.get("SEMANTIC_SIMILARITY_MEAN", 0.0)
            SEMANTIC_SIMILARITY_STD = reg_data.get("SEMANTIC_SIMILARITY_STD", 1.0)
            TOPIC_SIMILARITY_MEAN = reg_data.get("TOPIC_SIMILARITY_MEAN", 0.84609)
            TOPIC_SIMILARITY_STD = reg_data.get("TOPIC_SIMILARITY_STD", 0.03022)
    except:
        TAG_VECTOR_K = 100.0
        GLOBAL_POSITIVE_RATE = 0.8
        DOT_PRODUCT_LAMBDA = 0.5
        SEMANTIC_DOT_PRODUCT_LAMBDA = 0.1
        TOPIC_DOT_PRODUCT_LAMBDA = 0.1
        PLAYTIME_REGULARIZATION_C = 0.781171
        TAG_GLOBAL_SCALING_FACTOR = 1.0
        SEMANTIC_GLOBAL_SCALING_FACTOR = 2.0
        TOPIC_GLOBAL_SCALING_FACTOR = 26.5
        # SOFTMIN_TEMPERATURE already defined above
        
        QUALITY_TO_RATING_SLOPE = 3.008809
        QUALITY_TO_RATING_INTERCEPT = 3.277190
        # Revert to default if loading fails
        PLAYTIME_SENTIMENT_GAMMA = _DEFAULT_PLAYTIME_SENTIMENT_GAMMA
        PLAYTIME_SENTIMENT_S = _DEFAULT_PLAYTIME_SENTIMENT_S
        SEMANTIC_SIMILARITY_MEAN = 0.0
        SEMANTIC_SIMILARITY_STD = 1.0
        TOPIC_SIMILARITY_MEAN = 0.84609
        TOPIC_SIMILARITY_STD = 0.03022
else:
    TAG_VECTOR_K = 100.0
    GLOBAL_POSITIVE_RATE = 0.8
    DOT_PRODUCT_LAMBDA = 0.5
    SEMANTIC_DOT_PRODUCT_LAMBDA = 0.1
    TOPIC_DOT_PRODUCT_LAMBDA = 0.1
    PLAYTIME_REGULARIZATION_C = 0.781171
    TAG_GLOBAL_SCALING_FACTOR = 1.0
    SEMANTIC_GLOBAL_SCALING_FACTOR = 2.0
    TOPIC_GLOBAL_SCALING_FACTOR = 26.5
    QUALITY_TO_RATING_SLOPE = 3.008809
    QUALITY_TO_RATING_INTERCEPT = 3.277190
    # Also revert to default if file doesn't exist
    PLAYTIME_SENTIMENT_GAMMA = _DEFAULT_PLAYTIME_SENTIMENT_GAMMA
    PLAYTIME_SENTIMENT_S = _DEFAULT_PLAYTIME_SENTIMENT_S
    SEMANTIC_SIMILARITY_MEAN = 0.0
    SEMANTIC_SIMILARITY_STD = 1.0
    TOPIC_SIMILARITY_MEAN = 0.84609
    TOPIC_SIMILARITY_STD = 0.03022
    # Difficulty Model Constants
DIFFICULTY_NEUTRAL_FALLBACK = 5.0

# User Taste DNA Constants
DNA_UI_SCALING_FACTOR = 1.5
DNA_SOLVER_DOF_PROTECTION = 7 # Constraints p <= N - 7



# Pipeline / Research Constants
REG_RATE_MIN_REVIEWS_THRESHOLD = 100
REG_TAG_SAMPLE_SIZE = 1000
REG_TAG_MIN_VOTES_THRESHOLD = 1000
REG_PLAYTIME_REVIEWS_THRESHOLD = 80
REG_PLAYTIME_SAFETY_FACTOR = 2.0
REG_PLAYTIME_C_MIN = 0.1
REG_PLAYTIME_C_MAX = 5000.0

TAG_EM_ITERATIONS = 5
TAG_EM_RIDGE = 1e-4
TAG_OPT_SAMPLE_SIZE = 100000
CHI_FIT_NORM_THRESHOLD = 5.0
CHI_FIT_PERCENTILE = 0.95
USE_TAG_WHITENING = True
TAG_TRANSFORM_TYPE = 'clr' # 'anscombe', or 'clr', or 'none

# Adaptive DNA Complexity (Linear Scaling)
# Formula: K = clamp(BASE + SLOPE * N_ratings, BASE, MAX)
ADAPTIVE_DNA_BASE_K = 40
ADAPTIVE_DNA_SLOPE = 0.7

# Scraping Constants
API_KEY = os.getenv("STEAM_API_KEY")
if API_KEY is None:
    raise ValueError("STEAM_API_KEY environment variable must be set. See .env.example for setup instructions.")
SCRAPE_SLEEP_TIME = 1.0
SCRAPE_BACKOFF_BASE_DELAY = 5.0
SCRAPE_BACKOFF_MAX_RETRIES = 5
CHECKPOINT_INTERVAL = 20
MAX_ERROR_RETRIES = 3
ERROR_IDS_FILE = "data/error_ids.csv"
TRENDING_APPIDS_FILE = os.path.join(ROOT_DIR, "data", "trending_appids.json")
SCRAPE_LOG_FILE = "scrape_steam.log"
SCRAPE_INPROGRESS_SUFFIX = "_inprogress.csv"
SCRAPE_ARCHIVE_CSV_DIR = "scraping/archive_csv"
RAW_DOWNLOAD_PATH = os.getenv("RAW_DOWNLOAD_PATH", "/steam_raw_downloads")
RAW_DOWNLOAD_REVIEWS_PATH = os.path.join(RAW_DOWNLOAD_PATH, "reviews")
ARCHIVE_PATH = os.path.join(RAW_DOWNLOAD_PATH, "archive")

# Quality Scoring Constants
QUALITY_SCORE_S_CONST = 2000
QUALITY_SCORE_S_BASE = 80
QUALITY_SCORE_MIN_VOTES_FOR_RELIABLE = 100
QUALITY_SCORE_CLIP = 1e-6
QUALITY_SCORE_PIN_GROUP = 500
PIN_QUALITY_DISTRIBUTION = True

# Constants for UI and Logic
APP_TITLE = "### First time here? Press \"Surprise Me (Random)\" to get started! Then open Search Options to refine your preferences." # Markdown format
LOADING_APP_TEXT = "Loading massive neural networks..."
UPDATING_RESULTS_TEXT = "Thinking..."
TOP_RECOMMENDATIONS_HEADER = "Top {top_k} Recommendations"

# Sliders
AP_SLIDER_MIN = -2.0
AP_SLIDER_MAX = 2.0
AP_SLIDER_STEP = 0.01
AP_SLIDER_VALUES = [round(x * AP_SLIDER_STEP, 2) for x in range(int(AP_SLIDER_MIN/AP_SLIDER_STEP), int(AP_SLIDER_MAX/AP_SLIDER_STEP) + 1)]

DISC_SLIDER_MIN = -1.0
DISC_SLIDER_MAX = 1.0
DISC_SLIDER_STEP = 0.1
DISC_SLIDER_VALUES = [round(x * DISC_SLIDER_STEP, 1) for x in range(int(DISC_SLIDER_MIN/DISC_SLIDER_STEP), int(DISC_SLIDER_MAX/DISC_SLIDER_STEP) + 1)]

# Labels
APP_HEADER = "### Find your next favorite game"
SIDEBAR_HEADER = "Preferences"

SEMANTIC_WEIGHT_LABEL = "Semantic Match (Text)"
TAG_WEIGHT_LABEL = "Tag Match (Vibes)"
SEMANTIC_WEIGHT_HELP = "How much the description/prompt matches the game."
TAG_WEIGHT_HELP = "How much the game's tags match the seed games."

QUALITY_PREF_LABEL = "Quality Preference"
QUALITY_SLIDER_LABEL = "Quality Bias"
QUALITY_LOVED_LABEL = "Loved"
QUALITY_HATED_LABEL = "Hated"
QUALITY_PREF_HELP = "Bias towards higher rated games."

AGE_PREF_LABEL = "Release Date"
AGE_OLD_LABEL = "Old"
AGE_NEW_LABEL = "New"
AGE_PREF_HELP = "Bias towards newer (to the right) or older (to the left) games."

POP_PREF_LABEL = "Popularity Preference"
POP_SLIDER_LABEL = "Popularity Bias"
POP_NICHE_LABEL = "Niche"
POP_MAINSTREAM_LABEL = "Mainstream"
POP_PREF_HELP = "Bias towards popular games."

PRICE_PREF_LABEL = "Price Preference"
PRICE_SLIDER_LABEL = "Price Bias"
PRICE_CHEAP_LABEL = "Cheap"
PRICE_EXPENSIVE_LABEL = "Expensive"
PRICE_PREF_HELP = "Bias towards cheaper (left) or more expensive (right) games."

DISC_PREF_LABEL = "Discovery Setting"
DISC_SLIDER_LABEL = "Discovery"
DISCOVERY_LABEL_LEFT = "Known Quantities"
DISCOVERY_LABEL_RIGHT = "Wild Cards"
DISC_PREF_HELP = "Controls how much we trust games with few reviews. 'Wild Cards' allows hidden gems to surface."

LENGTH_PREF_LABEL = "Game Length Preference"
LENGTH_SLIDER_LABEL = "Length Bias"
LENGTH_SHORT_LABEL = "Short"
LENGTH_LONG_LABEL = "Long"
LENGTH_PREF_HELP = "Bias towards shorter or longer games."

DIFFICULTY_PREF_LABEL = "Difficulty Preference"
DIFFICULTY_SLIDER_LABEL = "Difficulty Bias"
DIFFICULTY_EASY_LABEL = "Easy"
DIFFICULTY_HARD_LABEL = "Hard"
DIFFICULTY_PREF_HELP = "Bias towards easier or harder games (predicted)."

REMOVE_VR_LABEL = "Filter VR Only"
REMOVE_VR_HELP = "Excludes games that require a VR headset."
REMOVE_VR_DEFAULT = True

ENGLISH_ONLY_LABEL = "Filter Non-English"
ENGLISH_ONLY_HELP = "Excludes games that do not list English as a supported language."
REMOVE_NON_ENGLISH_DEFAULT = True

REMOVE_NSFW_LABEL = "Filter Adult Content"
REMOVE_NSFW_HELP = "Excludes games with 'Mature', 'Nudity', or 'Sexual Content' tags."
REMOVE_NSFW_DEFAULT = True

REMOVE_UTILITIES_LABEL = "Filter Software/Utilities"
REMOVE_UTILITIES_HELP = "Excludes non-game software."
REMOVE_UTILITIES_DEFAULT = True

REMOVE_UNRELEASED_LABEL = "Filter Unreleased"
REMOVE_UNRELEASED_HELP = "Excludes games that haven't been released yet."
REMOVE_UNRELEASED_DEFAULT = True

REMOVE_DELISTED_LABEL = "Filter Delisted"
REMOVE_DELISTED_HELP = "Excludes games that are no longer available for purchase on Steam."
REMOVE_DELISTED_DEFAULT = True

GENRE_FILTER_LABEL = "Filter by Genre"
GENRE_FILTER_HELP = "Only show games that match at least one selected genre."

DEBUG_MODE_LABEL = "Debug Mode"
DEBUG_MODE_HELP = "Show detailed scoring breakdown."
DEBUG_MODE_DEFAULT = False

TOP_K_LABEL = "Number of Recommendations"
TOP_K_HELP = "How many games to show."
TOP_K_DEFAULT = 10
TOP_K_MAX = 500

PROMPT_LABEL = "Describe what you want to play:"
PROMPT_PLACEHOLDER = "e.g. 'a relaxing farming sim', 'fast paced shooter', 'visual novel about romance'"
PROMPT_HELP = "Enter a natural language description."

SEED_LABEL = "Games you like (Seeds):"
SEED_HELP = "Select games to use as a baseline for recommendations."

RESET_BUTTON_LABEL = "Reset to Defaults"
RANDOM_BUTTON_LABEL = "Surprise Me (Random)"
USE_SEED_BUTTON = "Use '{game_name}' as Seed"

# Display Labels
APPID_LABEL = "**AppID:**"
VIEW_ON_STEAM_LINK = "[View on Steam]"
RELEASE_DATE_LABEL = "**Release Date:**"
RELEASE_DATE_UNKNOWN_TEXT = "Unknown"
ESTIMATED_LENGTH_LABEL = "**Est. Length:**"
DIFFICULTY_SCORE_LABEL = "**Difficulty:**"
QUALITY_SCORE_LABEL = "**Rating:**"
SEMANTIC_SIMILARITY_LABEL = "**Semantic Match:**"
TAG_SIMILARITY_LABEL = "**Tag Match:**"
GENRES_LABEL = "**Genres:**"
TAGS_LABEL = "**Top Tags:**"

DEBUG_INFO_HEADER = "#### Debug Info"
DEBUG_SEMANTIC_LABEL = "**Semantic:**"
DEBUG_TAG_LABEL = "**Tag:**"
DEBUG_QUALITY_LABEL = "**Quality:**"
DEBUG_AGE_LABEL = "**Date:**"
DEBUG_POP_LABEL = "**Pop:**"
DEBUG_PRICE_LABEL = "**Price:**"
DEBUG_LENGTH_LABEL = "**Length:**"
DEBUG_DIFFICULTY_LABEL = "**Diff:**"
DEBUG_TOTAL_SUM_LABEL = "**Final Score:**"

DATA_SOURCE_CAPTION = "Data scraped from Steam. Not affiliated with Valve Corporation."
ERROR_LOADING_DATA = "Error loading data: {e}"

# Footer
FOOTER_COPYRIGHT = "© 2025 Steam Jackalope. All rights reserved."
FOOTER_GITHUB_LINK = "[GitHub](https://github.com/misterriley/SteamJackalope)"
FOOTER_TEXT = "{copyright} | {github_link}"

# Tabs
RECOMMENDER_TAB = "Recommender"
ABOUT_TAB = "About"
METHODOLOGY_TAB = "Methodology"
METHODOLOGY_ERROR = "Methodology file not found."
ABOUT_ERROR = "About file not found."

# Text
APP_IMAGE_TEXT = """
**Steam Jackalope** is a hybrid recommender system that blends:
*   **Semantic Search:** Understanding the vibes of games and what people say about them.
*   **Tag Matching:** Analyzing the similarity of tag distributions.
*   **Bayesian Scoring:** robust estimation of several factors, which handles 'hidden gems' and rarely played games.
"""

# Logic Constants
EPSILON = 1e-9
Z_SCORE_CLAMP_MIN = -8.0
Z_SCORE_CLAMP_MAX = 8.0
TOP_K_SORT_MULTIPLIER = 2  # Extract 2x requested K then sort fully to handle ties
SEMANTIC_PROMPT_SEED_BLEND = 0.5 # 50% prompt, 50% seeds for semantic vector
MODEL_NAME = 'all-mpnet-base-v2'
SENTENCE_TRANSFORMER_BACKEND = 'torch'
SENTENCE_TRANSFORMER_MODEL_KWARGS = {}

# Jackalope Kernel Constants
KERNEL_VETO_PENALTY = 0.001
KERNEL_RESCUE_THRESHOLD = 0.15
KERNEL_SOUL_MATCH_THRESHOLD = 0.4
KERNEL_MOOD_CLASH_PENALTY = 0.6
KERNEL_CINEMATIC_BOOST = 0.05
KERNEL_CRPG_BOOST = 0.05
KERNEL_SOFT_GATE_TEMP = 0.05

# Mechanical & Thematic Tag Sets
HORROR_MARKERS = {"Horror", "Survival Horror", "Psychological Horror", "Gore", "Violent"}
HARD_ANCHORS = {"Platformer", "Puzzle", "Roguelike", "Souls-like", "Metroidvania", "Survival", "FPS", "First-Person", "Third Person", "Third-Person Shooter", "Shooter", "Walking Simulator", "Isometric", "CRPG", "RPG"}
SYMMETRIC_ANCHORS = {"First-Person", "Third Person", "Isometric", "2D", "VR", "VR Only"}
SERIOUS_TAGS = {"Education", "Math", "Science", "Typing", "Spelling", "Programming", "Logic"}
CUTE_TAGS = {"Cute", "Colorful", "Family Friendly", "Relaxing", "Anime"}
SERIOUS_MOOD_TAGS = {"Emotional", "Cinematic", "Story Rich", "Atmospheric", "Beautiful", "Dark", "Realistic", "Horror", "Psychological Horror"}
LIGHT_MOOD_TAGS = {"Funny", "Comedy", "Dark Comedy", "Satire", "Parody", "Cartoony", "Cute", "Casual", "Relaxing"}
NARRATIVE_TAGS = {"Visual Novel", "Interactive Fiction", "Story Rich", "Multiple Endings", "Choices Matter", "Narrative", "Character Customization", "Lore-Rich", "Emotional", "Cinematic"}

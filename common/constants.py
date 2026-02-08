import json
import os

# Backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Weights & Multipliers
ABG_NOTCHES_ON_SLIDER = 100
SEMANTIC_WEIGHT_MULTIPLIER = 1.0
TAG_WEIGHT_MULTIPLIER = 1.0
QUALITY_WEIGHT_MULTIPLIER = 4.0
AGE_WEIGHT_MULTIPLIER = 1.4
POPULARITY_WEIGHT_MULTIPLIER = 1.0
LENGTH_WEIGHT_MULTIPLIER = 0.25
DIFFICULTY_WEIGHT_MULTIPLIER = 1.3

# Files
EMBEDDINGS_DESC_FILE = "embeddings_desc.npy"
EMBEDDINGS_TAG_FILE = "embeddings_structural.npy"
TAG_VECTORS_FILE = "steam_tag_vectors.npy"
QUALITY_GRID_FILE = "quality_scores_grid.npy"
METADATA_FILE = "metadata.parquet"
W_DESC_FILE = "w_desc.npy"
W_STRUCTURAL_FILE = "w_structural.npy"
MEAN_DESC_FILE = "mean_desc.npy"
MEAN_STRUCTURAL_FILE = "mean_structural.npy"
W_TAG_FILE = "w_tag.npy"

# Regularization & Constants
REGULARIZATION_FILE = "regularization_constants.json"
if os.path.exists(REGULARIZATION_FILE):
    try:
        with open(REGULARIZATION_FILE, "r") as f:
            reg_data = json.load(f)
            TAG_VECTOR_K = reg_data.get("TAG_VECTOR_K", 100.0)
            GLOBAL_POSITIVE_RATE = reg_data.get("GLOBAL_POSITIVE_RATE", 0.8)
            DOT_PRODUCT_LAMBDA = reg_data.get("DOT_PRODUCT_LAMBDA", 0.1)
            PLAYTIME_REGULARIZATION_C = reg_data.get("PLAYTIME_REGULARIZATION_C", 100.0)
    except:
        TAG_VECTOR_K = 100.0
        GLOBAL_POSITIVE_RATE = 0.8
        DOT_PRODUCT_LAMBDA = 0.5
        PLAYTIME_REGULARIZATION_C = 100.0
else:
    TAG_VECTOR_K = 100.0
    GLOBAL_POSITIVE_RATE = 0.8
    DOT_PRODUCT_LAMBDA = 0.5
    PLAYTIME_REGULARIZATION_C = 100.0

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

# Scraping Constants
API_KEY = os.getenv("STEAM_API_KEY", "7DFED2D02BD057C12AA22C994885F3C9")
SCRAPE_SLEEP_TIME = 1.0
SCRAPE_BACKOFF_BASE_DELAY = 5.0
SCRAPE_BACKOFF_MAX_RETRIES = 5
CHECKPOINT_INTERVAL = 20
MAX_ERROR_RETRIES = 3
ERROR_IDS_FILE = "data/error_ids.csv"
SCRAPE_LOG_FILE = "scrape_steam.log"
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
APP_TITLE = "Steam Jackalope v0.0.1 (pre-pre-alpha)"
LOADING_APP_TEXT = "Loading massive neural networks..."
UPDATING_RESULTS_TEXT = "Thinking..."
TOP_RECOMMENDATIONS_HEADER = "Top {top_k} Recommendations"

# Sliders
AP_SLIDER_MIN = -1.0
AP_SLIDER_MAX = 1.0
AP_SLIDER_STEP = 0.01
AP_SLIDER_VALUES = [round(x * AP_SLIDER_STEP, 2) for x in range(int(AP_SLIDER_MIN/AP_SLIDER_STEP), int(AP_SLIDER_MAX/AP_SLIDER_STEP) + 1)]

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

AGE_PREF_LABEL = "Release Date Preference"
AGE_OLD_LABEL = "Old"
AGE_NEW_LABEL = "New"
AGE_PREF_HELP = "Bias towards newer games."

POP_PREF_LABEL = "Popularity Preference"
POP_SLIDER_LABEL = "Popularity Bias"
POP_NICHE_LABEL = "Niche"
POP_MAINSTREAM_LABEL = "Mainstream"
POP_PREF_HELP = "Bias towards popular games."

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
DEBUG_AGE_LABEL = "**Age:**"
DEBUG_POP_LABEL = "**Pop:**"
DEBUG_LENGTH_LABEL = "**Length:**"
DEBUG_DIFFICULTY_LABEL = "**Diff:**"
DEBUG_TOTAL_SUM_LABEL = "**Final Score:**"

DATA_SOURCE_CAPTION = "Data scraped from Steam. Not affiliated with Valve Corporation."
ERROR_LOADING_DATA = "Error loading data: {e}"

# Tabs
RECOMMENDER_TAB = "Recommender"
METHODOLOGY_TAB = "Methodology"
METHODOLOGY_ERROR = "Methodology file not found."

# Text
APP_IMAGE_TEXT = """
**The Jackalope** is a hybrid recommender system that blends:
*   **Semantic Search:** Understanding the meaning of your request.
*   **Vibe Matching:** Analyzing tag distributions to find similar 'feels'.
*   **Bayesian Scoring:** robust quality estimation that handles 'hidden gems'.
"""

# Logic Constants
EPSILON = 1e-9
Z_SCORE_CLAMP_MIN = -8.0
Z_SCORE_CLAMP_MAX = 8.0
TOP_K_SORT_MULTIPLIER = 2  # Extract 2x requested K then sort fully to handle ties
SEMANTIC_PROMPT_SEED_BLEND = 0.5 # 50% prompt, 50% seeds for semantic vector
MODEL_NAME = 'all-MiniLM-L6-v2'

# NSFW Definitions
NSFW_TAGS = ['sexual content', 'nudity', 'nsfw', 'hentai', 'mature', 'adult only']
NSFW_NAME_PATTERNS = [r'\b18\+\b', 'uncensored', 'adult only']

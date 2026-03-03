# Welcome to the Steam Natural Language Search Repository

This project is a Steam game recommendation engine that combines semantic search, tag analysis, and Bayesian quality scoring to provide highly relevant game suggestions.

## Environment

- This is being executed in VS Code via the Cline extension. 
- The operating system is Windows 11.

## Before You Start
**CRITICAL**: Read `user.md` immediately. It contains the user's non-negotiable standards for **Mathematical Parity** and **Transparency Mode**. Do not attempt to refactor the scoring logic or UI sliders without understanding the user's preference for absolute weight control over hidden multipliers.

## Repository Overview

For detailed documentation of each directory, see the individual README files:
- `app/README.md` - Backend and legacy frontend application code
- `frontend/README.md` - Modern React + TypeScript frontend
- `common/README.md` - Shared utilities and constants
- `pipeline/README.md` - Data processing and artifact generation
- `scraping/README.md` - Steam data collection scripts
- `research/README.md` - Experimental and analytical scripts
- `tests/README.md` - Automated test suite
- `tools/README.md` - Debugging and maintenance utilities
- `deployment/README.md` - Deployment configurations
- `data/README.md` - Intermediate data files
- `user.md` - User profile, technical preferences, and expectations

### Key Components

- `app/server.py`: The backend FastAPI server that handles data loading and hybrid score calculations. Start with `.\venv310\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000`.
- `frontend/`: Modern React 19 + TypeScript + Vite + Tailwind CSS v4 frontend. Located in the `frontend/` directory. Start with `cd frontend; npm run dev`.
- `app/app.py`: Legacy frontend Streamlit UI that communicates with the backend server. Start with `streamlit run app/app.py`.
- `run_test_env.bat`: The primary Windows entry point for local development. It pulls the latest code, updates dependencies, and launches both the FastAPI backend and the React frontend.
- `run_all_tests.bat`: A convenience batch file to run the full test suite using `pytest`.
- `deployment/deploy.sh`: The production deployment script for Linux servers. It builds the React frontend for optimized serving via the unified FastAPI process.
- `onPush.md`: Guidelines for updating the changelog and maintaining versioning before pushing to the remote repository.
- `pipeline/run_pipeline.py`: Orchestrates the data processing pipeline (tags -> semantic vectors -> metadata -> quality scores). Uses `pipeline/pipeline_config.json` for path and interval settings.

### GPU & Environment (Blackwell Support)

- **Python 3.10 Venv (`venv310`)**: This project utilizes a specialized Python 3.10 virtual environment to ensure compatibility with **PyTorch Nightly (CUDA 12.8)**. This is required to support **NVIDIA Blackwell (RTX 50-series)** GPUs. Always use `.\venv310\Scripts\python.exe` for running scripts to ensure GPU acceleration is available.
- **GPU Acceleration**: Both the backend server and the data pipeline automatically detect and utilize CUDA if available. This significantly speeds up prompt embedding and artifact generation.
- **Semantic Model**: The project uses the high-quality **`all-mpnet-base-v2`** model (768 dimensions) for descriptive embeddings.
- **Descriptive-Only Path**: Semantic similarity is calculated strictly using narrative text (descriptions and reviews). Structural metadata (tags/genres) is excluded from the semantic slider to avoid redundant categorical matching and focus on qualitative "vibes."

### Push Workflow (`onPush.md`)

The `onPush.md` file serves as a protocol for all contributors. Before pushing changes to the remote, you MUST review the commit history since the last push and update `CHANGELOG.md`. This ensures that all algorithmic, UI, and deployment changes are properly documented and surfaced to the user via the website's Changelog view.
- `pipeline/pipeline_config.json`: Configuration file for `run_pipeline.py` defaults.
- `common/constants.py`: Centralized configuration and hyperparameters. Loads from `pipeline/regularization_constants.json` if present.
- `pipeline/calculate_regularization.py`: Script to derive data-driven regularization constants.
- `common/utils.py`: Shared utility functions (z-scoring, hybrid score calculation).
- `methodology.md`: Detailed explanation of the statistical and machine learning techniques used. Note: `app.py` renders this by splitting at `![` to inject `st.image` calls, so image placement should follow this pattern. Note that this file is intended for human viewers of the website, so it should not be a place where notes about debugging or information related to internal workings of the app of statistical pipeline should go. Those types of notes should be put into `orientation.md`. 
- `data/`: Contains intermediate data files (see `data/README.md`). Production artifacts (embeddings, tag vectors, metadata, quality scores) are stored in the root directory.
- `scraping/scrape_from_start.bat`: Convenience script to start scraping from the beginning (archives existing data).
- `scraping/scrape_from_last_checkpoint.bat`: Convenience script to resume scraping from the last checkpoint.
- `research/`: Contains experimental scripts and data analysis notebooks.
- `common/common_adjectives.txt`: A curated list of common English adjectives used for the "Random" button functionality.
- `tools/generate_random_recommendation.py`: A CLI tool for generating Discord-ready recommendations using randomized parameters and high discovery.

## Current Project Status

- Core Python scripts (`app/app.py`, `pipeline/run_pipeline.py`, `scraping/scrape_steam.py`, `pipeline/generate_tag_vectors.py`, `pipeline/generate_quality_scores_grid.py`) have been updated with descriptive docstrings.
- `scraping/scrape_steam.py` uses direct **Storefront Scraping** (HTML/Embedded JSON) to retrieve high-fidelity user tags and metadata, bypassing age gates with specific cookies. It includes robust error handling, exponential backoff, and a **Hierarchical Local Cache** with historical archiving to prevent redundant downloads. Review counts are prioritized from **English** sources to better reflect the perspective of the target audience, falling back to global counts if no English reviews are available.
- `scraping/get_steam_appids.py` retrieves the full list of Steam AppIDs. It includes an automatic fallback to the public `ISteamApps/GetAppList/v2` endpoint if the primary `IStoreService` API call fails (e.g., due to an invalid API key).
- The pipeline is functional but could benefit from better progress monitoring.
- **Client/Server Architecture:** The project uses a decoupled architecture where heavy data loading and vector computations are performed by a FastAPI backend (`app/server.py`), while a modern React frontend (`frontend/`) or a legacy Streamlit UI (`app/app.py`) provides the interface.
- **Metadata and Search:** The backend provides `/metadata`, `/genres`, `/term_links`, `/games/search`, and `/games/random` endpoints. `/games/search` enables fast autocomplete for seed game selection.
- **Personalization Engine:** The system features a sophisticated "Taste DNA" pipeline.
    - **Async Solver:** The `/user/solve` endpoint runs an asynchronous subprocess to build a user's mathematical profile from their Steam history.
    - **Archetypal Ridge Regression (Build 50)**: The solver uses **Archetypal Ridge Regression** across 45 structural features (38 MIGs, 1 Quality Probit, 6 Metadata Z-scores). This prevents overfitting and ensures that the learned model generalizes to the entire Steam population. $R^2$ is typically around 0.28.
    - **Differentiable Sigmoid-Based Kernel**: The Jackalope kernel has transitioned from hard Booleans to a **Differentiable Soft-Gate** framework. It uses sigmoid transitions ($1 / (1 + exp(-x))$) with controllable temperature to manage mechanical identity, mood clashes, and un-rescueable conflicts.
    - **Symmetric Hard Anchors**: Perspective (First-Person, Isometric) and strict interaction models (CRPG, RPG) are enforced symmetrically. If a candidate possesses a hard anchor that the seed lacks, it is strictly penalized, ensuring structural consistency.
    - **Status-Based Training (Build 55)**: The solver trains **exclusively** on games marked **'rated'** in the catalogue. Games marked 'played', 'backlog', or 'wishlist' provide library metadata for exclusion but do not influence the vibe vector calculation.
    - **Zero-Order Relevance Filtering**: To prevent overfitting and increase training $R^2$, the solver applies a Pearson correlation filter to all 700+ thematic dimensions. Only the top $P$ dimensions ($P \le N-7$, where $N$ is the number of ratings) are eligible for the LASSO model.
    - **Population-Correct Scaling**: Modalities are scaled based on population variance to ensure fair competition in the solver: Tags (1.0x), Semantics (~2.0x), and Topics (~26.4x).
    - **Adaptive Saturation:** Tag dimensionality scales dynamically with library size ($K = \text{clip}(N-6, 1, 243)$).
    - **Ordinal vs. Mathematical Parity:** While the system was designed for bit-perfect parity, the primary requirement is **Ordinal Parity**. The order of games retrieved from "Analyze My Catalogue" must match the order in "Recommendations" after exporting the profile. A linear transformation between the scoring spaces is acceptable as long as the relative ranking of games is preserved.
    - **Unified Pathway:** Both the Solver's preview and the Recommender's rankings utilize the exact same `calculate_linear_scores` logic in `common/utils.py`.
    - **Absolute Slider Control:** Build 39 refactored the UI so that sliders directly represent the solved absolute weights, allowing for transparent fine-tuning without "squaring" or multiplier confusion.
- **Frontend Features:** The modern frontend includes real-time filtering, clickable tag/genre links to Steam, weight contribution visualization, and **NSFW Blurring**. 
- **Profile Filtering:** Users can toggle between three exclusion modes: **None**, **Rated** (exclude games verify-rated in the UI), or **All** (exclude every game in their Steam library).
- **Verification:** Automated tests are handled via `run_all_tests.bat`. Manual verification of UI and discovery features is documented in `QA.md`.
- **Networking:** `127.0.0.1` is preferred over `localhost` for local backend connectivity on Windows to avoid latency and connection issues.
- **Tool Usage Constraints**: 
    - **PowerShell Separators**: When using `run_shell_command`, do not use `&&` as a statement separator, as the environment uses PowerShell by default. Use `;` instead.
    - **Path Resolution**: Relative paths are resolved against the project root. Use `common/constants.py` as the source of truth for all data paths.
    - **Environment Variables**: Use `$env:PYTHONPATH="."` prefix for shell commands requiring module imports from the root.
    - When using `run_shell_command` in this environment (PowerShell), avoid using the `&&` operator to chain commands as it will cause a `ParserError`. Execute commands sequentially instead.
    - Many data artifacts (`.npy`, `.parquet`) are memory-mapped by the server. If a script fails to update them with a "Permission Denied" or "File in Use" error, the FastAPI server MUST be stopped first.
    - Large files read via `read_file` will be truncated. Use `offset` and `limit` to paginate through long files like `server.py` or large CSVs.
    - `grep_search` and `read_file` need very precise `include` and `file_path` parameters respectively for efficient searching and reading.
    - `replace` tool requires a very exact `old_string` parameter, including surrounding context (e.g., 3 lines above and below) and precise whitespace/indentation, to ensure accurate modifications.
- **Memory Footprint Optimization**: The backend server is optimized for efficiency to support standard cloud deployment.
    - **Memory Mapping**: All large NumPy arrays (`embeddings_desc.npy`, `embeddings_structural.npy`, `tag_vectors`, `quality_grid`) now utilize `mmap_mode='r'`. This allows the OS to page data in and out as needed, keeping the active Resident Set Size (RSS) low.
    - **Pre-Normalization**: Semantic embeddings are pre-normalized to unit length.
    - **Dimensionality Reduction**: `steam_tag_vectors.npy` has been reduced to **243 dimensions** (via truncated PCA-ZCA whitening at a **95% variance threshold**) and `quality_scores_grid.npy` to 21 steps to minimize file size and cache pressure.
    - **Numerical Stability**: When computing statistics (mean/std) on large float16 arrays (like the quality grid or embedding matrices), always use `dtype=np.float64` to prevent numerical overflow and `inf` results. The `common.utils.to_z` function handles this automatically.
    - **Metadata Optimization**: Uses `pyarrow` backend for efficient string storage and avoids creating Python list objects for genres/tags where possible.
    - **Thread Limiting**: `OMP_NUM_THREADS` and related variables are set to `1` to prevent excessive buffer allocation by linear algebra libraries.
- **NSFW Blur Architecture**: Content sensitivity is managed via a "Blur, don't filter" policy. Games are flagged as `is_nsfw` based on metadata tags, and the frontend applies a CSS blur filter to header images based on the global `remove_nsfw` toggle.
- **Improved Debug Mode:** The recommender's game cards now feature a comprehensive debug section. It provides a full breakdown of the hybrid score calculation, including raw values, z-scores, and both slider and hard-coded weights for every component.
- **Unified Server (FastAPI Static Serving)**: The backend FastAPI server is configured to serve the modern React frontend's production build (`frontend/dist`) as a fallback for non-API routes. This enables a single-process deployment (Unified Server) where both the API and the UI are available on the same port (e.g., port 8000 in production). The `deployment/deploy.sh` script automates the frontend build and server restart on Linux.
- **Linux Shell Scripts**: Re-created Linux shell script equivalents (`.sh`) for all core Windows workflows (`.bat`), including `run_all_tests.sh`, `run_test_env.sh`, and scraping scripts, ensuring cross-platform development parity.
- **UI Stability & The "Single Card Bug"**: A known rendering failure where the recommendation grid collapses to a single card has been identified. This is triggered by `display: flex` on the `body` tag in `index.css` (which interferes with React's grid rendering) and by React key collisions. The fix involves ensuring `body` does not use flexbox and that all grid items use unique, persistent keys (like `appid`).

- **Tag Dimension Explainability**: To make the latent dimensions from PCA/ZCA whitening interpretable, the project now includes human-readable descriptions for each dimension (e.g., "Tactical Combat vs Adult Themes"). These are generated by `research/analyze_tag_dimensions.py` which inspects the whitening matrix, and the descriptions are served to the frontend via the `/tag_dimensions` endpoint.

- **Deployment**: The project is configured for deployment on **Render** using a single-container architecture (FastAPI backend + Streamlit frontend).
    - `Dockerfile`: Configured to run both services, binding Streamlit to the `$PORT` environment variable.
    - `render.yaml`: Blueprint for one-click deployment.
    - `run_test_env.bat`: Mirrors the production setup locally for testing.

- Data files like `embeddings.npy`, `steam_tag_vectors.npy`, and `metadata.parquet` are the core artifacts used by the recommender. 
    - **CRITICAL:** `metadata.parquet` should reside in `data/production/`. The path in `common/constants.py` is the single source of truth.
- `metadata.parquet` includes pre-calculated z-scores for release dates (`date_z`), popularity (`pop_z`), and game length (`playtime_z`), as well as a cleaned `release_year` for fast display.
    - **Age Z-Scoring**: Future release dates are clamped to today to prevent skewing the distribution. Unknown dates are assigned a neutral z-score of 0.
    - **Review Count Repair**: `generate_metadata.py` automatically repairs stale review counts using the raw individual reviews found in `scraped_reviews.csv` if they exceed the global reported counts.
- **CRITICAL:** All `.npy` artifacts and `metadata.parquet` must have perfectly synchronized row counts and ordering. The pipeline ensures this by using `data/pipeline_games_clean.csv` as the source of truth for appids. Manual script execution should point to this file (e.g., `python pipeline/generate_metadata.py data/pipeline_games_clean.csv`) to avoid `IndexError` in the app. Note that the pipeline explicitly filters out games with missing names (`dropna(subset=['appid', 'name'])`) to ensure consistency between scraped CSVs and downstream parquet/numpy artifacts; manual runs of `generate_metadata.py` on raw CSVs without this step have previously caused index shifts.
- **Tag Cleaning:** `pipeline/generate_semantic_vectors.py` extracts tag names from the dictionary string before embedding them in the structural vector. This ensures the LLM focuses on semantics rather than Python dictionary syntax.
- **Optimization:** `app/app.py` caches pre-normalized embedding matrices to speed up cosine similarity calculations.
- **Regularization Constants:** `pipeline/run_pipeline.py` calls `pipeline/calculate_regularization.py` to generate `pipeline/regularization_constants.json`. These values (like `TAG_VECTOR_K` and `GLOBAL_POSITIVE_RATE`) are used by subsequent pipeline stages and the app via `common/constants.py`.
- **Whitening Stability**: The tag vector pipeline uses **Truncated PCA-ZCA Whitening**. This is critical for stability in the CLR-transformed tag space, as it eliminates singular dimensions that would otherwise cause numerical noise to explode and create false similarities between disparate games.
- **Dependencies:** The lists page requires the `tabulate` library for markdown table generation.
    - **Note on `TAG_VECTOR_K`:** The solver for `TAG_VECTOR_K` must include the LOD imputation step in its cross-validation loop to be consistent with the pipeline. Ignoring imputation leads to a near-zero value because raw counts are highly distinct; including it correctly identifies the need to smooth the imputed "baseline" towards the global distribution.
    - **Note on `DOT_PRODUCT_LAMBDA`:** This regularization constant is calculated by fitting a Chi-distribution to the lengths (norms) of "low-tag" vectors (norms between 0 and 5). $\lambda$ is set to the 95th percentile of this fitted distribution. This represents the "noise floor" and ensures that a vector's length must be statistically significant before it can achieve high similarity scores.
    - **Tag Vector Normalization:** The 'Anscombe' and 'None' transforms in `generate_tag_vectors.py` now explicitly normalize the resulting vectors (and the prior) to sum to 1 before centering. This ensures the resulting embedding captures the shape of the distribution rather than the magnitude of counts.
- **Z-Scoring Logic:** Component scores (Release Date, Popularity, Length, Difficulty) are converted to Z-scores using the global distribution to ensure they are on the same 0-1.0 variance scale. In Build 16, **Tag Matching** was unified with the Linear Scorer; it uses a penalized dot product scaled by **11.283x** to achieve a similar variance without requiring neighborhood-dependent Z-scoring. This ensures that the contribution of a specific tag match is absolute and predictable across different search results. All component scores are clamped between `Z_SCORE_CLAMP_MIN` and `Z_SCORE_CLAMP_MAX` (defined in `common/constants.py`) to prevent outliers from dominating the hybrid ranking. 
- **Note on `quality_scores_grid.npy`:** Unlike other artifacts where rows correspond to games, this file is a 2D grid of shape `(num_steps, num_games)`, where each row represents a different popularity preference setting. The number of games is matched along the second dimension (axis 1).
- **Quality Score Normalization**: `generate_quality_scores_grid.py` supports a `PIN_QUALITY_DISTRIBUTION` toggle in `constants.py`. When enabled, it pins the top and bottom means of each distribution to a baseline spread, ensuring consistent scoring magnitude regardless of Bayesian regularization strength.
- **Tag Vector Norms**: The file `tag_vectors_norms.npy` contains the precalculated L2 norms of the tag vectors. It is automatically generated by `pipeline/generate_tag_vectors.py` to speed up similarity calculations in the app. If this file becomes out of sync with `steam_tag_vectors.npy`, it can be regenerated using `tools/precalculate_tag_norms.py`. The backend server includes robustness checks to detect and handle shape mismatches by recalculating norms on-the-fly if necessary.
- The project uses `all-MiniLM-L6-v2` for semantic embeddings. To improve recommendation precision, it uses a **Dual Semantic Vector** system: `embeddings_structural.npy` (Genres + Tags) and `embeddings_desc.npy` (Description + Reviews). The app blends prompt matches against both, but performs specialized seed matching (structural-to-structural and descriptive-to-descriptive) to ensure categorical and narrative vibes are handled independently.
- **Difficulty Estimation**: Game cards now display a predicted 1-5 difficulty score derived from a model trained on GameFAQs ground truth data.
- **UI Enhancements**: The recommender includes a sorted seed game selection list and a global "Reset to Defaults" button for tuning parameters.
- **ZCA Whitening**: Semantic embeddings are centered and whitened using ZCA (Zero-phase Component Analysis) to decorrelate dimensions and improve query precision. The mean vectors and transformation matrices are saved (e.g., `mean_desc.npy`, `w_desc.npy`) and are applied to user queries at runtime in `app.py`. Analysis of the whitened vectors confirms that they are successfully centered at zero and exhibit identity covariance on their active dimensions.
- **Note on Zero-Variance Dimensions:** The `all-MiniLM-L6-v2` model naturally has three zero-variance dimensions (indices 127, 223, and 319). These are preserved during whitening, resulting in 381 active semantic dimensions.

- **Playtime Sentiment Analysis**: We use a kernel smoothing model to predict review sentiment from playtime.
    - **Vectorization**: Leave-One-Out (LOO) calculations are fully vectorized by computing the full kernel matrix and zeroing the diagonal (`np.fill_diagonal(W, 0)`). This provides exact LOO estimates in $O(N^2)$ time with minimal Python overhead.
    - **Global Optimization**: Optimal parameters ($\gamma = 0.5109$, $s = 0.7812$) were derived using a parallelized grid search over the total log-likelihood of 100 randomly sampled games. The `research/optimize_global_playtime_params_parallel.py` script utilizes all CPU cores via `multiprocessing.Pool`.
    - **Stability**: Multiple runs confirm that $\gamma \approx 0.51$ is a robust global bandwidth for the Steam dataset.

- **Scraping Architecture**: The scraping process is split into three steps: download (`download_steam_data.py`), dataset build (`build_scraped_dataset.py`), and pipeline integration. It features robust interruption tolerance and a retry mechanism for file operations on Windows. The scraper writes to temporary "in-progress" files and only atomically replaces the production CSVs (`scraped_games.csv` and `scraped_reviews.csv`) upon successful completion. This ensures the pipeline can safely run concurrently with a long-running scrape without file locking issues or data inconsistency. Existing production files are archived to `scraping/archive_csv/` before replacement.

- **Git LFS for Production Data**: Large production artifacts (embeddings, tag vectors, quality grids) are tracked via **Git LFS**. This allows the repository to stay under GitHub's file size limits while ensuring Render can pull the necessary binary data for deployment.

- **Slider Impact Simulation**: A simulation study (`research/simulate_slider_impact.py`) quantifies the sensitivity of the recommendation leaderboard to different slider adjustments. The script is multithreaded (using up to 24 threads) for efficient execution. Results show that for the top 100 recommendations, the **Quality** slider has the highest impact (average shift of ~0.44 z-score units per notch), followed by **Length** and **Age**. The **Discovery** slider has the lowest relative impact (~60x smaller) because it influences the underlying data distribution rather than the component weights that drive the hybrid score normalization.

- **Distribution Analysis**: A utility `research/analyze_vector_distributions.py` is used to verify that vector lengths and cosine similarities follow theoretical normal distribution expectations.
- **Interrogation Tool**: `tools/interrogate_vectors.py` provides an interactive GUI (Streamlit) for inspecting vector distributions, game-to-game similarities, and prompt matching. Run with `streamlit run tools/interrogate_vectors.py`.
- The "tags" data in `scraped_games.csv` is censored at 20 entries, even though users may have entered tags beyond these top 20. Imputation for tag data is therefore only necessary if the tags have 20 nonzero count tags - if there are fewer than 20, then the data is exact and no imputation is needed.

## Research Findings (Difficulty Prediction)

A series of research tests were conducted to predict game difficulty, initially using Steam tags ($y = z_{\text{Difficult}} + z_{\text{Unforgiving}}$) and later integrating external data.

- **Supervised PCA**: Identified a strong "Precision/Speedrun" latent factor as the primary predictor of the "Difficult" signal.
- **Stepwise BIC Selection**: A parsimonious model was built using Bayesian Information Criterion. It converged on a set of ~30-100 tags (depending on constraints) that explain difficulty variance while avoiding overfitting.
- **Distributional Stability**: Difficulty scores on Steam are extremely sparse and heavy-tailed (Skewness: 19.2, Kurtosis: 747).
- **Rank-INT Transformation**: Applying Rank-Based Inverse Normal Transformation to both tags and difficulty scores significantly stabilized model coefficients and mitigated the influence of extreme tag-density outliers (like "Football" in specific cases).
- **Leverage/Influence**: Analysis identified high-leverage games (e.g., *Center2048*, *King of Texas*) that disproportionately skew linear model coefficients due to unusual tag profiles.
- **External Validation**: The final model (`pipeline/generate_difficulty_model.py`) is trained on ~3,200 games matched between GameFAQs (which has explicit 1-5 difficulty ratings) and Steam. It uses Rank-INT transformed tag proportions to predict the GameFAQs difficulty score, clamped to the [1, 5] range. This provides a ground-truth anchored difficulty metric rather than relying solely on tag semantics.

### Personalized Quality (Expected Experience)

We have developed a "Personalized Quality" model that adjusts a game's global quality score $Q$ (the probit transformed Bayesian rating) to reflect the expected experience for a specific player based on their target playtime $t$. 

- **Concept**: The model assumes a game's experiences are distributed as $N(Q, 1)$. While the global review score reflects the natural sampling from this distribution, a specific player's likelihood of enjoyment is guided by the **Playtime-Sentiment** kernel model ($p_+(t)$).
- **Mathematical Basis**: The "Personalized Quality" is the biased mean of the experience distribution:
  $$E[X \mid t] = Q + \phi(Q) \left[ \frac{p_+(t)}{\Phi(Q)} - \frac{1 - p_+(t)}{1 - \Phi(Q)} \right]$$
- **Implementation**: This is implemented in `common/utils.py` as `calculate_personalized_quality(q_global, p_plus_playtime)`.
- **Impact**: This correction allows the recommender to prioritize "Acquired Tastes" (games that improve over time) for long-playtime seekers, even if their early-game bounce rate is high. Conversely, it can penalize "Flash in the Pan" games for users seeking depth.
- **Next Steps**: Integration into the hybrid scoring pipeline will occur once user-specific data (Steam ID connection or manual playtime preference) is integrated (see `tasklist.md`).

## Feature: Analyze My Catalogue (Personalization Engine)

The "Analyze My Catalogue" feature allows users to solve for their personal preference weights by analyzing their existing Steam library. 

### Data Flow
1.  **Acquisition**: Fetch AppIDs and playtimes via SteamID64 API or manual HTML source paste (`scraping/get_user_stats.py`).
2.  **Soft-Labeling**: Generate 0-10 predicted ratings using the **Personalized Quality** formula ($Q_{pers}$) and the global **Playtime-Sentiment** model.
3.  **Verification (Ground Truth)**: User reviews the predicted ratings in a dense table UI, adjusting sliders or checking "Ignore" for games that don't reflect their taste.
4.  **Taste Solver**: Run an **Archetypal Ridge Regression** mapping the 0-10 ratings against:
    - **45 Structural Features** (38 MIG binary memberships, 1 Quality Probit, 6 Metadata Z-scores)
    - **Semantic Vibe Vector** (Blended from 235 ZCA Whitened dimensions)
    - **Tag Match Magnitude** (Derived from ZCA Whitened tag space)
5.  **Explainability Calibration**: The solver uses **Composite Word-Sum Calibration** to label the 235 semantic dimensions with high-contrast pairs like "Exploration + Terraform vs. Gunplay + Ricochet."
6.  **Generalization**: By reducing the feature count from $N$ to 45 archetypes, the solver prevents overfitting and ensures that the learned weights are representative of global Steam preferences ($R^2 \approx 0.28$).
7.  **Deployment**: Exported weights (including the Unit Semantic Vibe Vector) are used to initialize the recommendation sliders and the underlying scoring engine.


### UI Requirements (For React Implementation)
- **Dense Grid/Table**: Support for Steam banner images, expandable review text, and responsive columns.
- **Persistent State**: Use local storage or a backend database to save user verification progress.
- **Sorting & Filtering**: Real-time sorting by Playtime, Predicted Rating, Global Rating, or Ignore status.
- **Mapping**: All quality scores must be mapped to the 0-10 scale using the calibrated anchors ($m=3.0088, c=3.2772$).

## Technical Gotchas & Lessons Learned

- **PowerShell Command Chaining**: This environment uses PowerShell. Use `;` instead of `&&` to chain commands (e.g., `git add .; git commit`). Using `&&` will result in a `ParserError`.
- **Nginx & Unified Server**: When deploying to Linux, ensure the Nginx `proxy_pass` points to the unified server port (8000). Previous configurations used port 8501 for Streamlit, which will cause a 502 Bad Gateway if not updated.
- **Streamlit Vertical Alignment**: For centering elements in columns (like buttons next to cards), use `st.columns(..., vertical_alignment="center")`. This requires Streamlit >= 1.35.0.
- **Parquet Column Detection**: When checking for available columns in a Parquet file without loading the full dataset, do NOT use `pd.read_parquet(file, columns=[])`. In many versions of pandas/pyarrow, this returns an empty list regardless of the file's schema. Instead, use `pyarrow.parquet.read_schema(file).names` for a reliable and high-performance schema check.
- **Numpy Save Extensions**: The `numpy.save` function automatically appends a `.npy` extension if it is not present. This can cause issues with temporary files (e.g., `file.tmp` becoming `file.tmp.npy`). The `safe_save_npy` utility handles this by ensuring the target path ends in `.npy` and explicitly tracking the temp file's name.
- **Server Persistence**: The FastAPI backend (`app/server.py`) loads and caches metadata in RAM at startup. If the underlying `metadata.parquet` file is regenerated by the pipeline with a new schema (e.g., adding a new field), the server MUST be restarted to recognize and serve the new columns.

## Testing

- Unit tests are located in `tests/`.
- **Running Tests:** Use `run_all_tests.bat` to execute the full suite. This script ensures `PYTHONPATH` is set correctly and `pytest` is installed.
- **CRITICAL:** Ensure tests use temporary files or mock objects. Do not write to production data files (e.g., `.npy` files in root or `data/`) during tests.
- `tests/test_tag_vector_generation.py` uses explicit temporary file paths to prevent overwriting `steam_tag_vectors.npy`.

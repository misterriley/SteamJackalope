# Welcome to the Steam Natural Language Search Repository

This project is a Steam game recommendation engine that combines semantic search, tag analysis, and Bayesian quality scoring to provide highly relevant game suggestions.

## Environment

- This is being executed in VS Code via the Cline extension. 
- The operating system is Windows 11.

## Repository Overview

- `app/server.py`: The backend FastAPI server that handles data loading and hybrid score calculations. Start with `python -m uvicorn app.server:app --host 127.0.0.1 --port 8000`.
- `app/app.py`: The frontend Streamlit UI that communicates with the backend server. Start with `streamlit run app/app.py`.
- `run_test_env.bat`: A convenience batch file to launch both the server and frontend for local testing.
- `pipeline/run_pipeline.py`: Orchestrates the data processing pipeline (tags -> semantic vectors -> metadata -> quality scores). Uses `pipeline/pipeline_config.json` for path and interval settings.
- `pipeline/pipeline_config.json`: Configuration file for `run_pipeline.py` defaults.
- `common/constants.py`: Centralized configuration and hyperparameters. Loads from `pipeline/regularization_constants.json` if present.
- `pipeline/calculate_regularization.py`: Script to derive data-driven regularization constants.
- `common/utils.py`: Shared utility functions (z-scoring, hybrid score calculation).
- `methodology.md`: Detailed explanation of the statistical and machine learning techniques used. Note: `app.py` renders this by splitting at `![` to inject `st.image` calls, so image placement should follow this pattern. Note that this file is intended for human viewers of the website, so it should not be a place where notes about debugging or information related to internal workings of the app of statistical pipeline should go. Those types of notes should be put into `orientation.md`. 
- `data/`: Contains processed data files, including tag vectors and quality scores.
- `research/`: Contains experimental scripts and data analysis notebooks.
- `common/common_adjectives.txt`: A curated list of common English adjectives used for the "Random" button functionality.

## Current Project Status

- Core Python scripts (`app/app.py`, `pipeline/run_pipeline.py`, `scraping/scrape_steam.py`, `pipeline/generate_tag_vectors.py`, `pipeline/generate_quality_scores_grid.py`) have been updated with descriptive docstrings.
- `scraping/scrape_steam.py` uses direct **Storefront Scraping** (HTML/Embedded JSON) to retrieve high-fidelity user tags and metadata, bypassing age gates with specific cookies. It includes robust error handling, exponential backoff, and a **Hierarchical Local Cache** with historical archiving to prevent redundant downloads. Review counts are prioritized from **English** sources to better reflect the perspective of the target audience, falling back to global counts if no English reviews are available.
- `scraping/get_steam_appids.py` retrieves the full list of Steam AppIDs. It includes an automatic fallback to the public `ISteamApps/GetAppList/v2` endpoint if the primary `IStoreService` API call fails (e.g., due to an invalid API key).
- The pipeline is functional but could benefit from better progress monitoring.
- **Client/Server Architecture:** The project uses a decoupled architecture where heavy data loading and vector computations are performed by a FastAPI backend (`app/server.py`), while a lightweight Streamlit frontend (`app/app.py`) provides the UI. This ensures fast load times for the web interface and enables better scalability.
- **Metadata Retrieval:** The backend provides a `/metadata` endpoint to retrieve full metadata for specific games by name, enabling the frontend to display detailed game cards for user-selected seeds independently of the recommendation results.
- **Performance Optimization:** The backend caches pre-normalized embedding matrices and metadata. Similarity scores and hybrid calculations are vectorized using `numpy`. The frontend uses `st.cache_data` to minimize redundant API calls for static data like the game list.
- **Improved Debug Mode:** The recommender's game cards now feature a comprehensive debug section. It provides a full breakdown of the hybrid score calculation, including raw values, z-scores, and both slider and hard-coded weights for every component.

- Data files like `embeddings.npy`, `steam_tag_vectors.npy`, and `metadata.parquet` are the core artifacts used by the recommender. 
- `metadata.parquet` includes pre-calculated z-scores for release dates (`date_z`), popularity (`pop_z`), and game length (`playtime_z`), as well as a cleaned `release_year` for fast display.
    - **Age Z-Scoring**: Future release dates are clamped to today to prevent skewing the distribution. Unknown dates are assigned a neutral z-score of 0.
    - **Review Count Repair**: `generate_metadata.py` automatically repairs stale review counts using the raw individual reviews found in `scraped_reviews.csv` if they exceed the global reported counts.
- **CRITICAL:** All `.npy` artifacts and `metadata.parquet` must have perfectly synchronized row counts and ordering. The pipeline ensures this by using `data/pipeline_games_clean.csv` as the source of truth for appids. Manual script execution should point to this file (e.g., `python pipeline/generate_metadata.py data/pipeline_games_clean.csv`) to avoid `IndexError` in the app.
- **Tag Cleaning:** `pipeline/generate_semantic_vectors.py` extracts tag names from the dictionary string before embedding them in the structural vector. This ensures the LLM focuses on semantics rather than Python dictionary syntax.
- **Optimization:** `app/app.py` caches pre-normalized embedding matrices to speed up cosine similarity calculations.
- **Regularization Constants:** `pipeline/run_pipeline.py` calls `pipeline/calculate_regularization.py` to generate `pipeline/regularization_constants.json`. These values (like `TAG_VECTOR_K` and `GLOBAL_POSITIVE_RATE`) are used by subsequent pipeline stages and the app via `common/constants.py`.
- **Dependencies:** The lists page requires the `tabulate` library for markdown table generation.
    - **Note on `TAG_VECTOR_K`:** The solver for `TAG_VECTOR_K` must include the LOD imputation step in its cross-validation loop to be consistent with the pipeline. Ignoring imputation leads to a near-zero value because raw counts are highly distinct; including it correctly identifies the need to smooth the imputed "baseline" towards the global distribution.
    - **Note on `DOT_PRODUCT_LAMBDA`:** This regularization constant is calculated by fitting a Chi-distribution to the lengths (norms) of "low-tag" vectors (norms between 0 and 5). $\lambda$ is set to the 95th percentile of this fitted distribution. This represents the "noise floor" and ensures that a vector's length must be statistically significant before it can achieve high similarity scores.
    - **Tag Vector Normalization:** The 'Anscombe' and 'None' transforms in `generate_tag_vectors.py` now explicitly normalize the resulting vectors (and the prior) to sum to 1 before centering. This ensures the resulting embedding captures the shape of the distribution rather than the magnitude of counts.
- **Z-Scoring Logic:** When converting tag similarities to z-scores in `app/app.py`, zero values are ignored in the distribution calculation. This prevents the large number of games with no tag overlap from skewing the mean and standard deviation, ensuring more meaningful z-scores for actual matches. Additionally, all z-scores are clamped between `Z_SCORE_CLAMP_MIN` and `Z_SCORE_CLAMP_MAX` (defined in `common/constants.py`) to prevent outliers from dominating the hybrid ranking. Note that the pipeline generates raw z-scores for `playtime_z` and `pop_z` which are only clamped at runtime in the app to allow for parameter flexibility.
- **Note on `quality_scores_grid.npy`:** Unlike other artifacts where rows correspond to games, this file is a 2D grid of shape `(num_steps, num_games)`, where each row represents a different popularity preference setting. The number of games is matched along the second dimension (axis 1).
- **Quality Score Normalization**: `generate_quality_scores_grid.py` supports a `PIN_QUALITY_DISTRIBUTION` toggle in `constants.py`. When enabled, it pins the top and bottom means of each distribution to a baseline spread, ensuring consistent scoring magnitude regardless of Bayesian regularization strength.
- The project uses `all-MiniLM-L6-v2` for semantic embeddings. To improve recommendation precision, it uses a **Dual Semantic Vector** system: `embeddings_structural.npy` (Genres + Tags) and `embeddings_desc.npy` (Description + Reviews). The app blends prompt matches against both, but performs specialized seed matching (structural-to-structural and descriptive-to-descriptive) to ensure categorical and narrative vibes are handled independently.
- **Difficulty Estimation**: Game cards now display a predicted 1-5 difficulty score derived from a model trained on GameFAQs ground truth data.
- **UI Enhancements**: The recommender includes a sorted seed game selection list and a global "Reset to Defaults" button for tuning parameters.
- **ZCA Whitening**: Semantic embeddings are centered and whitened using ZCA (Zero-phase Component Analysis) to decorrelate dimensions and improve query precision. The mean vectors and transformation matrices are saved (e.g., `mean_desc.npy`, `w_desc.npy`) and are applied to user queries at runtime in `app.py`. Analysis of the whitened vectors confirms that they are successfully centered at zero and exhibit identity covariance on their active dimensions.
- **Note on Zero-Variance Dimensions:** The `all-MiniLM-L6-v2` model naturally has three zero-variance dimensions (indices 127, 223, and 319). These are preserved during whitening, resulting in 381 active semantic dimensions.

- **Scraping Architecture**: The scraping process is split into three steps: download (`download_steam_data.py`), dataset build (`build_scraped_dataset.py`), and pipeline integration. It features robust interruption tolerance and a retry mechanism for file operations on Windows.

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

## Testing

- Unit tests are located in `tests/`.
- **CRITICAL:** Ensure tests use temporary files or mock objects. Do not write to production data files (e.g., `.npy` files in root or `data/`) during tests.
- `tests/test_tag_vector_generation.py` uses explicit temporary file paths to prevent overwriting `steam_tag_vectors.npy`.

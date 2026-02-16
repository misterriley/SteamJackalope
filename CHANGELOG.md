# Changelog

All notable changes to the Steam Jackalope project will be documented in this file.

## [15] - 2026-02-15

### Added
- **Unified Linear Scorer**: Replaced the 'hybrid approximation' system with a direct linear scorer. The Discovery Engine now executes the user's solved Ridge Regression model ($Score = Intercept + \sum \beta_i x_i$) with 100% mathematical consistency between the Solver and Recommender.
- **Global Feature Standardization**: Implemented a global scaling factor (**11.283x**) for tag features and standardized metadata weights on Global Z-scores. This enables 'Beta Weights' that are portable and stable across hits and niche games.
- **Robust Error Handling**: Installed a global **ErrorBoundary** and a **Reset App** safety button to prevent and recover from 'white-screen' crashes caused by corrupted cache or invalid data.
- **Format-Agnostic Parsing**: Updated frontend game card parsing to handle Genres and Tags in multiple formats (Array, String, Object), preventing type errors during data hydration.
- **Personalization Persistence**: User DNA results (weights, vibe vector, intercept) now persist across tab switches and page reloads via `sessionStorage`.

### Fixed
- **Vibe Inflation Bug**: Resolved an issue where Z-scoring tag similarities caused niche North Star games to artificially jump to #1.
- **UnboundLocal & Name Errors**: Fixed several critical variable scope bugs in the Taste Solver pipeline.
- **JSON Safety**: Solver now recursively strips `NaN`/`Inf` values from output JSON to ensure frontend compatibility.

### Changed
- **Predictive Sliders**: In Personalized mode, sliders now act as **Multipliers** on the solved DNA weights (0.5 = 100% impact, 1.0 = 200% impact) rather than absolute biases.
- **Rating Calibration**: Recommender 'Match Scores' are now calibrated to the user's predicted **0-10 rating** scale.

## [13] - 2026-02-14

### Added
- **Steam Store Integration**: Implemented a comprehensive validation system for 512 tags, genres, and features.
- **Interactive Taxonomy**: Tags and genres in the UI are now clickable, linking directly to verified Steam Store pages.
- **Data Hygiene Protocol**: Backend now filters out "dead" or non-linkable tags (e.g., legacy or junk tags) during metadata load to ensure a clean UI and valid discovery paths.
- **Manual QA Suite**: Established `QA.md` as the source of truth for manual functional verification of navigation, integration, and UI features.
- **Automated Validation Tooling**: Created `tools/validate_steam_links.py` to handle pattern matching, slugification variants, and Steam category ID mapping.

### Fixed
- **Networking Stability**: Standardized backend calls to `127.0.0.1` on Windows to resolve DNS resolution latency.
- **Metadata Parsing**: Switched to `ast.literal_eval` for robust handling of complex tag/genre dictionary strings in the backend.

### Changed
- **Unified Startup**: Consolidated local development environment into `run_test_env.bat` (automated git pull, dep installs, and environment launch).
- **Push Protocol**: Added requirement for manual QA verification and changelog synchronization before push.

## [11] - 2026-02-14
### Added
- **Documentation**: Integrated `onPush.md` into `orientation.md`, `onShutdown.md`, and `gemini.md` for better onboarding.
- **UI**: Added a "Changelog" link to the header in both modern and legacy frontends.
- **Tooling**: Refactored `generate_random_recommendation.py` output to a multiline, labeled format optimized for Discord sharing.

## 2026-02-13
### Build 3-8
### Version 0.0.1-pre-alpha+build.3
- **Frontend**: Launched a modern React 19 + TypeScript + Vite + Tailwind CSS v4 frontend, replacing the legacy Streamlit interface as the primary UI.
- **Architecture**: Implemented a Unified Server architecture where FastAPI serves both the API and the production frontend build from a single process.
- **UI**: Added "About" and "Methodology" pages with KaTeX support for mathematical formulas.
- **UI**: Improved seed game management with removal buttons and better grid alignment.
- **Deployment**: Added automated Linux deployment scripts (`deploy.sh`) for unified server management.

## 2026-02-12
- **Algorithm**: Optimized playtime sentiment analysis using full vectorization and parallelized global parameter optimization ($\gamma \approx 0.51$).
- **UI**: Fixed a bug where difficulty scores of 0 were not being displayed in game cards.
- **UI**: Added color-coded weighting visualizations to game cards (Lesser Priority task).

## 2026-02-11
- **UI**: Split the methodology page into dedicated "About" and "Methodology" sections for better readability.
- **UI**: Implemented a compact sidebar layout and 3-column genre selection to maximize vertical space.
- **UI**: Added a custom header and footer with GitHub integration and copyright information.
- **Documentation**: Added detailed README files for all major project directories.

## 2026-02-10
- **Algorithm**: Implemented alphabetical fallback sorting for recommendations when all preference sliders are set to zero.
- **UI**: Refactored all preference sliders (Age, Quality, Popularity, etc.) to use discrete steps for more precise user control.
- **UI**: Added "Debug Mode" to game cards, providing a full breakdown of z-scores and weight contributions for transparency.

## 2026-02-09
- **Performance**: Migrated to quantized ONNX transformer models (`all-MiniLM-L6-v2`), reducing RAM usage and increasing inference speed.
- **Performance**: Converted large production artifacts to float16 precision, reducing the memory footprint by 50%.
- **Algorithm**: Applied dimensionality reduction to whitening matrices to optimize cache performance and reduce noise in high-dimensional similarities.

## 2026-02-14 (Build 12)
- **Algorithm**: Re-calibrated playtime sentiment hyperparameters ($\gamma \approx 0.688$, $s \approx 1.453$) using a full-dataset parallelized grid search (36k+ games) with per-review cross-entropy loss.
- **Infrastructure**: Removed the 512 MB RAM constraint across all documentation, tests, and deployment scripts.
- **Backend**: Switched `SentenceTransformer` from lazy to eager loading at startup to improve initial request latency.
- **Frontend**: Added SEO metadata (Open Graph, Twitter Cards) to the landing page.
- **UI**: Integrated Steam Store links into the "Lists" view for direct navigation.
- **UI**: Added "Trending" seed game selection option in the Recommendations view.
- **Scraping**: Implemented `scrape_trending.py` to fetch top 100 Steam games daily.

## 2026-02-08
- **Deployment**: Achieved a highly efficient RAM footprint for the backend server, enabling deployment on standard cloud tiers.
- **Algorithm**: Implemented thresholding in the z-score normalization function to mitigate noise from dense tag vectors.
- **Algorithm**: Resolved a critical tag similarity bug and eliminated genre contamination in semantic vector matching.
- **Infrastructure**: Configured Git LFS for production data files to ensure repository stability and efficient deployment.

## 2026-02-07
- **Algorithm**: Initial implementation of the hybrid recommendation engine combining semantic search, PCA-ZCA whitened tag vectors, and Bayesian quality scoring.
- **Algorithm**: Added support for multi-genre filtering in the recommendation loop.
- **UI**: Initial release of the legacy Streamlit frontend with interactive sliders and real-time result updates.
- **Scraping**: Established robust Steam storefront scraping architecture with hierarchical caching and automatic checkpointing.

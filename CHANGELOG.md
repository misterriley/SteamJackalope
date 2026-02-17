# Changelog

All notable changes to the Steam Jackalope project will be documented in this file.

## [29] - 2026-02-16

### Fixed
- **Tag Space Stabilization**: Resolved a critical bug where predictive tags in the "Analyze My Catalogue" view were disconnected from recommendations. The issue was caused by a missing `tag_names.json` file, forcing the solver to fall back to an unstable metadata scan that misaligned tag indices.
- **Explicit Tag Priors**: Updated the tag vector pipeline to explicitly save the Global Prior counts and Transformed Prior vector. This ensures that centering and transformation logic remain consistent across the pipeline, solver, and recommendation engine, even as the dataset grows.
- **North Star Alignment**: Verified that "North Star" recommendations are now perfectly aligned with the solved DNA's predictive tags.

### Added
- **Tag Prior Artifacts**: New production artifacts `tag_prior_counts.npy` and `tag_prior_transformed.npy` provide a stable anchor for the tag embedding space.

## [28] - 2026-02-16

### Added
- **Interactive Predictive Tags**: Predictive tags in the Taste DNA view are now clickable, linking directly to their validated Steam store pages.
- **Extended Solver Previews**: Increased the number of "Games You'll Love/Hate" in the personalization view from 20 to 30 items.

### Fixed
- **Rating Display Calibration**: Predicted ratings in the solver are now clamped to the 0-10 scale, preventing values outside the intuitive range.
- **Weights UI Cleanup**: Removed the redundant "semantic" entry from the metadata weights display in the personalization view.

## [27] - 2026-02-16

### Fixed
- **Vanity Name Resolution**: Fixed an infinite polling loop in the personalization engine when using Steam vanity names. The backend now synchronously resolves names to 64-bit SteamIDs before starting the background task, ensuring the frontend polls for the same ID used for file storage.

## [26] - 2026-02-16

### Changed
- **Expanded Discovery Logging**: The Taste Solver now prints the absolute correlation for **every step** in the discovery grid (21 values), allowing for detailed inspection of the correlation curve.

## [25] - 2026-02-16

### Added
- **Discovery Debug Logging**: The Taste Solver now prints a "Step-wise Correlation Scan" to the console, showing the absolute correlation for every 10th step of the discovery grid. This provides transparency into how the optimal setting is derived and helps diagnose flat signals.

## [24] - 2026-02-16

### Fixed
- **Solver NameError**: Fixed a crash in `solve_user_taste.py` where Bayesian variables `s` and `a` were undefined. The solver now correctly uses the precalculated quality grid from the optimal Discovery level for the entire dataset during "Top/Bottom" preview generation.

## [23] - 2026-02-16

### Changed
- **Refined Discovery Optimization**: The Taste Solver now uses the **maximum absolute correlation** to identify the optimal Discovery level. This ensures that the strongest signal is captured, even for users whose preferences are inversely correlated with global ratings (i.e., preference for "so-bad-it's-good" games).

## [22] - 2026-02-16

### Added
- **Discovery Optimization**: The Taste DNA Solver now automatically identifies the optimal Discovery setting for each user. It iterates through the quality grid to find the regularization strength that maximizes correlation with the user's provided ratings. This optimal setting is exported and automatically initializes the Discovery slider in the Recommender.

## [21] - 2026-02-16

### Changed
- **Score De-calibration**: Removed the regression `intercept` from the recommendation engine. While the Taste Solver continues to use the intercept for 1-10 rating prediction, the Recommender now returns purely unitless, relative scores for cleaner ranking.

## [20] - 2026-02-16

### Fixed
- **Tag Magnitude Double-Scaling**: Resolved a critical 11.28x over-correction bug. Since regression features were already scaled to "Rating Point equivalents," scaling the exported norm again in the UI created a massive mismatch. Sliders and Solver lists now match perfectly.
- **DNA Import Logic**: Replaced truthy fallbacks (`|| 1.0`) with nullish coalescing (`?? 1.0`) in the frontend to ensure that users who don't care about tags (solved magnitude near 0) aren't forced to a default weight of 1.0.

## [19] - 2026-02-16

### Fixed
- **Tag Weight Scaling**: Resolved a bug where the solved "Tag Match" magnitude was inversely scaled by the global scaling factor. Sliders now correctly display the absolute rating points importance (Magnitude * 11.283) for the vibe component.
- **Frontend Fallbacks**: Standardized profile import logic to ensure the solved tag magnitude is used in the slider, preventing hardcoded defaults (1.0 or 1.5) from overriding user DNA.

## [18] - 2026-02-16

### Changed
- **Unified Absolute Weighting (Rating Points)**: Redesigned the slider system to use absolute rating points rather than relative multipliers. Sliders now represent the direct contribution to the 0-10 predicted rating per standard deviation of a feature.
- **Import Parity**: Taste DNA profiles now import their solved coefficients directly into the sliders without any transformation, ensuring 100% UI and mathematical parity.
- **Expanded UI Bounds**: Metadata sliders now range from -5 to 5 (Rating Points), providing consistent "headroom" for both standard and personalized preferences.
- **Default Calibration**: Standard mode now defaults to established absolute weights (Quality=4.0, Age=1.4, Tags=1.0).

## [17] - 2026-02-16

### Changed
- **Unified Tag Scoring (Linear Mode)**: Unified the seed-based recommendation path with the Taste DNA solver. Seed games now act as regression coefficients ($\beta_{seed} = V / (\|V\| + \lambda)$), providing 100% mathematical parity between manual seeds and solved DNA profiles.
- **Eliminated Tag Z-Scoring**: Removed neighborhood-based Z-scoring for the tag component. Tag contributions are now calculated on an absolute rating scale (scaled by 11.283x), ensuring stable rankings that don't drift based on the current result set.
- **Seed/DNA Blending**: Implemented a robust 50/50 linear blend when both a Personal Taste Profile and manual seed games are provided.

## [16] - 2026-02-16

### Changed
- **UI Intuition (Labels)**: Renamed "Age" slider and debug labels to "Release Date" across the app (modern React and legacy Streamlit frontends) to improve user intuition.
- **UI**: Shortened "Release Date Preference" to "Release Date" and updated tooltips to explicitly state that the right side of the slider favors newer games.

### Added
- **Steam Integration**: Made the "Trending" label in the recommender a direct link to Steam's most played charts while preserving its checkbox functionality.

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

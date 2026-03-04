# Changelog

All notable changes to the Steam Jackalope project will be documented in this file.

## [51] - 2026-03-03
### Build 51
### Version 0.0.1-pre-alpha+build.51

- **Algorithm**: Upgraded to **Jackalope Kernel v4.2**, transitioning from simple feature agreement to **Jaccard Similarity (Intersection over Union)** for Mechanical Identity Groups (MIGs). This eliminates the "Shared Vacuum" bug and ensures recommendations are based on positive evidence of shared mechanics.
- **Algorithm**: Implemented **Thematic MIGs** (`SCI_FI`, `FANTASY`, `HISTORICAL`, `SURREAL`) and a **0.5x Thematic Clash Penalty** to prevent immersion-breaking genre pivots (e.g., Cyberpunk vs. Hogwarts Legacy).
- **Algorithm**: Increased **Mechanical Identity Power to 2.0**, creating a "precision cliff" that boosts games with perfect structural alignment and suppresses partial matches.
- **Algorithm**: Refined **Pollution Vetoes** by adding `POINT_AND_CLICK` and `DATING_SIM` to the Hard Veto list (0.1x), and moving `Romance` to the Narrative Story group to restore high-fidelity RPG matches.
- **UI/UX**: Launched the **Jackalope Kernel Explorer** diagnostic tool (`tools/kernel_explorer.py`), a specialized Streamlit interface for analyzing game similarity, managing ground truth labels, and verifying model selectivity.
- **UI/UX**: Implemented **Quick Exemplar Buttons** in the Kernel Explorer sidebar for 25 high-rated games (9+), facilitating rapid kernel verification across diverse genres.
- **UI/UX**: Enhanced the diagnostic sort logic by fetching the **Top 1000 Raw Matches** before applying the user's Predicted Rating re-sort, ensuring high-quality "vibe rescues" are visible in the UI.
- **Reliability**: Fully synchronized the **Taste DNA Regression Pipeline** with the new kernel math, expanding the archetypal solver to 41 features and achieving zero drift between analysis and recommendation.

## [50] - 2026-03-02
### Build 50
### Version 0.0.1-pre-alpha.50

- **Algorithm**: Exorcised low-fidelity matches for *Disco Elysium* (e.g., *Leons Identität*) and *Mass Effect* (e.g., *Front Mission Evolved*) by refining `MIGS` to require core verbs and implementing **Symmetric Hard Anchor** enforcement.
- **Algorithm**: Broadened the `CRPG` MIG to include general `RPG` and `Action RPG` tags, ensuring high-fidelity matches for narrative RPGs like *Mass Effect* and *Alpha Protocol*.
- **Algorithm**: Refined `HARD_ANCHORS` to focus on Perspective (Isometric, First-Person) and strict Interaction Models (CRPG, RPG), removing thematic descriptors (Surreal, Abstract) that caused false-positive vetoes for high-fidelity matches.
- **Algorithm**: Symmetric Perspective Enforcer: Candidates with a hard perspective anchor (e.g., First-Person) that the seed lacks are now strictly penalized (0.001x), ensuring structural consistency in recommendations.
- **Algorithm**: Cleaned up `SPATIAL_PUZZLE`, `LOGIC_PUZZLE`, and `WALKING_SIM` mechanical groups to prevent identity cross-contamination from broad tags like `Exploration` or `Surreal`.

## [49] - 2026-02-26
### Build 49
### Version 0.0.1-pre-alpha.49

- **Algorithm**: Integrated the **12 Gamer Motivation Archetypes** (Destruction, Excitement, Competition, Community, Challenge, Strategy, Completion, Power, Fantasy, Story, Discovery, Design) into the Taste DNA engine. 
- **Algorithm**: Implemented a **Hybrid Motivation Scoring** model that combines Tag, Semantic, and Topic modalities to calculate user alignment with psychological archetypes.
- **Algorithm**: Added **Motivation-Based Recommendations**: The system now identifies the top 1% of games loading onto each motivation's basis vector and ranks the top 3 using the user's personalized Taste DNA weights.
- **UI/UX**: Added **Psychological Motivation Bars** to the Taste DNA Insights page, providing a high-level view of the user's gamer personality.
- **UI/UX**: Implemented **Motivation Tooltips** that reveal the defining mechanics (synthetic tags) and modality breakdowns for each archetype.
- **UI/UX**: Added a new **"Top Recommendations by Motivation"** section at the bottom of the Insights page, similar to the existing tag-based recommendations.
- **Data**: Created `data/production/motivations_library.json` as a permanent artifact containing the basis vectors for the 12 archetypes.

## [48] - 2026-02-25
### Build 48
### Version 0.0.1-pre-alpha.48

- **Algorithm**: Synchronized the **Thematic Topic Match** signal between the Taste DNA page and the Recommender. The recommendation engine now correctly receives the `topic_vibe_vector` and its weight (`gamma_topic`) upon profile import, ensuring perfect ordinal parity with the solver's preview.
- **UI/UX**: Added a **"Topic Match"** slider to the Preferences sidebar, allowing users to manually tune the impact of atmospheric and thematic topics on their recommendations.
- **UI/UX**: Hardened the profile application and clear logic in `App.tsx` to handle all 3 similarity modalities (Tags, Semantics, Topics) and their respective vectors.

## [47] - 2026-02-25
### Build 47
### Version 0.0.1-pre-alpha.47

- **UI/UX**: Launched the **My Game Catalogue** (User Hub) for comprehensive library management. Users can now categorize games into **Rated**, **Played**, **Backlog**, **Wishlist**, or **Ignored** states.
- **UI/UX**: Resolved severe UI lag in large libraries (1000+ items) using **Lite Virtualization (Infinite Scroll)**, React memoization, and isolated search components.
- **UI/UX**: Standardized **Keyboard-Driven Data Entry** with auto-scrolling dropdowns and 'Enter-to-Commit' support across all search inputs.
- **Algorithm**: Integrated a **250-Topic BERTopic Layer** providing high-fidelity thematic matching using Jensen-Shannon Divergence (JSD).
- **Algorithm**: Implemented **Zero-Order Relevance Filtering** ($p \le N-7$) in the Taste DNA solver to prevent overfitting and stabilize $R^2$ for smaller libraries.
- **Algorithm**: Achieved **Population-Wide Modality Scaling**, calibrating Tags (1.0x), Semantics (~2.0x), and Topics (~26.5x) based on true data variance.
- **Algorithm**: Synchronized **Topic Match** scoring between the Python solver (Dot Product) and FastAPI backend, ensuring perfect ordinal parity.
- **Explainability**: Displayed top **Topic Keywords** in the UI to provide human-readable justification for atmospheric recommendations.
- **Stability**: Fixed multiple critical server and solver crashes related to the categorical status migration and NaN propagation in similarity signals.
- **Data**: Hardened the scraping pipeline with robust regex for language detection and English-prioritized review count repair.

## [46] - 2026-02-22
### Build 46
### Version 0.0.1-pre-alpha.46

- **Algorithm**: Implemented **Softmin Multi-Signal Blending** ($T=3.0$) for multi-signal targets (DNA + Seeds + Prompts). This rewards "consensus" among all active signals, preventing a single dominant match from overwhelming the final recommendations.
- **Algorithm**: Decoupled "Owned" status from "Discovery" eligibility. Added a dedicated **"From Your Backlog"** discovery list to the Insights page, allowing users to surface high-rated titles they already own but haven't yet played.
- **Algorithm**: Fixed a critical **Tag Similarity Normalization Bug** in the solver to ensure parity with the backend's penalized pathway, improving the quality of "Similar to Favorites" results.
- **Algorithm**: Implemented **Robust Release Date Filtering** using a dual-layer approach: build-time comparison and explicit text-search for placeholders ("Coming soon", "TBD"). This ensures unreleased games are correctly excluded even with stale metadata.
- **UI/UX**: Enhanced the **Insights UI** with a dedicated "Backlog" section and added lucide icons for better navigation.
- **UI/UX**: Removed confusing projected rating badges from horizontal similarity lists ("Tag Recommendations" and "Similar to Favorites") to eliminate mismatch with global profile-based ratings.
- **UI/UX**: Added a smooth **"Scroll to Top"** button on the Recommender page with responsive positioning and Framer Motion animations.
- **UI/UX**: Refactored the **"Reset All"** functionality to be profile-aware. It now snaps sliders back to the user's solved DNA baseline while clearing search-specific prompts and seeds.
- **Cleanup**: Excised redundant structural semantic and tag artifacts (`embeddings_structural.npy`, etc.) to minimize RAM footprint and reclaim disk space.

## [45] - 2026-02-22
### Build 45
### Version 0.0.1-pre-alpha.45

- **Algorithm**: Fixed the **"82% Match Bug"** by isolating the 5.0 neutral intercept from scaling division. High-affinity recommendations now correctly reach their true statistical potential (98-99% match).
- **Algorithm**: Standardized the **"Neutral Anchor"** (5.0 baseline) for both raw linear scoring and probability mapping, ensuring a consistent and intuitive "50% = Average" experience across the app.
- **Algorithm**: Implemented **Discovery-Aware Bias Correction** to the scoring pipeline, ensuring that "average games" (with no specific affinity) correctly result in a 5.0 score regardless of active feature weights.
- **UI/UX**: Launched **Transparency Mode (Absolute Weights)** for Taste DNA. UI sliders now directly populate with and control the absolute weights learned by the solver (e.g., Quality = 0.86). This eliminates "multiplier" confusion and fixes the "Squaring Bug" where weights were being multiplied by themselves upon import.
- **Reliability**: Achieved **Perfect Ordinal Parity** between the Python solver and the FastAPI backend. Verified with a new automated integration test (`tests/test_ordinal_parity.py`) that checks for bit-perfect ranking identity.
- **Data**: Refreshed `data/trending_appids.json` using the automated Steam storefront scraper.
- **Documentation**: Updated `methodology.md`, `orientation.md`, and `gemini.md` to reflect the Build 45 mathematical refinements and the shift to absolute slider control.

## [44] - 2026-02-21
### Build 44
### Version 0.0.1-pre-alpha.44

- **Stability**: Fixed a recurring **Data Corruption Bug** where the production `tag_names.json` was being overwritten by small mock datasets during test runs. Implemented a mandatory **Production Lock** (Read-Only) in the test runner to prevent future regressions.
- **Algorithm**: Implemented **Delisted Game Filtering** across the entire stack. The system now identifies games that are no longer for sale on Steam and excludes them from recommendations, category lists, and Taste DNA analysis by default.
- **Algorithm**: Launched **Personalized Quality Adjustments** in the recommender. For games in a user's library, the system now recalculates the quality component of the score using a **Kernel Smoothing Model** based on the user's playtime, providing a more accurate "expected experience" score.
- **UI/UX**: Added a **"Hide Delisted"** toggle to the preferences sidebar in both React and Streamlit frontends, allowing users to opt-back-in to seeing unpurchasable games if desired.
- **UI/UX**: Updated the **GameCard** to display a animated **"Match %"** with a sparkle icon when a personalized quality adjustment is active, visually distinguishing tailored ratings from global averages.
- **Data**: Updated `metadata.parquet` with a pre-calculated `is_delisted` flag for high-performance filtering.
- **Reliability**: Hardened the backend to automatically re-calculate missing boolean flags (VR-only, NSFW, Delisted, etc.) if the underlying metadata file is stale.

## [43] - 2026-02-20
### Build 43
### Version 0.0.1-pre-alpha.43

- **Algorithm**: Implemented **Hybrid Semantic-Tag Solver** using high-dimensional LASSO across both Steam tags and 235 descriptive "Vibe" dimensions.
- **Algorithm**: Achieved **Semantic Variance Parity** by applying a dynamic 11.25x scaling factor, ensuring descriptive vibes and categorical tags are treated as mathematically equal features.
- **Algorithm**: Transitioned to a **Case-Insensitive Semantic Model** by lowercasing all input text across the pipeline and backend, eliminating orthographic noise and consolidating conceptual variance.
- **Explainability**: Developed **Composite Word-Sum Labeling** for semantic dimensions, providing high-contrast titles (e.g., "Exploration + Terraform vs. Gunplay + Ricochet") derived from a 10,000-word descriptive vocabulary.
- **Explainability**: Updated **North Star & Abyss** logic to use weighted Hybrid Alignment (Tags + Semantics) for more qualitatively accurate taste anchors.
- **UI/UX**: Refactored the personalization insights view to display **Key Vibe Dimensions** with interactive hovers and rating correlation plots.
- **UI/UX**: Automated **Semantic Weight Import**, correctly mapping solved descriptive preferences to the Recommender's Semantic slider upon profile application.

## [41] - 2026-02-20
### Build 41
### Version 0.0.1-pre-alpha.41

- **Explainability**: Added **Tag Dimension Descriptions** to the personalization engine. The system now provides human-readable labels (e.g., "Tactical Combat vs Adult Themes") for the latent taste dimensions, offering clear insight into the "why" behind taste analysis.
- **Explainability**: Implemented a new API endpoint (`/tag_dimensions`) to serve the dimension descriptions to the frontend.
- **UI/UX**: The "Analyze My Catalogue" view now displays trendlines and descriptions for each of a user's top taste dimensions.
- **Analysis**: Created a new research script (`research/analyze_tag_dimensions.py`) to generate refined, high-level descriptions for each dimension by analyzing its most influential positive and negative tags.

## [40] - 2026-02-19
### Build 40
### Version 0.0.1-pre-alpha.40

- **Algorithm**: Upgraded semantic model to **`all-mpnet-base-v2`** (768 dimensions), providing superior qualitative and thematic matching.
- **Algorithm**: Implemented **Descriptive-Only Semantic Path**, focusing semantic search strictly on narrative text (descriptions and reviews) to reduce noise from categorical matches.
- **Algorithm**: Introduced **Uncentered ZCA Whitening** for semantic vectors, preserving the origin while decorrelating narrative features.
- **Algorithm**: Established a **Calibrated Natural Range** for semantic similarities based on a 10,000-pair simulation, ensuring stable slider impact.
- **Algorithm**: Integrated **Price** as a first-class feature across the stack, including Price Z-scoring in the metadata pipeline and Price weighting in the Taste DNA regression.
- **UI/UX**: Created a central **Splash Page** navigation hub for clearer user orientation between tools.
- **UI/UX**: Refactored scoring visualization to be **Always On**, providing transparent weight breakdowns on every game card by default.
- **UI/UX**: Optimized for **Mobile** with a collapsible preferences sidebar and removal of nested scrollbars.
- **Infrastructure**: Configured **GPU Acceleration (CUDA 12.8)** support for Blackwell-series (RTX 50) GPUs via a dedicated `venv310` environment.
- **Infrastructure**: Hardened metadata pipeline to clamp future release dates to "Today" for distribution stability.

## [39] - 2026-02-17
### Build 39
### Version 0.0.1-pre-alpha.39

- **Algorithm**: Achieved **bit-perfect parity** between Taste DNA solver and Recommender by unifying the scoring path in `common/utils.py`.
- **Algorithm**: Implemented **Absolute Slider Logic** across the entire stack, eliminating the "squaring bug" and multiplier confusion. Sliders now directly represent solved weights.
- **Algorithm**: Enhanced numerical stability with explicit `float32` casting during linear scoring to prevent precision-based ranking swaps.
- **UI/UX**: Refactored profile application in `App.tsx` to correctly map metadata weights to sliders and automatically enable **All-Profile Filtering**.
- **Stability**: Added extensive **Trace Logging** to both frontend and backend to monitor DNA profile data flow and vector integrity.
- **Data**: Verified **Dota 2 (AppID 570)** searchability and metadata integrity.

## [38] - 2026-02-17
### Build 38
### Version 0.0.1-pre-alpha.38

- **Algorithm**: Achieved 100% mathematical parity between Solver and Recommender using a unified scoring pathway in `common/utils.py`.
- **Algorithm**: Switched personalization to **LASSO Regression** with adaptive saturation dimensionality ($K = \text{clip}(N-6, 1, 243)$) for sparse, high-fidelity profiles.
- **Algorithm**: Fixed Discovery slider inversion and calibrated quality grid regularization mapping.
- **UI/UX**: Implemented **NSFW Blur** architecture: tag-based flagging with frontend CSS blur sync.
- **UI/UX**: Added **3-way Profile Filtering** (None, Rated, All) to the discovery engine.
- **UI/UX**: Calibrated Metadata Weights bars (Discovery at 1.0, others at 3.0) for better visual density.
- **Data**: Expanded library acquisition to include zero-playtime games for comprehensive filtering.
- **Stability**: Refactored Taste DNA solver to be asynchronous and improved backend error logging.
- **Stability**: Resolved Windows file-locking issues in the utility library.

## [36] - 2026-02-16

### Changed
- **Smooth Adaptive DNA**: Refined the adaptive dimensionality logic in `solve_user_taste.py` to use a smooth linear function ($K = \text{clamp}(40 + 0.7 \times N, 40, 243)$). This replaces the previous stepped logic, providing organic complexity growth and improved model stability as users add ratings to their library.

## [35] - 2026-02-16

### Added
- **Adaptive DNA Dimensionality**: Implemented a dynamic feature scaling system in `solve_user_taste.py`. The solver now automatically adjusts the number of tag components ($K$) based on the user's library size:
    - **Micro (<30)**: 40 components
    - **Small (30-100)**: 80 components
    - **Standard (100-300)**: 160 components
    - **Enthusiast (>300)**: Full 243 components
- This prevents overfitting for new users while maintaining high-fidelity modeling for power users, with all profiles remaining 100% compatible with the recommendation engine via zero-padding.

## [34] - 2026-02-16

### Added
- **Support-Based Sanity Check**: Implemented a filtering mechanism in `solve_user_taste.py` that prevents "phantom" tags from appearing in the DNA view. Tags must now appear in at least one game in the user's library to be eligible for the "Love/Hate" lists, effectively eliminating aliasing artifacts.

### Changed
- **Whitening Optimization**: Reverted the tag vector pipeline to use a **95% variance threshold**. This provides the best balance between information density and regularization, significantly reducing overfitting in the personalization engine.

## [33] - 2026-02-16

### Added
- **Centralized Regularization Logic**: Moved `calculate_dot_product_lambda` to `common/utils.py` for consistent Chi-distribution fitting across the pipeline.

### Fixed
- **Windows File Locking (WinError 32/5)**: Further hardened `safe_save_npy` with a multi-step retry loop, unique PID-based temporary files, and a fallback rename-to-garbage strategy. This ensures that the pipeline can update artifacts even under heavy contention from OneDrive or a running FastAPI server.
- **Numpy Save Extensions**: Fixed a bug where `safe_save_npy` failed to find temp files because `np.save` automatically appends `.npy`.
- **Pipeline Restoration**: Restored the missing `whiten` function and fixed mangled imports/indentation in `generate_tag_vectors.py`, `generate_semantic_vectors.py`, and `generate_quality_scores_grid.py`.
- **Whitening Calibration**: Re-centered the whitening threshold to 99% in `generate_tag_vectors.py` to ensure optimal noise reduction.

## [32] - 2026-02-16

### Added
- **Robust Artifact Saving**: Implemented `safe_save_npy` in `common/utils.py` to handle Windows file locking issues. This allows the data pipeline to update `.npy` artifacts (tag vectors, semantic embeddings, quality grid) even while the FastAPI server is running and memory-mapping those files.

## [31] - 2026-02-16

### Fixed
- **Tag List Recovery**: Manually regenerated `tag_names.json` to resolve a critical mismatch where only 2 tags were recognized despite the pipeline expecting 455. This ensures that predictive tags in the Personalization view are correctly mapped and displayed.

## [30] - 2026-02-16

### Changed
- **DNA Solver Refinement**: Increased the ridge regression alpha range from $10^{-2}-10^{4}$ to $10^{-2}-10^{6}$ with higher step density (81 steps). This ensures the optimal regularization constant is captured even for highly noisy or sparse user libraries.
- **Inclusive Soft-Labeling**: Removed playtime filters that previously excluded games with playtime below the minimum review threshold. This allows the personalization engine to utilize more of a user's library for taste analysis, while still filtering out games with zero playtime to ensure valid data points.

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

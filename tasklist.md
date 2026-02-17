# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] **Unified Scoring Pathway (Build 40)**: Consolidated personalization math into `common/utils.py`. Both the Solver and Recommender now use the exact same code path for Taste DNA scoring, eliminating implementation drift.
- [x] **Solver/Recommender Parity Sync (Build 39)**: Achieved 100% mathematical and environmental parity by fixing discovery mapping, restoring DNA scaling, implementating NSFW filters, and unifying preprocessing (clamping).
- [x] **Solver/Recommender Synchronization (Build 38)**: Achieved 100% ranking parity by unifying the feature space (Raw Z-Scores) and implementing a "Pure Linear Mode" in the backend that directly utilizes solver coefficients and intercepts.
- [x] **LASSO DNA Solver (Build 37)**: Switched the personalization engine to use LassoCV with K=N-6 dimensionality, enabling sparse, high-fidelity taste profiles that outperform Ridge in predictive stability.
- [x] **Solver/Recommender Synchronization (Build 37)**: Achieved 100% ranking parity between library analysis and discovery by synchronizing Z-score clamping, lexicographical sorting, and tag normalization logic.
- [x] **Smooth Adaptive DNA (Build 36)**: Implemented a linear complexity function ($K = 40 + 0.7 \times N$) to organically scale Taste DNA fidelity with library size.
- [x] **Adaptive DNA Dimensionality (Build 35)**: Implemented dynamic feature scaling based on library size to prevent overfitting.
- [x] **Predictive Tag Sanity Check (Build 34)**: Implemented support-based filtering in `solve_user_taste.py` to eliminate statistical aliasing (phantom tags) from the DNA view.
- [x] **Whitening Optimization**: Re-calibrated the tag pipeline to a 95% variance threshold for improved regularization.
- [x] **Windows File Locking Fix (Build 33)**: Resolved persistent `OSError [Errno 22]` and `PermissionError` in the data pipeline by implementing a robust rename-and-replace strategy for `.npy` artifacts.
- [x] **Pipeline Stability**: Fixed `NameError` and `ImportError` in tag vector generation; restored missing `whiten` function and centralized `calculate_dot_product_lambda`.
- [x] **Robust Artifact Saving**: Implemented a "temp-and-move" strategy (`safe_save_npy`) for all pipeline artifacts to resolve Windows file locking errors when the FastAPI server is active.
- [x] **Tag List Recovery**: Resolved a mismatch between the tag whitening matrix and the master tag name list, ensuring correct predictive tag mapping.
- [x] **DNA Solver Refinement**: Increased the ridge regression alpha range to $10^{-2}-10^{6}$ to ensure optimal regularization is captured for all user profiles.
- [x] **Inclusive Personalization**: Removed the "minimum review playtime" filter from soft-labeling, allowing all games with positive playtime to contribute to taste DNA.
- [x] **Tag Space Stabilization**: Fixed misaligned predictive tags by ensuring `tag_names.json` is always saved and introducing explicit prior anchors (`tag_prior_counts.npy`, `tag_prior_transformed.npy`).
- [x] **UI Intuition (Labels)**: Renamed "Age" slider to "Release Date" across the app (modern and legacy) to improve user intuition.
- [x] **Steam Integration**: Made the "Trending" label a direct link to Steam's most played charts while preserving checkbox functionality.
- [x] **Unified Personalized Engine**: Unified the Taste DNA Solver and Discovery Engine into a single mathematical system. Implemented **Linear Scorer Mode**, **Penalized Normalization**, and **Global Scaling (11.28x)** for 100% ranking consistency.
- [x] **UI Robustness & Recovery**: Added a global **ErrorBoundary**, **Reset App** safety button, and hardened all slider/parsing logic against null/NaN values.
- [x] **Analyze My Catalogue (Phase 1-3)**: Implemented data acquisition, soft-labeling, verification UI, and regression-based DNA solving.

### Bugs (work on these first)

### Next Priority
**⚠️ MANDATORY POLISH CONSTRAINT**: DO NOT start these tasks until the current Unified Personalized Engine (Build 15) is confirmed 100% bug-free, stable, and mathematically perfect in all edge cases.

#### Phase 1: High Impact / Low Effort (The "Agency Update")
- [ ] **Visual Previews**: Implement "Hover-to-Play" gameplay clips in `GameCard` using the `movies` metadata.
- [ ] **Instant Blacklist**: Add an "X" / "Hide" button to game cards that persists to `localStorage` and immediately removes the game from view.
- [ ] **Overlaid Action Buttons**: Add hover-triggered buttons to `GameCard` for quick categorization: "Mark as Played", "Add to Backlog", "Wishlist", and "Hide".
- [ ] **Explainable AI (XAI)**: Add deep justifications to the "Taste DNA Solved" page and Recommender. For tags, point to specific library games that drove the weight. For recommendations, show a breakdown of what influenced the score (DNA vs Meta vs Seed).
- [ ] **Personal Wishlist**: Add a "Bookmark" / "Star" feature to save games for later discovery.
- [ ] **Fuzzy Search**: Allow for non-exact matches to strings in multiselects.
- [ ] **"Mark as Played"**: Explicit toggle to exclude a game from recommendations without blacklisting its "vibe."

#### Phase 2: Medium Effort (Strategic Alignment)
- [ ] **User Management Hub**: Implement a dedicated "User" page for managing identity context. View and edit ratings, ground truth, backlog, wishlist, and ignore lists in a centralized, searchable interface.
- [ ] **Softmin Multi-Target Blending**: Replace additive similarity blending with a Softmin-based approach for multi-comparator searches (DNA + Seeds, Seed + Seed). This rewards "between-ness" and ensures results align with *all* active targets rather than just one.
- [ ] **Architectural Decoupling**: Separate "User Identity" (library, ratings, blacklist) from "Taste DNA" (mathematical profiles). Allow multiple independent DNA profiles to be hot-swapped or blended while maintaining a stable exclusion context.
- [ ] **Psychological Taxonomy**: Implement a "Mood" filter based on the Quantic Foundry model (Mastery, Immersion, etc.) mapped to Steam tags.
- [ ] **Motivation Profiling**: Add "Why I Play" sliders (Destruction, Strategy, Fantasy, Discovery) that map to underlying tag clusters.
- [ ] **Cross-Domain Seeds**: Enable searching for games using Movie or Book titles by leveraging semantic embedding similarity.
- [ ] **Universal Importer**: Create a tool to import Epic/GOG libraries via CSV/Text paste to avoid duplicate recommendations.
- [ ] **Client-Side Pricing**: Fetch real-time price/discount data for the top 20 visible recommendations using the Steam Storefront API.

#### Phase 3: High Effort (Moonshots)
- [ ] **Taste Twins**: Opt-in social discovery to find users with similar rating histories.
- [ ] **Visual-First Mode**: Implement a "Gallery View" that prioritizes high-res screenshots and environmental aesthetic (Color Palette Matching).
- [ ] **The "Wildcard" Slot**: Intentionally inject one recommendation per page that matches the "Vibe" but purposefully violates genre preferences.
- [ ] **AI Shovelware Filter**: Explicit developer-history and asset-flip detection beyond Bayesian scores.

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] **Price Implementation**: Added a Price slider across the stack. Calculated price Z-scores in the metadata pipeline, integrated price into the Taste DNA regression (with adaptive dimensionality adjustment), and added Price contribution visualization to game cards.
- [x] **Navigation Hub (Splash Page)**: Created a central splash page to route users between the "Recommender," "Taste DNA Solver," and "Steam Data Analyzer." Mapped the header logo/text to return to this hub.
- [x] **Mobile Optimization**: Enhanced mobile UX by making the options window collapsible and removing inner scroll bars from complex UI elements.
- [x] **Persistent Contributions**: Made "Visualize Contributions" always on by removing the UI toggle and ensuring scoring breakdowns are permanently visible on game cards.
- [x] **Methodology Audit**: Synchronized `methodology.md` and `orientation.md` with the current codebase, documenting the MPNet upgrade, GPU acceleration, and descriptive-only semantic path.
- [x] **Semantic Model Upgrade**: Switched to `all-mpnet-base-v2` (768 dimensions) for significantly better qualitative and thematic matching.
- [x] **GPU Acceleration support**: Added automatic CUDA detection to both the pipeline and the server to utilize NVIDIA GPUs for embedding generation and prompt processing.
- [x] **Descriptive-Only Semantic Path**: Decoupled structural (tag/genre) embeddings from the semantic similarity slider to reduce noise from categorical matches (e.g., 'Dark Humor' matching 'happiness') and focus strictly on narrative 'vibes.'
- [x] **Parallelized Dataset Builder**: Reduced `build_scraped_dataset.py` execution time from 5 hours to 20 minutes using multiprocessing.
- [x] **Fuzzy Game Search**: Implemented accent/symbol normalization (™/®) and popularity-aware ranking to surface high-profile games like *Dragon Quest* more reliably.
- [x] **Uncentered Unit Semantic Path**: Refactored semantic embeddings to use uncentered ZCA and unit-normalization for consistent, origin-preserving similarity scoring.
- [x] **Natural Range Z-Scoring**: Established a calibrated Z-scoring model for semantic similarities based on a 10,000-pair simulation, ensuring consistent slider impact.
- [x] **Refined NSFW Filter**: Updated logic to only trigger on "Adult Only" flags and "Hentai" tags, preventing over-filtering of standard mature content.
- [x] **Genre Leakage Fix**: Hardened HTML parsing to prevent developer/publisher names from being captured as genres.
- [x] **Bit-Perfect Parity**: Achieved perfect scoring agreement between the Taste DNA solver and the Recommender by unifying the scoring path in `common/utils.py`.
- [x] **Absolute Slider Logic**: Refactored `server.py` and `App.tsx` so sliders act as absolute weights in both DNA and standard modes, eliminating the "squaring bug" and multiplier confusion.
- [x] **Profile Exclusion**: Updated `App.tsx` to set `profile_filter: 'all'` by default when applying a profile, ensuring library games are automatically excluded from the "Love List" in the recommender.
- [x] **Robust Error Handling**: Added extensive logging and type-checking to the profile application pipeline to prevent metadata-related crashes and UI drift.
- [x] **Dota 2 Searchability**: Verified AppID 570 presence and ensured searching for "Dota" or "Dota 2" works correctly.
- [x] **Sample Bias Mitigation**: Removed `StandardScaler` from the Taste DNA solver; shifted to **Global Standardization** to improve generalization (~4% MSE reduction).
- [x] **Tie-Aware Quantile Normalization**: Replaced `t-to-z` transformation with a rank-based mapping that handles large clusters of identical Bayesian scores (e.g., zero-review games) as discrete spikes in a smooth normal envelope.
- [x] **Interactive Explainability**: Implemented animated scatter plots and Discovery optimization bar charts in the Personalization insights view.
- [x] **Visual Analytics Refinement**: Added log scaling, dynamic timeline capping (2026), and clear axis labeling to all explainability visualizations.

### Bugs (work on these first)

- [ ] No need for adding the coefficient in the recommender page. Scale the end result so that the maximum value for the largest slider is +3 or -3. 
- [ ] Games You'll love isn't filtering out added games

### Next Priority

#### Phase 1: Mathematical & Analytical Refinement
- [ ] **Methodology Audit**: Perform a close reading of the current codebase and synchronize `methodology.md` to ensure it accurately reflects the implemented algorithms and scoring logic.
- [ ] **Robust Logistic Tag Profiling**: Shift Taste DNA tag prediction to univariate logistic regression (Tag Presence ~ User Rating). Implement "Sandbagging" (Rating 0/10 anchors) to stabilize slopes against low-frequency overfitting.
- [ ] **Semantic-Aware Difficulty Model**: Transition difficulty predictions to a penalized regression model using whitened tag and semantic vectors. Perform nested cross-validation to benchmark L1 vs L2 generalization performance.
- [ ] **Review Sentiment Summarization**: Implement Pros/Cons extraction from game review sets to provide immediate qualitative feedback in the UI. (Note: Can leverage local `gpt-oss-20b` via `ollama`; process will be highly time-intensive).

#### Phase 2: High Impact / Low Effort (The "Agency Update")
- [ ] **Persistent Contributions**: Make "Visualize Contributions" always on by removing the UI toggle and ensuring scoring breakdowns are permanently visible on game cards.
- [ ] **Navigation Hub (Splash Page)**: Create a central splash page to route users between the "Recommender" and "Steam Data Analyzer." Map the header logo/text to return to this hub.
- [ ] **Mobile Optimization**: Enhance mobile UX by making the options window collapsible and removing inner scroll bars from complex UI elements.
- [ ] **Visual Previews**: Implement "Hover-to-Play" gameplay clips in `GameCard` using the `movies` metadata.
- [ ] **Instant Blacklist**: Add an "X" / "Hide" button to game cards that persists to `localStorage` and immediately removes the game from view.
- [ ] **Overlaid Action Buttons**: Add hover-triggered buttons to `GameCard` for quick categorization: "Mark as Played", "Add to Backlog", "Wishlist", and "Hide".
- [ ] **Explainable AI (XAI)**: Add deep justifications to the "Taste DNA Solved" page and Recommender. For tags, point to specific library games that drove the weight. For recommendations, show a breakdown of what influenced the score (DNA vs Meta vs Seed).
- [ ] **Personal Wishlist**: Add a "Bookmark" / "Star" feature to save games for later discovery.
- [ ] **Fuzzy Search**: Allow for non-exact matches to strings in multiselects.
- [ ] **"Mark as Played"**: Explicit toggle to exclude a game from recommendations without blacklisting its "vibe."

#### Phase 3: Strategic Alignment
- [ ] **User Management Hub**: Implement a dedicated "User" page for managing identity context. View and edit ratings, ground truth, backlog, wishlist, and ignore lists in a centralized, searchable interface.
- [ ] **Softmin Multi-Target Blending**: Replace additive similarity blending with a Softmin-based approach for multi-comparator searches (DNA + Seeds, Seed + Seed). This rewards "between-ness" and ensures results align with *all* active targets rather than just one.
- [ ] **Architectural Decoupling**: Separate "User Identity" (library, ratings, blacklist) from "Taste DNA" (mathematical profiles). Allow multiple independent DNA profiles to be hot-swapped or blended while maintaining a stable exclusion context.
- [ ] **Psychological Taxonomy**: Implement a "Mood" filter based on the Quantic Foundry model (Mastery, Immersion, etc.) mapped to Steam tags.
- [ ] **Motivation Profiling**: Add "Why I Play" sliders (Destruction, Strategy, Fantasy, Discovery) that map to underlying tag clusters.
- [ ] **Cross-Domain Seeds**: Enable searching for games using Movie or Book titles by leveraging semantic embedding similarity.
- [ ] **Universal Importer**: Create a tool to import Epic/GOG libraries via CSV/Text paste to avoid duplicate recommendations.
- [ ] **Client-Side Pricing**: Fetch real-time price/discount data for the top 20 visible recommendations using the Steam Storefront API.

#### Phase 4: Moonshots
- [ ] **Taste Twins**: Opt-in social discovery to find users with similar rating histories.
- [ ] **Visual-First Mode**: Implement a "Gallery View" that prioritizes high-res screenshots and environmental aesthetic (Color Palette Matching).
- [ ] **The "Wildcard" Slot**: Intentionally inject one recommendation per page that matches the "Vibe" but purposefully violates genre preferences.
- [ ] **AI Shovelware Filter**: Explicit developer-history and asset-flip detection beyond Bayesian scores.

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

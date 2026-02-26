# How to use this file

- Only tackle one item from this list at a time. 
- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed
- [x] **Build 48: Topic Sync Fix**: Synchronized the thematic topic match signal between the Taste DNA page and the Recommender by ensuring `topic_vibe_vector` and `gamma_topic` are correctly mapped in the frontend. Added a "Topic Match" slider to the sidebar for manual control.
- [x] **Build 55-68: Library Management Update**: Implemented categorical status filtering (Rated/Played/Backlog/Ignored/Wishlist) across engine and UI. Resolved severe UI lag using Lite Virtualization (Infinite Scroll), React memoization, and isolated search components. Standardized keyboard-driven data entry with auto-scroll across search dropdowns. Fixed multiple server and solver crashes related to the status migration.
- [x] **Catalogue View (User Hub)**: Created a high-performance management page for categorized game identity. Features infinite scroll for 1000+ items, multi-column sorting, and real-time status syncing.       
- [x] **Build 49: Precision Update**: Implemented population-wide modality scaling ($1.0, 2.0, 26.5$) and Zero-Order Relevance Filtering ($p \le N-7$) for the Taste Solver. Synchronized all similarity signals across pipeline and recommender.
- [x] **Build 48: High-Fidelity Thematic Search**: Integrated 250-topic BERTopic layer with JSD-based similarity and population Z-scoring. Added Topic Match signal to Taste DNA and backend.
- [x] **Consensus Blending (Softmin)**: Integrated Softmin similarity blending (T=3.0) for multi-signal targets (DNA + Seeds + Prompts) to enforce a "consensus" requirement.
- [x] **Backlog Integration**: Decoupled owned-but-unplayed games from exclusion filters. Added a dedicated "From Your Backlog" discovery section to the Insights page.
- [x] **Profile-Aware Reset**: Refactored the "Reset All" functionality to snap sliders to the solved DNA baseline while clearing search-specific prompts and seeds.
- [x] **Scroll to Top**: Implemented a floating, animated "Scroll to Top" button on the Recommender page for better navigation.
- [x] **Perfect Parity**: Synchronized Analyzer's "Similar to Favorites" scoring with Recommender's manual mode, including personalized p+t quality and normalization fixes.
- [x] **Robust Metadata Filtering**: Implemented text-based placeholder detection ("Coming Soon", "TBD") for unreleased game filtering to supplement build-time cutoffs.
- [x] **Legacy Removal**: Excised unused structural semantic/tag artifacts to reclaim RAM and minimize Git footprint.
- [x] **Mathematical Parity (Solver/API)**: Harmonized the scoring pipeline between the Python solver and FastAPI backend. Fixed the "82% Bug" by isolating the 5.0 neutral intercept from scaling division.       
- [x] **Transparency Mode (Absolute Weights)**: Refactored the Taste DNA mode to use UI sliders as absolute weights rather than multipliers.
- [x] **Delisted Game Filter**: Implemented a global filter to exclude games that are no longer available for purchase on Steam.

### Bugs (work on these first)
- [ ] **Filter Failure (VR/English)**: Investigate reports that "Filter VR Only" and "Filter Non-English" are not correctly excluding games. (Examples needed).

### Next Priority

#### Phase 1: Mathematical & Analytical Refinement
- [ ] **Lognormal Rating Prediction**: Update rating predictions to use lognormal smoothing to estimate the probability of a positive review, then map that percentage to a 0-10 score (synced with the Bayesian quality scale).
- [ ] **LASSO Difficulty Model**: Train a new difficulty model using both whitened tag and semantic vectors via 10-fold CV LASSO. Re-establish tag-level explainability for the new coefficients.
- [ ] **Greedy Semantic Labeling**: Implement iterative "bag of words" vector approximation (top 10k words) to label semantic dimensions. Use greedy addition/subtraction of word vectors to maximize similarity to dimensions.
- [ ] **Methodology Audit**: Perform a close reading of the current codebase and synchronize `methodology.md` to ensure it accurately reflects implementing logic.

#### Phase 2: High Impact / Low Effort (The "Agency Update")
- [ ] **Infinite Scroll**: Implement infinite scrolling in the Recommender view to allow seamless browsing of deep results without manual pagination or "Load More" interruptions.
- [ ] **Thematic Breakdown UI**: Display the top 3 topic keywords on the `GameCard` or Detail view for better explainability.
- [ ] **Review Count Indicators**: Add color-coded thumbs up/down counts to `GameCard`. Use significant-digit rounding for values > 1000 (e.g., 1.1k, 120k).
- [ ] **Explainable Hover Tooltips**: Implement contribution-aware hover explanations on the Insights page. Display top 3 contributors (Quality, Pop, etc.) with valence-aware text.
- [ ] **Kernel-Based Vibe Attribution**: For Vibe-led recommendations, identify and list the two training games contributing most to the prediction using `sim(A,B) * (rating_A - rating_mean)`.
- [ ] **Mobile Optimization**: Enhance mobile UX by making the options window collapsible and removing inner scroll bars.
- [ ] **Visual Previews**: Implement "Hover-to-Play" gameplay clips in `GameCard` using the `movies` metadata.

#### Phase 3: Strategic Alignment (User Hub & Identity)
- [ ] **Global Visual Cues**: Reflect user categories (icons/overlays) on game images throughout the site and add quick-action buttons to `GameCard` for categorization.
- [ ] **Wishlist External Links**: Integrate isthereanydeal.com links for wishlisted titles.
- [ ] **Architectural Decoupling**: Separate "User Identity" from "Taste DNA". Allow multiple independent DNA profiles to be hot-swapped while maintaining a stable exclusion context.
- [ ] **Shareable Taste Cards**: Generate a visual "Taste DNA" summary card optimized for sharing on social media.

#### Phase 4: Psychological & Market Extensions
- [ ] **Psychological Taxonomy**: Implement a "Mood" filter based on the Quantic Foundry model mapped to Steam tags.
- [ ] **Motivation Profiling**: Add "Why I Play" sliders (Destruction, Strategy, Fantasy, Discovery) that map to underlying tag clusters.
- [ ] **Pace & Intensity Metrics**: Derive "Pace" metadata (Slow-burn vs. High-intensity) from review text and tag clusters.
- [ ] **Balanced Review Snapshots**: Display representative positive and negative reviews side-by-side in the `GameCard`.
- [ ] **Cross-Domain Seeds**: Enable searching for games using Movie or Book titles by leveraging semantic embedding similarity.

#### Phase 5: Moonshots
- [ ] **Taste Twins**: Opt-in social discovery to find users with similar rating histories.
- [ ] **Visual-First Mode**: Implement a "Gallery View" that prioritizes high-res screenshots and environmental aesthetic (Color Palette Matching).
- [ ] **The "Wildcard" Slot**: Intentionally inject one recommendation per page that matches the "Vibe" but purposefully violates genre preferences.
- [ ] **AI Shovelware Filter**: Explicit developer-history and asset-flip detection beyond Bayesian scores.

# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] **UI Intuition (Labels)**: Renamed "Age" slider to "Release Date" across the app (modern and legacy) to improve user intuition.
- [x] **Steam Integration**: Made the "Trending" label a direct link to Steam's most played charts while preserving checkbox functionality.
- [x] **Unified Personalized Engine**: Unified the Taste DNA Solver and Discovery Engine into a single mathematical system. Implemented **Linear Scorer Mode**, **Penalized Normalization**, and **Global Scaling (11.28x)** for 100% ranking consistency.
- [x] **UI Robustness & Recovery**: Added a global **ErrorBoundary**, **Reset App** safety button, and hardened all slider/parsing logic against null/NaN values. Fixed the 'white screen of death' random failure state.
- [x] **Analyze My Catalogue (Phase 1-3)**: Implemented data acquisition, soft-labeling, verification UI, and regression-based DNA solving. 

### Bugs (work on these first)

### Next Priority
**⚠️ MANDATORY POLISH CONSTRAINT**: DO NOT start these tasks until the current Unified Personalized Engine (Build 15) is confirmed 100% bug-free, stable, and mathematically perfect in all edge cases.

#### Phase 1: High Impact / Low Effort (The "Agency Update")
- [ ] **Visual Previews**: Implement "Hover-to-Play" gameplay clips in `GameCard` using the `movies` metadata.
- [ ] **Instant Blacklist**: Add an "X" / "Hide" button to game cards that persists to `localStorage` and immediately removes the game from view.
- [ ] **Personal Wishlist**: Add a "Bookmark" / "Star" feature to save games for later discovery.

#### Phase 2: Medium Effort (Strategic Alignment)
- [ ] **Psychological Taxonomy**: Implement a "Mood" filter based on the Quantic Foundry model (Mastery, Immersion, etc.) mapped to Steam tags.
- [ ] **Universal Importer**: Create a tool to import Epic/GOG libraries via CSV/Text paste to avoid duplicate recommendations.
- [ ] **Client-Side Pricing**: Fetch real-time price/discount data for the top 20 visible recommendations using the Steam Storefront API.

#### Phase 3: High Effort (Moonshots)
- [ ] **Taste Twins**: Opt-in social discovery to find users with similar rating histories.
- [ ] **AI Shovelware Filter**: Explicit developer-history and asset-flip detection beyond Bayesian scores.

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

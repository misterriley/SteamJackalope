# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed
- [x] **Mathematical Parity (Solver/API)**: Harmonized the scoring pipeline between the Python solver and FastAPI backend. Fixed the "82% Bug" by isolating the 5.0 neutral intercept from scaling division and established perfect ordinal parity in top-10 lists.
- [x] **Transparency Mode (Absolute Weights)**: Refactored the Taste DNA mode to use UI sliders as absolute weights rather than multipliers. This ensures the weights shown in the UI (e.g., Quality = 0.86) are exactly what the recommendation engine uses, eliminating "hidden" scaling and the "Squaring Bug."
- [x] **Neutral Anchor Implementation**: Standardized the use of a 5.0 baseline for both raw scoring and probability mapping, ensuring that "Match %" correctly reflects deviations from an "average game" prior.
- [x] **Production Data Corruption Fix**: Identified and patched the bug causing `tag_names.json` to be overwritten during tests. Added an automated `Read-Only` lock to the `data/production` directory during test execution.
- [x] **Delisted Game Filter**: Implemented a global filter to exclude games that are no longer available for purchase on Steam from recommendations and lists.
- [x] **Personalized Quality UI**: Integrated kernel-smoothed quality adjustments into the React UI and Recommender pathway, displaying tailored "Match %" for library games.

### Bugs (work on these first)

### Next Priority

#### Phase 1: Mathematical & Analytical Refinement
- [ ] **Methodology Audit**: Perform a close reading of the current codebase and synchronize `methodology.md` to ensure it accurately reflects the implemented algorithms and scoring logic.
- [ ] **Semantic-Aware Difficulty Model**: Transition difficulty predictions to a penalized regression model using whitened tag and semantic vectors. Perform nested cross-validation to benchmark L1 vs L2 generalization performance.
- [ ] **Review Sentiment Summarization**: Implement Pros/Cons extraction from game review sets to provide immediate qualitative feedback in the UI. (Note: Can leverage local `gpt-oss-20b` via `ollama`; process will be highly time-intensive).

#### Phase 2: High Impact / Low Effort (The "Agency Update")
- [ ] **Mobile Optimization**: Enhance mobile UX by making the options window collapsible and removing inner scroll bars from complex UI elements.
- [ ] **Visual Previews**: Implement "Hover-to-Play" gameplay clips in `GameCard` using the `movies` metadata.
- [ ] **Instant Blacklist**: Add an "X" / "Hide" button to game cards that persists to `localStorage` and immediately removes the game from view.
- [ ] **Overlaid Action Buttons**: Add hover-triggered buttons to `GameCard` for quick categorization: "Mark as Played", "Add to Backlog", "Wishlist", and "Hide".
- [ ] **Personal Wishlist**: Add a "Bookmark" / "Star" feature to save games for later discovery.
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

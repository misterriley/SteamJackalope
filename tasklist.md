# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed
- **Game Exclusion Logic Refinement**: Implemented robust exclusion logic in the Taste DNA solver to ensure manually added and rated games are filtered from recommendation lists, preventing self-recommendations.
- **UI Explainability Enhancements**:
    - Implemented a visual reversal of negatively weighted tag dimensions (e.g., "NSFW/Mature vs. Tanks/Wargame") with absolute positive values, green bars, and swapped positive/negative associated tags for improved clarity, including chart data and title.
    - Changed alignment values on Taste Anchors (North Stars, The Abyss) to display raw sum of tag and semantic contributions (e.g., "Alignment: 1.50" or "Alignment: -1.50") instead of percentages.
    - Removed "Key Vibe Dimensions" section from Personalization Insights.
    - Updated semantic hover text for clearer explanations.
    - Modified difficulty hover graph X-axis ticks to 0, 5, and 10.
    - Fixed corrupted tag dimension display and predictive tags.
- **Slider Sensitivity Study & Parameter Tuning**: Completed a simulation study to analyze slider impact on game rankings, leading to refined default slider values (Semantic=0.25, Tag=1.5, Quality=1.0) and other parameters for improved recommendation stability.
- **DNA Weight Import**: Automated the mapping of solved semantic weights to the recommender's UI sliders.
- **Case-Insensitive Semantic Model**: Lowercased all input text across the pipeline and backend to eliminate orthographic noise.
- **Hybrid Taste Anchors**: Updated North Star and Abyss logic to use weighted alignment of both semantic and tag components.
- **Semantic Variance Parity**: Integrated dynamic scaling (11.25x) to ensure vibes and tags are treated equally in the DNA solve.
- **Semantic Dimension Labeling**: Developed composite word-sum calibration (e.g., "Exploration + Terraform") using a 10,000-word vocabulary.
- **Hybrid Semantic-Tag Solver**: Implemented high-dimensional LASSO regression across 235 descriptive dimensions.

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

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

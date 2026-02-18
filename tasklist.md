# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] **Bit-Perfect Parity**: Achieved perfect scoring agreement between the Taste DNA solver and the Recommender by unifying the scoring path in `common/utils.py`.
- [x] **Absolute Slider Logic**: Refactored `server.py` and `App.tsx` so sliders act as absolute weights in both DNA and standard modes, eliminating the "squaring bug" and multiplier confusion.
- [x] **Profile Exclusion**: Updated `App.tsx` to set `profile_filter: 'all'` by default when applying a profile, ensuring library games are automatically excluded from the "Love List" in the recommender.
- [x] **Robust Error Handling**: Added extensive logging and type-checking to the profile application pipeline to prevent metadata-related crashes and UI drift.
- [x] **Dota 2 Searchability**: Verified AppID 570 presence and ensured searching for "Dota" or "Dota 2" works correctly.

### Bugs (work on these first)

- [ ] No need for adding the coefficient in the recommender page. Scale the end result so that the maximum value for the largest slider is +3 or -3. 
- [ ] Games You'll love isn't filtering out added games

### Next Priority

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

# How to use this file

- Only tackle one item from this list at a time. 
- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed
- [DONE] **Differentiable Temperature-Dependent Kernel**: Refactored hard veto logic into a soft-constraint framework using sigmoid-based penalty functions and symmetric anchor enforcement.
- [DONE] **Archetypal Solver Refinement**: Refactored the Taste DNA solver to use 45 fixed Structural Archetypes (MIGs, Metadata) as features, successfully reducing overfit ($R^2$ dropped from > 0.7 to ~0.28) and improving model generalizability.
- [DONE] **Identity & Perspective Discipline**: Implemented Symmetric Hard Anchor enforcement for Perspective (First-Person, Isometric) and strict Interaction Models (CRPG, RPG), ensuring structural consistency in recommendations.
- [DONE] **MIG Refinement & Cleanup**: Refined Mechanical Identity Groups (MIGs) to require core verbs and eliminated identity leaks from broad tags like 'Exploration' or 'Surreal'.
- [DONE] **Mood Clash & Un-rescueable Conflicts**: Integrated differentiable penalties for mood mismatches (Serious vs. Lighthearted) and strict mechanical barriers (Shooter vs. Narrative) into the production kernel.
- [DONE] **Jackalope Kernel Production Port**: Integrated the high-fidelity Mechanical Identity Group (MIG) kernel into `app/server.py`. Added support for 32 granular skill silos, Title Hijacking protection, and "Adult Only" banner-based siloing.

### Bugs (work on these first)
- [ ] **Filter Failure (VR/English)**: Investigate reports that "Filter VR Only" and "Filter Non-English" are not correctly excluding games. (Examples needed).

### Next Priority

#### Phase 1: Mathematical & Analytical Refinement
- [ ] **Kernel-Based Recommender Overhaul**: Port the backend recommendation logic (`app/server.py`) to use the new Hybrid Ridge regression model used in the Solver. This will ensure perfect parity between the "Taste DNA" preview and the actual recommendations.
- [ ] **Lognormal Rating Prediction**: Update rating predictions to use lognormal smoothing to estimate the probability of a positive review, then map that percentage to a 0-10 score (synced with the Bayesian quality scale).

#### Phase 2: High Impact / Low Effort (The "Agency Update")
- [ ] **Infinite Scroll**: Implement infinite scrolling in the Recommender view to allow seamless browsing of deep results without manual pagination or "Load More" interruptions.
- [ ] **Thematic Breakdown UI**: Display the top 3 topic keywords on the `GameCard` or Detail view for better explainability.
- [ ] **Explainable Hover Tooltips**: Implement contribution-aware hover explanations on the Insights page. Display top 3 contributors (Quality, Pop, etc.) with valence-aware text.
- [ ] **Kernel-Based Vibe Attribution**: For Vibe-led recommendations, identify and list the two training games contributing most to the prediction using `sim(A,B) * (rating_A - rating_mean)`.
- [ ] **Mobile Optimization**: Enhance mobile UX by making the options window collapsible and removing inner scroll bars.
- [ ] **Visual Previews**: Implement "Hover-to-Play" gameplay clips in `GameCard` using the `movies` metadata.

#### Phase 2: High Impact / Low Effort (The "Agency Update")
- [ ] **Infinite Scroll**: Implement infinite scrolling in the Recommender view to allow seamless browsing of deep results without manual pagination or "Load More" interruptions.
- [ ] **Thematic Breakdown UI**: Display the top 3 topic keywords on the `GameCard` or Detail view for better explainability.
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

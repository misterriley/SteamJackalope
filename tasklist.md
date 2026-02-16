# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] **Unified Personalized Engine**: Unified the Taste DNA Solver and Discovery Engine into a single mathematical system. Implemented **Linear Scorer Mode**, **Penalized Normalization**, and **Global Scaling (11.28x)** for 100% ranking consistency.
- [x] **UI Robustness & Recovery**: Added a global **ErrorBoundary**, **Reset App** safety button, and hardened all slider/parsing logic against null/NaN values. Fixed the 'white screen of death' random failure state.
- [x] **Analyze My Catalogue (Phase 1-3)**: Implemented data acquisition, soft-labeling, verification UI, and regression-based DNA solving. 

### Bugs (work on these first)

- [ ] "Age" is counterintuitive as a label for the slider on the page - I keep thinking that to the right = higher age = weighting toward older games. Let's rename the slider to "Release Date", which makes it more intuitive with the slider direction.  

### Next Priority

- [ ] The "Trending" label (next to the checkbox), when clicked on, should open a tab to the most played games list for Steam: https://store.steampowered.com/charts/mostplayed. This should happen only when the text is clicked - the box itself should work as a standard checkbox. 
- [ ] Implement "Analyze My Catalogue" (Personalization Engine)
    - [x] **Data Acquisition**: Create a backend module to fetch a user's library (AppID + Playtime) via SteamID64. Support both OpenID login and manual AppID list input. (Implemented: `scraping/get_user_stats.py`)
    - [x] **Soft-Labeling Engine**: Implement the math to predict user ratings (1-10) using the global Playtime Sentiment Model ($\gamma, s$) and global Quality Scores. (Implemented: `pipeline/generate_user_soft_labels.py`)
    - [ ] **User Verification UI**: Build a frontend view to display predicted ratings and allow the user to quickly verify or override them (Ground Truth generation).
    - [ ] **Taste Solve**: Implement a Ridge Regression solver with LOOCV on the backend to map verified ratings against the 128-dim tag vectors and metadata factors.
    - [ ] **Recommender Integration**: Feed the resulting regression weights back into the search sliders and identify "Ideal Match" seed games based on the user's solved taste vector.
    - [ ] **QA**: Add manual verification steps to `QA.md` for the catalogue analysis flow.

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

- [ ] Start on "analyze my catalogue" project?
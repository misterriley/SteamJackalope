# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] SEO for the website (Metadata, Open Graph, Twitter Cards).
- [x] Validated Steam store endpoints for tags/genres/categories and made them clickable in the UI.
    - [x] Created validation script `tools/validate_steam_links.py` (running).
    - [x] Updated backend to serve term links and filter "dead" tags.
    - [x] Updated frontend (`GameCard`, `ListsView`) to render clickable links.

### Top Priority

- [ ] Implement "Analyze My Catalogue" (Personalization Engine)
    - [ ] **Data Acquisition**: Create a backend module to fetch a user's library (AppID + Playtime) via SteamID64. Support both OpenID login and manual AppID list input.
    - [ ] **Soft-Labeling Engine**: Implement the math to predict user ratings (1-10) using the global Playtime Sentiment Model ($\gamma, s$) and global Quality Scores.
    - [ ] **User Verification UI**: Build a frontend view to display predicted ratings and allow the user to quickly verify or override them (Ground Truth generation).
    - [ ] **Taste Solve**: Implement a Ridge Regression solver with LOOCV on the backend to map verified ratings against the 128-dim tag vectors and metadata factors.
    - [ ] **Recommender Integration**: Feed the resulting regression weights back into the search sliders and identify "Ideal Match" seed games based on the user's solved taste vector.
    - [ ] **QA**: Add manual verification steps to `QA.md` for the catalogue analysis flow.

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

- [ ] Start on "analyze my catalogue" project?
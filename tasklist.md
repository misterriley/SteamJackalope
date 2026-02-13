# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] Difficulty values of 0 aren't displaying in the game cards.
- [x] **Playtime-Sentiment Correlation Research:**
    - Developed a kernel smoothing model to predict review sentiment from playtime.
    - **Vectorization:** Implemented a fully vectorized, loop-free version of the LOO estimation logic in `research/analyze_playtime_sentiment.py`.
    - **Global Optimization:** Created a parallelized optimization script (`research/optimize_global_playtime_params_parallel.py`) to find defaults that generalize across games.
    - **Stability Analysis:** Ran 10 independent optimization runs on random samples of 100 games. Converged on stable global defaults: $\gamma = 0.5109$ and $s = 0.7812$.
    - **Visualization:** Created `research/visualize_playtime_curve.py` to generate smooth probability curves for any game. Generated charts for the top 10 most popular games.
    - **Integration:** Externalized optimized parameters to `common/constants.py`.

### Top Priority

### Lesser Priority

- [ ] Maintain a consistent session state so that switching between pages does not lose slider settings, seeds, etc.
- [ ] Make game cards on website same color as banner (light purple) with black font. Make Search Options card yellow with black font. Buttons should be teal with a black font.
- [ ] Have an x-out button on seed games directly to the left of the card where the seed game is displayed.
- [ ] Add a build versioning system for me to keep track of where we're at in the deployment cycle. Current version is 0.0.1 (pre-pre-alpha).
- [ ] Create a quick onboarding for new users. 
        - [ ] Create a guide page with simple instructions for how to use the site. 
        - [ ] Link it in the same way the other pages are linked, in the hamburger menu.
- [ ] Display image files linked in the download data on the game cards if it can be done without expanding memory footprint too much.
- [ ] Generate a changelog and link it as a page on the website. Make updating the changelog one of the tasks on onShutdown.md.
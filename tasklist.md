# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- Optimized sidebar vertical space by refactoring sliders into a compact layout with labels on the left and removing redundant bound labels for semantic/tag match sliders.
- Successfully configured Git LFS and pushed all data/production files to GitHub, including CSV, NPY, PARQUET, and JSON files with proper large file storage tracking.

### Top Priority

### Lesser Priority

- [ ] Zeros in the game cards are displaying weirdly.
- [ ] Make game cards on website same color as banner (light purple) with black font. Make Search Options card yellow with black font. Buttons should be teal with a black font.
- [ ] Have an x-out button on seed games directly next to the card where the seed game is displayed.
- [ ] Add a build versioning system for me to keep track of where we're at in the deployment cycle. Current version is 0.0.1 (pre-pre-alpha).
- [ ] Create a quick onboarding for new users. 
        - [ ] Create a guide page with simple instructions for how to use the site. 
        - [ ] Link it in the same way the other pages are linked, in the hamburger menu.
- [ ] New research to do: discriminating between whether a game is liked or not based on playtime.
        - [ ] Pick a game with a large number of reviews.
        - [ ] Load all of the reviews and review times associated with that game.
        - [ ] Build out distributions of these reviews.
        - [ ] Train a logistic regression predicting the probability that a review will be positive as a function of the logarithm of its length + 1.
        - [ ] Test weighting based on the inverse of the rates of positive and negative reviews to combat unbalanced data.
        - [ ] Generate a global prior (equally weighted per game, not per rating) and run a simulation study.
        - [ ] Find the best combination of prior and game data for predicting outcomes in small sythetic games. 
- [ ] Display image files linked in the download data on the game cards if it can be done without expanding memory footprint too much.
- [ ] Generate a changelog and link it as a page on the website. Make updating the changelog one of the tasks on onShutdown.md.
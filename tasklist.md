# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] Remove the API key for steam scarping into a private file that doesn't get added to github.
- [x] Add a copyright notice and a github link (https://github.com/misterriley/SteamJackalope) to the bottom of each page.
- [x] Fix startup race condition - backend connection errors on first page load

### Top Priority

- [ ] Fix the lists page - the discovery preference dropdown doesn't seem to be affecting the results at all. The quality scores shown should be different depending on the dropdown selection.

### Lesser Priority

- [ ] Split the methodology page. One should be "About", which should contain what is now the first part of the Methodology page, and the rest should stay where it is.
- [ ] The left pane of the page is too long vertically. Place the slider labels to the left of the sliders rather than below them.
- [ ] The main folder is getting full. Let's move some of the top level files to folders. 
        - [ ] Make a folder for the production data files (e.g., tag_vectors_norms.npy, anything else written by scripts in pipeline). Make sure that everything that reads the production files now gets data from the production files directory. Make sure that tests are not allowed to write to this directory.
        - [ ] Keep .md and .bat files in place.
        - [ ] Find appropriate locations for anything else that does not need to be in the top level folder.
- [ ] Create readme.md files for each directory. 
- [ ] Add a build versioning system for me to keep track of where we're at in the deployment cycle. Current version is 0.0.1 (pre-pre-alpha, if that's a thing)
- [ ] Create a banner at the top of the page.
        - [ ] Give the name of the website.
        - [ ] Make it a pretty color, maybe a light purple.
        - [ ] Use the jackalope icon.
        - [ ] Remove the jackalope icon from the Recommender header. 
        - [ ] Link the secondary pages from a hamburger menu rather than from links at the top of the page. 
- [ ] Create a quick onboarding for new users. 
        - [ ] Create a guide page with simple instructions for how to use the site. 
- [ ] New research to do: discriminating between whether a game is liked or not based on playtime.
        - [ ] Pick a game with a large number of reviews.
        - [ ] Load all of the reviews and review times associated with that game.
        - [ ] Build out distributions of these reviews.
        - [ ] Train a logistic regression predicting the probability that a review will be positive as a function of the logarithm of its length + 1.
        - [ ] Test weighting based on the inverse of the rates of positive and negative reviews to combat unbalanced data.
        - [ ] Generate a global prior (equally weighted per game, not per rating) and run a simulation study.
        - [ ] Find the best combination of prior and game data for predicting outcomes in small sythetic games. 
- [ ] Add the short description of the game to the game cards.
- [ ] Display image files linked in the download data on the game cards if it can be done without expanding memory footprint too much.
- [ ] Generate a changelog and link it as a page on the website. Make updating the changelog one of the tasks on onShutdown.md.
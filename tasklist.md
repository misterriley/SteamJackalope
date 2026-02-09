# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

## Recently Completed

## Top Priority

- [x] Reduce the dimension of whitening matrices. The purpose of this task is to reduce memory load and decrease noise sensitivity. 
        - [x] To begin, print the size and shape of all production files.
        - [x] For whitening steps, check the ZCA eigenvalues for the percent variance explained by each dimension. Order the dimensions by percent variance explained, and cut off the smallest explained-variance dimensions after a total of 80% of the variance explained has been retained.
        - [x] Ensure that these reduced dimension matrices are being produced in all scripts in the `pipeline` directory.
        - [x] Ensure that when data from these files is used while the website is running, the lower dimension matrices do not interfere with how the website functions.
        - [x] Ensure that this change does not interfere with the 'float16' formatting currently in place.
        - [x] When finished, rerun `pipeline\run_pipeline.py`, then print out the size and shape of the new production files. 
        - [x] Report the change in size.
        - [x] Call `./run_test_env.bat` to launch the website.

### Lesser Priority

- [ ] Add a build versioning system for me to keep track of where we're at in the deployment cycle. Current version is 0.0.1 (pre-pre-alpha, if that's a thing)
- [ ] Create a banner at the top of the page.
        - [ ] Give the name of the website.
        - [ ] Make it a pretty color, maybe a light purple.
        - [ ] Use the jackalope icon.
        - [ ] Remove the jackalope icon from the Recommender header. 
        - [ ] Link the secondary pages from a hamburger menu rather than from links at the top of the page. 
- [ ] Create a quick onboarding for new users. 
        - [ ] Have prominent text at the top that says "New here? Click the 'Random' button to try out the features!"
        - [ ] Create a guide page with simple instructions for how to use the site. 
- [ ] Remove the API key for steam scarping into a private file that doesn't get added to github.
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

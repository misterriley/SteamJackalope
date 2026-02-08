# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing. 

# Task List

- [x] In generate_tag_vectors.apply_tag_transform, the regularization step has to be handled differently for CLR and anscombe transforms due to the change in geometry. 
    - [x] The vote counts should be regularized toward the prior first, since the prior was determined on raw count data. 
    - [x] Then, after vote counts have been regularized, apply the transform.
    - [x] Also apply the transform to the prior.
    - [x] The resultant vectors should be transform(regularized_vector) - transform(prior). If I am correct, then this should not change the output under the 'none' transform.
    - [x] We will need to rerun generate_tag_vectors from scratch as well as any code that depends on the tag vectors data (e.g., calculating difficulty values). 

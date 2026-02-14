# Git Push Protocol

This file contains a procedure for steps to take alongside pushing commits to git. 

## Updating the Changelog

Before pushing changes to the remote repository, ensure that `CHANGELOG.md` is updated to reflect all significant user-facing or algorithmic changes.

### Instructions

1.  **Review History**: Use `git log origin/master..HEAD` to review all commits made since the last push.
2.  **Identify Relevant Changes**: Summarize changes that fall into the following categories:
    *   **Algorithm**: Changes to recommendation logic, scoring, search precision, or data processing.
    *   **UI/UX**: New pages, layout updates, styling changes, or functional improvements to the web interface.
    *   **Deployment**: Updates to Docker configurations, deployment scripts, or environment settings.
3.  **Update CHANGELOG.md**:
    *   Append new entries to the **top** of the file in reverse-chronological order.
    *   Include the **Build Number** (found in `build_count.json`) and the **Version String** (constructed in `common/constants.py`).
    *   **Daily Merging**: If `CHANGELOG.md` already contains an entry for today's date, merge the new summaries into the existing entry instead of creating a new section.
4.  **Commit the Changelog**: Stage and commit the updated `CHANGELOG.md` before executing the push.
5. Execute ```git push origin main```. 

### Format Example
```markdown
## YYYY-MM-DD
### Build XXX
### Version XXX
- **Category**: Concise summary of the change and its impact.
```

# Have a safe push!
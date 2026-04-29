# GitHub Workflow Guide (Team Standard)

This guide defines a **simple, clean, and professional workflow** for our team.

Our goals:

* Avoid merge conflicts
* Keep work organized
* Track progress using Issues
* Maintain clean Pull Requests (PRs)

---

# Core Principle

> **1 Issue = 1 Branch = 1 Pull Request**

Each task is:

* Tracked as a GitHub Issue
* Implemented in a separate branch
* Merged through a Pull Request

---

# Branch Structure

```text
main
└── develop
    └── feature/data-shahd
        ├── feature/data-load-7
        ├── feature/preprocessing-10

    └── feature/training-ibrahim
        ├── feature/train-baseline-16

    └── feature/deployment-islam
        ├── feature/api-22
```

---

# Workflow Overview

1. Pick an Issue
2. Sync your repository
3. Create a new branch from `develop`
4. Implement the task
5. Commit and push
6. Open a Pull Request to `develop`
7. Get review and merge

---

# Step-by-Step Workflow

## 1) Sync Your Repository

Always start by updating `develop`.

```bash
git checkout develop
git pull origin develop
```

Then switch to your personal branch:

```bash
git checkout feature/deployment-islam
git pull origin develop
```

This ensures:

* You have the latest changes
* Fewer merge conflicts
* Clean history

---

## 2) Create a New Branch (Per Issue)

Create a new branch **from develop**.

```bash
git checkout develop
git pull origin develop

git checkout -b feature/<task-name>-<issueID>
```

### Examples

```bash
feature/data-loading-7
feature/gitignore-8
feature/training-baseline-16
feature/contributing-43
```

---

## 3) Work on Your Task

✔ Write code
✔ Add features
✔ Fix bugs

### Rules

* Keep changes focused on the issue
* Do NOT mix multiple tasks in one branch
* Avoid modifying unrelated files
* Commit frequently

---

## 4) Commit Your Changes

```bash
git add .
git commit -m "<type>: short description (#issueID)"
```

### Commit Types

```text
feat   → new feature
fix    → bug fix
docs   → documentation
chore  → maintenance
test   → tests
config → configuration
deploy → deployment
```

### Examples

```bash
feat: add data loading pipeline (#7)
chore: update .gitignore (#8)
fix: handle null values (#10)
docs: add CONTRIBUTING guide (#43)
```

---

## 5) Push Your Branch

First time pushing a new branch:

```bash
git push -u origin feature/<task-name>-<issueID>
```

After that:

```bash
git push
```

---

## 6) Create a Pull Request

On GitHub:

FROM:

```text
feature/<task-name>-<issueID>
```

TO:

```text
develop
```

---

# Pull Request Template

## Title

```
[area] short description (#issueID)
```

### Example

```
[docs] Add CONTRIBUTING guide (#43)
```

---

## Description

```
## Description
Explain what this PR does.

## Changes
- Key change 1
- Key change 2

## Related Issue
Fixes #<issueID>
```

---

# 7) Code Review

Before merging:

* Review code carefully
* Check for bugs
* Ensure task is complete
* Resolve all comments
* Confirm CI/CD passes

---

# 8) Merge

Once approved:

* Merge into `develop`
* Delete the branch

The issue will automatically close if linked:

```
Fixes #ID
```

---

# Working on Multiple Issues at the Same Time

Correct workflow:

```bash
feature/deployment-islam   (personal branch)

feature/gitignore-8        → PR #1
feature/contributing-43    → PR #2
feature/dockerfile-50      → PR #3
```

Each issue gets:

* Its own branch
* Its own PR
* Its own review
---
# Role of the Personal Branch

Each team member has a **personal branch** (for example: `feature/deployment-islam`).
This branch serves as a **working workspace**, not a delivery branch.

---

## Purpose of the Personal Branch

The personal branch is used to:

* Save ongoing work safely
* Keep a backup of progress on GitHub
* Experiment or develop features before finalizing them
* Prepare clean changes before creating issue branches

It allows developers to work continuously without affecting the main workflow.

---

## Important Clarification

The personal branch:

* **Is NOT used for Pull Requests**
* **Is NOT merged into `develop`**
* **Is NOT tied to a specific issue**

Instead, it acts as:

> A personal development workspace.

---

## How It Fits Into the Workflow

```text
develop
   │
   └── feature/deployment-islam        (personal branch — workspace)
            │
            ├── feature/api-22         (issue branch → Pull Request)
            ├── feature/docker-25      (issue branch → Pull Request)
            └── feature/logging-30     (issue branch → Pull Request)
```

---

## Typical Usage Flow

1. Work and save progress on your personal branch

```bash
git checkout feature/deployment-islam
git add .
git commit -m "deploy: progress on API"
git push
```

2. When a task is ready, create a new issue branch

```bash
git checkout develop
git pull origin develop

git checkout -b feature/api-22
```

3. Commit only the changes related to that issue

4. Create a Pull Request to `develop`

---

## Why We Use a Personal Branch

Using a personal branch helps:

* Reduce risk of losing work
* Keep Pull Requests clean and focused
* Allow working on multiple tasks safely
* Maintain organized development

---

## Key Rule

> **Personal branch = workspace**
> **Issue branch = deliverable**
> **Develop branch = integration**


---

# If You Already Created a PR and Need Another One

Do NOT reuse the same branch.

Instead:

```bash
git checkout develop
git pull origin develop

git checkout -b feature/<new-task>-<issueID>
```

---

# If Git Shows This Error

```
fatal: The current branch has no upstream branch
```

Run:

```bash
git push -u origin <branch-name>
```

Example:

```bash
git push -u origin feature/contributing-43
```

---

# Branch Rules

## main

* Protected
* No direct pushes
* Requires PR + approvals

---

## develop

* Main working branch
* All features merge here

---

# Important Rules

## Always

* Create a new branch per issue
* Pull latest develop before starting
* Link PR to an issue (`Fixes #ID`)
* Keep PRs small and focused
* Delete branch after merge

---

## Never

* Push directly to main or develop
* Reuse the same branch for multiple issues
* Open PRs without linking an issue
* Mix unrelated changes in one PR

---

# Example Workflow

## Issue #8 — Configure `.gitignore`

```bash
git checkout develop
git pull origin develop

git checkout -b feature/gitignore-8

# make changes

git add .
git commit -m "chore: update .gitignore (#8)"

git push -u origin feature/gitignore-8
```

Create PR → merge → branch deleted → issue closes

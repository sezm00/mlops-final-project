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

#  Branch Structure

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

1. Create / pick an Issue
2. Create a new branch from `develop`
3. Implement the task
4. Commit and push
5. Open a Pull Request to `develop`
6. Get review and merge

---

#  Step-by-Step Workflow

## 1️ Sync your repository

```bash
git checkout develop
git pull origin develop

git checkout feature/deployment-islam
git pull origin develop
```

---

## 2️ Create a new branch (per issue)

```bash
git checkout -b feature/<task-name>-<issueID>
```

### Examples:

```bash
feature/data-loading-7
feature/gitignore-8
feature/training-baseline-16
```

---

## 3️ Work on your task

✔ Write code
✔ Add features
✔ Fix bugs

### Rules:

* Keep changes **focused on the issue**
* Do NOT mix multiple tasks in one branch
* Avoid modifying unrelated files

---

## 4️ Commit your changes

```bash
git add .
git commit -m "<type>: short description (#issueID)"
```

### Examples:

```bash
feat: add data loading pipeline (#7)
chore: update .gitignore (#8)
fix: handle null values (#10)
```

---

## 5️ Push your branch

```bash
git push -u origin feature/<task-name>-<issueID>
```

---

## 6️ Create a Pull Request

On GitHub:

**FROM:** `feature/<task-name>-<issueID>`
**TO:** `develop`

---

##  Pull Request Template

### Title:

```
[area] short description (#issueID)
```

### Example:

```
[data] Add data loading pipeline (#7)
```

---

### Description:

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

## 7️ Code Review

Before merging:

* Review code carefully
* Check for bugs
* Ensure task is complete
* Resolve all comments

---

## 8️ Merge

Once approved:

* Merge into `develop`
* Delete the branch

✔ The issue will automatically close if linked

---

# 🔒 Branch Rules

## `main`

* Protected
* No direct pushes
* Requires PR + approvals

## `develop`

* Main working branch
* All features are merged here via PR

---

# Important Rules

## Always

* Create a **new branch per issue**
* Pull latest `develop` before starting
* Link PR to an issue (`Fixes #ID`)
* Keep PRs small and focused

---

## Never

* Push directly to `main` or `develop`
* Reuse the same branch for multiple issues
* Open PRs without linking an issue
* Mix unrelated changes in one PR

---

# Example Workflow

## Issue #8 — Configure `.gitignore`

```bash
git checkout develop
git pull origin develop

git checkout feature/deployment-islam
git pull origin develop

git checkout -b feature/gitignore-8

# make changes

git add .
git commit -m "chore: update .gitignore (#8)"
git push
```

Create PR → merge → issue closes ✅


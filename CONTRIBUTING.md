# GitHub Workflow Guide

This guide defines a **clean and simple workflow** for our team.  
The goal is to **avoid conflicts, stay organized, and move fast**.

---

# Core Idea

Each developer works on **their own branch**, then creates a **Pull Request to `develop`**.

No extra branches. No file copying. Keep it simple.

---

# Workflow Overview

1. Pull latest changes from develop  
2. Switch to your personal branch  
3. Work on your task  
4. Commit and push your changes  
5. Create a Pull Request to develop  
6. Review and merge  

---

# Branch Structure

develop ← main working branch
├── feature/deployment-islam
├── feature/training-ibrahim
└── feature/data-shahd


---

# Before You Start (Always Sync)

```bash
git checkout develop
git pull origin develop

git checkout feature/<your-branch>
git pull origin develop

```
##### Example:

```bash 

git checkout feature/deployment-islam
git pull origin develop

```
---

# Work on Your Task (Issue)

##### You can:

- Write code
- Add features
- Fix bugs
- Update configs

##### Rules:

- Work only in your assigned module
- Don’t modify unrelated files
- Keep changes focused

---

# Commit Your Changes

Use clear commit messages:

```bash 
<label>: short description
```
##### Examples:

```bash
git commit -m "feature: add model training pipeline"
git commit -m "fix: resolve null exception"
git commit -m "docs: update README"
git commit -m "config: add yaml file"
```
---

# Push Your Work

```bash
git push origin feature/<your-branch>

```
---
# Create a Pull Request
##### Create a Pull Request:

FROM: feature/<your-branch>
TO: develop

## Pull Request Template
### Description

What does this PR do?

### Changes

- Key changes
- Files added/updated

## Related Issue

Fixes #<issue-number>

---

# Code Review
##### Before merging:

- Review code carefully
- Check for bugs
- Ensure no unrelated changes
- Suggest improvements if needed

---

# Merge

##### Once approved:

- Merge into develop

---
# Important Rules
##### Always:
- Pull develop before working
- Work on your own branch
- Commit frequently
- Keep PRs small and clear

##### Never:
- Push directly to develop
- Mix unrelated changes
- Work on someone else’s module without coordination



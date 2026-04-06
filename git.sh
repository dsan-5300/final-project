#!/usr/bin/env bash

set -e

REPO_URL="https://github.com/dsan-5300/final-project.git"
BRANCH="main"

# Optional commit message passed as first argument
COMMIT_MSG="${1:-update project files}"

# Make sure we're in a git repo
if [ ! -d ".git" ]; then
  git init
fi

# Set or fix remote origin
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

# Ensure branch name is main
git branch -M "$BRANCH"

# Add all files, commit, and push
git add .
git commit -m "$COMMIT_MSG" || echo "No new changes to commit."
git push -u origin "$BRANCH"
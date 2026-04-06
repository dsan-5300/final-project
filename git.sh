#!/usr/bin/env bash
set -euo pipefail

BRANCH="main"
REMOTE="origin"
MSG="${1:-update project files}"

echo "Current directory: $(pwd)"
echo

# Make sure this is a git repo
if [ ! -d ".git" ]; then
  echo "Error: this folder is not a git repository."
  echo "Run: git init"
  exit 1
fi

# Make sure origin exists
if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Error: remote '$REMOTE' is not set."
  echo "Run:"
  echo "git remote add origin https://github.com/dsan-5300/final-project.git"
  exit 1
fi

# Make sure branch is main
git branch -M "$BRANCH"

echo "Fetching latest remote changes..."
git fetch "$REMOTE"

echo
echo "Status before staging:"
git status --short
echo

# Refuse to continue if there are unresolved merge conflicts
if git diff --name-only --diff-filter=U | grep -q .; then
  echo "Error: unresolved merge conflicts detected."
  echo "Resolve conflicts first, then run the script again."
  exit 1
fi

# Pull remote changes first
echo "Pulling latest changes with rebase..."
git pull --rebase "$REMOTE" "$BRANCH"

echo
echo "Staging all tracked and untracked project files..."
git add .

echo
echo "Staged changes:"
git diff --cached --name-status
echo

# Stop if nothing changed
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

echo "Creating commit..."
git commit -m "$MSG"

echo "Pushing to $REMOTE/$BRANCH..."
git push "$REMOTE" "$BRANCH"

echo
echo "Push complete."
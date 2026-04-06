#!/usr/bin/env bash
set -e

BRANCH="main"
MSG="${1:-update project files}"

git branch -M "$BRANCH"
git pull --rebase origin "$BRANCH"
git add .
git commit -m "$MSG" || echo "No new changes to commit."
git push origin "$BRANCH"
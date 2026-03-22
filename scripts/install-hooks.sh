#!/usr/bin/env bash
# Install git hooks for The Dark Candle
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_SRC="${REPO_ROOT}/scripts/pre-commit"
HOOK_DST="${REPO_ROOT}/.git/hooks/pre-commit"

cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"
echo "✓ Pre-commit hook installed."

#!/usr/bin/env bash
set -euo pipefail

# Push the current branch directly to a Git URL without configuring a git remote.
# Usage: ./scripts/push_without_remote.sh https://github.com/OWNER/REPO.git [BRANCH]

URL="${1:-}"
BRANCH="${2:-$(git branch --show-current)}"

if [[ -z "$URL" ]]; then
  echo "Usage: $0 <git-url> [branch]" >&2
  exit 2
fi
if [[ -z "$BRANCH" ]]; then
  echo "Cannot determine current branch; pass branch explicitly." >&2
  exit 2
fi

git diff --check

git push "$URL" "HEAD:refs/heads/$BRANCH"

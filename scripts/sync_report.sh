#!/bin/bash
# Sync report/ on GitHub's main with the Overleaf project, in both directions.
#
# Run it from a checkout of main with the Overleaf project added as the remote
# "overleaf":  bash scripts/sync_report.sh
#
# It finds the last report version both sides had, then:
#   only GitHub changed    -> pushes report/ to Overleaf
#   only Overleaf changed  -> commits Overleaf's version to main and pushes it
#   both changed           -> merges the two; on a conflict it stops so you can fix it
# It never overwrites edits it has not merged. git subtree is not used, because
# Overleaf's history does not line up with the repository's.
set -euo pipefail

OVERLEAF_REMOTE=${OVERLEAF_REMOTE:-overleaf}
GITHUB_REMOTE=${GITHUB_REMOTE:-origin}
BRANCH=main
FOLDER=report

stop() { echo "Stopped: $1" >&2; exit 1; }

[ "$(git branch --show-current)" = "$BRANCH" ] || stop "check out $BRANCH first."
git diff --quiet HEAD -- "$FOLDER" || stop "$FOLDER/ has uncommitted changes; commit or discard them first."

git pull --quiet --ff-only "$GITHUB_REMOTE" "$BRANCH"
git fetch --quiet "$OVERLEAF_REMOTE" "$BRANCH"
overleaf_commit=$(git rev-parse FETCH_HEAD)
overleaf_tree=$(git rev-parse "$overleaf_commit^{tree}")
github_tree=$(git rev-parse "HEAD:$FOLDER")

if [ "$overleaf_tree" = "$github_tree" ]; then
  echo "Already in sync."
  exit 0
fi

# The newest version of report/ that both sides had: a version on main that
# Overleaf also had, or the Overleaf version a merge commit on main records
# that it includes (its "Overleaf-tree:" line).
overleaf_trees=$(git log --format=%T "$overleaf_commit")
shared_tree=""
for commit in $(git rev-list HEAD -- "$FOLDER"); do
  for tree in $(git log -1 --format=%B "$commit" | sed -n 's/^Overleaf-tree: //p') \
              $(git rev-parse "$commit:$FOLDER" 2>/dev/null || true); do
    if grep -qx "$tree" <<< "$overleaf_trees"; then
      shared_tree=$tree
      break 2
    fi
  done
done
[ -n "$shared_tree" ] || stop "GitHub and Overleaf never had the same version of $FOLDER/; merge them by hand."

push_to_overleaf() {  # push_to_overleaf <tree>
  local commit
  commit=$(git commit-tree "$1" -p "$overleaf_commit" -m "Sync report from GitHub ($(git rev-parse --short HEAD))")
  git push --quiet "$OVERLEAF_REMOTE" "$commit:$BRANCH"
  echo "Pushed $FOLDER/ to Overleaf."
}

replace_folder_with() {  # replace_folder_with <tree>: puts the tree's files in report/ and stages them
  git rm -rq "$FOLDER"
  git read-tree --prefix="$FOLDER/" -u "$1"
}

if [ "$overleaf_tree" = "$shared_tree" ]; then
  echo "Only GitHub changed."
  push_to_overleaf "$github_tree"
  exit 0
fi

if [ "$github_tree" = "$shared_tree" ]; then
  echo "Only Overleaf changed."
  replace_folder_with "$overleaf_tree"
  git commit --quiet -m "Sync report from Overleaf"
  git push --quiet "$GITHUB_REMOTE" "$BRANCH"
  echo "Committed Overleaf's version to $BRANCH and pushed it to GitHub."
  exit 0
fi

echo "Both changed; merging."
as_commit() { git commit-tree "$1" -m "$2"; }
shared=$(as_commit "$shared_tree" "last shared version")
if merged_tree=$(git merge-tree --write-tree --merge-base="$shared" \
    "$(as_commit "$github_tree" "GitHub")" "$(as_commit "$overleaf_tree" "Overleaf")" | head -n 1); then
  replace_folder_with "$merged_tree"
  git commit --quiet -m "Merge Overleaf edits into the report" -m "Overleaf-tree: $overleaf_tree"
  git push --quiet "$GITHUB_REMOTE" "$BRANCH"
  push_to_overleaf "$merged_tree"
else
  replace_folder_with "$merged_tree"
  git reset --quiet
  echo "The same lines changed on both sides. Fix the <<<<<<< markers in $FOLDER/, then run:" >&2
  echo "  git add -A $FOLDER && git commit -m \"Merge Overleaf edits into the report\" -m \"Overleaf-tree: $overleaf_tree\"" >&2
  echo "  bash scripts/sync_report.sh" >&2
  exit 1
fi

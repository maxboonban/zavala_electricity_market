#!/bin/bash
#
# Push OSCAR results to GitHub. RUN THIS ON A LOGIN NODE.
#
#   ./push_results.sh
#
# Compute nodes have no outbound network, so the `git push` at the end of a
# SLURM script always fails. `git commit` works fine there (it is purely
# local), so the jobs commit and this script does the one step that needs the
# network. Safe to run any time: it is a no-op when there is nothing to send.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

BRANCH=hrithik_final
echo "repo:   $PWD"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"

if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]]; then
    echo "[FAIL] not on $BRANCH. Switch first:  git checkout $BRANCH"
    exit 1
fi

git config user.email "hrithik_nambiar@brown.edu"
git config user.name  "Hrithik Nambiar"

# Sweep up results a job did not commit itself. Two cases produce these:
# a cancelled/timed-out job that never reached its commit, and -- always --
# the running job's own .out/.log, which keep growing after `git add` runs.
# -A so deletions are picked up too.
if [[ -n "$(git status --porcelain outputs/ logs/ figures/ 2>/dev/null)" ]]; then
    echo
    echo "Uncommitted results found:"
    git status --short outputs/ logs/ figures/
    git add -A outputs/ logs/ figures/
    git commit -F - <<MSGEOF
OSCAR results swept up by push_results.sh

Committed from a login node on $(date '+%Y-%m-%d %H:%M:%S'). These are
outputs a SLURM job produced but did not commit itself -- either because it
was cancelled before its commit step, or because it kept writing to its own
log files after that step ran.
MSGEOF
fi

# Anything still dirty is NOT a result, so it is not ours to commit. A rebase
# would fail on it with a message that does not say which file is at fault.
DIRTY=$(git status --porcelain --untracked-files=no 2>/dev/null)
if [[ -n "$DIRTY" ]]; then
    echo
    echo "[FAIL] tracked files outside outputs/, logs/ and figures/ are modified:"
    echo "$DIRTY" | sed 's/^/       /'
    echo
    echo "       These are source edits, not results, so this script will not"
    echo "       commit them for you. Either keep them:"
    echo "           git add -u && git commit"
    echo "       or discard them:"
    echo "           git checkout -- <file>"
    echo "       then re-run ./push_results.sh"
    exit 1
fi

echo
echo "Syncing with origin/$BRANCH (rebase, so OSCAR commits stay on top)..."
if ! git pull --rebase origin "$BRANCH"; then
    echo
    echo "[FAIL] rebase stopped. Resolve, then:"
    echo "         git rebase --continue && ./push_results.sh"
    echo "       or abandon the sync with:  git rebase --abort"
    exit 1
fi

AHEAD=$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo 0)
if [[ "$AHEAD" -eq 0 ]]; then
    echo "Nothing to push — origin/$BRANCH is already up to date."
    exit 0
fi

echo
echo "Pushing $AHEAD commit(s):"
git log --oneline "origin/$BRANCH..HEAD" | sed 's/^/  /'
echo
if git push origin "$BRANCH"; then
    echo
    echo "Pushed. Pull on your laptop with:  git pull origin $BRANCH"
else
    echo
    echo "[FAIL] push rejected. If this is a credentials problem, OSCAR needs"
    echo "       an SSH key or a cached token for GitHub. Check:"
    echo "         git remote -v"
    exit 1
fi

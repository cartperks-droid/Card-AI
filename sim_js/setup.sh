#!/usr/bin/env bash
# Fetch DaddyDrago's engine at the pinned commit (ENGINE_COMMIT) and build the label version of its battle code.
# His repository has no license file, so it is downloaded here and never committed (sim_js/vendor is gitignored).
set -euo pipefail
cd "$(dirname "$0")"
commit=$(tr -d '[:space:]' < ENGINE_COMMIT)
repo=vendor/CardRngExpansionDepths
if [ ! -d "$repo/.git" ]; then
  mkdir -p vendor
  git clone -q https://github.com/daddydrag0/CardRngExpansionDepths "$repo"
fi
git -C "$repo" fetch -q origin "$commit" 2>/dev/null || git -C "$repo" fetch -q origin
git -C "$repo" checkout -q "$commit"
node codemod.mjs "$repo/src/engine/battle-v2.ts" "$repo/src/engine/battle-v2.label.ts"
[ -d node_modules/tsx ] || npm install --no-audit --no-fund --silent
echo "engine ready at $commit"

#!/usr/bin/env bash
# Push the built dashboard to the static Hugging Face Space. A static Space is
# free, and it runs no process, so the Space serves these files and nothing
# else. No corpus file and no key travel with this deploy.
set -euo pipefail

SPACE="${KB_ARENA_HF_SPACE:-xmpuspus/kb-arena}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "$SOURCE_DIR/../.." && pwd)/kb_arena/static"
TOKEN="${HF_TOKEN:-$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || true)}"

if [ -z "$TOKEN" ]; then
  echo "Set HF_TOKEN, or log in so that ~/.cache/huggingface/token exists." >&2
  exit 1
fi

if [ ! -f "$BUNDLE/index.html" ]; then
  echo "No dashboard at $BUNDLE. Run python3 scripts/sync_frontend_bundle.py first." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# A missing Space answers 404, and the prompt setting turns that into a fast
# failure instead of a wait on a credential prompt. Create the Space first.
GIT_TERMINAL_PROMPT=0 git clone --depth 1 "https://huggingface.co/spaces/${SPACE}" "$WORK_DIR/space"

# Clear the old dashboard, so a file that left the build also leaves the Space.
find "$WORK_DIR/space" -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +
cp -R "$BUNDLE"/. "$WORK_DIR/space"/
cp "$SOURCE_DIR/README.md" "$WORK_DIR/space/README.md"

cd "$WORK_DIR/space"
git add -u
git add -- README.md
while IFS= read -r entry; do
  git add -- "$entry"
done < <(cd "$BUNDLE" && ls -A)

if git diff --cached --quiet; then
  echo "The Space already matches this dashboard build. Nothing to push."
  exit 0
fi

git commit -q -m "Deploy the KB Arena static dashboard

Co-Authored-By: Xavier Puspus"

# The token rides in the push URL and never lands in a config file. Any git
# message goes through the redaction below before it reaches the terminal.
if ! push_log=$(git push "https://user:${TOKEN}@huggingface.co/spaces/${SPACE}" HEAD:main 2>&1); then
  echo "${push_log//${TOKEN}/REDACTED}" >&2
  exit 1
fi

echo "Pushed the dashboard and README.md to ${SPACE}."

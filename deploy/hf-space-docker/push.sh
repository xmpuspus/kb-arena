#!/usr/bin/env bash
# Push this directory's README.md and Dockerfile to the Hugging Face Space.
# The Space holds those two files only. Everything it serves comes from the
# published kb-arena wheel, so no corpus file travels with this deploy.
set -euo pipefail

SPACE="${KB_ARENA_HF_SPACE:-xmpuspus/kb-arena}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="${HF_TOKEN:-$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || true)}"

if [ -z "$TOKEN" ]; then
  echo "Set HF_TOKEN, or log in so that ~/.cache/huggingface/token exists." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# A missing Space answers 404, and the prompt setting turns that into a fast
# failure instead of a wait on a credential prompt. Create the Space first.
GIT_TERMINAL_PROMPT=0 git clone --depth 1 "https://huggingface.co/spaces/${SPACE}" "$WORK_DIR/space"
cp "$SOURCE_DIR/README.md" "$SOURCE_DIR/Dockerfile" "$WORK_DIR/space/"
cd "$WORK_DIR/space"
git add README.md Dockerfile

if git diff --cached --quiet; then
  echo "The Space already matches deploy/hf-space-docker. Nothing to push."
  exit 0
fi

git commit -q -m "Deploy the read-only KB Arena demo

Co-Authored-By: Xavier Puspus"

# The token rides in the push URL and never lands in a config file. Any git
# message goes through the redaction below before it reaches the terminal.
if ! push_log=$(git push "https://user:${TOKEN}@huggingface.co/spaces/${SPACE}" HEAD:main 2>&1); then
  echo "${push_log//${TOKEN}/REDACTED}" >&2
  exit 1
fi

echo "Pushed README.md and Dockerfile to ${SPACE}."

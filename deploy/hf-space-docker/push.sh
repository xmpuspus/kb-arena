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
# Clear the old tree first, so a file that left this directory also leaves the
# Space. Copying the two files on top of an existing Space kept whatever else
# was there, including a file an earlier deploy left behind.
find "$WORK_DIR/space" -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +
cp "$SOURCE_DIR/README.md" "$SOURCE_DIR/Dockerfile" "$WORK_DIR/space/"
cd "$WORK_DIR/space"
git add -u
git add -- README.md Dockerfile

if git diff --cached --quiet; then
  echo "The Space already matches deploy/hf-space-docker. Nothing to push."
  exit 0
fi

git commit -q -m "Deploy the read-only KB Arena demo

Co-Authored-By: Xavier Puspus"

# The token stays out of the command line. A URL carrying it shows up in `ps`
# for every user on the machine, and it lands in the shell history. Git reads it
# from the environment through the credential helper below instead. Any git
# message still goes through the redaction before it reaches the terminal.
export HF_PUSH_TOKEN="$TOKEN"
HELPER='!f() { echo username=hf; echo "password=${HF_PUSH_TOKEN}"; }; f'
if ! push_log=$(git -c "credential.helper=${HELPER}" \
    push "https://huggingface.co/spaces/${SPACE}" HEAD:main 2>&1); then
  echo "${push_log//${TOKEN}/REDACTED}" >&2
  exit 1
fi
unset HF_PUSH_TOKEN

echo "Pushed README.md and Dockerfile to ${SPACE}."

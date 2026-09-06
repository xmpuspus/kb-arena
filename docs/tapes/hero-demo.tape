# KB Arena README hero. Every command runs the published 0.11.0 wheel and needs no API key.
# Build the recording environment, then record, both from the repository root:
#   python3 -m venv /tmp/release-verify
#   /tmp/release-verify/bin/pip install kb-arena==0.11.0
#   vhs docs/tapes/hero-demo.tape

Output docs/demo.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 25ms
Set Shell zsh

Hide
Type "setopt interactive_comments"
Enter
Type "export VIRTUAL_ENV_DISABLE_PROMPT=1"
Enter
# Keep INFO logs off stderr, because they collide with the rich progress bars.
Type "export KB_ARENA_LOG_LEVEL=WARNING"
Enter
Type "source /tmp/release-verify/bin/activate"
Enter
# Warm the corpus caches, so the recorded run keeps the pace of a second run.
Type "kb-arena ingest datasets/aws-compute/raw --corpus aws-compute >/dev/null 2>&1"
Enter
Sleep 5s
Type "kb-arena build-vectors --corpus aws-compute --strategy bm25 >/dev/null 2>&1"
Enter
Sleep 6s
Type "kb-arena retriever-lab --corpus aws-compute --strategies bm25 --top-k 5 --min-recall 0 >/dev/null 2>&1"
Enter
Sleep 25s
Type "clear"
Enter
Sleep 500ms
Show

Type "kb-arena --version"
Enter
Sleep 3s

Type "# Public benchmark corpora, each with its licence and its pinned revision"
Enter
Sleep 800ms
Type "kb-arena datasets"
Enter
Sleep 6s

Type "clear"
Enter
Sleep 800ms

Type "# Or score your own documents. BM25 needs no key, so this whole run is free."
Enter
Sleep 1200ms

Type "kb-arena ingest datasets/aws-compute/raw --corpus aws-compute"
Enter
Sleep 3s

Type "kb-arena build-vectors --corpus aws-compute --strategy bm25"
Enter
Sleep 4s

Type "kb-arena retriever-lab --corpus aws-compute --strategies bm25 --top-k 5 --min-recall 0"
Enter
Sleep 9s

Type "kb-arena evidence --check results/run_422209dd/evidence.json"
Enter
Sleep 3s

Type "# Every run carries corpus, questions, code, seed, and command."
Enter
Sleep 2500ms

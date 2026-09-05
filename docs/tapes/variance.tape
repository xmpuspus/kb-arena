# Spread across repeats — regenerate with:
#   vhs docs/tapes/variance.tape
# Output: docs/demo-variance.gif
#
# Uses a fresh KB_ARENA_RESULTS_PATH so the recording shows this run's repeats
# and not every run the repository already holds. Runs the module from this
# checkout, because an editable install points at whichever copy made it.

Output docs/demo-variance.gif

Set FontSize 16
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 30ms
Set Shell zsh

Hide
Type "cd /Users/xavier/Desktop/kb-arena/.worktrees/slice-p6-docs-gap"
Enter
Type "source /Users/xavier/Desktop/kb-arena/.venv/bin/activate 2>/dev/null; setopt interactive_comments"
Enter
Type "export KB_ARENA_RESULTS_PATH=/tmp/kbdemo KB_ARENA_EMBEDDING_PROVIDER=bge; alias kb-arena='python3 -m kb_arena.cli'; clear"
Enter
Sleep 800ms
Show

Type "# One run is a point. Repeat it to tell a difference from noise."
Enter
Sleep 1500ms

Type "kb-arena retriever-lab --corpus aws-compute --strategies bm25 | tail -3"
Enter
Sleep 6000ms

Type "kb-arena variance --corpus aws-compute --metric mean_recall_at_k"
Enter
Sleep 6000ms

Type "# Zero spread here: BM25 is deterministic, so the repeats agree."
Enter
Sleep 3000ms

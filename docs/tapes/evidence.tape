# What a run says about itself. Regenerate from the repository root with:
#   vhs docs/tapes/evidence.tape
# Output: docs/demo-evidence.gif
#
# Runs the module from this checkout, not the installed console script. An
# editable install points at whichever working copy it was made from, and the
# first recording of this tape captured "No such command 'evidence'" because of
# exactly that. The tape carries no `cd` for the same reason: a hard-coded
# worktree path went stale the moment that worktree was removed.

Output docs/demo-evidence.gif

Set FontSize 16
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 30ms
Set Shell zsh

Hide
Type "source .venv/bin/activate 2>/dev/null; setopt interactive_comments; alias kb-arena='python3 -m kb_arena.cli'; clear"
Enter
Sleep 800ms
Show

Type "# A benchmark number is worth what sits beside it."
Enter
Sleep 1500ms

Type "kb-arena evidence --check results/run_b84eba57/evidence.json"
Enter
Sleep 4000ms

Type "# So the bundle says whether the run can be cited, and over which questions."
Enter
Sleep 1500ms

Type "python3 scripts/show_evidence.py"
Enter
Sleep 5000ms

Type "# The run set and the reviewed set match, so the number is citable."
Enter
Sleep 3000ms

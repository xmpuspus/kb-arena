# What a run says about itself — regenerate with:
#   vhs docs/tapes/evidence.tape
# Output: docs/demo-evidence.gif
#
# Runs the module from this checkout, not the installed console script. An
# editable install points at whichever working copy it was made from, and the
# first recording of this tape captured "No such command 'evidence'" because of
# exactly that.

Output docs/demo-evidence.gif

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
Type "source /Users/xavier/Desktop/kb-arena/.venv/bin/activate 2>/dev/null; setopt interactive_comments; alias kb-arena='python3 -m kb_arena.cli'; clear"
Enter
Sleep 800ms
Show

Type "# A benchmark number is worth what sits beside it."
Enter
Sleep 1500ms

Type "kb-arena evidence --check results/run_59b5b60d/evidence.json --corpus aws-compute --run-id 59b5b60d"
Enter
Sleep 4000ms

Type "# So the bundle says whether the run may be cited, and why not."
Enter
Sleep 1500ms

Type "python3 scripts/show_evidence.py"
Enter
Sleep 5000ms

Type "# Nobody reviewed those questions. The run is a signal, not evidence."
Enter
Sleep 3000ms

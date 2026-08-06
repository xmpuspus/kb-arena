# KB Arena historical result. Regenerate from the repository root with:
# vhs docs/tapes/hero-demo.tape

Output docs/demo.gif

Set FontSize 16
Set Width 1200
Set Height 650
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 25ms
Set Shell zsh

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 500ms
Show

Type "echo 'KB Arena | historical retrieval evidence'"
Enter
Sleep 1200ms

Type "echo 'run 855aac4e | 75 questions | top-k 5'"
Enter
Sleep 1400ms

Type "sed -n '5,10p;13,16p' results/run_855aac4e/retriever_lab.md"
Enter
Sleep 5500ms

Type "echo 'Historical sample: compare on your corpus before choosing an architecture'"
Enter
Sleep 2200ms

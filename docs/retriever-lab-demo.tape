Output docs/demo-retriever-lab.gif

Set Shell "zsh"
Set FontSize 14
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set PlaybackSpeed 1.0

Hide
Type "export ANONYMIZED_TELEMETRY=False CHROMA_TELEMETRY=False"
Enter
Type "clear"
Enter
Show

Type "kb-arena retriever-lab --corpus aws-compute --top-k 5 --strategies bm25 --min-recall 0.0"
Sleep 600ms
Enter
Sleep 7s

Type "kb-arena retriever-lab --corpus aws-compute --top-k 5 --strategies contextual_vector --min-recall 0.0"
Sleep 400ms
Enter
Sleep 13s

Type "cat results/run_9beeb4aa/retriever_lab.md"
Sleep 400ms
Enter
Sleep 5s

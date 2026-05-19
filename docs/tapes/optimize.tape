# kb-arena optimize walkthrough — regenerate with:
#   vhs docs/tapes/optimize.tape
# Output: docs/demo-optimize.gif
# Keyless: uses --dry-run + --help only, no API credit burned.

Output docs/demo-optimize.gif

Set FontSize 16
Set Width 1280
Set Height 760
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 32ms
Set Shell zsh

Hide
Type "cd /Users/xavier/Desktop/kb-arena && source .venv/bin/activate && clear"
Enter
Sleep 500ms
Show

Type "kb-arena optimize --help | head -n 18"
Sleep 400ms
Enter
Sleep 2600ms

Type "# Preview the search space + rebuild cost — no API keys needed"
Sleep 800ms
Enter

Type "kb-arena optimize --corpus aws-compute --strategies naive_vector,bm25,knowledge_graph,rerank_vector --top-ks 3,5,10 --chunk-sizes 256,512,1024 --embedding-providers openai,bge --dry-run"
Sleep 500ms
Enter
Sleep 3500ms

Type "# Each strategy only sweeps the dims it actually consumes."
Sleep 1200ms
Enter
Type "# Run for real:  kb-arena optimize --corpus aws-compute --top-ks 3,5,10 --metric ndcg"
Sleep 2200ms
Enter

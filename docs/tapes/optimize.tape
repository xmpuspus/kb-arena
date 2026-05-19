# kb-arena optimize walkthrough — regenerate with:
#   vhs docs/tapes/optimize.tape
# Output: docs/demo-optimize.gif
# Keyless: shows the corpus-prep recipe (not executed — needs API credit),
# then runs the real keyless --dry-run preview.

Output docs/demo-optimize.gif

Set FontSize 16
Set Width 1280
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 28ms
Set Shell zsh

Hide
Type "cd /Users/xavier/Desktop/kb-arena && source .venv/bin/activate && setopt interactive_comments && clear"
Enter
Sleep 500ms
Show

Type "# 1. Prepare the corpus once (same pipeline as benchmark/retriever-lab):"
Sleep 900ms
Enter
Type "#    kb-arena ingest datasets/my-docs/raw/ --corpus my-docs"
Sleep 500ms
Enter
Type "#    kb-arena build-vectors --corpus my-docs"
Sleep 500ms
Enter
Type "#    kb-arena build-graph --corpus my-docs        # only for knowledge_graph/hybrid"
Sleep 500ms
Enter
Type "#    kb-arena generate-questions --corpus my-docs --count 50"
Sleep 500ms
Enter
Type "#    kb-arena label-chunks --corpus my-docs       # chunk-level IR ground truth"
Sleep 1400ms
Enter

Type "# 2. Preview the search space + rebuild cost — no keys, no corpus prep needed:"
Sleep 1000ms
Enter
Type "kb-arena optimize --corpus aws-compute --strategies naive_vector,bm25,knowledge_graph,rerank_vector --top-ks 3,5,10 --chunk-sizes 256,512,1024 --embedding-providers openai,bge --dry-run"
Sleep 500ms
Enter
Sleep 3500ms

Type "# 3. Run it on the prepared corpus:"
Sleep 800ms
Enter
Type "#    kb-arena optimize --corpus my-docs --top-ks 3,5,10 --metric ndcg"
Sleep 2200ms
Enter

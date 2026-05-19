# kb-arena optimize walkthrough — regenerate with:
#   vhs docs/tapes/optimize.tape
# Output: docs/demo-optimize.gif
# Mixes the keyless --dry-run preview with a real BM25 sweep on the bundled
# aws-compute corpus so the GIF carries actual NDCG numbers, not just a plan.

Output docs/demo-optimize.gif

Set FontSize 15
Set Width 1280
Set Height 820
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 26ms
Set Shell zsh

Hide
Type "cd /Users/xavier/Desktop/kb-arena && source .venv/bin/activate && setopt interactive_comments && clear"
Enter
Sleep 400ms
Show

Type "# 1. Prepare the corpus once (same pipeline as benchmark / retriever-lab):"
Sleep 700ms
Enter
Type "#    kb-arena ingest datasets/my-docs/raw/ --corpus my-docs"
Sleep 350ms
Enter
Type "#    kb-arena build-vectors --corpus my-docs"
Sleep 350ms
Enter
Type "#    kb-arena build-graph --corpus my-docs        # only for knowledge_graph / hybrid"
Sleep 350ms
Enter
Type "#    kb-arena generate-questions --corpus my-docs --count 50"
Sleep 350ms
Enter
Type "#    kb-arena label-chunks --corpus my-docs       # chunk-level IR ground truth"
Sleep 1000ms
Enter

Type "# 2. Preview the search space — keyless, instant, no corpus needed:"
Sleep 700ms
Enter
Type "kb-arena optimize --corpus aws-compute --strategies bm25,naive_vector,contextual_vector,raptor --top-ks 3,5,10 --dry-run"
Sleep 300ms
Enter
Sleep 3000ms

Type "# 3. Real run on the bundled aws-compute corpus (75 questions, top-k sweep):"
Sleep 900ms
Enter
Type "kb-arena optimize --corpus aws-compute --strategies bm25 --top-ks 3,5,10 --metric ndcg"
Sleep 300ms
Enter
Sleep 4500ms

Type "# Top-k=10 lifts BM25 NDCG by +0.031 (+11% relative). Full 4-strategy sweep"
Sleep 900ms
Enter
Type "# (bm25, naive_vector, contextual_vector, raptor) is in the README."
Sleep 2200ms
Enter

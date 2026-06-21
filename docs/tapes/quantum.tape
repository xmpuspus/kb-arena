# kb-arena quantum strategies (#10 qiss, #11 sqr) — regenerate with:
#   vhs docs/tapes/quantum.tape
# Output: docs/demo-quantum.gif
# Runs the real quantum-diagnostics command on the bundled aws-compute corpus
# so the GIF carries actual honest-caveat numbers (PCA variance per qubit
# count, SWAP-test error vs shots, quantum overhead), not a mockup.

Output docs/demo-quantum.gif

Set FontSize 14
Set Width 1200
Set Height 1000
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 24ms
Set Shell zsh

Hide
Type "cd /Users/xavier/Desktop/kb-arena && source .venv/bin/activate && setopt interactive_comments && clear"
Enter
Sleep 400ms
Show

Type "# Strategies #10 qiss (quantum-inspired, pure NumPy) and #11 sqr (Qiskit Aer SWAP test)"
Sleep 600ms
Enter
Type "# rerank naive_vector candidates and flow through the same IR metrics. Honest caveats:"
Sleep 1200ms
Enter

Type "kb-arena quantum-diagnostics --corpus aws-compute --sample-questions 2"
Sleep 300ms
Enter
Sleep 16000ms

Type "# 4-qubit/16-dim amplitude encoding keeps ~0.68 of corpus variance; SWAP error"
Sleep 800ms
Enter
Type "# falls ~1/sqrt(shots); statevector mode is exact. aws-compute Recall@5: qiss/sqr 0.35."
Sleep 2500ms
Enter

# KB Arena own-documents ingestion. Regenerate from the repository root with:
# vhs docs/tapes/own-docs.tape

Output docs/demo-own-docs.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 25ms
Set Shell zsh

Hide
Type "source .venv/bin/activate && cd `mktemp -d` && clear"
Enter
Sleep 500ms
Show

Type "kb-arena init-corpus handbook"
Enter
Sleep 5000ms

Type "printf '# Operations Handbook\n\n## Incident review\nRecord the owner, impact, timeline, and follow-up actions.\n' > datasets/handbook/raw/handbook.md"
Enter
Sleep 1800ms

Type "kb-arena ingest datasets/handbook/raw --corpus handbook"
Enter
Sleep 5000ms

Type "jq -r .title datasets/handbook/processed/documents.jsonl && wc -l datasets/handbook/processed/documents.jsonl"
Enter
Sleep 2200ms

Type "echo 'Next: configure providers, then run kb-arena run --corpus handbook'"
Enter
Sleep 2600ms

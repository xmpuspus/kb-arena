# Retriever Lab Demo Validation

**Source:** `docs/retriever-lab-demo.tape`
**Output:** `docs/demo-retriever-lab.gif` (343 KB, 784 frames)

## Keyframe inspection (frame-by-frame)

Frames extracted with `ffmpeg -i docs/demo-retriever-lab.gif -vsync 0 /tmp/rlab-frames/%04d.png`.
Six keyframes inspected by reading the PNG into context.

| % | Frame | Expected | Observed | Pass |
|---|-------|----------|----------|------|
| 1 | 0008 | First keystroke of `kb-arena retriever-lab` | Prompt with "k" being typed | YES |
| 20 | 0156 | BM25 result table — Recall@5=0.275, MRR=0.352, NDCG@5=0.278 (n=75) | Exactly that table, plus run dir written | YES |
| 40 | 0314 | Second command being typed — `--strategies contextual_vector` | Yes, command typing in progress | YES |
| 60 | 0470 | contextual_vector partial table (~17 questions in) | Partial table at n=17, R@5=0.360 — live updates working | YES |
| 80 | 0628 | contextual_vector complete (n=75); cat command being typed | Final n=75, R@5=0.355 visible | YES |
| 99 | 0780 | Multi-strategy markdown table from run 9beeb4aa rendered | Eight-strategy table visible: naive_vector / contextual_vector / qna_pairs / knowledge_graph / hybrid / raptor / pageindex / bm25 with their numbers | YES |

## Numerical sanity

- contextual_vector final aggregate matches `results/run_9beeb4aa/retriever_lab.md`:
  R@5=0.355, P@5=0.245, Hit@5=0.467, MRR=0.433, NDCG@5=0.388
- BM25 final aggregate matches: R@5=0.275, MRR=0.352, NDCG@5=0.278

## Repro

```bash
vhs docs/retriever-lab-demo.tape
ffmpeg -i docs/demo-retriever-lab.gif -vsync 0 /tmp/rlab-frames/%04d.png
# Read /tmp/rlab-frames/{0008,0156,0314,0470,0628,0780}.png
```

## Notes

- Telemetry warnings from chromadb 0.5.23 suppressed in `kb_arena/benchmark/retriever_lab.py`
  via `ANONYMIZED_TELEMETRY=False` plus logger level adjustment. Without this, every Chroma
  client init prints a `Failed to send telemetry event` warning.
- Live Rich.Table updates only render when stdout is a TTY. VHS provides one, so the demo shows
  progressive table updates. In CI / piped contexts, retriever-lab falls back to plain progress
  prints (`bm25: n=75, mean Recall@5=0.275`).

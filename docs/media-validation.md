# Canonical media validation

Validated on 2026-08-05. The README references only the three assets below.

## Capture environment

- KB Arena 0.10.0 release candidate from the repository virtual environment
- VHS 0.11.0
- FFmpeg 8.1
- Node.js 25.2.1
- Google Chrome 150.0.7871.189

## Historical retrieval recording

- Asset: `docs/demo.gif`
- Source: `docs/tapes/hero-demo.tape`
- Duration: 15.48 seconds
- SHA-256: `424b6ceaa630bba8be955519198719014eff6e928d3184bcd40a23ca5ea7be5a`
- Command: `vhs docs/tapes/hero-demo.tape`
- Numerical source: `results/run_855aac4e/retriever_lab.json` and its generated Markdown report
- Reviewed frames: 1.5, 7.0, 13.0, and 15.1 seconds

The reviewed frames show run `855aac4e`, 75 questions, top-k 5, and the six mapped strategy rows.
The displayed Recall@5, P@5, Hit@5, MRR, and NDCG@5 values match the report. Q&A Pairs and
Knowledge Graph are not shown because their zero chunk scores came from incomplete identifier
mappings in this historical run.

## Own-documents recording

- Asset: `docs/demo-own-docs.gif`
- Source: `docs/tapes/own-docs.tape`
- Duration: 26.24 seconds
- SHA-256: `9fa37e80a7e9286ee2326c7a3a98e57f6a8a3f95e2b35b33f0826fea219ee22a`
- Command: `vhs docs/tapes/own-docs.tape`
- Reviewed frames: 3.0, 12.0, 19.0, and 25.0 seconds

The tape uses a temporary directory, creates an `Operations Handbook` Markdown file, and runs the
real `init-corpus` and `ingest` commands. The final frame confirms one document and two sections.

## Historical benchmark visual

- Asset: `docs/benchmark-evidence.png`
- Source: `scripts/render_benchmark_evidence.mjs`
- Dimensions: 1200 by 675 pixels
- SHA-256: `aefc2c94c1400e6bc7e11ab94007dff3219910acebf76cde8cc30d104260b55a`
- Data source: `results/run_855aac4e/retriever_lab.json`

Regenerate the HTML with `node scripts/render_benchmark_evidence.mjs`, then capture it with a
1200 by 675 headless browser viewport. The full-resolution PNG was inspected after capture. Its six
rows and three metrics match the JSON to three decimal places, and the interpretation limit remains
visible at README size.

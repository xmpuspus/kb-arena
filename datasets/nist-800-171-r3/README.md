# NIST SP 800-171 Revision 3 corpus

This corpus supports a larger, source-auditable retrieval comparison than the bundled AWS sample.
It uses the official NIST publication about protecting Controlled Unclassified Information in
nonfederal systems and organizations.

## Source and license

- Publication: NIST SP 800-171 Revision 3, May 2024
- DOI: <https://doi.org/10.6028/NIST.SP.800-171r3>
- Snapshot date: 2026-08-05
- SHA-256: `7ff0e0301a248f820b0779509bfbdd1369c5fb10592c4d8abc0d4b769bee0acf`
- License statement: not subject to copyright in the United States

See `source-manifest.json` for the source URL and NIST license policy.

## Transform

Run:

```bash
python3 scripts/build_nist_corpus.py
```

The script creates one `Document` for each official control identifier. It keeps the official
family, control text, discussion, references, source anchor, and explicit cross-control references.
It does not add inferred relationships.

## Evaluation status

The question set is a machine-generated draft. Each question has a source anchor, rationale, answer,
qrel, split, and review status. Do not publish a benchmark winner from this corpus until a human
reviewer checks every question and changes `review_status` to `human-reviewed`.

The planned distribution is:

| Type | Count |
|---|---:|
| Direct control lookup | 20 |
| Paraphrased control question | 20 |
| Applied scenario | 20 |
| Boundary or negative question | 10 |
| Cross-control or multi-hop question | 10 |

The split is 48 development, 12 validation, and 20 holdout questions. Do not tune on the holdout.

This corpus is an evaluation fixture. It is not compliance advice.

# Dataset adapters

A benchmark number is only as citable as the corpus under it. An adapter turns a
public dataset into a KB Arena corpus and records what a reader needs in order
to check the result.

```bash
kb-arena datasets                              # what is available, and its terms
kb-arena datasets --name crag --destination ~/data/crag
```

## What an adapter records

Every adapter emits a `DatasetManifest` beside the corpus it produces:

| Field | Why a reader needs it |
|---|---|
| `attribution` | Who made the data, cited as they ask |
| `source_url` | Where to check the original |
| `revision` | Which version was scored |
| `license`, `license_url` | Whether the reader may use it the same way |
| `checksum_sha256` | Whether what arrived is what the manifest describes |
| `preprocessing_version` | What this repository did before scoring |
| `split_rules` | How a question lands in development, validation or holdout |
| `documents`, `questions` | The size, without loading the corpus |

## Three rules the code enforces

**A moving revision is refused.** `latest`, `HEAD`, `main` and `master` are all
rejected. A run against a moving target cannot be repeated, so it cannot be
cited. Name a tag, a commit, or a dated snapshot.

**A checksum mismatch stops the run.** A corpus that is not what its manifest
describes is not evidence of anything.

**A non-commercial licence is never vendored.** Some datasets let a user
download and use them, and do not let this project redistribute them. CRAG is
published under CC BY-NC 4.0. Its adapter is marked download-only, ships no
data, fetches nothing on your behalf, and refuses to write inside this
checkout. Fetching it is your act under the licence you accept, not this
project's.

That last rule is enforced by where the file lands, not by a comment, because a
comment does not stop anything:

```
$ kb-arena datasets --name crag --destination ./datasets/crag
crag is download-only under its licence, and .../datasets/crag is inside the
repository. Point --destination outside the checkout.
```

## Adding an adapter

Subclass `DatasetAdapter`, provide `name`, `manifest_template` and `build`, and
register it in `kb_arena/adapters/__init__.py`. The base class carries the
licence check, the checksum check and the digest helper, so an adapter cannot
skip them by leaving them out.

Tests live in `tests/adapters/` and commit no data. A dataset large enough to be
worth adapting is too large to commit, so the fixtures are small files written
in the test and what is checked is the contract.

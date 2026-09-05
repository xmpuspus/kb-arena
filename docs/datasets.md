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

## The adapters this repository ships

| Adapter | Licence | Revision pin | Note |
|---|---|---|---|
| `crag` | CC-BY-NC-4.0 | `v1.0.0` | Non-commercial. Never vendored. |
| `multihop-rag` | ODC-BY-1.0 | commit `71ac0d0b` | News corpus plus multi-hop questions. |
| `frames` | Apache-2.0 | commit `58d9fb63` | Questions over Wikipedia pages, no separate corpus file. |
| `bright` | CC-BY-4.0 | commit `3066d29c` | Twelve reasoning domains, chunked article ids. |
| `beir-scifact` | ODC-BY-1.0 (corpus), CC-BY-4.0 (claims) | zip md5 `5f7d1de6` | One BEIR slice, licensed by its own source, not by BEIR. |
| `miracl` | CC-BY-SA-4.0 | corpus commit `d921ec7e`, topics commit `5be20db9` | Sixteen Wikipedia-language slices. Packaging is tagged Apache-2.0, but the passage text is Wikipedia's, so its licence is what is recorded. |
| `longbench-v2` | Apache-2.0 | commit `2b48e494` | Long-context multiple choice, one inline context per question. |

Every adapter above is download-only, the same as `crag`. Fetching the data,
whatever the licence allows, stays the user's own act. Each adapter's
`build` refuses and names the source, the pinned revision, and the licence.

Revisions marked "commit" are the dataset's own Hugging Face repository
commit hash, read from that repository's own API. `beir-scifact` pins the
md5 its own BEIR README names for `scifact.zip`, because that corpus zip
carries no software release tag of its own; a package version tag such as
`v2.2.0` would name the BEIR pip package, not the zip's bytes. None of these
values is `latest`, `main`, or invented.

**BEIR covers one slice on purpose.** BEIR repackages many source
collections and does not relicense them, so each slice needs its own
confirmed licence. Only `scifact` publishes its terms directly, in
`allenai/scifact`'s own `LICENSE.md`. NFCorpus publishes no licence on its
homepage. TREC-COVID's source corpus, CORD-19, tags licence per document,
from `cc0` to `no-cc`, so no single licence describes it. FiQA and Quora's
original terms could not be confirmed from a page this project could read.
Those five are left out, rather than shipped with a guessed licence.

**MIRACL's `license` field names the corpus, not the package.** MIRACL's
own Hugging Face card tags its packaging Apache-2.0. The passage text in
every language slice is drawn from Wikipedia, which Wikipedia itself
licenses CC BY-SA 4.0. A run scores against the passage text, so that is the
licence recorded, and the packaging licence is named in the attribution
field instead.

## Adding an adapter

Subclass `DatasetAdapter`, provide `name`, `manifest_template` and `build`, and
register it in `kb_arena/adapters/__init__.py`. The base class carries the
licence check, the checksum check and the digest helper, so an adapter cannot
skip them by leaving them out.

Tests live in `tests/adapters/` and commit no data. A dataset large enough to be
worth adapting is too large to commit, so the fixtures are small files written
in the test and what is checked is the contract.

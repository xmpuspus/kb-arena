"""BEIR: a benchmark suite this repository may point at and must fetch nowhere near.

BEIR (Thakur et al., 2021) repackages many retrieval test collections into one
format. BEIR does not relicense the collections it repackages, so this adapter
records the licence of the underlying source, not a blanket BEIR licence.

Of the collections BEIR ships, this adapter covers only `scifact`: its source
repository states its terms directly, in `allenai/scifact`'s own `LICENSE.md`. The
other common slices (NFCorpus, FiQA, SciDocs, Quora, TREC-COVID) either publish no
licence on their own homepage or publish a heterogeneous, per-document one, such as
TREC-COVID's CORD-19 corpus, so they are left out rather than given a guessed
licence.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

BEIR_URL = "https://github.com/beir-cellar/beir"
# scifact.zip is not tagged by a BEIR software release; the pip package version
# and the corpus zip on the TU Darmstadt host are two different things. BEIR's own
# README names an md5 for the zip, so that md5 is the pin: it names the exact
# bytes a run would score, the way a software release tag cannot.
SCIFACT_ZIP_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_ZIP_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
BEIR_REVISION = f"scifact.zip-md5-{SCIFACT_ZIP_MD5}"

SCIFACT_URL = "https://github.com/allenai/scifact"
# scifact's own LICENSE.md splits the corpus from the claims. The corpus, which
# is what a retrieval run scores against, is what this adapter's `license` field
# names. The claims carry a separate, more permissive licence, named here instead
# of guessed into a single field.
SCIFACT_CORPUS_LICENSE = "ODC-BY-1.0"
SCIFACT_CLAIMS_LICENSE = "CC-BY-4.0"


class BeirScifactAdapter(DatasetAdapter):
    """Download-only. This repository fetches and bundles no dataset automatically."""

    download_only = True

    @property
    def name(self) -> str:
        return "beir-scifact"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "BEIR: A Heterogeneous Benchmark, Thakur et al. (2021), repackaging "
                "SciFact, Wadden et al. (2020). The corpus abstracts are part of the "
                f"Semantic Scholar S2ORC dataset, under {SCIFACT_CORPUS_LICENSE}. The "
                f"claims are Allen AI's own, under {SCIFACT_CLAIMS_LICENSE}. Cite both "
                "papers, as their dataset cards ask."
            ),
            source_url=BEIR_URL,
            revision=BEIR_REVISION,
            license=SCIFACT_CORPUS_LICENSE,
            license_url="https://opendatacommons.org/licenses/by/1-0/",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                "train": "qrels/train.tsv relevance judgments",
                "test": "qrels/test.tsv relevance judgments, the scored benchmark split",
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead."""
        raise NotImplementedError(
            "beir-scifact is download-only. KB Arena will not fetch or bundle it. "
            f"Download {SCIFACT_ZIP_URL} yourself and check its md5 against "
            f"{SCIFACT_ZIP_MD5}, the digest BEIR's own README names for this file. "
            f"See {SCIFACT_URL} for the original corpus and its licence terms, "
            "already named in the manifest, and point --destination at a directory "
            "outside this checkout."
        )

    @staticmethod
    def parse_document(raw: dict) -> dict:
        """Turn one `corpus.jsonl` record into a section.

        A BEIR corpus record is one abstract with no further sub-sectioning, so
        `source_doc_id` is the record's own `_id` and the section is always "0".
        """
        return {
            "source_doc_id": raw["_id"],
            "source_section_id": "0",
            "title": raw.get("title", ""),
            "text": raw["text"],
        }

    @staticmethod
    def parse_qrel(raw: dict) -> dict:
        """Turn one qrels row into a judgment, keeping the corpus id it points at.

        `source_doc_id` here is the same `_id` `parse_document` emits for the
        corpus record the query is judged against.
        """
        return {
            "query_id": raw["query-id"],
            "source_doc_id": raw["corpus-id"],
            "source_section_id": "0",
            "relevance": int(raw["score"]),
        }

"""MIRACL: a benchmark this repository may point at and must fetch nowhere near.

MIRACL (Zhang et al., 2022) is a multilingual retrieval dataset built from
Wikipedia dumps, one language slice at a time. Its own Hugging Face card tags the
packaging as Apache-2.0, but the passage text in every slice is Wikipedia article
text, which Wikipedia itself publishes under CC BY-SA 4.0. That is the licence that
governs the bytes a run would score against, so it is what this adapter's `license`
field names; the packaging licence is recorded in the attribution instead.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

MIRACL_CORPUS_URL = "https://huggingface.co/datasets/miracl/miracl-corpus"
MIRACL_TOPICS_URL = "https://huggingface.co/datasets/miracl/miracl"
# The corpus and the topics/qrels live in two separate Hugging Face repos, each
# with its own commit history, so each needs its own pin. Both are confirmed from
# the repos' own API (`sha` field). The manifest's `revision` field names the
# corpus, since that is what `license` and `checksum_sha256` describe.
MIRACL_CORPUS_REVISION = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"
MIRACL_TOPICS_REVISION = "5be20db9509754dadad47689368639fcec739c00"

# The corpus text is Wikipedia article text, so Wikipedia's own licence governs it,
# not the Apache-2.0 tag MIRACL's own packaging carries.
MIRACL_CORPUS_LICENSE = "CC-BY-SA-4.0"
MIRACL_PACKAGING_LICENSE = "Apache-2.0"

# The 16 "known languages" MIRACL has released. Two "surprise languages" are
# withheld by the dataset's own card, so they are not listed here.
MIRACL_LANGUAGES = (
    "ar",
    "bn",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "id",
    "ja",
    "ko",
    "ru",
    "sw",
    "te",
    "th",
    "zh",
)


class MiraclAdapter(DatasetAdapter):
    """Download-only. This repository fetches and bundles no dataset automatically."""

    download_only = True

    @property
    def name(self) -> str:
        return "miracl"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "MIRACL: Multilingual Information Retrieval Across a Continuum of "
                "Languages, Zhang et al. (2022). The corpus text is Wikipedia "
                f"content under {MIRACL_CORPUS_LICENSE}; MIRACL's own packaging of "
                f"queries and judgments is tagged {MIRACL_PACKAGING_LICENSE}. Cite "
                "the paper and miracl/miracl-corpus on Hugging Face, as the dataset "
                "card asks."
            ),
            source_url=MIRACL_CORPUS_URL,
            revision=MIRACL_CORPUS_REVISION,
            license=MIRACL_CORPUS_LICENSE,
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                "train": "qrels/train relevance judgments, per language",
                "dev": "qrels/dev relevance judgments, per language",
                "testA": "qrels/testA, only for the languages MIRACL shares with Mr. TyDi",
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead."""
        raise NotImplementedError(
            "miracl is download-only. KB Arena will not fetch or bundle it. Download "
            f"a language slice yourself from {MIRACL_CORPUS_URL} at revision "
            f"{MIRACL_CORPUS_REVISION}, and its topics and qrels from "
            f"{MIRACL_TOPICS_URL} at revision {MIRACL_TOPICS_REVISION}. Accept the "
            "licence terms named in the manifest, and point --destination at a "
            "directory outside this checkout."
        )

    @staticmethod
    def parse_document(raw: dict) -> dict:
        """Turn one corpus record into a section, splitting MIRACL's own docid.

        MIRACL names each passage `X#Y`, where `X` is the Wikipedia article and `Y`
        is the passage's position in it. Those become `source_doc_id` and
        `source_section_id`.
        """
        doc_id, _, section_id = raw["docid"].partition("#")
        return {
            "source_doc_id": doc_id,
            "source_section_id": section_id,
            "title": raw["title"],
            "text": raw["text"],
        }

    @staticmethod
    def parse_qrel(raw: dict) -> dict:
        """Turn one TREC-format qrels line into a judgment.

        The line is `qid Q0 docid relevance`. `docid` is split the same way
        `parse_document` splits it, so a judgment always names a real section.
        """
        query_id, _, docid, relevance = raw["line"].split()
        doc_id, _, section_id = docid.partition("#")
        return {
            "query_id": query_id,
            "source_doc_id": doc_id,
            "source_section_id": section_id,
            "relevance": int(relevance),
        }

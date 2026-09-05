"""MultiHop-RAG: a benchmark this repository may point at and must fetch nowhere near.

MultiHop-RAG (Tang & Yang, 2024) pairs a news corpus with questions whose answers need
evidence from more than one article. The corpus is published under ODC-BY 1.0, a licence
that permits redistribution with attribution. This repository still does not auto-fetch
or vendor it: fetching stays the user's own act, on their own destination outside the
checkout, so no run here can silently point at an unpinned copy of the data.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

MULTIHOP_RAG_LICENSE = "ODC-BY-1.0"
MULTIHOP_RAG_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG"
# The Hugging Face dataset repo commit this adapter was written against, confirmed
# from the repo's own API (`sha` field). Not "main", so a run against it can be repeated.
MULTIHOP_RAG_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"


class MultiHopRagAdapter(DatasetAdapter):
    """Download-only. This repository fetches and bundles no dataset automatically."""

    download_only = True

    @property
    def name(self) -> str:
        return "multihop-rag"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "MultiHop-RAG, Tang & Yang (2024). Cite the paper and "
                "yixuantt/MultiHopRAG on Hugging Face, as the dataset card asks."
            ),
            source_url=MULTIHOP_RAG_URL,
            revision=MULTIHOP_RAG_REVISION,
            license=MULTIHOP_RAG_LICENSE,
            license_url="https://opendatacommons.org/licenses/by/1-0/",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                "benchmark": (
                    "every one of the released queries; MultiHop-RAG ships one "
                    "evaluation set and no train split of its own"
                ),
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead.

        There is no download here on purpose. This repository does not run automated
        fetches against a dataset's own host, whatever the licence allows, so a corpus
        is only ever the copy the user chose to pull, at the revision they pinned.
        """
        raise NotImplementedError(
            "multihop-rag is download-only. KB Arena will not fetch or bundle it. "
            f"Download it yourself from {MULTIHOP_RAG_URL} at revision "
            f"{MULTIHOP_RAG_REVISION}, accept the ODC-BY 1.0 terms there, and point "
            "--destination at a directory outside this checkout. The manifest template "
            "records the attribution and the terms."
        )

    @staticmethod
    def parse_document(raw: dict) -> dict:
        """Turn one corpus record into a section this repository can score against.

        The corpus config has no sub-document structure, so one article is one
        section. `source_doc_id` is the article URL, the only stable identifier the
        dataset provides.
        """
        return {
            "source_doc_id": raw["url"],
            "source_section_id": "0",
            "title": raw["title"],
            "text": raw["body"],
        }

    @staticmethod
    def parse_question(raw: dict) -> dict:
        """Turn one query record into a question, keeping its evidence trail.

        Each evidence entry names the article URL it came from, so the question
        keeps a link back to the same `source_doc_id` values `parse_document` emits.
        """
        return {
            "question": raw["query"],
            "answer": raw["answer"],
            "question_type": raw["question_type"],
            "source_doc_ids": [evidence["url"] for evidence in raw["evidence_list"]],
        }

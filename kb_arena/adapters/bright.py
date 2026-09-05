"""BRIGHT: a benchmark this repository may point at and must fetch nowhere near.

BRIGHT (Su et al., 2024) is a reasoning-intensive retrieval benchmark across twelve
domains, from biology to competition math. It is published under CC BY 4.0, a
licence that permits redistribution. This repository still does not auto-fetch or
vendor it, for the same reason every adapter here does not: fetching stays the
user's own act.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

BRIGHT_LICENSE = "CC-BY-4.0"
BRIGHT_URL = "https://huggingface.co/datasets/xlangai/BRIGHT"
# The Hugging Face dataset repo commit this adapter was written against, confirmed
# from the repo's own API (`sha` field).
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"

# The twelve domains the "examples" and "documents" configs both split by.
BRIGHT_DOMAINS = (
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
)


class BrightAdapter(DatasetAdapter):
    """Download-only. This repository fetches and bundles no dataset automatically."""

    download_only = True

    @property
    def name(self) -> str:
        return "bright"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive "
                "Retrieval, Su et al. (2024). Cite the paper and xlangai/BRIGHT on "
                "Hugging Face, as the dataset card asks."
            ),
            source_url=BRIGHT_URL,
            revision=BRIGHT_REVISION,
            license=BRIGHT_LICENSE,
            license_url="https://creativecommons.org/licenses/by/4.0/",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                domain: "the documents and examples config for this domain"
                for domain in BRIGHT_DOMAINS
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead."""
        raise NotImplementedError(
            "bright is download-only. KB Arena will not fetch or bundle it. Download "
            f"it yourself from {BRIGHT_URL} at revision {BRIGHT_REVISION}, accept the "
            "CC BY 4.0 terms there, and point --destination at a directory outside "
            "this checkout. The manifest template records the attribution and the "
            "terms."
        )

    @staticmethod
    def _split_chunk_id(chunk_id: str) -> tuple[str, str]:
        """Split `topic/Article_3.txt` into its article id and its chunk number.

        BRIGHT names each chunk of an article with a trailing `_<index>` before the
        extension. The article id is what `parse_document` and `parse_question` share
        as `source_doc_id`, so a chunk and a gold reference to it always agree.
        """
        stem = chunk_id.removesuffix(".txt")
        doc_id, _, section_id = stem.rpartition("_")
        return doc_id, section_id

    @staticmethod
    def parse_document(raw: dict) -> dict:
        """Turn one `documents` config row into a section, keeping the chunk number."""
        doc_id, section_id = BrightAdapter._split_chunk_id(raw["id"])
        return {
            "source_doc_id": doc_id,
            "source_section_id": section_id,
            "text": raw["content"],
        }

    @staticmethod
    def parse_question(raw: dict) -> dict:
        """Turn one `examples` config row into a question and its gold sections."""
        gold = [BrightAdapter._split_chunk_id(gold_id) for gold_id in raw["gold_ids"]]
        return {
            "question": raw["query"],
            "answer": raw["gold_answer"],
            "gold_sections": [
                {"source_doc_id": doc_id, "source_section_id": section_id}
                for doc_id, section_id in gold
            ],
        }

"""LongBench v2: a benchmark this repository may point at and must fetch nowhere near.

LongBench v2 (Bai et al., 2024) is a long-context, multiple-choice benchmark, with
contexts from 8k to 2M words. It ships as one JSON file with no separate corpus, so
each question carries its own context inline. It is published under Apache-2.0, a
licence that permits redistribution. This repository still does not auto-fetch or
vendor it, for the same reason every adapter here does not: fetching stays the
user's own act.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

LONGBENCH_V2_LICENSE = "Apache-2.0"
# The dataset repo moved from THUDM to zai-org on Hugging Face; this is the
# current, live location.
LONGBENCH_V2_URL = "https://huggingface.co/datasets/zai-org/LongBench-v2"
# The Hugging Face dataset repo commit this adapter was written against, confirmed
# from the repo's own API (`sha` field).
LONGBENCH_V2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"

# The context is one long blob per question. This adapter cuts it into fixed-size
# sections so a question's evidence can name a section instead of the whole context.
_SECTION_CHARS = 4000


class LongBenchV2Adapter(DatasetAdapter):
    """Download-only. This repository fetches and bundles no dataset automatically."""

    download_only = True

    @property
    def name(self) -> str:
        return "longbench-v2"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "LongBench v2: Towards Deeper Understanding and Reasoning on "
                "Realistic Long-context Multitasks, Bai et al. (2024). Cite the "
                "paper and zai-org/LongBench-v2 on Hugging Face (formerly published "
                "as THUDM/LongBench-v2), as the dataset card asks."
            ),
            source_url=LONGBENCH_V2_URL,
            revision=LONGBENCH_V2_REVISION,
            license=LONGBENCH_V2_LICENSE,
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                "easy": "every question where difficulty == easy",
                "hard": "every question where difficulty == hard",
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead."""
        raise NotImplementedError(
            "longbench-v2 is download-only. KB Arena will not fetch or bundle it. "
            f"Download it yourself from {LONGBENCH_V2_URL} at revision "
            f"{LONGBENCH_V2_REVISION}, accept the Apache-2.0 terms there, and point "
            "--destination at a directory outside this checkout. The manifest "
            "template records the attribution and the terms."
        )

    @staticmethod
    def parse_document(raw: dict) -> list[dict]:
        """Cut one question's inline context into fixed-size sections.

        LongBench v2 ships no corpus file, so the question's own `_id` becomes
        `source_doc_id`, and each chunk of its context gets a sequential
        `source_section_id`.
        """
        context = raw["context"]
        chunks = [context[i : i + _SECTION_CHARS] for i in range(0, len(context), _SECTION_CHARS)]
        return [
            {
                "source_doc_id": raw["_id"],
                "source_section_id": str(index),
                "text": chunk,
            }
            for index, chunk in enumerate(chunks)
        ]

    @staticmethod
    def parse_question(raw: dict) -> dict:
        """Turn one row into a multiple-choice question over its own context."""
        return {
            "question": raw["question"],
            "choices": {
                "A": raw["choice_A"],
                "B": raw["choice_B"],
                "C": raw["choice_C"],
                "D": raw["choice_D"],
            },
            "answer": raw["answer"],
            "difficulty": raw["difficulty"],
            "source_doc_id": raw["_id"],
        }

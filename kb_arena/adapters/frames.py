"""FRAMES: a benchmark this repository may point at and must fetch nowhere near.

FRAMES (Google DeepMind, 2024) is a factuality, retrieval, and reasoning benchmark
built from Wikipedia. It ships questions and the Wikipedia pages that answer them,
with no separate corpus file. It is published under Apache-2.0, a licence that
permits redistribution. This repository still does not auto-fetch or vendor it, for
the same reason every adapter here does not: fetching stays the user's own act.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

FRAMES_LICENSE = "Apache-2.0"
FRAMES_URL = "https://huggingface.co/datasets/google/frames-benchmark"
# The Hugging Face dataset repo commit this adapter was written against, confirmed
# from the repo's own API (`sha` field).
FRAMES_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"

# The dataset stores each gold link in its own numbered column instead of a list.
_LINK_COLUMNS = [f"wikipedia_link_{i}" for i in range(1, 11)] + ["wikipedia_link_11+"]


class FramesAdapter(DatasetAdapter):
    """Download-only. This repository fetches and bundles no dataset automatically."""

    download_only = True

    @property
    def name(self) -> str:
        return "frames"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "FRAMES: Factuality, Retrieval, And reasoning MEasurement Set, "
                "Google DeepMind. Cite the paper and google/frames-benchmark on "
                "Hugging Face, as the dataset card asks."
            ),
            source_url=FRAMES_URL,
            revision=FRAMES_REVISION,
            license=FRAMES_LICENSE,
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                "test": "every question in the single released split, the whole set",
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead."""
        raise NotImplementedError(
            "frames is download-only. KB Arena will not fetch or bundle it. Download "
            f"it yourself from {FRAMES_URL} at revision {FRAMES_REVISION}, accept the "
            "Apache-2.0 terms there, and point --destination at a directory outside "
            "this checkout. The manifest template records the attribution and the "
            "terms."
        )

    @staticmethod
    def parse_question(raw: dict) -> dict:
        """Turn one row into a question and the Wikipedia pages it depends on.

        FRAMES ships no separate corpus, so each gold Wikipedia page stands in for
        one document, with a single section, since the dataset names no passage
        inside the page. `source_doc_id` is the page URL.
        """
        evidence = [
            {"source_doc_id": link, "source_section_id": "0"}
            for column in _LINK_COLUMNS
            if (link := raw.get(column))
        ]
        return {
            "question": raw["Prompt"],
            "answer": raw["Answer"],
            "reasoning_type": raw["reasoning_types"],
            "evidence": evidence,
        }

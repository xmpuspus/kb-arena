"""CRAG: a benchmark this repository may point at and must never carry.

The Comprehensive RAG Benchmark is published under CC BY-NC 4.0. That licence
lets a user download and use the data, and it does not let this project
redistribute it. So the adapter fetches nothing on import, ships nothing, and
refuses to write inside the checkout.
"""

from __future__ import annotations

from pathlib import Path

from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import DatasetManifest

CRAG_LICENSE = "CC-BY-NC-4.0"
CRAG_URL = "https://github.com/facebookresearch/CRAG"


class CragAdapter(DatasetAdapter):
    """Download-only. The licence forbids this repository from shipping the data."""

    download_only = True

    @property
    def name(self) -> str:
        return "crag"

    def manifest_template(self) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            attribution=(
                "CRAG: Comprehensive RAG Benchmark, Meta AI. Cite the authors as the "
                "dataset card asks."
            ),
            source_url=CRAG_URL,
            # A release tag, so a run names the data it scored. The adapter
            # refuses a moving revision, so this cannot quietly become "latest".
            revision="v1.0.0",
            license=CRAG_LICENSE,
            license_url="https://creativecommons.org/licenses/by-nc/4.0/",
            checksum_sha256="0" * 64,
            preprocessing_version="1",
            split_rules={
                "development": "every question the release marks as public",
                "holdout": "reserved, and never opened by an ordinary run",
            },
            cache_path="",
            documents=0,
            questions=0,
        )

    def build(self, destination: Path) -> DatasetManifest:
        """Refuse, and say what the caller has to do instead.

        There is no download here on purpose. Fetching CRAG is the user's act
        under the licence they accepted, not this project's, and an adapter
        that fetched it automatically would blur exactly that line.
        """
        raise NotImplementedError(
            "CRAG is download-only under CC BY-NC 4.0. KB Arena will not fetch or "
            f"bundle it. Download it yourself from {CRAG_URL}, accept the licence "
            "there, and point --destination at a directory outside this checkout. "
            "The manifest template records the attribution and the terms."
        )

"""What a dataset adapter must record before its data can be cited.

A benchmark number is only as citable as the corpus under it. These fields are
the ones a reader needs to answer "what exactly was measured, and may I use
it": who made the data, where it came from, which revision, under what licence,
and what this repository did to it before scoring.

Nothing here is optional by accident. A field that could be left blank would be
left blank, and the manifest would then describe a corpus nobody can reproduce.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator, model_validator

# A licence that forbids commercial use, or forbids redistribution, cannot be
# vendored into this repository. The adapter downloads it at the user's request
# instead. CRAG is the case that forced this: CC BY-NC 4.0.
NON_REDISTRIBUTABLE = frozenset({"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-ND-4.0"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetManifest(BaseModel):
    """The record a dataset adapter writes beside the corpus it produced."""

    # Who to credit, and where a reader goes to check the original.
    name: str = Field(min_length=1)
    attribution: str = Field(min_length=1, description="Who made this data, cited as they ask")
    source_url: str = Field(min_length=1)
    # A tag, a commit, or a dated snapshot. "latest" is not a revision, because
    # a run against "latest" cannot be repeated.
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1, description="SPDX identifier where one exists")
    license_url: str = ""

    # What arrived, and what this repository did to it.
    checksum_sha256: str = Field(min_length=64, max_length=64)
    preprocessing_version: str = Field(min_length=1)
    split_rules: dict[str, str] = Field(
        default_factory=dict,
        description="Split name to the rule that assigns a question to it",
    )
    # Empty in a template, which is everything known BEFORE the data arrives.
    # Required once the manifest describes real data, because a corpus nobody
    # can locate is a corpus nobody can check.
    cache_path: str = ""

    # Counts, so a reader can tell a full corpus from a slice without loading it.
    documents: int = Field(ge=0)
    questions: int = Field(ge=0)
    downloaded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("revision")
    @classmethod
    def _revision_is_pinned(cls, value: str) -> str:
        if value.strip().lower() in {"latest", "head", "main", "master", ""}:
            raise ValueError(
                f"revision {value!r} is a moving target, so a run against it cannot be "
                "repeated. Name a tag, a commit, or a dated snapshot."
            )
        return value

    @field_validator("checksum_sha256")
    @classmethod
    def _checksum_is_a_digest(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("checksum_sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _a_described_corpus_can_be_found(self) -> DatasetManifest:
        if (self.documents or self.questions) and not self.cache_path.strip():
            raise ValueError(
                "a manifest that counts documents or questions must say where they "
                "are. Leave the counts at zero for a template."
            )
        return self

    @property
    def redistributable(self) -> bool:
        """Whether this repository may ship the data, rather than fetch it."""
        return self.license.strip().upper() not in NON_REDISTRIBUTABLE

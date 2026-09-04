"""The contract every dataset adapter follows.

An adapter turns a public dataset into a KB Arena corpus. The rules below are
the ones that keep the result citable, and each exists because getting it wrong
produces a number nobody can defend.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from kb_arena.models.adapter import DatasetManifest


class LicenseRefusalError(RuntimeError):
    """The licence forbids this repository from shipping the data."""


class ChecksumMismatchError(RuntimeError):
    """What arrived is not what the manifest describes."""


def sha256_of(path: Path) -> str:
    """The digest of a file, read in chunks so a large corpus does not load."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetAdapter(ABC):
    """Fetch a public dataset and describe exactly what was fetched.

    A subclass provides `manifest_template` and `build`. Everything that keeps
    the result honest lives here, so an adapter cannot skip it by omission.
    """

    #: Set on a subclass whose licence forbids redistribution. The adapter then
    #: refuses to write inside the repository, whatever the caller asks for.
    download_only: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The corpus name this adapter produces."""

    @abstractmethod
    def manifest_template(self) -> DatasetManifest:
        """Everything known before the data arrives, with a placeholder checksum."""

    @abstractmethod
    def build(self, destination: Path) -> DatasetManifest:
        """Write the corpus under `destination` and return what was written."""

    def check_destination(self, destination: Path, repo_root: Path) -> None:
        """Refuse to write non-redistributable data inside the repository.

        A licence such as CC BY-NC 4.0 permits a user to download the data and
        forbids this project from shipping it. The difference is where the file
        lands, so that is what gets checked, rather than trusting a comment.
        """
        if not self.download_only:
            return
        target = destination.resolve()
        root = repo_root.resolve()
        if target == root or root in target.parents:
            raise LicenseRefusalError(
                f"{self.name} is download-only under its licence, and {target} is "
                f"inside the repository at {root}. Point --destination outside the "
                "checkout. The data stays yours to fetch and is never vendored here."
            )

    def verify(self, path: Path, expected_sha256: str) -> None:
        """Refuse data that does not match the digest the manifest names."""
        actual = sha256_of(path)
        if actual != expected_sha256:
            raise ChecksumMismatchError(
                f"{path} has digest {actual}, and the manifest names {expected_sha256}. "
                "A corpus that is not what its manifest describes cannot be cited."
            )

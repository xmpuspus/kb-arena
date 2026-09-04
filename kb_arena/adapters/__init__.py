"""Adapters that turn a public dataset into a KB Arena corpus."""

from kb_arena.adapters.base import (
    ChecksumMismatchError,
    DatasetAdapter,
    LicenseRefusalError,
    sha256_of,
)
from kb_arena.adapters.crag import CragAdapter

#: Every adapter this package ships, by corpus name.
ADAPTERS: dict[str, type[DatasetAdapter]] = {"crag": CragAdapter}

__all__ = [
    "ADAPTERS",
    "ChecksumMismatchError",
    "DatasetAdapter",
    "LicenseRefusalError",
    "CragAdapter",
    "sha256_of",
]

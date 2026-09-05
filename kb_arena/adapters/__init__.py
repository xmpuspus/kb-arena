"""Adapters that turn a public dataset into a KB Arena corpus."""

from kb_arena.adapters.base import (
    ChecksumMismatchError,
    DatasetAdapter,
    LicenseRefusalError,
    sha256_of,
)
from kb_arena.adapters.beir import BeirScifactAdapter
from kb_arena.adapters.bright import BrightAdapter
from kb_arena.adapters.crag import CragAdapter
from kb_arena.adapters.frames import FramesAdapter
from kb_arena.adapters.longbench import LongBenchV2Adapter
from kb_arena.adapters.miracl import MiraclAdapter
from kb_arena.adapters.multihop_rag import MultiHopRagAdapter

#: Every adapter this package ships, by corpus name.
ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "crag": CragAdapter,
    "multihop-rag": MultiHopRagAdapter,
    "frames": FramesAdapter,
    "bright": BrightAdapter,
    "beir-scifact": BeirScifactAdapter,
    "miracl": MiraclAdapter,
    "longbench-v2": LongBenchV2Adapter,
}

__all__ = [
    "ADAPTERS",
    "ChecksumMismatchError",
    "DatasetAdapter",
    "LicenseRefusalError",
    "BeirScifactAdapter",
    "BrightAdapter",
    "CragAdapter",
    "FramesAdapter",
    "LongBenchV2Adapter",
    "MiraclAdapter",
    "MultiHopRagAdapter",
    "sha256_of",
]
